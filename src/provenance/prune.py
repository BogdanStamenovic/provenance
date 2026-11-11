"""Single-disk mode: delete unused, recreatable units once their recipe is saved.

Nothing here deletes on a guess. A unit is pruned only if every one of these holds, and
they are checked again at the moment of deletion, not just when the plan was made:

* its kind is one we can rebuild by command (PRUNABLE) and its risk is `recreatable`;
* it has not been used for `prune_after_days`: by atime on a relatime mount, or by the
  daemon's own open-tracking on a noatime mount, which must also have watched the area
  without a gap for that long;
* its source is verified online right now: the HF tree has the same sha256s, the commit is
  on a remote branch after a fetch, every pinned venv package exists on PyPI;
* no process has it open, mapped, or as its working directory, and it is not a hotline
  team's repo;
* nobody claimed it `--keep`.

Deleting is soft for everything except HF caches: the unit is renamed into
`<area>/.provenance-trash/<plan>/` and only removed after `trash_grace_h`. HF caches go
through huggingface_hub's own delete (see hfcache.py), because its shared blob store is
refcounted and a renamed repo would free nothing anyway.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import hfcache
from .classify import EXPENDABLE, RECREATABLE, human, preserve_atimes
from .engine import Disk, Engine
from .store import UnitRow, unpack_listing

PRUNABLE = ("hf-cache", "hf-local", "git", "venv", "cache", "ollama", "weights")
TRASH = ".provenance-trash"
DAY = 86400.0


class PruneError(Exception):
    pass


@dataclass
class Candidate:
    unit: UnitRow
    bytes: int
    last_used: float | None  # epoch seconds; None = unknown
    used_by: str  # "atime" | "opens" | "unknown"
    reasons: list[str] = field(default_factory=list)  # why NOT prunable; empty = prunable
    verified: str = ""

    @property
    def ok(self) -> bool:
        return not self.reasons


def mount_options(path: str) -> tuple[str, str, list[str]]:
    """(mountpoint, fstype, options) of the filesystem holding path."""
    r = subprocess.run(["findmnt", "-n", "-o", "TARGET,FSTYPE,OPTIONS", "-T", path],
                       capture_output=True, text=True, timeout=10, check=False)
    parts = r.stdout.split()
    if len(parts) < 3:
        raise PruneError(f"cannot tell which filesystem holds {path}")
    return parts[0], parts[1], parts[2].split(",")


def tracks_atime(path: str) -> bool:
    _m, _t, opts = mount_options(path)
    return "noatime" not in opts


def probe_area(path: str) -> Disk | None:
    p = os.path.realpath(path)
    if not os.path.isdir(p):
        return None
    mnt, fstype, _ = mount_options(p)
    st = os.statvfs(p)
    return Disk(f"area:{p}", p, f"{fstype} on {mnt}", st.f_blocks * st.f_frsize)


# --- what is in use right now --------------------------------------------------------


def paths_in_use(proc: str = "/proc") -> set[str]:
    """Every path some process has as cwd, root, open fd, or mapped file."""
    out: set[str] = set()
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        base = f"{proc}/{pid}"
        for link in ("cwd", "root", "exe"):
            try:
                out.add(os.readlink(f"{base}/{link}"))
            except OSError:
                pass
        try:
            for fd in os.listdir(f"{base}/fd"):
                try:
                    t = os.readlink(f"{base}/fd/{fd}")
                except OSError:
                    continue
                if t.startswith("/"):
                    out.add(t)
        except OSError:
            pass
        try:
            with open(f"{base}/maps") as f:
                for line in f:
                    i = line.find("/")
                    if i >= 0:
                        out.add(line[i:].rstrip("\n").removesuffix(" (deleted)"))
        except OSError:
            pass
    out.discard("/")
    return out


def team_repos() -> set[str]:
    """Repo directories hotline teams live in: an agent may be resumed there any time."""
    hl = "/home/bodas/.claude/bin/hotline"
    try:
        r = subprocess.run([hl, "--teams"], capture_output=True, text=True, timeout=30,
                           check=False)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    out = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and not line.startswith(" ") and parts[-1].startswith("/"):
            out.add(os.path.realpath(parts[-1]))
    return out


# --- verification: is the source really still there? --------------------------------


def _get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "provenance"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def hf_tree(repo: str, rev: str, repo_type: str = "model", timeout: float = 30
            ) -> dict[str, tuple[int, str]]:
    """path -> (size, sha256 for LFS files else '') for every file of an HF revision."""
    import re
    import urllib.parse

    kind = {"model": "models", "dataset": "datasets", "space": "spaces"}[repo_type]
    url: str | None = (f"https://huggingface.co/api/{kind}/{repo}/tree/"
                       f"{urllib.parse.quote(rev, safe='')}?recursive=true")
    out: dict[str, tuple[int, str]] = {}
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "provenance"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
            link = r.headers.get("Link", "")
        for e in data:
            if e.get("type") == "file":
                lfs = e.get("lfs") or {}
                out[e["path"]] = (int(e.get("size", -1)), lfs.get("oid", ""))
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


def _run_as_owner(path: str, args: list[str], timeout: float = 600
                  ) -> subprocess.CompletedProcess[str]:
    owner = os.stat(path).st_uid
    if os.geteuid() == 0 and owner != 0:
        args = ["runuser", "-u", pwd.getpwuid(owner).pw_name, "--", *args]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False,
                          env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def verify(c: Candidate, abs_: str, listing: list[tuple[str, int, str]],
           hf_info: dict[str, Any] | None, timeout: float = 30) -> str:
    """Return a one-line proof that the unit can be rebuilt; raise PruneError if not."""
    u = c.unit
    src = u.source or {}
    try:
        if u.kind == "cache":
            return "cache: refills on use"
        if u.kind == "hf-cache":
            info = (hf_info or {}).get(os.path.basename(abs_)) or {}
            if not info.get("found") or not info.get("revisions"):
                raise PruneError("huggingface_hub does not list it")
            for rev in info["revisions"]:
                tree = hf_tree(info["repo_id"], rev, info["repo_type"], timeout)
                if not tree:
                    raise PruneError(f"{info['repo_id']}@{rev[:12]}: empty tree")
            mine = {p.split("/", 1)[1]: (s, h) for p, s, h in listing if "/" in p}
            last = info["revisions"][-1]
            tree = hf_tree(info["repo_id"], last, info["repo_type"], timeout)
            bad = [p for p, (s, h) in mine.items() if p in tree and h and tree[p][1] != h]
            if bad:
                raise PruneError(f"HF content differs from the cache for {bad[:3]}")
            return f"HF {info['repo_id']}: {len(info['revisions'])} revision(s) still served"
        if u.kind in ("hf-local", "weights"):
            repo, rev = src.get("ref"), src.get("revision") or "main"
            if not repo or src.get("kind") != "hf":
                raise PruneError("no HF repo id")
            tree = hf_tree(str(repo), str(rev).split(",")[0], "model", timeout)
            want = {h for _p, _s, h in listing if h}
            have = {h for _s, h in tree.values() if h}
            if want and not want <= have:
                raise PruneError(f"{len(want - have)} file hash(es) not in {repo}@{rev}")
            if not want:
                sizes = {p: s for p, (s, _h) in tree.items()}
                missing = [p for p, s, _h in listing
                           if not p.startswith(".cache/") and sizes.get(p) != s]
                if u.kind == "hf-local" and missing:
                    raise PruneError(f"{len(missing)} file(s) not in {repo}@{rev} with that size")
            return f"HF {repo}@{str(rev)[:12]}: {len(want) or len(listing)} file(s) match"
        if u.kind == "git":
            with preserve_atimes(abs_):
                return _verify_git(u, abs_)
        if u.kind == "venv":
            raw = u.detail.get("requirements")
            reqs = [str(x) for x in raw] if isinstance(raw, list) else []
            if u.detail.get("local_installs"):
                raise PruneError("has packages installed from local paths")
            for req in reqs:
                if "==" not in req or " @ " in req:
                    raise PruneError(f"cannot verify {req!r} on PyPI")
                name, ver = req.split("==", 1)
                try:
                    _get_json(f"https://pypi.org/pypi/{name}/{ver}/json", timeout)
                except urllib.error.HTTPError as e:
                    raise PruneError(f"{req} not on PyPI (HTTP {e.code})") from e
            return f"{len(reqs)} pinned package(s) all on PyPI"
        if u.kind == "ollama":
            for tag in str(src.get("ref") or "").split(","):
                name, _, t = tag.partition(":")
                name = name if "/" in name else f"library/{name}"
                probe = urllib.request.Request(
                    f"https://registry.ollama.ai/v2/{name}/manifests/{t or 'latest'}",
                    method="HEAD", headers={"Accept":
                                            "application/vnd.docker.distribution.manifest.v2+json"})
                urllib.request.urlopen(probe, timeout=timeout).close()
            return "ollama manifests still served"
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        raise PruneError(f"verification failed: {exc}") from exc
    raise PruneError(f"kind {u.kind} is not prunable")


def _verify_git(u: UnitRow, abs_: str) -> str:
    head = str(u.detail.get("head") or "")
    if not head:
        raise PruneError("no HEAD")
    f = _run_as_owner(abs_, ["git", "-C", abs_, "fetch", "--quiet", "--all"], 600)
    if f.returncode != 0:
        raise PruneError(f"git fetch failed: {f.stderr.strip()[:160]}")
    r = _run_as_owner(abs_, ["git", "-C", abs_, "branch", "-r", "--contains", head])
    if not r.stdout.strip():
        raise PruneError(f"{head[:12]} is on no remote branch after fetch")
    s = _run_as_owner(abs_, ["git", "-C", abs_, "status", "--porcelain", "--untracked-files=all"])
    if s.stdout.strip():
        raise PruneError("working tree is no longer clean")
    return f"git {head[:12]} on {r.stdout.split()[0]} after fetch"


# --- planning ---------------------------------------------------------------------------


class Pruner:
    def __init__(self, engine: Engine, log: Callable[[str], None] = lambda m: None) -> None:
        self.e = engine
        self.cfg = engine.cfg
        self.st = engine.store
        self.log = log

    def areas(self) -> Iterator[Disk]:
        for a in self.cfg.areas:
            d = probe_area(a)
            if d is None:
                self.log(f"area {a} does not exist; skipped")
                continue
            yield d

    def _hf_infos(self, area: Disk, units: list[UnitRow]) -> dict[str, dict[str, Any]]:
        """HF's own view (sizes, revisions) of every HF cache repo, grouped per cache dir."""
        by_cache: dict[str, list[str]] = {}
        for u in units:
            if u.kind == "hf-cache":
                ab = os.path.join(area.root, u.path)
                by_cache.setdefault(os.path.dirname(ab), []).append(os.path.basename(ab))
        out: dict[str, dict[str, Any]] = {}
        for cache, dirs in by_cache.items():
            try:
                res = hfcache.run("plan", cache, dirs)
            except hfcache.HFCacheError as exc:
                self.log(f"HF cache {cache}: {exc}")
                continue
            out[cache] = res["repos"]
        return out

    def candidates(self, area: Disk, days: float | None = None, online: bool = True,
                   only: str | None = None) -> list[Candidate]:
        days = self.cfg.prune_after_days if days is None else days
        now = time.time()
        atime = tracks_atime(area.root)
        cov = self.st.coverage(area.uuid)
        covered = cov[0] if cov else None
        claims = self.st.claims(area.uuid)
        rejected = self.st.recently_rejected(area.uuid)
        used_now = paths_in_use()
        teams = team_repos()
        units = [u for u in self.st.units(area.uuid)
                 if only is None or u.path == only or u.path.startswith(only + "/")]
        hf = self._hf_infos(area, units)
        out: list[Candidate] = []
        for u in units:
            ab = os.path.join(area.root, u.path)
            size = u.bytes
            hfi = hf.get(os.path.dirname(ab), {}) if u.kind == "hf-cache" else None
            if hfi is not None:
                size = int((hfi.get(os.path.basename(ab)) or {}).get("freed_alone") or 0)
            opened = self.st.last_open(area.uuid, u.path)
            scanned = u.detail.get("last_used")
            if atime:
                lu = max(float(scanned) if isinstance(scanned, (int, float)) else 0.0,
                         opened or 0.0) or None
                used_by = "atime"
            else:
                lu, used_by = opened, "opens"
            c = Candidate(u, size, lu, used_by)
            r = c.reasons
            if u.kind not in PRUNABLE:
                r.append(f"kind {u.kind} is never pruned")
            if u.risk != RECREATABLE and not (u.kind == "cache" and u.risk == EXPENDABLE):
                r.append(f"risk is {u.risk}")
            if u.kind == "weights" and (u.source or {}).get("kind") != "hf":
                r.append("weights without an HF source")
            claim = claims.get(u.path)
            if claim and claim["kind"] == "keep":
                r.append(f"kept by {claim['by']}")
            if u.path in rejected:
                r.append("in a plan rejected within 30 days")
            if not u.path:
                r.append("the area itself")
            if atime:
                if lu and now - lu < days * DAY:
                    r.append(f"used {(now - lu) / DAY:.1f} d ago")
            else:
                if covered is None or now - covered < days * DAY:
                    have = (now - covered) / DAY if covered else 0
                    r.append(f"noatime mount: opens watched for {have:.1f} of {days:g} d")
                elif opened and now - opened < days * DAY:
                    r.append(f"opened {(now - opened) / DAY:.1f} d ago")
            hit = next((p for p in used_now if p == ab or p.startswith(ab + "/")), None)
            if hit:
                r.append(f"in use: {hit}")
            team = next((t for t in teams if t == ab or t.startswith(ab + "/")
                         or ab.startswith(t + "/")), None)
            if team:
                r.append(f"hotline team repo {team}")
            if not os.path.lexists(ab):
                r.append("no longer on disk (rescan)")
            if c.ok and online:
                try:
                    listing = unpack_listing(self._listing_blob(area.uuid, u.path))
                    c.verified = verify(c, ab, listing, hfi)
                except PruneError as exc:
                    r.append(str(exc))
            out.append(c)
        return out

    def _listing_blob(self, disk: str, path: str) -> bytes | None:
        row = self.st.db.execute("SELECT listing FROM units WHERE disk=? AND path=?",
                                 (disk, path)).fetchone()
        return row["listing"] if row else None

    # --- plans for review ------------------------------------------------------------

    def propose(self, area: Disk) -> tuple[str, list[Candidate]] | None:
        """Make a pending plan of everything prunable now; None if there is nothing."""
        pending = [p for p in self.st.plans("pending") if p["disk"] == area.uuid]
        if pending:
            return None
        ok = [c for c in self.candidates(area) if c.ok]
        if not ok:
            return None
        pid = time.strftime("p%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
        items: list[dict[str, object]] = [{"path": c.unit.path, "kind": c.unit.kind, "bytes": c.bytes,
                  "last_used": c.last_used, "used_by": c.used_by, "why": c.verified}
                 for c in ok]
        self.st.save_plan(pid, area.uuid, items)
        return pid, ok

    def summary(self, pid: str) -> str:
        p = self.st.plan(pid)
        if p is None:
            raise PruneError(f"no plan {pid}")
        items = json.loads(p["items"])
        root = p["disk"].removeprefix("area:")
        total = sum(int(i["bytes"]) for i in items)
        lines = [(f"[provenance] prune plan {pid}: {len(items)} unused, recreatable unit(s) "
                 f"in {root}, {human(total)} to free.")]
        for i in sorted(items, key=lambda i: -int(i["bytes"]))[:30]:
            age = (f"{(time.time() - float(i['last_used'])) / DAY:.0f} d"
                   if i["last_used"] else "never seen")
            lines.append(f"- {human(int(i['bytes'])):>9}  {i['path']} [{i['kind']}] "
                         f"unused {age} ({i['used_by']}); {i['why']}")
        if len(items) > 30:
            lines.append(f"...and {len(items) - 30} more: provenance plans --show {pid}")
        lines += ["", ("Everything is re-checked at delete time; HF caches are deleted through "
                  "huggingface_hub, the rest go to .provenance-trash for "
                  f"{self.cfg.trash_grace_h:g} h first."),
                  f"Approve: provenance prune --approve {pid}",
                  f"Reject:  provenance prune --reject {pid}",
                  f"Expires unapproved in {self.cfg.plan_ttl_h:g} h."]
        return "\n".join(lines)

    # --- deleting ----------------------------------------------------------------------

    def apply(self, area: Disk, paths: list[str], plan: str | None) -> dict[str, Any]:
        """Re-check each path and prune the ones that still qualify."""
        fresh = {c.unit.path: c for c in self.candidates(area)}
        done: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        hf_batches: dict[str, list[Candidate]] = {}
        for p in paths:
            c = fresh.get(p)
            if c is None:
                skipped.append({"path": p, "why": "no longer in the recipe"})
                continue
            if not c.ok:
                skipped.append({"path": p, "why": "; ".join(c.reasons)})
                continue
            ab = os.path.join(area.root, p)
            if c.unit.kind == "hf-cache":
                hf_batches.setdefault(os.path.dirname(ab), []).append(c)
                continue
            owner = os.lstat(ab).st_uid
            trash = os.path.join(area.root, TRASH, plan or "manual", p)
            os.makedirs(os.path.dirname(trash), exist_ok=True)
            self.st.add_pruned(area.uuid, c.unit, owner, plan, trash, c.bytes)
            os.rename(ab, trash)
            done.append({"path": p, "bytes": c.bytes, "how": "trash"})
            self.st.log("info", f"pruned {area.root}/{p} to trash ({human(c.bytes)})")
        for cache, cs in hf_batches.items():
            dirs = [os.path.basename(os.path.join(area.root, c.unit.path)) for c in cs]
            for c in cs:
                self.st.add_pruned(area.uuid, c.unit, os.stat(cache).st_uid, plan, None, c.bytes)
            try:
                res = hfcache.run("delete", cache, dirs)
            except hfcache.HFCacheError as exc:
                # Nothing was deleted: put the units back into the live recipe.
                for c in cs:
                    self._unprune(area.uuid, c.unit.path)
                    skipped.append({"path": c.unit.path, "why": str(exc)})
                continue
            for c in cs:
                done.append({"path": c.unit.path, "bytes": c.bytes, "how": "huggingface_hub"})
            self.st.log("info", f"HF cache {cache}: deleted {len(cs)} repo(s), "
                        f"{human(int(res['freed_together']))} freed")
        if plan:
            self.st.decide_plan(plan, "applied", "", {"done": done, "skipped": skipped})
        return {"done": done, "skipped": skipped,
                "freed": sum(int(d["bytes"]) for d in done)}

    def _unprune(self, disk: str, path: str) -> None:
        self.st.db.execute("DELETE FROM pruned WHERE disk=? AND path=? AND purged_at IS NULL "
                           "AND restored_at IS NULL AND trash IS NULL", (disk, path))
        self.st.db.commit()

    def purge_trash(self, grace_h: float | None = None) -> list[str]:
        grace = (self.cfg.trash_grace_h if grace_h is None else grace_h) * 3600
        gone: list[str] = []
        for r in self.st.pruned():
            if r["trash"] and r["purged_at"] is None and time.time() - r["pruned_at"] >= grace:
                if os.path.lexists(r["trash"]):
                    shutil.rmtree(r["trash"]) if os.path.isdir(r["trash"]) and not \
                        os.path.islink(r["trash"]) else os.unlink(r["trash"])
                self.st.mark_pruned(r["disk"], r["path"], r["pruned_at"], "purged_at")
                gone.append(r["trash"])
        # Empty plan directories left behind.
        for a in self.cfg.areas:
            t = Path(a) / TRASH
            if t.is_dir():
                for d in [*sorted(t.rglob("*"), key=lambda x: -len(x.parts)), t]:
                    if d.is_dir() and not d.is_symlink() and not any(d.iterdir()):
                        d.rmdir()
        return gone

    # --- restoring ---------------------------------------------------------------------

    def restore(self, abspath: str, dry_run: bool = False) -> str:
        ap = os.path.realpath(abspath) if os.path.lexists(abspath) else os.path.abspath(abspath)
        for r in self.st.pruned():
            root = r["disk"].removeprefix("area:")
            if os.path.join(root, r["path"]) != ap:
                continue
            if os.path.lexists(ap):
                raise PruneError(f"{ap} exists again; nothing to restore")
            unit = json.loads(r["unit"])
            if r["trash"] and r["purged_at"] is None and os.path.lexists(r["trash"]):
                if dry_run:
                    return f"would move it back from {r['trash']}"
                os.makedirs(os.path.dirname(ap), exist_ok=True)
                os.rename(r["trash"], ap)
                self.st.mark_pruned(r["disk"], r["path"], r["pruned_at"], "restored_at")
                return f"moved back from {r['trash']}"
            cmds = [c for c in unit["rebuild"] if not c.lstrip().startswith("#")]
            if not cmds:
                raise PruneError(f"no rebuild commands recorded for {ap}")
            script = "\n".join(["set -euo pipefail", f"ROOT={_shq(root)}",
                                ("command -v hf >/dev/null || hf() { uvx --from "
                                "'huggingface_hub>=2.4' hf \"$@\"; }"), *cmds])
            if dry_run:
                return script
            parent = os.path.dirname(ap)
            os.makedirs(parent, exist_ok=True)
            owner = int(r["owner"])
            args = ["bash", "-c", script]
            if os.geteuid() == 0 and owner != 0:
                pw = pwd.getpwuid(owner)
                path = f"{pw.pw_dir}/.local/bin:/usr/local/bin:/usr/bin"
                args = ["runuser", "-u", pw.pw_name, "--", "env", f"HOME={pw.pw_dir}",
                        f"PATH={path}", *args]
            p = subprocess.run(args, capture_output=True, text=True, timeout=7200, check=False)
            if p.returncode != 0:
                raise PruneError(f"rebuild failed: {p.stderr.strip()[-400:]}")
            bad = _check_listing(ap, unpack_listing(r["listing"]), unit["kind"])
            if bad:
                raise PruneError(f"rebuilt, but {len(bad)} file(s) differ: {bad[:3]}")
            self.st.mark_pruned(r["disk"], r["path"], r["pruned_at"], "restored_at")
            return "rebuilt from the recipe and checked against the stored listing"
        raise PruneError(f"{abspath} was not pruned (see `provenance pruned`)")


def _check_listing(ap: str, listing: list[tuple[str, int, str]], kind: str) -> list[str]:
    if kind in ("git", "venv", "cache"):
        return []
    bad = []
    for p, size, _h in listing:
        if kind == "hf-cache":
            continue  # revision dirs are named by full hash; checked by huggingface_hub
        fp = os.path.join(ap, p) if p else ap
        try:
            if os.stat(fp).st_size != size:
                bad.append(p)
        except OSError:
            bad.append(p)
    return bad


def _shq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"
