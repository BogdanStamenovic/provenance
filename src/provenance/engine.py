"""Glue: scan disks into the store, watch them, tell writers about at-risk units."""

from __future__ import annotations

import json
import os
import pwd
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import attribution, classify
from .classify import AT_RISK, Context, Unit
from .config import Config
from .store import Store, hf_hub_entries, sha256_file

# Our own git calls must not write .git/index, or every scan would trigger the next.
os.environ["GIT_OPTIONAL_LOCKS"] = "0"

FILE_KINDS = ("weights", "archive", "disk-image", "file")
SELF = -1  # writer id meaning "this daemon or one of its children"


class EngineError(Exception):
    pass


def hf_tree_entries(repo: str, rev: str, timeout: float
                    ) -> Iterator[tuple[str, str, str, str]]:
    """(sha256, repo, revision, path) for every LFS file of an HF model repo."""
    import urllib.parse
    import urllib.request

    url: str | None = (f"https://huggingface.co/api/models/{repo}/tree/"
                       f"{urllib.parse.quote(rev, safe='')}?recursive=true&expand=false")
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "provenance"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
            link = r.headers.get("Link", "")
        for e in data:
            lfs = e.get("lfs") or {}
            if lfs.get("oid"):
                yield (lfs["oid"], repo, rev, e["path"])
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None


@dataclass
class Disk:
    uuid: str
    root: str
    label: str
    size: int


def probe_disk(root: str) -> Disk | None:
    """The filesystem mounted exactly at `root`, or None if nothing is mounted there.

    Refusing a non-mountpoint matters: with the HDD unplugged, /mnt/hdd is an empty
    directory on the SSD, and scanning it would "learn" that the disk is empty.
    """
    root = os.path.realpath(root)
    try:
        out = subprocess.run(
            ["findmnt", "-n", "-b", "-o", "TARGET,UUID,LABEL,SIZE", "-M", root],
            check=False, capture_output=True, text=True, timeout=10,
        ).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out or out[0] != root:
        return None
    uuid = out[1] if len(out) > 1 else ""
    label = out[2] if len(out) > 3 else ""
    size = int(out[-1]) if out[-1].isdigit() else 0
    if not uuid:
        st = os.stat(root)
        uuid = f"dev-{st.st_dev:x}"
    return Disk(uuid, root, label, size)


class Engine:
    def __init__(self, cfg: Config, store: Store,
                 log: Callable[[str], None] = lambda m: None,
                 warn: Callable[[str], None] = lambda m: None) -> None:
        self.cfg = cfg
        self.store = store
        self.log = log
        self.warn = warn

    # --- context for the classifiers -------------------------------------------------

    def context(self, disk: Disk) -> Context:
        claims = self.store.claims(disk.uuid)

        def source_for(rel: str) -> dict[str, str | None] | None:
            rows = self.store.writers_under(disk.uuid, rel)
            for r in rows:
                inside = (r["dir"] == rel or r["dir"].startswith(rel + "/")) if rel else True
                if r["source"] and (inside or r["dir"] == rel):
                    src: dict[str, str | None] = json.loads(r["source"])
                    return src
            return None

        def file_sha(rel: str, size: int, mtime_ns: int, ino: int) -> str | None:
            sha = self.store.cached_sha(disk.uuid, rel, size, mtime_ns, ino)
            if sha or not self.cfg.hash_weights:
                return sha
            self.log(f"hashing {rel} ({classify.human(size)}) to look for a known source")
            try:
                sha = sha256_file(os.path.join(disk.root, rel))
            except OSError as exc:
                self.warn(f"cannot hash {rel}: {exc}")
                return None
            self.store.put_sha(disk.uuid, rel, size, mtime_ns, ino, sha)
            return sha

        return Context(
            root=disk.root, source_for=source_for, hf_blob=self.store.hf_blob,
            file_sha=file_sha, claim_for=claims.get, warn=self.warn,
        )

    def refresh_hf_index(self, extra: list[str] = ()) -> int:  # type: ignore[assignment]
        n = 0
        for hub in [*self.cfg.hf_hubs, *extra]:
            if os.path.isdir(hub):
                n += self.store.index_hf(hf_hub_entries(hub))
        return n

    # --- scanning --------------------------------------------------------------------

    def scan(self, disk: Disk, prefix: str = "", shallow: bool = False) -> list[Unit]:
        self.store.touch_disk(disk.uuid, disk.root, disk.label, disk.size)
        if not prefix:
            hubs = {os.path.join(disk.root, os.path.dirname(u.path))
                    for u in self.store.units(disk.uuid) if u.kind == "hf-cache"}
            self.refresh_hf_index(sorted(hubs))
        ctx = self.context(disk)
        t0 = time.time()
        if shallow:
            units = classify.walk_shallow(ctx, prefix)
            self.store.replace_direct(disk.uuid, prefix, units, FILE_KINDS)
        else:
            units = classify.walk(ctx, prefix)
            self.store.replace_units(disk.uuid, prefix, units)
        self.store.compact_writes(disk.uuid, [u.path for u in units if u.kind != "data"])
        if not prefix:
            self.store.mark_full_scan(disk.uuid)
        dropped = self.store.enforce_budget(self.cfg.budget)
        if dropped:
            self.store.log("warn", f"over budget: dropped {len(dropped)} listing(s): "
                           + ", ".join(dropped[:10]))
            self.warn(f"recipe over {self.cfg.budget_mb:g} MB budget: dropped "
                      f"{len(dropped)} file listing(s)")
        self.store.log("info", f"scanned {disk.root}/{prefix} ({'shallow' if shallow else 'full'}"
                       f"): {len(units)} unit(s) in {time.time() - t0:.1f}s")
        return units

    # --- online identification of unknown weights ------------------------------------

    def candidate_repos(self) -> list[str]:
        """Every HF model repo id this box has ever been seen touching."""
        q = self.store.db
        repos = {r[0] for r in q.execute("SELECT DISTINCT repo FROM hfblobs")}
        for (raw,) in q.execute("SELECT source FROM writers WHERE source IS NOT NULL"):
            src = json.loads(raw)
            if src.get("kind") == "hf" and src.get("ref"):
                repos.add(src["ref"])
        for (raw,) in q.execute("SELECT source FROM units WHERE source IS NOT NULL"):
            src = json.loads(raw)
            if src.get("repo_type", "model") != "model":
                continue  # the lookup uses the models API; datasets answer 401 there
            if src.get("kind") == "hf" and src.get("ref"):
                repos.add(str(src["ref"]))
        return sorted(r for r in repos if r and "/" in r and not r.startswith("datasets/"))

    def identify_online(self, disk: Disk, timeout: float = 20.0, max_repos: int = 200,
                        recheck_days: float = 7.0) -> list[str]:
        """Pull sha256 lists of candidate repos from the HF API; return units now matched."""
        # Loose weights, and --local-dir HF downloads whose repo id was never recorded:
        # their metadata still carries each file's sha256, which is enough to match.
        unknown = [u for u in self.store.units(disk.uuid, risk=AT_RISK)
                   if u.kind == "weights" or (u.kind == "hf-local" and u.source
                                              and not u.source.get("ref"))]
        if not unknown:
            return []
        want = {}
        for u in unknown:
            for _p, _s, sha in self.store.listing(disk.uuid, u.path):
                if sha:
                    want[sha] = u.path
        if not want:
            return []
        cutoff = time.time() - recheck_days * 86400
        recent = {r[0] for r in self.store.db.execute(
            "SELECT repo FROM hfrepo_checked WHERE at > ?", (cutoff,))}
        matched: list[str] = []
        for repo in [r for r in self.candidate_repos() if r not in recent][:max_repos]:
            try:
                entries = list(hf_tree_entries(repo, "main", timeout))
                ok = 1
            except OSError as exc:
                self.warn(f"HF tree {repo}: {exc}")
                entries, ok = [], 0
            self.store.index_hf(entries)
            self.store.db.execute("INSERT OR REPLACE INTO hfrepo_checked (repo, at, ok) VALUES (?,?,?)",
                                  (repo, time.time(), ok))
            self.store.db.commit()
            for sha, *_ in entries:
                if sha in want:
                    matched.append(want.pop(sha))
            if not want:
                break
        if matched:
            self.store.log("info", f"identified {len(matched)} weights file(s) online")
        return matched

    # --- claims ----------------------------------------------------------------------

    def resolve(self, abspath: str) -> tuple[Disk, str]:
        ap = os.path.realpath(abspath)
        for root in self.cfg.disks:
            r = os.path.realpath(root)
            if ap == r or ap.startswith(r + "/"):
                d = probe_disk(r)
                if d is None:
                    raise EngineError(f"{r} is not mounted")
                return d, os.path.relpath(ap, r) if ap != r else ""
        for a in self.cfg.areas:
            if ap == a or ap.startswith(a.rstrip("/") + "/"):
                from .prune import probe_area

                ad = probe_area(a)
                if ad is None:
                    raise EngineError(f"area {a} does not exist")
                return ad, os.path.relpath(ap, a) if ap != a else ""
        where = ", ".join([*self.cfg.disks, *self.cfg.areas])
        raise EngineError(f"{abspath} is not on a watched disk or area ({where})")

    def apply_claim(self, abspath: str, kind: str, value: str, by: str) -> tuple[Disk, str]:
        disk, rel = self.resolve(abspath)
        if kind != "clear" and not os.path.lexists(abspath):
            raise EngineError(f"{abspath} does not exist")
        self.store.set_claim(disk.uuid, rel, kind, value, by)
        self.store.log("info", f"claim {kind} on {rel} by {by}: {value}")
        return disk, rel

    def ingest_spool(self) -> list[tuple[Disk, str]]:
        """Apply claims unprivileged users dropped into the spool directory."""
        done: list[tuple[Disk, str]] = []
        sp = self.cfg.spool
        if not sp.is_dir():
            return done
        for f in sorted(sp.glob("*.json")):
            try:
                st = f.stat()
                req = json.loads(f.read_text())
                user = pwd.getpwuid(st.st_uid).pw_name
                by = user + (f" ({req['agent']})" if req.get("agent") else "")
                if req["kind"] in ("plan-approve", "plan-reject"):
                    self.decide_plan(req["value"], req["kind"] == "plan-approve", by)
                    continue
                done.append(self.apply_claim(req["path"], req["kind"], req.get("value", ""), by))
            except (OSError, ValueError, KeyError, EngineError) as exc:
                self.warn(f"bad claim {f.name}: {exc}")
                self.store.log("warn", f"rejected claim {f.name}: {exc}")
            finally:
                f.unlink(missing_ok=True)
        return done

    def decide_plan(self, pid: str, approve: bool, by: str) -> None:
        p = self.store.plan(pid)
        if p is None:
            raise EngineError(f"no prune plan {pid}")
        if p["status"] != "pending":
            raise EngineError(f"plan {pid} is {p['status']}, not pending")
        if time.time() - p["created"] > self.cfg.plan_ttl_h * 3600:
            self.store.decide_plan(pid, "expired", "")
            raise EngineError(f"plan {pid} expired")
        self.store.decide_plan(pid, "approved" if approve else "rejected", by)
        self.store.log("info", f"plan {pid} {'approved' if approve else 'rejected'} by {by}")

    # --- telling the writer ------------------------------------------------------------

    def pending_notices(self, disk: Disk) -> dict[str, list[str]]:
        """agent -> at-risk unit paths nobody has been told about yet."""
        out: dict[str, list[str]] = {}
        now = time.time()
        for u in self.store.units(disk.uuid, risk=AT_RISK):
            if u.notified_at:
                continue
            if now - (u.at_risk_since or now) < self.cfg.notify_after_s:
                continue
            agent = None
            # Weights/archives/images are recorded per file; other files per directory.
            if u.kind in FILE_KINDS:
                look, ok = os.path.dirname(u.path), {u.path, os.path.dirname(u.path)}
            else:
                look, ok = u.path, None
            for r in self.store.writers_under(disk.uuid, look):
                if ok is not None:
                    inside = r["dir"] in ok
                else:
                    inside = r["dir"] == look or r["dir"].startswith(look + "/")
                if inside and r["agent"]:
                    agent = r["agent"]
                    break
            to = agent or self.cfg.notify_fallback
            if to:
                out.setdefault(to, []).append(u.path)
        return out

    def notify(self, disk: Disk, dry_run: bool = False) -> list[str]:
        sent: list[str] = []
        if disk.uuid.startswith("area:"):
            # In a single-disk area, AT RISK only means "never pruned": nothing is lost, so
            # nobody is asked. Asking writers is for disks that may die (cfg.disks).
            return sent
        for agent, paths in self.pending_notices(disk).items():
            lines = []
            for p in paths[:20]:
                u = self.store.unit_at(disk.uuid, p)
                note = u.note if u else ""
                lines.append(f"- {os.path.join(disk.root, p)} [{u.kind if u else '?'}]: {note}")
            more = f"\n...and {len(paths) - 20} more" if len(paths) > 20 else ""
            msg = (
                f"[provenance] You wrote data to {disk.root} (the untrusted HDD) that cannot "
                "be recreated if that disk dies: no known source.\n" + "\n".join(lines) + more
                + "\n\nFor each one, either give its source:\n"
                "  provenance claim PATH --source hf:org/name[@rev] | git:URL | https://URL "
                "| 'free text: how to recreate it'\n"
                "or say losing it is fine:\n"
                "  provenance claim PATH --expendable 'why'\n"
                "Ignore this and it stays listed as AT RISK in `provenance status`."
            )
            if dry_run:
                sent.append(f"{agent}: {len(paths)} unit(s)")
                continue
            if self._send(agent, msg):
                for p in paths:
                    self.store.mark_notified(disk.uuid, p, agent)
                sent.append(f"{agent}: {len(paths)} unit(s)")
                self.store.log("info", f"told {agent} about {len(paths)} at-risk unit(s)")
        return sent

    def _send(self, agent: str, msg: str) -> bool:
        cmd = [self.cfg.notify_cmd, "--to", agent, "--no-wait", msg]
        if os.geteuid() == 0 and self.cfg.notify_user:
            home = pwd.getpwnam(self.cfg.notify_user).pw_dir
            path = f"{home}/.claude/bin:{home}/.local/bin:/usr/local/bin:/usr/bin"
            cmd = ["runuser", "-u", self.cfg.notify_user, "--", "env", f"HOME={home}",
                   f"PATH={path}", *cmd]
        try:
            r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.warn(f"notify {agent}: {exc}")
            return False
        if r.returncode not in (0, 3):  # 3 = delivered, unanswered
            self.warn(f"notify {agent} failed ({r.returncode}): {r.stderr.strip()[:200]}")
            self.store.log("warn", f"notify {agent} failed: {r.stderr.strip()[:200]}")
            return False
        return True


# --- the daemon ------------------------------------------------------------------------


def affected_prefix(rel: str, unit_paths: set[str], containers: set[str]
                    ) -> tuple[str, bool]:
    """Which subtree to rescan for a change at `rel`, and whether shallowly.

    Inside a known unit: rescan that unit. In a known container: rescan the new child
    entry if it is a directory, or the container's own loose files (shallow) if it is a
    file sitting directly in it.
    """
    parts = rel.split("/")
    for i in range(len(parts), 0, -1):
        p = "/".join(parts[:i])
        if p in unit_paths:
            return p, False
    for i in range(len(parts) - 1, -1, -1):
        parent = "/".join(parts[:i])
        if parent in containers:
            child = "/".join(parts[: i + 1])
            if i + 1 == len(parts):
                return parent, True  # a file or dir directly in the container
            return child, False
    return parts[0], False


class Daemon:
    def __init__(self, engine: Engine) -> None:
        self.e = engine
        self.cfg = engine.cfg
        self.me = os.getpid()
        self.pid_cache: dict[int, tuple[float, int | None]] = {}
        self.writes: dict[str, dict[tuple[str, int], tuple[float, float, int]]] = {}
        self.dirty: dict[tuple[str, str, bool], float] = {}
        self.last_identify: dict[str, float] = {}
        self.openers: dict[str, Any] = {}  # mountpoint -> OpenWatcher (noatime areas)
        self.usage: dict[str, dict[str, float]] = {}  # area uuid -> unit path -> last open
        self.area_units: dict[str, set[str]] = {}
        self.last_prune_cycle: dict[str, float] = {}

    def _writer(self, pid: int) -> int | None:
        now = time.time()
        hit = self.pid_cache.get(pid)
        if hit and now - hit[0] < 10:
            return hit[1]
        w = attribution.capture(pid)
        wid: int | None
        if any(p.pid == self.me for p in w.chain):
            wid = SELF
        elif not w.chain:
            wid = None
        else:
            src = attribution.source_of(w)
            sd = ({"kind": src.kind, "ref": src.ref, "revision": src.revision,
                   "command": src.command, "file": src.file} if src else None)
            named = attribution.hf_blobs_named_in(w)
            if named:
                self.e.store.index_hf(named)
            wid = self.e.store.writer_id(w.key, w.to_json(), sd, w.session_id,
                                         attribution.agent_name(w.session_id), w.user)
            self.e.store.db.commit()
        self.pid_cache[pid] = (now, wid)
        if len(self.pid_cache) > 4096:
            self.pid_cache = {k: v for k, v in self.pid_cache.items() if now - v[0] < 10}
        return wid

    def run(self, once: bool = False) -> None:
        from .fan import Watcher  # Linux-only; keep the rest importable elsewhere

        watchers: dict[str, tuple[Disk, Any]] = {}
        last_check = 0.0
        cfg_mtime = _mtime(self.cfg.path)
        while True:
            now = time.time()
            if now - last_check > 60 or once:
                last_check = now
                if not once and _mtime(self.cfg.path) != cfg_mtime:
                    # Simplest correct reload: start over with the new config. State is all
                    # in the db, so nothing is lost; pending writes are flushed first.
                    self.e.log(f"{self.cfg.path} changed; restarting")
                    for d, w in watchers.values():
                        self._flush(d)
                        w.close()
                    os.execv(sys.executable, [sys.executable, "-m", "provenance", *sys.argv[1:]])
                self._sync_disks(watchers, Watcher)
                self._sync_areas(watchers, Watcher)
                for disk, rel in self.e.ingest_spool():
                    self.dirty[(disk.uuid, rel, False)] = 0
                self._flush_usage(now)
                for d, _w in watchers.values():
                    # Cheap (milliseconds): learn HF blobs before the cache gets wiped.
                    self.e.refresh_hf_index()
                    self._full_scan_if_due(d)
                    self._identify_if_due(d)
                    self._flush(d)
                    self.e.notify(d)
                    if d.uuid.startswith("area:"):
                        self.area_units[d.uuid] = {
                            u.path for u in self.e.store.units(d.uuid) if u.path}
                        self._prune_cycle(d)
            self._read_opens()
            for disk, w in list(watchers.values()):
                try:
                    events = w.read(timeout=1.0 / max(len(watchers), 1))
                except OSError as exc:
                    self.e.warn(f"{disk.root}: watcher failed: {exc}")
                    w.close()
                    del watchers[disk.uuid]
                    continue
                self._take(disk, events)
                if w.overflowed:
                    w.overflowed = False
                    self.e.warn(f"{disk.root}: event queue overflowed; full rescan")
                    self.dirty[(disk.uuid, "", False)] = time.time()
            if not watchers:
                time.sleep(5)
            self._process_dirty({d.uuid: d for d, _ in watchers.values()})
            if once:
                return

    def _sync_disks(self, watchers: dict[str, tuple[Disk, Any]], watcher_cls: type) -> None:
        for root in self.cfg.disks:
            d = probe_disk(root)
            known = [k for k, (dd, _) in watchers.items() if dd.root == os.path.realpath(root)]
            if d is None:
                for k in known:
                    self.e.warn(f"{root} unmounted; stopped watching")
                    watchers[k][1].close()
                    del watchers[k]
                continue
            if d.uuid in watchers:
                continue
            for k in known:  # a different disk is now mounted there
                watchers[k][1].close()
                del watchers[k]
            if os.stat(d.root).st_dev == os.stat(self.cfg.state_dir).st_dev:
                raise EngineError(f"{d.root} is the same filesystem as state_dir")
            watchers[d.uuid] = (d, watcher_cls(d.root))
            self.e.log(f"watching {d.root} (uuid {d.uuid})")
            self.e.store.log("info", f"watching {d.root} (uuid {d.uuid})")

    # --- single-disk mode --------------------------------------------------------------

    def _sync_areas(self, watchers: dict[str, tuple[Disk, Any]], watcher_cls: type) -> None:
        from .fan import FanotifyError, OpenWatcher
        from .prune import mount_options, probe_area

        now = time.time()
        for a in self.cfg.areas:
            d = probe_area(a)
            if d is None or d.uuid in watchers:
                continue
            try:
                w = watcher_cls(d.root)
            except FanotifyError as exc:
                self.e.warn(f"area {a}: no write attribution ({exc})")
                w = _NullWatcher()
            watchers[d.uuid] = (d, w)
            mnt, _fs, opts = mount_options(d.root)
            if "noatime" in opts and mnt not in self.openers:
                try:
                    self.openers[mnt] = OpenWatcher(mnt)
                except FanotifyError as exc:
                    self.e.warn(f"area {a}: cannot track opens on {mnt} ({exc}); "
                                "nothing there will ever count as unused")
            self._start_coverage(d, now)
            self.e.log(f"area {d.root} ({d.label}); unused = "
                       + ("observed opens" if "noatime" in opts else "atime"))

    def _start_coverage(self, d: Disk, now: float) -> None:
        """Keep the "watched since" clock across a restart or a power-off, not a long stop.

        A gap is harmless when the daemon was merely restarted (< 1 h) or the machine
        was off (it booted after the last heartbeat and we started soon after boot).
        Any other gap means opens may have been missed, so the clock starts over.
        """
        cov = self.e.store.coverage(d.uuid)
        since = now
        if cov:
            prev_since, beat = cov
            boot = _boot_time()
            if now - beat < 3600 or (boot > beat and now - boot < 900):
                since = prev_since
        self.e.store.set_coverage(d.uuid, since, now)

    def _read_opens(self) -> None:
        if not self.openers:
            return
        now = time.time()
        areas = {d: os.path.realpath(d.removeprefix("area:")) for d in self.area_units} or {
            f"area:{a}": a for a in self.cfg.areas}
        for mnt, w in list(self.openers.items()):
            try:
                events = w.read(timeout=0)
            except OSError as exc:
                self.e.warn(f"open tracking on {mnt} failed: {exc}")
                w.close()
                del self.openers[mnt]
                continue
            for path, pid in events:
                if pid == self.me or "/.git/" in path:
                    continue
                for uuid, root in areas.items():
                    if path.startswith(root + "/"):
                        rel = path[len(root) + 1:]
                        self.usage.setdefault(uuid, {})[self._unit_of(uuid, rel)] = now
                        break
            if w.overflowed:
                # Missed opens: anything might have been used. Restart the clock.
                w.overflowed = False
                for uuid in areas:
                    self.e.store.set_coverage(uuid, now, now)
                self.e.warn(f"open tracking on {mnt} overflowed; unused-clock restarted")

    def _unit_of(self, uuid: str, rel: str) -> str:
        units = self.area_units.get(uuid, set())
        parts = rel.split("/")
        for i in range(len(parts), 0, -1):
            p = "/".join(parts[:i])
            if p in units:
                return p
        return parts[0]

    def _flush_usage(self, now: float) -> None:
        for uuid, seen in self.usage.items():
            if seen:
                self.e.store.record_usage(uuid, seen)
        self.usage = {}
        for a in self.cfg.areas:
            uuid = f"area:{a}"
            cov = self.e.store.coverage(uuid)
            if cov:
                self.e.store.set_coverage(uuid, cov[0], now)

    def _prune_cycle(self, d: Disk) -> None:
        """Daily: purge expired trash, expire stale plans, propose one, apply approved."""
        from .prune import PruneError, Pruner

        pr = Pruner(self.e, self.e.log)
        for p in self.e.store.plans("approved"):
            if p["disk"] != d.uuid:
                continue
            paths = [i["path"] for i in json.loads(p["items"])]
            try:
                res = pr.apply(d, paths, p["id"])
            except (PruneError, OSError) as exc:
                self.e.store.decide_plan(p["id"], "failed", "", {"error": str(exc)})
                self._report(f"[provenance] plan {p['id']} FAILED: {exc}")
                continue
            self._report(
                f"[provenance] plan {p['id']} applied: {len(res['done'])} unit(s) pruned, "
                f"{classify.human(res['freed'])} freed"
                + (f"; {len(res['skipped'])} skipped at delete time: "
                   + "; ".join(f"{s['path']}: {s['why']}" for s in res["skipped"][:5])
                   if res["skipped"] else "")
                + ". Undo one: provenance restore PATH")
        if not self.cfg.auto_prune:
            return
        now = time.time()
        if now - self.last_prune_cycle.get(d.uuid, 0) < 86400:
            return
        self.last_prune_cycle[d.uuid] = now
        pr.purge_trash()
        for p in self.e.store.plans("pending"):
            if now - p["created"] > self.cfg.plan_ttl_h * 3600:
                self.e.store.decide_plan(p["id"], "expired", "")
        try:
            made = pr.propose(d)
        except (PruneError, OSError) as exc:
            self.e.warn(f"prune plan for {d.root}: {exc}")
            return
        if made:
            pid, _ = made
            self.e._send(self.cfg.prune_approver, pr.summary(pid))
            self.e.store.log("info", f"proposed plan {pid} to {self.cfg.prune_approver}")

    def _report(self, msg: str) -> None:
        for to in {self.cfg.prune_approver, self.cfg.prune_report_to} - {""}:
            self.e._send(to, msg)

    def _identify_if_due(self, d: Disk) -> None:
        now = time.time()
        if now - self.last_identify.get(d.uuid, 0) < 3600:
            return
        self.last_identify[d.uuid] = now
        hits = self.e.identify_online(d)
        for rel in hits:
            self.dirty[(d.uuid, rel, False)] = 0

    def _full_scan_if_due(self, d: Disk) -> None:
        row = next((r for r in self.e.store.disks() if r["uuid"] == d.uuid), None)
        last = row["last_full_scan"] if row else None
        if not last or time.time() - last > self.cfg.full_scan_every_h * 3600:
            self.e.log(f"full scan of {d.root}")
            self.e.scan(d)

    def _take(self, disk: Disk, events: Sequence[Any]) -> None:
        if not events:
            return
        root = disk.root.rstrip("/") + "/"
        batch = self.writes.setdefault(disk.uuid, {})
        now = time.time()
        for ev in events:
            path = getattr(ev, "path", "")
            if not path.startswith(root):
                continue
            rel = path[len(root):]
            if rel.split("/", 1)[0] in classify.SKIP_TOP:
                continue
            if "/.git/" in f"/{rel}/" and not rel.endswith("/.git"):
                # Git internals: the unit is the repo; attribute to the repo dir.
                rel = rel.split("/.git/")[0] if "/.git/" in rel else rel
            wid = self._writer(getattr(ev, "pid", 0))
            if wid == SELF:
                continue  # our own scan (git, hashing) must not schedule another scan
            if wid is not None and not getattr(ev, "is_removal", False):
                if getattr(ev, "is_dir", False):
                    d = rel
                elif classify.file_kind_by_name(rel):
                    # Files that get a source of their own are recorded individually, so
                    # one file's download URL is never pinned on its neighbours. Anything
                    # else is recorded per directory, which keeps a folder of a million
                    # images from becoming a million rows.
                    d = rel
                else:
                    d = os.path.dirname(rel)
                k = (d, wid)
                first, _l, n = batch.get(k, (now, now, 0))
                batch[k] = (first, now, n + 1)
            self._mark_dirty(disk, rel, now)

    def _mark_dirty(self, disk: Disk, rel: str, now: float) -> None:
        units = {u.path for u in self.e.store.units(disk.uuid) if u.kind != "data"}
        containers = {""}
        for u in units:
            parts = u.split("/")
            containers.update("/".join(parts[:i]) for i in range(1, len(parts)))
        prefix, shallow = affected_prefix(rel, units, containers)
        self.dirty[(disk.uuid, prefix, shallow)] = now

    def _flush(self, d: Disk) -> None:
        batch = self.writes.pop(d.uuid, None)
        if batch:
            self.e.store.record_writes(d.uuid, batch)

    def _process_dirty(self, disks: dict[str, Disk]) -> None:
        now = time.time()
        ready = [k for k, t in self.dirty.items() if now - t >= self.cfg.debounce_s]
        if not ready:
            return
        for uuid in {k[0] for k in ready}:
            if uuid in disks:
                self._flush(disks[uuid])
        # A full rescan subsumes everything else queued for that disk.
        for k in sorted(ready, key=lambda k: (k[0], len(k[1]))):
            if k not in self.dirty:
                continue
            uuid, prefix, shallow = k
            del self.dirty[k]
            d = disks.get(uuid)
            if d is None:
                continue
            if not prefix and not shallow:
                for other in [o for o in self.dirty if o[0] == uuid]:
                    del self.dirty[other]
            if prefix and not os.path.lexists(os.path.join(d.root, prefix)):
                self.e.store.replace_units(uuid, prefix, [])
                continue
            try:
                self.e.scan(d, prefix, shallow)
            except classify.ClassifyError as exc:
                self.e.warn(str(exc))
        for d in disks.values():
            self.e.notify(d)


class _NullWatcher:
    """Stands in when a filesystem cannot take a write mark: no attribution, no events."""

    overflowed = False

    def read(self, timeout: float | None = None) -> list[Any]:
        time.sleep(timeout or 0)
        return []

    def close(self) -> None:
        pass


def _boot_time() -> float:
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except OSError:
        pass
    return 0.0


def _mtime(path: str) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def ensure_state(cfg: Config) -> None:
    Path(cfg.state_dir).mkdir(parents=True, exist_ok=True)
    os.chmod(cfg.state_dir, 0o755)
    cfg.spool.mkdir(exist_ok=True)
    # World-writable drop box with the sticky bit, like /tmp: anyone may leave a claim,
    # nobody can remove someone else's. The daemon reads the owner uid as "who asked".
    os.chmod(cfg.spool, 0o1733)
