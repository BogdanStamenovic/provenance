"""Recognise what a directory or file on the watched disk is, and how to get it back.

The walk descends from the disk root. A directory a classifier recognises (git repo, HF
download, ollama store, venv, cache, timeshift) becomes a *unit* and the walk does not go
inside it. Unrecognised directories are containers: the walk goes in, and whatever loose
files it finds there are judged on their own (weights, archives, disk images) or pooled
into one residual "data" unit for that directory.

Every unit ends in one of four risk states:

    recreatable  a known source and a command that brings it back
    redundant    a copy of something that still lives elsewhere (root backups, SSD copies)
    at_risk      unique, no known source -- gone for good if the disk dies
    expendable   someone said losing it does not matter
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

RECREATABLE = "recreatable"
REDUNDANT = "redundant"
AT_RISK = "at_risk"
EXPENDABLE = "expendable"
RISKS = (AT_RISK, RECREATABLE, REDUNDANT, EXPENDABLE)

WEIGHT_EXT = {".safetensors", ".gguf", ".ggml", ".pt", ".pth", ".ckpt", ".onnx", ".h5",
              ".keras", ".pb", ".tflite", ".mlmodel", ".pte", ".engine", ".bin", ".npz",
              ".msgpack", ".nemo"}
# .bin and .npz are only weights when big; plenty of small .bin files are not.
WEIGHT_MIN_AMBIGUOUS = 50 * 1024 * 1024
AMBIGUOUS_EXT = {".bin", ".npz", ".msgpack"}
ARCHIVE_RE = re.compile(r"\.(tar|tgz|tbz2|txz|tzst|zip|7z|rar|zst|xz|gz|bz2|lz4)$", re.IGNORECASE)
IMAGE_EXT = {".iso", ".img", ".qcow2", ".vmdk", ".vdi", ".vhd", ".vhdx", ".raw"}
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "uv-cache",
               ".uv-cache", ".npm", ".cargo-cache", ".gradle", "pip-cache", ".tox", ".nox"}
LOCKFILES = ("package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb", "bun.lock")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ClassifyError(Exception):
    pass


@dataclass
class Unit:
    path: str  # relative to the disk root, "" for the root itself
    kind: str
    risk: str
    files: int = 0
    bytes: int = 0
    source: dict[str, str | None] | None = None
    rebuild: list[str] = field(default_factory=list)
    note: str = ""
    detail: dict[str, object] = field(default_factory=dict)
    listing: list[tuple[str, int, str]] = field(default_factory=list)  # relpath, size, sha
    writer_ids: list[int] = field(default_factory=list)


@dataclass
class Context:
    """What the walk may consult. Kept as callables so tests can fake each of them."""

    root: str
    # dir (relative) -> best known Source dict for writes inside it, or None
    source_for: Callable[[str], dict[str, str | None] | None] = lambda rel: None
    # sha256 -> {"repo","revision","file"} for weights we can match to an HF blob
    hf_blob: Callable[[str], dict[str, str] | None] = lambda sha: None
    # (rel, size, mtime_ns, ino) -> sha256 hex, hashing the file if needed (or None)
    file_sha: Callable[[str, int, int, int], str | None] = lambda rel, s, m, i: None
    # rel -> claim dict ({"kind": "source"|"expendable", "value": ..., "by": ...})
    claim_for: Callable[[str], dict[str, str] | None] = lambda rel: None
    warn: Callable[[str], None] = lambda msg: None


def _q(rel: str) -> str:
    return '"$ROOT/' + rel.replace('"', '\\"') + '"' if rel else '"$ROOT"'


def _git(repo: str, *args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", repo, *args],
            check=False, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def _du(abspath: str, dev: int) -> tuple[int, int, list[tuple[str, int, str]]]:
    """Files, bytes and a (relpath, size) listing for everything under abspath."""
    files = total = 0
    listing: list[tuple[str, int, str]] = []
    stack = [abspath]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if e.is_dir(follow_symlinks=False):
                    if st.st_dev == dev:
                        stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    files += 1
                    total += st.st_size
                    listing.append((os.path.relpath(e.path, abspath), st.st_size, ""))
    return files, total, listing


# --- directory classifiers ------------------------------------------------------------
# Each takes (abs, rel, ctx, dev) and returns a Unit or None.


def _classify_git(abs_: str, rel: str, ctx: Context, dev: int) -> Unit | None:
    if not os.path.lexists(os.path.join(abs_, ".git")):
        return None
    files, total, _ = _du(abs_, dev)
    u = Unit(rel, "git", RECREATABLE, files, total)
    head = (_git(abs_, "rev-parse", "HEAD") or "").strip()
    branch = (_git(abs_, "rev-parse", "--abbrev-ref", "HEAD") or "").strip()
    remotes = (_git(abs_, "remote") or "").split()
    remote = "origin" if "origin" in remotes else (remotes[0] if remotes else None)
    url = (_git(abs_, "remote", "get-url", remote) or "").strip() if remote else ""
    u.source = {"kind": "git", "ref": url or None, "revision": head or None}
    u.detail = {"branch": branch, "head": head}
    problems: list[str] = []
    if not url:
        problems.append("no git remote: the history exists only here")
    if head:
        unpushed = (_git(abs_, "log", "--format=%h %s", "HEAD", "--not", "--remotes") or "")
        commits = [c for c in unpushed.splitlines() if c]
        if commits and url:
            u.detail["unpushed"] = commits[:50]
            problems.append(f"{len(commits)} commit(s) on no remote branch")
    else:
        problems.append("no commits")
    porcelain = _git(abs_, "status", "--porcelain=v1", "-z", "--untracked-files=all") or ""
    dirty = [e[3:] for e in porcelain.split("\0") if e and not e.startswith("??")]
    untracked = [e[3:] for e in porcelain.split("\0") if e.startswith("??")]
    if dirty:
        u.detail["modified"] = dirty[:50]
        problems.append(f"{len(dirty)} uncommitted change(s)")
    if untracked:
        u.detail["untracked"] = untracked[:50]
        problems.append(f"{len(untracked)} untracked file(s)")

    # Ignored content is not brought back by a clone. Judge each ignored path.
    ignored = _git(abs_, "ls-files", "-z", "--others", "--ignored", "--exclude-standard",
                   "--directory") or ""
    lost_bytes = 0
    lost: list[str] = []
    sub: list[dict[str, object]] = []
    extra: list[str] = []
    for ip in [p for p in ignored.split("\0") if p]:
        ip = ip.rstrip("/")
        ipa = os.path.join(abs_, ip)
        name = os.path.basename(ip)
        child_rel = f"{rel}/{ip}" if rel else ip
        if os.path.isdir(ipa) and not os.path.islink(ipa):
            if name == "node_modules" and any(
                    os.path.exists(os.path.join(os.path.dirname(ipa), lf)) for lf in LOCKFILES):
                inner_units = [Unit(child_rel, "node_modules", RECREATABLE,
                                    rebuild=[f"(cd {_q(os.path.dirname(child_rel))} && npm ci)"])]
            else:
                # Walk it like any container: finds venvs, weights, HF downloads inside.
                inner_units = []
                _walk_dir(ipa, child_rel, ctx, dev, inner_units)
                if not inner_units:
                    inner_units = []  # empty dir
            for inner in inner_units:
                ir = os.path.relpath(inner.path, rel) if rel else inner.path
                sub.append({"path": ir, "kind": inner.kind, "risk": inner.risk,
                            "bytes": inner.bytes})
                extra += inner.rebuild
                if inner.risk == AT_RISK:
                    if inner.kind == "data":
                        lost_bytes += inner.bytes
                        lost.append(ir + "/")
                    else:
                        problems.append(f"ignored {ir} [{inner.kind}]: {inner.note}")
        elif os.path.isfile(ipa):
            st = os.lstat(ipa)
            if _is_weights(ipa, st.st_size):
                w = _classify_file(ipa, child_rel, ctx, st)
                sub.append({"path": ip, "kind": w.kind, "risk": w.risk})
                extra += w.rebuild
                if w.risk == AT_RISK:
                    problems.append(f"ignored {ip}: {w.note}")
                continue
            if name.endswith((".pyc", ".log", ".tmp")) or name == ".DS_Store":
                continue
            lost_bytes += st.st_size
            lost.append(ip)
    if sub:
        u.detail["ignored_units"] = sub
    if lost:
        u.detail["ignored_unknown"] = lost[:50]
        problems.append(f"{len(lost)} gitignored path(s) a clone does not restore "
                        f"({_human(lost_bytes)})")

    if url and head:
        # A clone can only reach commits some remote has: restore to the newest of those.
        target = head if not u.detail.get("unpushed") else _pushed_base(abs_, head)
        if target:
            u.rebuild = [f"git clone {shlex.quote(url)} {_q(rel)}"]
            co = (f"checkout -B {shlex.quote(branch)} {target}"
                  if branch and branch != "HEAD" else f"checkout {target}")
            u.rebuild.append(f"git -C {_q(rel)} {co}")
            if target != head:
                u.detail["restores_to"] = target
                u.rebuild.append(f"# local commits after {target[:12]} are lost")
    u.rebuild += extra
    if problems:
        u.risk = AT_RISK
        u.note = "; ".join(problems)
    else:
        u.note = f"clean clone of {url} at {head[:12]}"
    return u


def _pushed_base(repo: str, head: str) -> str | None:
    """Newest commit in HEAD's history that some remote-tracking ref also contains."""
    refs = (_git(repo, "for-each-ref", "--format=%(objectname)", "refs/remotes") or "").split()
    best: tuple[int, str] | None = None
    for r in set(refs):
        mb = (_git(repo, "merge-base", head, r) or "").strip()
        if not mb:
            continue
        t = (_git(repo, "show", "-s", "--format=%ct", mb) or "0").strip()
        cand = (int(t or 0), mb)
        if best is None or cand > best:
            best = cand
    return best[1] if best else None


def _hf_cache_repo(abs_: str, rel: str, dev: int) -> Unit | None:
    name = os.path.basename(abs_)
    m = re.match(r"^(models|datasets|spaces)--(.+?)--(.+)$", name)
    if not m or not os.path.isdir(os.path.join(abs_, "snapshots")):
        return None
    rtype, org, repo = m.groups()
    repo_id = f"{org}/{repo}"
    files, total, _ = _du(abs_, dev)
    listing: list[tuple[str, int, str]] = []
    revs = sorted(os.listdir(os.path.join(abs_, "snapshots")))
    for rev in revs:
        sd = os.path.join(abs_, "snapshots", rev)
        for dp, _dn, fn in os.walk(sd):
            for f in fn:
                p = os.path.join(dp, f)
                tgt = os.path.basename(os.path.realpath(p))
                try:
                    size = os.stat(p).st_size
                except OSError:
                    size = -1
                listing.append((f"{rev[:12]}/{os.path.relpath(p, sd)}", size,
                                tgt if HEX64.match(tgt) else ""))
    hub = os.path.dirname(rel)
    flag = "" if rtype == "models" else f" --repo-type {rtype[:-1]}"
    u = Unit(rel, "hf-cache", RECREATABLE, files, total, listing=listing)
    u.source = {"kind": "hf", "ref": repo_id, "revision": ",".join(revs) or None}
    u.rebuild = [f"hf download {repo_id}{flag} --revision {r} --cache-dir {_q(hub)}"
                 for r in revs]
    u.note = f"HF {rtype[:-1]} {repo_id}, {len(revs)} revision(s)"
    if not revs:
        u.risk = EXPENDABLE
        u.note += " (empty: only partial blobs)"
    return u


def _hf_local_dir(abs_: str, rel: str, ctx: Context, dev: int) -> Unit | None:
    meta = os.path.join(abs_, ".cache", "huggingface", "download")
    if not os.path.isdir(meta):
        return None
    commits: set[str] = set()
    etags: dict[str, str] = {}
    for dp, _dn, fn in os.walk(meta):
        for f in fn:
            if not f.endswith(".metadata"):
                continue
            try:
                lines = Path(dp, f).read_text().splitlines()
            except OSError:
                continue
            if lines and HEX40.match(lines[0].strip()):
                commits.add(lines[0].strip())
            if len(lines) > 1:
                relf = os.path.relpath(os.path.join(dp, f[: -len(".metadata")]), meta)
                etags[relf] = lines[1].strip().strip('"')
    files, total, raw = _du(abs_, dev)
    listing = [(p, s, etags.get(p, "") if HEX64.match(etags.get(p, "")) else "")
               for p, s, _ in raw if not p.startswith(".cache/")]
    u = Unit(rel, "hf-local", RECREATABLE, files, total, listing=listing)
    src = ctx.source_for(rel)
    rev = min(commits) if len(commits) == 1 else None
    repo_id = src["ref"] if src and src.get("kind") == "hf" else None
    if repo_id is None:
        # Fall back to matching file hashes against HF blobs we know of.
        for _p, _s, sha in listing:
            hit = ctx.hf_blob(sha) if sha else None
            if hit:
                repo_id = hit["repo"]
                break
    u.source = {"kind": "hf", "ref": repo_id, "revision": rev or ",".join(sorted(commits))}
    u.detail = {"commits": sorted(commits)}
    if repo_id:
        r = f" --revision {rev}" if rev else ""
        u.rebuild = [f"hf download {repo_id}{r} --local-dir {_q(rel)}"]
        u.note = f"HF download of {repo_id}" + (f" at {rev[:12]}" if rev else "")
    else:
        u.risk = AT_RISK
        u.note = ("HF --local-dir download but the repo id is unknown (HF stores only the "
                  f"commit: {', '.join(c[:12] for c in sorted(commits)) or 'none'})")
    return u


def _ollama(abs_: str, rel: str, dev: int) -> Unit | None:
    man = os.path.join(abs_, "manifests")
    if not (os.path.isdir(man) and os.path.isdir(os.path.join(abs_, "blobs"))):
        return None
    tags: list[str] = []
    for dp, _dn, fn in os.walk(man):
        for f in fn:
            parts = os.path.relpath(os.path.join(dp, f), man).split(os.sep)
            if len(parts) < 4:
                continue
            host, ns, model, tag = parts[0], parts[1], "/".join(parts[2:-1]), parts[-1]
            name = model if ns == "library" else f"{ns}/{model}"
            if host != "registry.ollama.ai":
                name = f"{host}/{ns}/{model}"
            tags.append(f"{name}:{tag}")
    files, total, _ = _du(abs_, dev)
    u = Unit(rel, "ollama", RECREATABLE, files, total)
    u.source = {"kind": "ollama", "ref": ",".join(sorted(tags)), "revision": None}
    u.rebuild = [f"OLLAMA_MODELS={_q(rel)} ollama pull {shlex.quote(t)}" for t in sorted(tags)]
    u.note = f"ollama store, {len(tags)} model(s)"
    return u


def _venv(abs_: str, rel: str, dev: int) -> Unit | None:
    cfg = os.path.join(abs_, "pyvenv.cfg")
    if not os.path.isfile(cfg):
        return None
    pyver = ""
    try:
        for line in Path(cfg).read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("version", "version_info"):
                pyver = v.strip()
    except OSError:
        pass
    reqs: list[str] = []
    local: list[str] = []
    for sp in Path(abs_).glob("lib/python*/site-packages"):
        for di in sp.glob("*.dist-info"):
            name = ver = ""
            try:
                for line in (di / "METADATA").read_text(errors="replace").splitlines()[:20]:
                    if line.startswith("Name: "):
                        name = line[6:].strip()
                    elif line.startswith("Version: "):
                        ver = line[9:].strip()
            except OSError:
                continue
            if not name:
                continue
            du = di / "direct_url.json"
            if du.exists():
                try:
                    url = json.loads(du.read_text()).get("url", "")
                except (OSError, ValueError):
                    url = ""
                if url.startswith("file://"):
                    local.append(f"{name} ({url[7:]})")
                    continue
                if url:
                    reqs.append(f"{name} @ {url}")
                    continue
            reqs.append(f"{name}=={ver}")
    files, total, _ = _du(abs_, dev)
    u = Unit(rel, "venv", RECREATABLE, files, total)
    short = ".".join(pyver.split(".")[:2]) if pyver else ""
    u.detail = {"python": pyver, "requirements": sorted(reqs, key=str.lower)}
    pyflag = f" --python {short}" if short else ""
    u.rebuild = [
        f"uv venv{pyflag} {_q(rel)}",
        f"uv pip install --python {_q(rel + '/bin/python')} -r - <<'REQ'\n"
        + "\n".join(u.detail["requirements"]) + "\nREQ",  # type: ignore[arg-type]
    ]
    u.note = f"Python {pyver or '?'} venv, {len(reqs)} package(s)"
    if local:
        u.detail["local_installs"] = local
        u.note += f"; {len(local)} installed from local paths (rebuild needs those paths)"
    return u


def _cache(abs_: str, rel: str, dev: int) -> Unit | None:
    name = os.path.basename(abs_)
    parts = rel.split("/")
    is_cache = name in CACHE_NAMES or (len(parts) >= 2 and parts[-2] == ".cache"
                                       and name in ("uv", "pip", "pnpm", "yarn", "go-build"))
    if not is_cache:
        return None
    files, total, _ = _du(abs_, dev)
    u = Unit(rel, "cache", EXPENDABLE, files, total)
    u.note = f"{name} cache: refills on use"
    return u


def _timeshift(abs_: str, rel: str, dev: int) -> Unit | None:
    snaps = os.path.join(abs_, "snapshots")
    if os.path.basename(abs_) != "timeshift" or not os.path.isdir(snaps):
        return None
    taken: list[str] = []
    for s in sorted(os.listdir(snaps)):
        info = os.path.join(snaps, s, "info.json")
        if os.path.isfile(info):
            taken.append(s)
    # Snapshots are hardlink farms; a full du of millions of links is slow and the number
    # is meaningless (each snapshot "contains" all of /). Count apparent size of the top.
    u = Unit(rel, "timeshift", REDUNDANT, len(taken), 0)
    u.detail = {"snapshots": taken}
    u.note = (f"{len(taken)} timeshift snapshot(s) of the root SSD: a backup, the live "
              "system is the original. Losing it loses history, not data.")
    # Deliberately a comment: a restore script writes only under $ROOT. Running
    # `timeshift --create` from it would snapshot the live root (it did, once, in testing).
    u.rebuild = [("# timeshift backups: take a fresh one with `sudo timeshift --create` "
                 "once the new disk is set as the timeshift device")]
    return u


DIR_CLASSIFIERS: list[Callable[[str, str, int], Unit | None]] = [
    _hf_cache_repo, _ollama, _venv, _cache, _timeshift,
]


def _classify_dir_unit(abs_: str, rel: str, ctx: Context, dev: int) -> Unit | None:
    u = _classify_git(abs_, rel, ctx, dev)
    if u:
        return u
    u = _hf_local_dir(abs_, rel, ctx, dev)
    if u:
        return u
    for fn in DIR_CLASSIFIERS:
        u = fn(abs_, rel, dev)
        if u:
            return u
    return None


# --- loose files ----------------------------------------------------------------------


def _is_weights(path: str, size: int) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext not in WEIGHT_EXT:
        return False
    return ext not in AMBIGUOUS_EXT or size >= WEIGHT_MIN_AMBIGUOUS


def file_kind_by_name(path: str) -> bool:
    """Could this file get a unit of its own (decided before its size is known)?"""
    ext = os.path.splitext(path)[1].lower()
    return ext in WEIGHT_EXT or ext in IMAGE_EXT or bool(ARCHIVE_RE.search(path))


def _file_kind(path: str, size: int) -> str | None:
    if _is_weights(path, size):
        return "weights"
    if ARCHIVE_RE.search(path):
        return "archive"
    if os.path.splitext(path)[1].lower() in IMAGE_EXT:
        return "disk-image"
    return None


def _classify_file(abs_: str, rel: str, ctx: Context, st: os.stat_result) -> Unit:
    kind = _file_kind(abs_, st.st_size) or "file"
    u = Unit(rel, kind, AT_RISK, 1, st.st_size, listing=[("", st.st_size, "")])
    parent = os.path.dirname(rel)
    # Only this file's own writes count. A sibling's download is not this file's source.
    src = ctx.source_for(rel)
    if src and src.get("kind") == "url":
        u.risk = RECREATABLE
        u.source = src
        u.rebuild = [f"mkdir -p {_q(parent)}",
                     f"curl -fL -o {_q(rel)} {shlex.quote(str(src['ref']))}"]
        u.note = f"downloaded from {src['ref']}"
        return u
    if src and src.get("kind") == "hf":
        u.risk = RECREATABLE
        u.source = src
        rev = f" --revision {src['revision']}" if src.get("revision") else ""
        fname = str(src.get("file") or os.path.basename(rel))
        u.rebuild = [(f"hf download {src['ref']} {shlex.quote(fname)}{rev} "
                     f"--local-dir {_q(parent + '/.provenance-fetch')}"),
                     f"mv {_q(parent + '/.provenance-fetch/' + fname)} {_q(rel)}",
                     f"rm -rf {_q(parent + '/.provenance-fetch')}"]
        u.note = f"file {fname} from HF {src['ref']}"
        return u
    if kind == "weights":
        sha = ctx.file_sha(rel, st.st_size, st.st_mtime_ns, st.st_ino)
        hit = ctx.hf_blob(sha) if sha else None
        if sha:
            u.listing = [("", st.st_size, sha)]
        if hit:
            u.risk = RECREATABLE
            u.source = {"kind": "hf", "ref": hit["repo"], "revision": hit.get("revision")}
            fname = hit["file"]
            u.rebuild = [
                (f"hf download {hit['repo']} {shlex.quote(fname)} --revision {hit['revision']} "
                f"--local-dir {_q(parent + '/.provenance-fetch')}"),
                f"mv {_q(parent + '/.provenance-fetch/' + fname)} {_q(rel)}",
                f"rm -rf {_q(parent + '/.provenance-fetch')}",
            ]
            u.note = f"identical (sha256) to {hit['repo']}:{fname}"
            return u
        u.note = ("weights with no known source: not an HF file we know by hash and no "
                  "recognised download wrote it. Probably trained here.")
        return u
    if src and src.get("kind") == "copy":
        u.source = src
        u.note = f"copied from {src['ref']}"
        if _exists_off_disk(str(src["ref"]), ctx.root):
            u.risk = REDUNDANT
            u.note += " (still there)"
            u.rebuild = [f"rsync -a {shlex.quote(str(src['ref']))} {_q(rel)}"]
        return u
    u.note = f"{kind} with no known source"
    return u


def _exists_off_disk(path: str, root: str) -> bool:
    for p in path.split():
        if p.startswith(root):
            return False
        if os.path.exists(p):
            return True
    return False


# --- the walk -------------------------------------------------------------------------


def walk(ctx: Context, start: str = "") -> list[Unit]:
    """Classify everything under `start` (relative to ctx.root)."""
    root_abs = os.path.realpath(ctx.root)
    start_abs = os.path.join(root_abs, start) if start else root_abs
    try:
        dev = os.lstat(root_abs).st_dev
    except OSError as exc:
        raise ClassifyError(f"cannot stat {root_abs}: {exc}") from exc
    units: list[Unit] = []
    if os.path.isfile(start_abs):
        units.append(_classify_file(start_abs, start, ctx, os.lstat(start_abs)))
        return _apply_claims(units, ctx)
    _walk_dir(start_abs, start, ctx, dev, units, top=True)
    return _apply_claims(units, ctx)


SKIP_TOP = {"lost+found", ".Trash-1000", ".provenance"}


def walk_shallow(ctx: Context, start: str) -> list[Unit]:
    """Only the loose files directly in container `start`, plus any new child directory
    that is not yet known (the caller replaces just these, leaving deeper units alone)."""
    root_abs = os.path.realpath(ctx.root)
    abs_ = os.path.join(root_abs, start) if start else root_abs
    out: list[Unit] = []
    residual: list[tuple[str, int, str]] = []
    try:
        entries = sorted(os.scandir(abs_), key=lambda e: e.name)
    except OSError as exc:
        raise ClassifyError(f"cannot read {abs_}: {exc}") from exc
    for e in entries:
        if not start and e.name in SKIP_TOP:
            continue
        crel = f"{start}/{e.name}" if start else e.name
        try:
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        if e.is_file(follow_symlinks=False):
            if _file_kind(e.path, st.st_size) or ctx.claim_for(crel):
                out.append(_classify_file(e.path, crel, ctx, st))
            else:
                residual.append((e.name, st.st_size, ""))
    if residual:
        out.append(_residual_unit(start, residual, ctx))
    return _apply_claims(out, ctx)


def _residual_unit(rel: str, residual: list[tuple[str, int, str]], ctx: Context,
                   subtree: bool = False) -> Unit:
    u = Unit(rel, "data", AT_RISK, len(residual), sum(s for _, s, _ in residual),
             listing=residual)
    if subtree:
        u.detail["subtree"] = True
    src = ctx.source_for(rel)
    if src:
        u.source = src
        if src.get("kind") == "copy" and _exists_off_disk(str(src["ref"]), ctx.root):
            u.risk = REDUNDANT
            u.note = f"loose files copied from {src['ref']} (still there)"
        else:
            u.note = f"loose files; written by: {src.get('command')}"
    else:
        u.note = "loose files with no known source"
    return u


def _walk_dir(abs_: str, rel: str, ctx: Context, dev: int, out: list[Unit],
              top: bool = False) -> None:
    if rel:
        u = _classify_dir_unit(abs_, rel, ctx, dev)
        if u:
            out.append(u)
            return
    try:
        entries = sorted(os.scandir(abs_), key=lambda e: e.name)
    except OSError as exc:
        ctx.warn(f"cannot read {abs_}: {exc}")
        return
    residual: list[tuple[str, int, str]] = []
    mark = len(out)
    for e in entries:
        if not rel and e.name in SKIP_TOP:
            continue
        crel = f"{rel}/{e.name}" if rel else e.name
        try:
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        if e.is_dir(follow_symlinks=False):
            if st.st_dev != dev:
                continue
            _walk_dir(e.path, crel, ctx, dev, out)
        elif e.is_file(follow_symlinks=False):
            if _file_kind(e.path, st.st_size) or ctx.claim_for(crel):
                out.append(_classify_file(e.path, crel, ctx, st))
            else:
                residual.append((e.name, st.st_size, ""))
    if residual:
        out.append(_residual_unit(rel, residual, ctx))
    # A subtree holding nothing but unclassified data is one unit at its top, not one per
    # leaf folder: 10k image-class folders are "a dataset", not 10k questions. Stops at
    # the disk's top level so each top-level folder still gets its own entry.
    below = out[mark:]
    if rel and len(below) > 1 and all(u.kind == "data" for u in below) and \
            not any(ctx.claim_for(u.path) for u in below if u.path != rel):
        merged: list[tuple[str, int, str]] = []
        for u in below:
            sub = os.path.relpath(u.path, rel) if u.path != rel else ""
            merged += [(os.path.join(sub, p) if sub else p, sz, h) for p, sz, h in u.listing]
        del out[mark:]
        out.append(_residual_unit(rel, merged, ctx, subtree=True))


def _apply_claims(units: list[Unit], ctx: Context) -> list[Unit]:
    for u in units:
        c = ctx.claim_for(u.path)
        if not c:
            continue
        u.detail["claim"] = c
        if c["kind"] == "expendable":
            u.risk = EXPENDABLE
            u.note = f"marked expendable by {c.get('by', '?')}: {c.get('value') or 'no reason'}"
        elif c["kind"] == "source":
            src = parse_source(c["value"])
            u.source = src
            u.risk = RECREATABLE
            u.rebuild = rebuild_for_claim(u, src)
            u.note = f"source given by {c.get('by', '?')}: {c['value']}"
    return units


def parse_source(text: str) -> dict[str, str | None]:
    t = text.strip()
    m = re.match(r"^(?:hf:|https?://huggingface\.co/)([\w.-]+/[\w.-]+)(?:@(\S+))?/?$", t)
    if m:
        return {"kind": "hf", "ref": m.group(1), "revision": m.group(2)}
    if t.startswith("git:"):
        return {"kind": "git", "ref": t[4:], "revision": None}
    if re.match(r"^(https?|ftp)://\S+$", t):
        if re.search(r"(github|gitlab|codeberg)\.\w+/[^/]+/[^/]+?(\.git)?/?$", t):
            return {"kind": "git", "ref": t, "revision": None}
        return {"kind": "url", "ref": t, "revision": None}
    return {"kind": "manual", "ref": t, "revision": None}


def rebuild_for_claim(u: Unit, src: dict[str, str | None]) -> list[str]:
    ref = str(src["ref"])
    if src["kind"] == "hf":
        rev = f" --revision {src['revision']}" if src.get("revision") else ""
        if u.kind in ("weights", "file", "archive", "disk-image"):
            return [(f"hf download {ref} {shlex.quote(os.path.basename(u.path))}{rev} "
                    f"--local-dir {_q(os.path.dirname(u.path))}")]
        return [f"hf download {ref}{rev} --local-dir {_q(u.path)}"]
    if src["kind"] == "git":
        return [f"git clone {shlex.quote(ref)} {_q(u.path)}"]
    if src["kind"] == "url":
        return [f"mkdir -p {_q(os.path.dirname(u.path))}",
                f"curl -fL -o {_q(u.path)} {shlex.quote(ref)}"]
    return ["# manual: " + ref.replace("\n", " ")]


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{n} B"


human = _human
