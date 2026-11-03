"""Who wrote a file, and what that tells us about where it came from.

A write event carries only a pid. We walk that pid's ancestry in /proc while it is still
alive, keep each process's argv and cwd, and pull out two things:

* the Claude Code session it belongs to (CLAUDE_CODE_SESSION_ID is inherited by every
  child of a session), which hotline maps to an agent name we can ask;
* a recognisable fetch command anywhere in the chain (hf download, git clone, wget, ...),
  which is the source.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_DEPTH = 12
AGENTS_JSON = Path("/home/bodas/.local/state/hotline/agents.json")


@dataclass
class Proc:
    pid: int
    argv: list[str]
    cwd: str


@dataclass
class Writer:
    chain: list[Proc] = field(default_factory=list)
    session_id: str | None = None
    user: str | None = None

    @property
    def key(self) -> str:
        return " | ".join(" ".join(p.argv)[:200] for p in self.chain[:4])

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> Writer:
        d = json.loads(raw)
        return cls(
            chain=[Proc(**p) for p in d.get("chain", [])],
            session_id=d.get("session_id"),
            user=d.get("user"),
        )


@dataclass
class Source:
    kind: str  # hf | git | url | ollama | pip | copy
    ref: str  # repo id, URL, model tag, or source path
    revision: str | None = None
    command: str | None = None  # the original command, verbatim
    file: str | None = None  # for a single file out of an HF repo


# .../models--org--name/snapshots/<rev>/<file>  (an HF hub cache path)
HF_SNAPSHOT = re.compile(r"/models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]{40})(?:/(.+))?$")


def hf_snapshot_path(path: str) -> tuple[str, str, str | None] | None:
    """(repo_id, revision, file-or-None) if `path` points into an HF hub snapshot."""
    m = HF_SNAPSHOT.search(path.rstrip("/"))
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}", m.group(3), m.group(4)


def _read(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def _session_of(pid: int) -> str | None:
    env = _read(f"/proc/{pid}/environ")
    if not env:
        return None
    for item in env.split(b"\0"):
        if item.startswith(b"CLAUDE_CODE_SESSION_ID="):
            return item.split(b"=", 1)[1].decode(errors="replace")
    return None


def _ppid(pid: int) -> int | None:
    stat = _read(f"/proc/{pid}/stat")
    if not stat:
        return None
    # comm may contain spaces and parens; the fields after the last ')' are fixed.
    rest = stat.rsplit(b")", 1)[-1].split()
    try:
        return int(rest[1])
    except (IndexError, ValueError):
        return None


def capture(pid: int, proc_root: str = "/proc") -> Writer:
    """Snapshot the ancestry of `pid`. Cheap; call it as soon as an event arrives."""
    w = Writer()
    seen: set[int] = set()
    cur: int | None = pid
    while cur and cur > 1 and cur not in seen and len(w.chain) < MAX_DEPTH:
        seen.add(cur)
        raw = _read(f"{proc_root}/{cur}/cmdline")
        if raw is None:
            break
        argv = [a.decode(errors="replace") for a in raw.rstrip(b"\0").split(b"\0") if a]
        try:
            cwd = os.readlink(f"{proc_root}/{cur}/cwd")
        except OSError:
            cwd = ""
        w.chain.append(Proc(pid=cur, argv=argv, cwd=cwd))
        if w.session_id is None:
            w.session_id = _session_of(cur)
        if w.user is None:
            try:
                uid = os.stat(f"{proc_root}/{cur}").st_uid
                import pwd

                w.user = pwd.getpwuid(uid).pw_name
            except (OSError, KeyError):
                pass
        cur = _ppid(cur)
    return w


def agent_name(session_id: str | None, agents_json: Path = AGENTS_JSON) -> str | None:
    if not session_id:
        return None
    try:
        data = json.loads(agents_json.read_text())
    except (OSError, ValueError):
        return None
    for a in data.get("agents", []):
        if a.get("session_id") == session_id:
            name = a.get("name")
            return str(name) if name else None
    return None


# --- recognising fetch commands -------------------------------------------------------

_HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_URL = re.compile(r"^(https?|ftp)://\S+$")


def _strip_interp(argv: list[str]) -> list[str]:
    """Drop `python -m`, `uv run`, `uvx`, env assignments, so argv[0] is the tool."""
    a = list(argv)
    while a and "=" in a[0] and not a[0].startswith("-") and "/" not in a[0].split("=")[0]:
        a = a[1:]
    if a and os.path.basename(a[0]) in ("env", "nice", "ionice", "nohup", "sudo", "timeout"):
        a = a[1:]
        while a and (a[0].startswith("-") or "=" in a[0] or a[0].replace(".", "").isdigit()):
            a = a[1:]
    if a and re.match(r"python[\d.]*$", os.path.basename(a[0])):
        if len(a) > 2 and a[1] == "-m":
            a = a[2:]
        elif len(a) > 1:
            a = a[1:]
    if a and os.path.basename(a[0]) in ("uv", "uvx"):
        a = a[2:] if len(a) > 1 and a[1] in ("run", "tool") else a[1:]
        if a and a[0] == "run":
            a = a[1:]
    return a


def _opt(argv: list[str], *names: str) -> str | None:
    for i, tok in enumerate(argv):
        for n in names:
            if tok == n and i + 1 < len(argv):
                return argv[i + 1]
            if tok.startswith(n + "="):
                return tok.split("=", 1)[1]
    return None


def _positionals(argv: list[str], with_value: tuple[str, ...] = ()) -> list[str]:
    out: list[str] = []
    skip = False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok.startswith("-"):
            if tok in with_value:
                skip = True
            continue
        out.append(tok)
    return out


def source_from_argv(argv: list[str], cwd: str = "") -> Source | None:
    a = _strip_interp(argv)
    if not a:
        return None
    tool = os.path.basename(a[0])
    cmd = shlex.join(argv)
    rest = a[1:]

    if tool in ("hf", "huggingface-cli") and rest[:1] == ["download"]:
        hv = ("--revision", "--local-dir", "--cache-dir", "--repo-type", "--include",
              "--exclude", "--token", "--max-workers")
        pos = _positionals(rest[1:], hv)
        if pos and _HF_ID.match(pos[0]):
            rtype = _opt(rest, "--repo-type") or "model"
            ref = pos[0] if rtype == "model" else f"{rtype}s/{pos[0]}"
            return Source("hf", ref, _opt(rest, "--revision"), cmd)

    if tool == "git" and "clone" in rest:
        i = rest.index("clone")
        pos = _positionals(rest[i + 1 :], ("-b", "--branch", "--depth", "-o", "--origin",
                                           "--reference", "-c", "--config", "--filter"))
        if pos:
            return Source("git", pos[0], _opt(rest, "-b", "--branch"), cmd)

    if tool in ("wget", "curl", "aria2c", "axel", "yt-dlp", "gdown"):
        urls = [t for t in rest if _URL.match(t)]
        if urls:
            return Source("url", urls[0], None, cmd)

    if tool == "ollama" and rest[:1] == ["pull"] and len(rest) > 1:
        return Source("ollama", rest[1], None, cmd)

    if tool in ("rsync", "cp", "mv", "ln") and rest:
        pos = _positionals(rest, ("-e", "--rsh", "--exclude", "--include", "-t",
                                  "--target-directory"))
        if len(pos) >= 2:
            srcs = pos[:-1]
            hits = [hf_snapshot_path(os.path.normpath(os.path.join(cwd or "/", x)))
                    for x in srcs]
            if len(srcs) == 1 and hits[0]:
                repo, rev, f = hits[0]
                return Source("hf", repo, rev, cmd, f)
            return Source("copy", " ".join(srcs), None, cmd)
    return None


def hf_blobs_named_in(w: Writer) -> list[tuple[str, str, str, str]]:
    """(sha256, repo, rev, file) for HF snapshot files any process in the chain names.

    Read at event time, while the cache still exists: the symlink in snapshots/ points at
    blobs/<sha256>. This is what lets a copy be identified even if the cache is wiped a
    second later.
    """
    out: list[tuple[str, str, str, str]] = []
    for p in w.chain:
        for tok in p.argv[1:]:
            if tok.startswith("-"):
                continue
            # Resolve against the process cwd first: `cp model.safetensors /hdd/` run from
            # inside a snapshot directory names no snapshot path in its argv at all.
            path = os.path.normpath(tok if os.path.isabs(tok) else os.path.join(p.cwd, tok))
            hit = hf_snapshot_path(path)
            if not hit:
                continue
            repo, rev, f = hit
            base = path[: path.index("/snapshots/") + len("/snapshots/") + 40]
            targets = [os.path.join(base, f)] if f else [
                os.path.join(dp, x) for dp, _d, fn in os.walk(base) for x in fn]
            for t in targets:
                blob = os.path.basename(os.path.realpath(t))
                if re.fullmatch(r"[0-9a-f]{64}", blob):
                    out.append((blob, repo, rev, os.path.relpath(t, base)))
    return out


def source_of(w: Writer) -> Source | None:
    """The first recognisable fetch command in the chain, innermost first.

    A `copy` match is weaker than a real fetch further up (rsync spawned by a download
    script), so we keep looking past it.
    """
    weak: Source | None = None
    for p in w.chain:
        s = source_from_argv(p.argv, p.cwd)
        if s is None:
            continue
        if s.kind == "copy":
            weak = weak or s
            continue
        return s
    return weak
