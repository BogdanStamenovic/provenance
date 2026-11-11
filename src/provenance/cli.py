"""Command-line interface for provenance.

Keeps stdout clean for real output (tables, JSON, rebuild scripts); all progress,
warnings and errors go to stderr.
Exit codes: 0 success, 1 one or more operations failed (or, for `status --strict`,
something is at risk), 2 usage error / user aborted.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

from . import __version__, attribution
from .classify import AT_RISK, EXPENDABLE, RECREATABLE, REDUNDANT, RISKS, human
from .config import Config, ConfigError, load
from .engine import Daemon, Disk, Engine, EngineError, ensure_state, probe_disk
from .store import Store, StoreError, UnitRow, sha256_file

Log = Callable[[str], None]

RISK_LABEL = {AT_RISK: "AT RISK", RECREATABLE: "recreatable", REDUNDANT: "redundant",
              EXPENDABLE: "expendable"}


class _UsageError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    p = _ArgumentParser(
        prog="provenance",
        description="Keep a recipe on the SSD for recreating everything on an untrusted disk.",
    )
    p.add_argument("--config", help="config file (default /etc/provenance.toml)")
    p.add_argument("-v", "--verbose", action="store_true", help="print detailed progress")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress non-error output")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_ArgumentParser)

    s = sub.add_parser("status", help="what is on each disk, by risk")
    s.add_argument("--json", action="store_true")
    s.add_argument("--strict", action="store_true", help="exit 1 if anything is at risk")

    s = sub.add_parser("ls", help="list units")
    s.add_argument("path", nargs="?", default="", help="only units at or under this path")
    s.add_argument("--risk", choices=RISKS)
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("show", help="everything known about one unit")
    s.add_argument("path")
    s.add_argument("--files", action="store_true", help="print the stored file listing")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("scan", help="scan now (root); --dry-run classifies without storing")
    s.add_argument("path", nargs="?", help="only this subtree (default: every watched disk)")
    s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("claim", help="give a source for something, or mark it expendable")
    s.add_argument("path")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="hf:org/name[@rev], git:URL, https://URL, or free text")
    g.add_argument("--expendable", nargs="?", const="", metavar="REASON")
    g.add_argument("--clear", action="store_true")
    g.add_argument("--keep", nargs="?", const="", metavar="REASON",
                   help="never prune this (single-disk mode)")

    s = sub.add_parser("prune", help="single-disk mode: what could be deleted (dry run)")
    s.add_argument("area", nargs="?", help="only this area (default: all configured)")
    s.add_argument("--days", type=float, help="unused for at least this long")
    s.add_argument("--all", action="store_true", help="also list what is NOT prunable, and why")
    s.add_argument("--offline", action="store_true", help="skip the online source checks")
    s.add_argument("--json", action="store_true")
    g2 = s.add_mutually_exclusive_group()
    g2.add_argument("--propose", action="store_true",
                    help="save the plan and send it to prune_approver for review (root)")
    g2.add_argument("--approve", metavar="PLAN")
    g2.add_argument("--reject", metavar="PLAN")
    g2.add_argument("--apply", action="store_true",
                    help="delete now, without a review step (root; asks first)")
    s.add_argument("--yes", action="store_true", help="do not ask before --apply")

    s = sub.add_parser("plans", help="prune plans and their status")
    s.add_argument("--show", metavar="PLAN", help="print one plan in full")

    s = sub.add_parser("pruned", help="what single-disk mode deleted, and how to get it back")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("restore", help="bring back one pruned unit (from trash or the recipe)")
    s.add_argument("path")
    s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("rebuild", help="print (or --run) the script that recreates a disk")
    s.add_argument("path", nargs="?", default="", help="only this subtree")
    s.add_argument("--disk", help="disk uuid (default: the only/most recent one)")
    s.add_argument("--into", help="target root (default: the disk's own mount point)")
    s.add_argument("--run", action="store_true", help="execute it instead of printing")
    s.add_argument("--yes", action="store_true", help="do not ask before --run")

    s = sub.add_parser("verify-restore", help="compare a restored tree with the recipe")
    s.add_argument("root", help="where the disk was rebuilt")
    s.add_argument("--disk")
    s.add_argument("--hash", action="store_true", help="also check sha256 where recorded")

    s = sub.add_parser("check-sources", help="check that recorded sources still exist online")
    s.add_argument("--disk")
    s.add_argument("--timeout", type=float, default=20.0)

    s = sub.add_parser("notify", help="tell writers about at-risk data now")
    s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("export", help="write the whole recipe as one .json.xz file")
    s.add_argument("-o", "--output", required=True)

    s = sub.add_parser("identify", help="look up unknown weights online via HF repo trees")
    s.add_argument("--timeout", type=float, default=20.0)

    sub.add_parser("watch", help="run the daemon (root; normally via systemd)")
    sub.add_parser("index-hf", help="re-index HF hub caches by blob sha256 (root)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"provenance: error: {exc}", file=sys.stderr)
        return 2

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr)

    def vlog(message: str) -> None:
        if args.verbose and not args.quiet:
            print(message, file=sys.stderr)

    try:
        cfg = load(args.config)
        return _dispatch(args, cfg, log, vlog)
    except (ConfigError, StoreError, EngineError) as exc:
        print(f"provenance: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 2


def _open(cfg: Config, write: bool) -> Store:
    if write:
        if os.geteuid() != 0 and not os.access(cfg.state_dir, os.W_OK):
            raise EngineError(f"writing the recipe in {cfg.state_dir} needs root (sudo)")
        ensure_state(cfg)
        return Store(cfg.db_path)
    return Store(cfg.db_path, readonly=True)


def _pick_disk(store: Store, uuid: str | None) -> str:
    disks = store.disks()
    if uuid:
        for d in disks:
            if d["uuid"].startswith(uuid):
                return str(d["uuid"])
        raise EngineError(f"no disk {uuid} in the recipe")
    if not disks:
        raise EngineError("the recipe knows no disks yet")
    return str(disks[0]["uuid"])


def _dispatch(args: argparse.Namespace, cfg: Config, log: Log, vlog: Log) -> int:
    c = args.cmd
    if c == "status":
        return _status(cfg, args)
    if c == "ls":
        return _ls(cfg, args)
    if c == "show":
        return _show(cfg, args)
    if c == "scan":
        return _scan(cfg, args, log, vlog)
    if c == "claim":
        return _claim(cfg, args, log)
    if c == "rebuild":
        return _rebuild(cfg, args, log)
    if c == "verify-restore":
        return _verify_restore(cfg, args, log)
    if c == "check-sources":
        return _check_sources(cfg, args, log)
    if c == "notify":
        st = _open(cfg, write=not args.dry_run)
        eng = Engine(cfg, st, log, log)
        for root in cfg.disks:
            d = probe_disk(root)
            if d:
                for line in eng.notify(d, dry_run=args.dry_run):
                    print(line)
        return 0
    if c == "prune":
        return _prune(cfg, args, log)
    if c == "plans":
        return _plans(cfg, args)
    if c == "pruned":
        return _pruned(cfg, args)
    if c == "restore":
        return _restore(cfg, args, log)
    if c == "export":
        return _export(cfg, args, log)
    if c == "watch":
        st = _open(cfg, write=True)
        eng = Engine(cfg, st, log, lambda m: log(f"warning: {m}"))
        if not cfg.disks:
            log(f"no disks configured in {cfg.path}; idle until the config names one")
        else:
            log(f"provenance {__version__}: recipe at {cfg.db_path}, watching {cfg.disks}")
        Daemon(eng).run()
        return 0
    if c == "identify":
        st = _open(cfg, write=True)
        eng = Engine(cfg, st, log, lambda m: log(f"warning: {m}"))
        for root in cfg.disks:
            d = probe_disk(root)
            if d is None:
                continue
            hits = eng.identify_online(d, timeout=args.timeout, recheck_days=0)
            for rel in hits:
                for u in eng.scan(d, rel):
                    print(f"{RISK_LABEL[u.risk]:<12} {u.path}  {u.note}")
            log(f"{d.root}: {len(hits)} weights file(s) identified")
        return 0
    if c == "index-hf":
        st = _open(cfg, write=True)
        print(Engine(cfg, st, log, log).refresh_hf_index())
        return 0
    raise _UsageError(c)


# --- status / ls / show ------------------------------------------------------------------


def _status(cfg: Config, args: argparse.Namespace) -> int:
    st = _open(cfg, write=False)
    out: list[dict[str, Any]] = []
    at_risk_any = False
    for d in st.disks():
        units = st.units(d["uuid"])
        mounted = probe_disk(d["root"])
        by: dict[str, list[int]] = {r: [0, 0] for r in RISKS}
        for u in units:
            by[u.risk][0] += 1
            by[u.risk][1] += u.bytes
        risky = [u for u in units if u.risk == AT_RISK]
        # "Recreatable" from a source that has since vanished is not recreatable.
        gone = [u for u in units if u.risk == RECREATABLE and _source_gone(u)]
        at_risk_any |= bool(risky or gone)
        out.append({
            "uuid": d["uuid"], "root": d["root"], "label": d["label"], "size": d["size"],
            "mounted": bool(mounted and mounted.uuid == d["uuid"]),
            "last_full_scan": d["last_full_scan"],
            "risk": {r: {"units": n, "bytes": b} for r, (n, b) in by.items()},
            "at_risk": [{"path": u.path, "kind": u.kind, "bytes": u.bytes, "note": u.note,
                         "told": u.notified_to} for u in risky]
                       + [{"path": u.path, "kind": u.kind, "bytes": u.bytes,
                           "note": "SOURCE GONE: " + str(_source_gone(u))
                           + f" ({json.dumps(u.source)})", "told": None} for u in gone],
        })
    size = st.size()
    if args.json:
        print(json.dumps({"recipe_bytes": size, "budget_bytes": cfg.budget, "disks": out},
                         indent=2))
        return 1 if args.strict and at_risk_any else 0
    print(f"recipe: {cfg.db_path}  {human(size)} of {cfg.budget_mb:g} MB budget")
    pr_rows = st.pruned()
    if cfg.areas or pr_rows:
        print(f"single-disk mode: areas {', '.join(cfg.areas) or 'none'}; "
              f"auto_prune {'on' if cfg.auto_prune else 'off'}; {len(pr_rows)} unit(s) pruned "
              f"({human(sum(r['bytes'] for r in pr_rows))}); "
              f"{len(st.plans('pending'))} plan(s) awaiting review")
    if not out:
        print("no disks scanned yet")
    for info in out:
        when = (time.strftime("%Y-%m-%d %H:%M", time.localtime(float(info["last_full_scan"])))
                if info["last_full_scan"] else "never")
        print(f"\n{info['root']}  uuid {info['uuid']}  "
              f"{'mounted' if info['mounted'] else 'NOT MOUNTED'}  last full scan {when}")
        for r in RISKS:
            n = info["risk"][r]
            print(f"  {RISK_LABEL[r]:<12} {n['units']:>6} unit(s)  {human(n['bytes']):>10}")
        risky = info["at_risk"]
        if risky:
            print("\n  would be lost if this disk died:")
            for u in sorted(risky, key=lambda x: -int(x["bytes"]))[:25]:
                told = f"  (asked {u['told']})" if u["told"] else ""
                print(f"    {human(int(u['bytes'])):>9}  {u['path'] or '/'}  [{u['kind']}] "
                      f"{u['note']}{told}")
            if len(risky) > 25:
                print(f"    ...and {len(risky) - 25} more: provenance ls --risk at_risk")
    for entry in st.recent_log(5) if args.verbose else []:
        at = time.strftime("%H:%M:%S", time.localtime(entry["at"]))
        print(f"log {at} {entry['level']}: {entry['msg']}")
    return 1 if args.strict and at_risk_any else 0


def _source_gone(u: UnitRow) -> str | None:
    """Why the recorded source failed its last online check, or None if it did not."""
    chk = u.detail.get("source_check")
    if isinstance(chk, dict) and chk.get("ok") is False:
        return str(chk.get("why") or "unreachable")
    return None


def _rel(cfg: Config, st: Store, path: str) -> tuple[str, str]:
    ap = os.path.realpath(path) if path else ""
    for d in st.disks():
        r = os.path.realpath(d["root"])
        if ap and (ap == r or ap.startswith(r + "/")):
            return d["uuid"], "" if ap == r else os.path.relpath(ap, r)
    if not path:
        return _pick_disk(st, None), ""
    # Not under a known root: treat it as a path relative to the most recent disk root.
    return _pick_disk(st, None), path.strip("/")


def _unit_dict(u: UnitRow) -> dict[str, Any]:
    return {"path": u.path, "kind": u.kind, "risk": u.risk, "files": u.files,
            "bytes": u.bytes, "source": u.source, "note": u.note, "rebuild": u.rebuild,
            "detail": u.detail, "listing_files": u.listing_n if u.has_listing else None}


def _ls(cfg: Config, args: argparse.Namespace) -> int:
    st = _open(cfg, write=False)
    disk, rel = _rel(cfg, st, args.path)
    units = st.units(disk, rel, args.risk)
    if args.json:
        print(json.dumps([_unit_dict(u) for u in units], indent=2))
        return 0
    for u in units:
        print(f"{RISK_LABEL[u.risk]:<12} {human(u.bytes):>9}  {u.kind:<10} {u.path or '/'}"
              f"  {u.note}")
    return 0


def _show(cfg: Config, args: argparse.Namespace) -> int:
    st = _open(cfg, write=False)
    disk, rel = _rel(cfg, st, args.path)
    u = st.unit_at(disk, rel)
    if u is None:
        print(f"provenance: nothing recorded at {args.path}", file=sys.stderr)
        return 1
    writers = st.writers_under(disk, u.path)
    if args.json:
        d = _unit_dict(u)
        d["writers"] = [{"agent": w["agent"], "user": w["user"], "source": w["source"],
                         "dir": w["dir"], "writes": w["n"]} for w in writers[:10]]
        if args.files:
            d["listing"] = st.listing(disk, u.path)
        print(json.dumps(d, indent=2))
        return 0
    if u.path != rel:
        print(f"({rel} is inside this unit)")
    print(f"{u.path or '/'}\n  kind     {u.kind}\n  risk     {RISK_LABEL[u.risk]}\n"
          f"  size     {human(u.bytes)} in {u.files} file(s)\n  note     {u.note}")
    if u.source:
        print(f"  source   {json.dumps(u.source)}")
    for k, v in u.detail.items():
        val = v if isinstance(v, str) else json.dumps(v)
        print(f"  {k:<8} {val[:600]}")
    if u.notified_to:
        print(f"  asked    {u.notified_to} at "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(u.notified_at or 0))}")
    for w in writers[:5]:
        who = w["agent"] or w["user"] or "?"
        try:
            argv = attribution.Writer.from_json(w["json"]).chain[0].argv
        except (ValueError, IndexError, KeyError, TypeError):
            argv = []
        print(f"  writer   {who} in {w['dir'] or '/'} ({w['n']} write(s)): "
              f"{' '.join(argv)[:160]}")
    print("  rebuild")
    for line in u.rebuild or ["# (nothing: see risk)"]:
        print("    " + line.replace("\n", "\n    "))
    if args.files:
        for p, s, h in st.listing(disk, u.path):
            print(f"  {s:>14}  {p}" + (f"  {h}" if h else ""))
    return 0


# --- scan / claim --------------------------------------------------------------------------


def _scan(cfg: Config, args: argparse.Namespace, log: Log, vlog: Log) -> int:
    if not cfg.disks:
        raise EngineError(f"no disks configured in {cfg.path}")
    st = _open(cfg, write=not args.dry_run)
    eng = Engine(cfg, st, vlog, lambda m: log(f"warning: {m}"))
    targets: list[tuple[Disk, str]] = []
    if args.path:
        targets.append(eng.resolve(args.path))
    else:
        for root in cfg.disks:
            d = probe_disk(root)
            if d is None:
                log(f"warning: {root} is not mounted; skipped")
                continue
            targets.append((d, ""))
    rc = 0
    for d, rel in targets:
        if args.dry_run:
            from . import classify

            units = classify.walk(eng.context(d), rel)
        else:
            units = eng.scan(d, rel)
        for u in units:
            print(f"{RISK_LABEL[u.risk]:<12} {human(u.bytes):>9}  {u.kind:<10} "
                  f"{u.path or '/'}  {u.note}")
        log(f"{d.root}/{rel}: {len(units)} unit(s)"
            + ("" if args.dry_run else f", recipe now {human(st.size())}"))
    return rc


def _claim(cfg: Config, args: argparse.Namespace, log: Log) -> int:
    if args.clear:
        kind, value = "clear", ""
    elif args.keep is not None:
        kind, value = "keep", args.keep
    elif args.expendable is not None:
        kind, value = "expendable", args.expendable
    else:
        kind, value = "source", args.source
    path = os.path.realpath(args.path)
    agent = attribution.agent_name(os.environ.get("CLAUDE_CODE_SESSION_ID"))
    if os.geteuid() == 0:
        st = _open(cfg, write=True)
        eng = Engine(cfg, st, log, log)
        by = os.environ.get("SUDO_USER") or "root"
        disk, rel = eng.apply_claim(path, kind, value, by + (f" ({agent})" if agent else ""))
        units = eng.scan(disk, rel)
        for u in units:
            print(f"{RISK_LABEL[u.risk]:<12} {u.path}  {u.note}")
        return 0
    if not cfg.spool.is_dir():
        raise EngineError(f"{cfg.spool} does not exist (is provenance installed?)")
    req = {"path": path, "kind": kind, "value": value, "agent": agent}
    name = cfg.spool / f"{time.time():.3f}-{os.getpid()}-{secrets.token_hex(4)}.json"
    tmp = name.with_suffix(".tmp")
    tmp.write_text(json.dumps(req))
    tmp.rename(name)
    log(f"claim queued; the daemon applies it within a minute ({name.name})")
    return 0


# --- rebuild ---------------------------------------------------------------------------------


def _script(st: Store, disk: str, rel: str, into: str) -> tuple[str, int, int]:
    units = st.units(disk, rel)
    lines = ["#!/usr/bin/env bash", "# Generated by provenance: recreates what is recreatable.",
             "# AT RISK units have no source; they are listed at the end as comments.",
             "set -u", f"ROOT={_shq(into)}", "FAILED=0",
             "step() { echo \"== $1\" >&2; }", ""]
    done = 0
    lost: list[UnitRow] = []
    for u in units:
        if u.risk == AT_RISK or (not u.rebuild and u.risk != EXPENDABLE):
            lost.append(u)
            continue
        if not u.rebuild or u.kind == "cache" or all(c.startswith("#") for c in u.rebuild):
            lines.append(f"# {u.path or '/'}: {u.note}")
            lines.extend(c for c in u.rebuild if c.startswith("#"))
            continue
        lines.append(f"step {_shq(u.path or '/')}")
        lines.append("( set -e")
        lines.extend("  " + cmd.replace("\n", "\n  ") if "<<" not in cmd else cmd
                     for cmd in u.rebuild)
        lines.append(f") || {{ echo \"FAILED: {u.path}\" >&2; FAILED=$((FAILED+1)); }}")
        lines.append("")
        done += 1
    if lost:
        lines.append("# ---- AT RISK: no known source, cannot be recreated ----")
        for u in lost:
            lines.append(f"# {u.path or '/'}  ({human(u.bytes)}, {u.kind}): {u.note}")
    lines.append('exit $(( FAILED > 0 ))')
    return "\n".join(lines) + "\n", done, len(lost)


def _shq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _rebuild(cfg: Config, args: argparse.Namespace, log: Log) -> int:
    st = _open(cfg, write=False)
    disk = _pick_disk(st, args.disk)
    row = next(d for d in st.disks() if d["uuid"] == disk)
    into = args.into or row["root"]
    script, n, lost = _script(st, disk, args.path.strip("/"), into)
    log(f"{n} unit(s) to recreate, {lost} at risk (listed as comments)")
    if not args.run:
        sys.stdout.write(script)
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            log("provenance: error: refusing to --run without --yes (stdin is not a TTY)")
            return 2
        log(f"This downloads and writes into {into}. Type 'yes' to confirm: ")
        try:
            reply = input()
        except EOFError:
            reply = ""
        if reply.strip().lower() != "yes":
            log("Aborted.")
            return 2
    r = subprocess.run(["bash", "-c", script], check=False)
    return 0 if r.returncode == 0 else 1


def _verify_restore(cfg: Config, args: argparse.Namespace, log: Log) -> int:
    st = _open(cfg, write=False)
    disk = _pick_disk(st, args.disk)
    missing = mismatched = checked = no_listing = 0
    for u in st.units(disk):
        if u.risk in (AT_RISK, EXPENDABLE) or u.kind in ("timeshift", "cache", "git", "venv"):
            continue
        rows = st.listing(disk, u.path)
        if not rows:
            no_listing += 1
            continue
        base = os.path.join(args.root, u.path)
        for p, size, sha in rows:
            if u.kind == "hf-cache":
                rev12, _, f = p.partition("/")
                snaps = os.path.join(base, "snapshots")
                revs = [r for r in (os.listdir(snaps) if os.path.isdir(snaps) else [])
                        if r.startswith(rev12)]
                fp = os.path.join(snaps, revs[0], f) if revs else os.path.join(snaps, p)
            else:
                fp = os.path.join(base, p) if p else base
            checked += 1
            try:
                got = os.stat(fp).st_size
            except OSError:
                missing += 1
                print(f"missing   {fp}")
                continue
            if size >= 0 and got != size:
                mismatched += 1
                print(f"size      {fp}: {got} != {size}")
            elif args.hash and sha and sha256_file(fp) != sha:
                mismatched += 1
                print(f"sha256    {fp}")
    log(f"checked {checked} file(s): {missing} missing, {mismatched} differ; "
        f"{no_listing} unit(s) had no listing (dropped for budget, or none kept)")
    return 1 if missing or mismatched else 0


# --- source liveness ------------------------------------------------------------------------


def _http_ok(url: str, timeout: float) -> tuple[bool, str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "provenance"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400, str(r.status)
    except urllib.error.HTTPError as e:
        if e.code == 405:  # HEAD not allowed; a ranged GET is the cheapest fallback
            try:
                g = urllib.request.Request(url, headers={"User-Agent": "provenance",
                                                         "Range": "bytes=0-0"})
                with urllib.request.urlopen(g, timeout=timeout) as r:
                    return r.status < 400, str(r.status)
            except (urllib.error.URLError, OSError) as e2:
                return False, str(e2)
        return False, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return False, str(e)


def _check_one(u: UnitRow, timeout: float) -> tuple[bool | None, str]:
    s = u.source or {}
    kind, ref, rev = s.get("kind"), s.get("ref"), s.get("revision")
    if not ref:
        return None, "no source"
    if kind == "hf":
        rtype = "models"
        first = (rev or "main").split(",")[0]
        return _http_ok(f"https://huggingface.co/api/{rtype}/{ref}/revision/{first}", timeout)
    if kind == "git":
        try:
            r = subprocess.run(["git", "ls-remote", ref], check=False, capture_output=True, text=True,
                               timeout=timeout, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)
        if r.returncode != 0:
            return False, r.stderr.strip()[:200]
        if rev and rev not in r.stdout:
            return True, "remote up; recorded commit is not a ref tip (may still exist)"
        return True, "ok"
    if kind == "url":
        return _http_ok(str(ref), timeout)
    if kind == "ollama":
        bad = []
        for t in str(ref).split(","):
            name, _, tag = t.partition(":")
            if "/" not in name:
                name = f"library/{name}"
            ok, why = _http_ok(f"https://registry.ollama.ai/v2/{name}/manifests/{tag or 'latest'}",
                               timeout)
            if not ok:
                bad.append(f"{t}: {why}")
        return (not bad), "; ".join(bad) or "ok"
    return None, f"{kind}: not checkable"


def _check_sources(cfg: Config, args: argparse.Namespace, log: Log) -> int:
    st = _open(cfg, write=os.geteuid() == 0)
    disk = _pick_disk(st, args.disk)
    bad = 0
    for u in st.units(disk):
        if u.risk != RECREATABLE or not u.source:
            continue
        ok, why = _check_one(u, args.timeout)
        if ok is None:
            continue
        mark = "ok  " if ok else "GONE"
        print(f"{mark} {u.path}  {u.source.get('kind')}:{u.source.get('ref')}  {why}")
        if not ok:
            bad += 1
        if os.geteuid() == 0:
            st.db.execute("UPDATE units SET detail=json_set(detail,'$.source_check',json(?)) "
                          "WHERE disk=? AND path=?",
                          (json.dumps({"at": time.time(), "ok": ok, "why": why}), disk, u.path))
            st.db.commit()
    log(f"{bad} source(s) unreachable")
    return 1 if bad else 0


def _export(cfg: Config, args: argparse.Namespace, log: Log) -> int:
    st = _open(cfg, write=False)
    doc: dict[str, Any] = {"provenance": __version__, "exported_at": time.time(), "disks": []}
    for d in st.disks():
        units = []
        for u in st.units(d["uuid"]):
            ud = _unit_dict(u)
            ud["listing"] = st.listing(d["uuid"], u.path)
            units.append(ud)
        doc["disks"].append({"uuid": d["uuid"], "root": d["root"], "label": d["label"],
                             "size": d["size"], "units": units})
    raw = json.dumps(doc, separators=(",", ":")).encode()
    with open(args.output, "wb") as f:
        f.write(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))
    log(f"wrote {args.output} ({human(os.path.getsize(args.output))})")
    return 0


# --- single-disk mode ------------------------------------------------------------------


def _age(t: float | None) -> str:
    return f"{(time.time() - t) / 86400:.0f} d" if t else "never seen"


def _spool_request(cfg: Config, kind: str, value: str, log: Log) -> int:
    if not cfg.spool.is_dir():
        raise EngineError(f"{cfg.spool} does not exist (is provenance installed?)")
    agent = attribution.agent_name(os.environ.get("CLAUDE_CODE_SESSION_ID"))
    name = cfg.spool / f"{time.time():.3f}-{os.getpid()}-{secrets.token_hex(4)}.json"
    tmp = name.with_suffix(".tmp")
    tmp.write_text(json.dumps({"path": "", "kind": kind, "value": value, "agent": agent}))
    tmp.rename(name)
    log(f"queued; the daemon applies it within a minute ({name.name})")
    return 0


def _prune(cfg: Config, args: argparse.Namespace, log: Log) -> int:
    from .prune import PruneError, Pruner, probe_area

    if args.approve or args.reject:
        pid = args.approve or args.reject
        if os.geteuid() != 0:
            return _spool_request(cfg, "plan-approve" if args.approve else "plan-reject",
                                  pid, log)
        st = _open(cfg, write=True)
        by = os.environ.get("SUDO_USER") or "root"
        Engine(cfg, st, log, log).decide_plan(pid, bool(args.approve), by)
        log(f"plan {pid} {'approved: the daemon applies it within a minute' if args.approve else 'rejected'}")
        return 0
    write = bool(args.propose or args.apply)
    st = _open(cfg, write=write)
    eng = Engine(cfg, st, log, lambda m: log(f"warning: {m}"))
    pr = Pruner(eng, log)
    areas = [probe_area(args.area)] if args.area else list(pr.areas())
    if not cfg.areas:
        raise EngineError(f"no areas configured in {cfg.path} (single-disk mode is off)")
    rc = 0
    for area in areas:
        if area is None:
            raise EngineError(f"{args.area} does not exist")
        if write:
            eng.scan(area)
        if args.propose:
            made = pr.propose(area)
            if made is None:
                log(f"{area.root}: nothing to propose (or a plan is already pending)")
                continue
            pid, _ = made
            print(pr.summary(pid))
            eng._send(cfg.prune_approver, pr.summary(pid))
            continue
        cands = pr.candidates(area, days=args.days, online=not args.offline)
        ok = [c for c in cands if c.ok]
        if args.json:
            print(json.dumps([{"path": c.unit.path, "kind": c.unit.kind, "bytes": c.bytes,
                               "last_used": c.last_used, "used_by": c.used_by,
                               "prunable": c.ok, "why_not": c.reasons, "verified": c.verified}
                              for c in cands], indent=2))
        else:
            print(f"{area.root}  ({area.label})")
            for c in sorted(ok, key=lambda c: -c.bytes):
                print(f"  PRUNE {human(c.bytes):>9}  {c.unit.path} [{c.unit.kind}] unused "
                      f"{_age(c.last_used)} ({c.used_by}); {c.verified}")
            if args.all:
                for c in sorted([c for c in cands if not c.ok], key=lambda c: -c.bytes):
                    print(f"  keep  {human(c.bytes):>9}  {c.unit.path} [{c.unit.kind}] "
                          f"{'; '.join(c.reasons)}")
            print(f"  -> {len(ok)} unit(s), {human(sum(c.bytes for c in ok))} prunable"
                  + ("" if args.apply else " (dry run)"))
        if args.apply and ok:
            if not args.yes:
                if not sys.stdin.isatty():
                    log("provenance: error: refusing to --apply without --yes (no TTY)")
                    return 2
                log("Delete these now? Type 'yes': ")
                try:
                    reply = input()
                except EOFError:
                    reply = ""
                if reply.strip().lower() != "yes":
                    log("Aborted.")
                    return 2
            try:
                res = pr.apply(area, [c.unit.path for c in ok], None)
            except (PruneError, OSError) as exc:
                log(f"provenance: error: {exc}")
                rc = 1
                continue
            log(f"pruned {len(res['done'])} unit(s), {human(res['freed'])}; "
                f"{len(res['skipped'])} skipped at delete time")
            for sk in res["skipped"]:
                log(f"  skipped {sk['path']}: {sk['why']}")
    return rc


def _plans(cfg: Config, args: argparse.Namespace) -> int:
    from .prune import Pruner

    st = _open(cfg, write=False)
    if args.show:
        print(Pruner(Engine(cfg, st)).summary(args.show))
        return 0
    for p in st.plans():
        items = json.loads(p["items"])
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(p["created"]))
        tot = sum(int(i["bytes"]) for i in items)
        by = f" by {p['decided_by']}" if p["decided_by"] else ""
        print(f"{p['id']}  {when}  {p['status']:<9}{by}  {len(items)} unit(s) "
              f"{human(tot)}  {p['disk'].removeprefix('area:')}")
    return 0


def _pruned(cfg: Config, args: argparse.Namespace) -> int:
    st = _open(cfg, write=False)
    rows = st.pruned()
    if args.json:
        print(json.dumps([{"path": os.path.join(r["disk"].removeprefix("area:"), r["path"]),
                           "kind": r["kind"], "bytes": r["bytes"], "pruned_at": r["pruned_at"],
                           "plan": r["plan"], "in_trash": bool(r["trash"] and not r["purged_at"]),
                           "rebuild": json.loads(r["unit"])["rebuild"]} for r in rows], indent=2))
        return 0
    for r in rows:
        where = ("in trash until purge" if r["trash"] and not r["purged_at"]
                 else "rebuild from recipe")
        print(f"{human(r['bytes']):>9}  {os.path.join(r['disk'].removeprefix('area:'), r['path'])}"
              f"  [{r['kind']}] {time.strftime('%Y-%m-%d', time.localtime(r['pruned_at']))}"
              f"  {where}")
    print(f"{len(rows)} pruned unit(s), {human(sum(r['bytes'] for r in rows))}")
    return 0


def _restore(cfg: Config, args: argparse.Namespace, log: Log) -> int:
    from .prune import PruneError, Pruner

    st = _open(cfg, write=not args.dry_run)
    try:
        msg = Pruner(Engine(cfg, st, log, log), log).restore(args.path, dry_run=args.dry_run)
    except PruneError as exc:
        log(f"provenance: error: {exc}")
        return 1
    print(msg)
    return 0
