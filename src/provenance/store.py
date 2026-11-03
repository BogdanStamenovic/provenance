"""The recipe: one sqlite file on the trusted SSD.

Nothing here is ever written to the watched disk. Listings (path + size [+ sha256]) are
stored lzma-compressed per unit; they are the "what exactly was there" half of the
recipe and the only part that grows with file count, so the budget trims them first.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .classify import AT_RISK, EXPENDABLE, RECREATABLE, REDUNDANT, Unit

SCHEMA = """
CREATE TABLE IF NOT EXISTS disks (
  uuid TEXT PRIMARY KEY, root TEXT, label TEXT, size INTEGER,
  first_seen REAL, last_seen REAL, last_full_scan REAL
);
CREATE TABLE IF NOT EXISTS units (
  disk TEXT, path TEXT, kind TEXT, risk TEXT, files INTEGER, bytes INTEGER,
  source TEXT, rebuild TEXT, detail TEXT, note TEXT, listing BLOB, listing_n INTEGER,
  scanned_at REAL, notified_at REAL, notified_to TEXT, at_risk_since REAL,
  PRIMARY KEY (disk, path)
);
CREATE TABLE IF NOT EXISTS writers (
  id INTEGER PRIMARY KEY, key TEXT UNIQUE, json TEXT, source TEXT,
  session_id TEXT, agent TEXT, user TEXT, first_seen REAL
);
CREATE TABLE IF NOT EXISTS writes (
  disk TEXT, dir TEXT, writer_id INTEGER, first_seen REAL, last_seen REAL, n INTEGER,
  PRIMARY KEY (disk, dir, writer_id)
);
CREATE TABLE IF NOT EXISTS claims (
  disk TEXT, path TEXT, kind TEXT, value TEXT, by TEXT, at REAL,
  PRIMARY KEY (disk, path)
);
CREATE TABLE IF NOT EXISTS hfblobs (sha TEXT PRIMARY KEY, repo TEXT, revision TEXT, file TEXT);
CREATE TABLE IF NOT EXISTS filehash (
  disk TEXT, path TEXT, size INTEGER, mtime_ns INTEGER, ino INTEGER, sha TEXT,
  PRIMARY KEY (disk, path)
);
CREATE TABLE IF NOT EXISTS log (at REAL, level TEXT, msg TEXT);
CREATE TABLE IF NOT EXISTS hfrepo_checked (repo TEXT PRIMARY KEY, at REAL, ok INTEGER);
"""

OPTIONAL_UNIT_COLS = ("at_risk_since",)

# Which listings to give up first when over budget: the cheapest to lose.
TRIM_ORDER = (EXPENDABLE, REDUNDANT, RECREATABLE, AT_RISK)


class StoreError(Exception):
    pass


def pack_listing(rows: Iterable[tuple[str, int, str]]) -> tuple[bytes, int]:
    rows = sorted(rows)
    text = "\n".join(f"{p}\t{s}\t{h}" if h else f"{p}\t{s}" for p, s, h in rows)
    return lzma.compress(text.encode(), preset=9 | lzma.PRESET_EXTREME), len(rows)


def unpack_listing(blob: bytes | None) -> list[tuple[str, int, str]]:
    if not blob:
        return []
    out: list[tuple[str, int, str]] = []
    for line in lzma.decompress(blob).decode().splitlines():
        parts = line.split("\t")
        out.append((parts[0], int(parts[1]), parts[2] if len(parts) > 2 else ""))
    return out


@dataclass
class UnitRow:
    disk: str
    path: str
    kind: str
    risk: str
    files: int
    bytes: int
    source: dict[str, str | None] | None
    rebuild: list[str]
    detail: dict[str, object]
    note: str
    listing_n: int
    has_listing: bool
    scanned_at: float
    notified_at: float | None
    notified_to: str | None
    at_risk_since: float | None = None


class Store:
    def __init__(self, path: str | Path, readonly: bool = False) -> None:
        self.path = Path(path)
        if readonly:
            if not self.path.exists():
                raise StoreError(f"no recipe database at {self.path} (is the daemon set up?)")
            self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, timeout=30)
            # Rollback journal, not WAL: unprivileged readers of a root-owned db cannot
            # create the -shm file WAL needs, and would fail to open it at all.
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.executescript(SCHEMA)
            self._migrate()
            os.chmod(self.path, 0o644)
        self.db.row_factory = sqlite3.Row

    def close(self) -> None:
        self.db.close()

    def _migrate(self) -> None:
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(units)")}
        if "at_risk_since" not in cols:
            self.db.execute("ALTER TABLE units ADD COLUMN at_risk_since REAL")
            self.db.commit()

    # --- disks ---------------------------------------------------------------------

    def touch_disk(self, uuid: str, root: str, label: str, size: int) -> None:
        now = time.time()
        self.db.execute(
            "INSERT INTO disks(uuid, root, label, size, first_seen, last_seen) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(uuid) DO UPDATE SET root=excluded.root, "
            "label=excluded.label, size=excluded.size, last_seen=excluded.last_seen",
            (uuid, root, label, size, now, now),
        )
        self.db.commit()

    def disks(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM disks ORDER BY last_seen DESC"))

    def mark_full_scan(self, uuid: str) -> None:
        self.db.execute("UPDATE disks SET last_full_scan=? WHERE uuid=?", (time.time(), uuid))
        self.db.commit()

    # --- units ---------------------------------------------------------------------

    def replace_units(self, disk: str, prefix: str, units: list[Unit]) -> None:
        """Replace every unit at or under `prefix` with `units` (one transaction)."""
        old = {r["path"]: r for r in self.db.execute(
            "SELECT path, risk, notified_at, notified_to, at_risk_since FROM units WHERE disk=? AND "
            "(?='' OR path=? OR path LIKE ? ESCAPE '\\')",
            (disk, prefix, prefix, _like_prefix(prefix)),
        )}
        now = time.time()
        with self.db:
            self.db.execute(
                "DELETE FROM units WHERE disk=? AND (?='' OR path=? OR path LIKE ? ESCAPE '\\')",
                (disk, prefix, prefix, _like_prefix(prefix)),
            )
            for u in units:
                blob, n = pack_listing(u.listing) if u.listing else (None, 0)
                prev = old.get(u.path)
                # Keep "already told someone" only while the unit is still at risk.
                told_at = told_to = None
                since = now if u.risk == AT_RISK else None
                if prev is not None and prev["risk"] == AT_RISK and u.risk == AT_RISK:
                    told_at, told_to = prev["notified_at"], prev["notified_to"]
                    since = prev["at_risk_since"] or now
                self.db.execute(
                    # Named columns: an older version (ownbox rollback) must still be able
                    # to write into a table a newer version added columns to.
                    "INSERT INTO units ""(disk,path,kind,risk,files,bytes,source,rebuild,detail,note,listing,listing_n,"
                    "scanned_at,notified_at,notified_to,at_risk_since) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (disk, u.path, u.kind, u.risk, u.files, u.bytes,
                     json.dumps(u.source) if u.source else None, json.dumps(u.rebuild),
                     json.dumps(u.detail), u.note, blob, n, now, told_at, told_to, since),
                )

    def replace_direct(self, disk: str, prefix: str, units: list[Unit],
                       file_kinds: tuple[str, ...]) -> None:
        """Replace the residual unit at `prefix` and file units directly inside it."""
        direct = [u for u in units if u.path == prefix or
                  os.path.dirname(u.path) == prefix and u.kind in file_kinds]
        with self.db:
            self.db.execute("DELETE FROM units WHERE disk=? AND path=? AND kind='data'",
                            (disk, prefix))
            ph = ",".join("?" * len(file_kinds))
            for r in list(self.db.execute(
                    f"SELECT path FROM units WHERE disk=? AND kind IN ({ph})",
                    (disk, *file_kinds))):
                if os.path.dirname(r["path"]) == prefix:
                    self.db.execute("DELETE FROM units WHERE disk=? AND path=?",
                                    (disk, r["path"]))
        for u in direct:
            self.replace_units(disk, u.path, [u]) if u.kind != "data" else None
        data = [u for u in direct if u.kind == "data"]
        if data:
            with self.db:
                blob, n = pack_listing(data[0].listing)
                now = time.time()
                self.db.execute(
                    "INSERT OR REPLACE INTO units ""(disk,path,kind,risk,files,bytes,source,rebuild,detail,note,listing,listing_n,"
                    "scanned_at,notified_at,notified_to,at_risk_since) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (disk, prefix, "data", data[0].risk, data[0].files, data[0].bytes,
                     json.dumps(data[0].source) if data[0].source else None, "[]",
                     json.dumps(data[0].detail), data[0].note, blob, n, now,
                     None, None, now if data[0].risk == AT_RISK else None))

    def units(self, disk: str | None = None, under: str = "",
              risk: str | None = None) -> list[UnitRow]:
        # Columns added by later versions are optional, so a read-only CLI works on a db
        # the daemon has not migrated yet (update order, or after an ownbox rollback).
        have = {r[1] for r in self.db.execute("PRAGMA table_info(units)")}
        extra = ",".join(c if c in have else f"NULL AS {c}" for c in OPTIONAL_UNIT_COLS)
        q = ("SELECT disk,path,kind,risk,files,bytes,source,rebuild,detail,note,listing_n,"
             "listing IS NOT NULL AS has_listing,scanned_at,notified_at,notified_to,"
             f"{extra} FROM units WHERE 1=1")
        args: list[object] = []
        if disk:
            q += " AND disk=?"
            args.append(disk)
        if under:
            q += " AND (path=? OR path LIKE ? ESCAPE '\\')"
            args += [under, _like_prefix(under)]
        if risk:
            q += " AND risk=?"
            args.append(risk)
        q += " ORDER BY path"
        return [_row(r) for r in self.db.execute(q, args)]

    def unit_at(self, disk: str, path: str) -> UnitRow | None:
        """The unit that is `path` or contains it."""
        best: UnitRow | None = None
        for u in self.units(disk):
            if u.path == path:
                return u
            if (u.path and path.startswith(u.path + "/")
                    and (best is None or len(u.path) > len(best.path))):
                best = u
        return best

    def listing(self, disk: str, path: str) -> list[tuple[str, int, str]]:
        r = self.db.execute("SELECT listing FROM units WHERE disk=? AND path=?",
                            (disk, path)).fetchone()
        return unpack_listing(r["listing"]) if r else []

    def mark_notified(self, disk: str, path: str, to: str) -> None:
        self.db.execute("UPDATE units SET notified_at=?, notified_to=? WHERE disk=? AND path=?",
                        (time.time(), to, disk, path))
        self.db.commit()

    # --- writers -------------------------------------------------------------------

    def writer_id(self, key: str, raw: str, source: dict[str, str | None] | None,
                  session_id: str | None, agent: str | None, user: str | None) -> int:
        r = self.db.execute("SELECT id FROM writers WHERE key=?", (key,)).fetchone()
        if r:
            return int(r["id"])
        cur = self.db.execute(
            "INSERT INTO writers(key, json, source, session_id, agent, user, first_seen) "
            "VALUES (?,?,?,?,?,?,?)",
            (key, raw, json.dumps(source) if source else None, session_id, agent, user,
             time.time()),
        )
        return int(cur.lastrowid or 0)

    def record_writes(self, disk: str, batch: dict[tuple[str, int], tuple[float, float, int]]
                      ) -> None:
        with self.db:
            for (d, wid), (first, last, n) in batch.items():
                self.db.execute(
                    "INSERT INTO writes (disk,dir,writer_id,first_seen,last_seen,n) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(disk,dir,writer_id) "
                    "DO UPDATE SET last_seen=excluded.last_seen, n=n+excluded.n",
                    (disk, d, wid, first, last, n),
                )

    def writers_under(self, disk: str, path: str) -> list[sqlite3.Row]:
        """Writers seen at `path`, inside it, or in an ancestor of it; nearest first."""
        rows = list(self.db.execute(
            "SELECT w.*, x.dir, x.n, x.last_seen AS at FROM writes x JOIN writers w "
            "ON w.id=x.writer_id WHERE x.disk=? AND (x.dir=? OR x.dir LIKE ? ESCAPE '\\' "
            "OR ? LIKE x.dir || '/%' OR x.dir='')",
            (disk, path, _like_prefix(path), path),
        ))
        def rank(r: sqlite3.Row) -> tuple[int, int, float]:
            inside = r["dir"] == path or r["dir"].startswith(path + "/") if path else True
            src = json.loads(r["source"]) if r["source"] else None
            strength = 0 if src and src.get("kind") != "copy" else (1 if src else 2)
            return (0 if inside else 1, strength, -float(r["at"]))
        rows.sort(key=rank)
        return rows

    def compact_writes(self, disk: str, unit_paths: list[str]) -> int:
        """Collapse per-directory write rows inside each unit onto the unit itself.

        Keeps the best writer per unit (plus any with an agent attached, so we can still
        ask them) and drops the rest. This is what stops the writes table from growing
        with directory count.
        """
        dropped = 0
        with self.db:
            for p in unit_paths:
                if not p:
                    continue
                rows = self.writers_under(disk, p)
                inside = [r for r in rows if r["dir"] == p or r["dir"].startswith(p + "/")]
                if len(inside) <= 1 and all(r["dir"] == p for r in inside):
                    continue
                keep = {inside[0]["id"]} if inside else set()
                keep |= {r["id"] for r in inside if r["agent"]}
                cur = self.db.execute(
                    "DELETE FROM writes WHERE disk=? AND (dir=? OR dir LIKE ? ESCAPE '\\')",
                    (disk, p, _like_prefix(p)),
                )
                dropped += cur.rowcount
                now = time.time()
                for wid in keep:
                    self.db.execute(
                        "INSERT OR IGNORE INTO writes (disk,dir,writer_id,first_seen,last_seen,n) "
                        "VALUES (?,?,?,?,?,?)",
                        (disk, p, wid, now, now, 1),
                    )
                    dropped -= 1
        return dropped

    # --- claims --------------------------------------------------------------------

    def set_claim(self, disk: str, path: str, kind: str, value: str, by: str) -> None:
        with self.db:
            if kind == "clear":
                self.db.execute("DELETE FROM claims WHERE disk=? AND path=?", (disk, path))
            else:
                self.db.execute(
                    "INSERT OR REPLACE INTO claims (disk,path,kind,value,by,at) VALUES (?,?,?,?,?,?)",
                    (disk, path, kind, value, by, time.time()),
                )

    def claims(self, disk: str) -> dict[str, dict[str, str]]:
        return {r["path"]: {"kind": r["kind"], "value": r["value"], "by": r["by"]}
                for r in self.db.execute("SELECT * FROM claims WHERE disk=?", (disk,))}

    # --- HF blob index and file hash cache -----------------------------------------

    def index_hf(self, entries: Iterable[tuple[str, str, str, str]]) -> int:
        with self.db:
            cur = self.db.executemany("INSERT OR IGNORE INTO hfblobs (sha,repo,revision,file) VALUES (?,?,?,?)", entries)
        return cur.rowcount

    def hf_blob(self, sha: str) -> dict[str, str] | None:
        r = self.db.execute("SELECT * FROM hfblobs WHERE sha=?", (sha,)).fetchone()
        return {"repo": r["repo"], "revision": r["revision"], "file": r["file"]} if r else None

    def cached_sha(self, disk: str, path: str, size: int, mtime_ns: int, ino: int) -> str | None:
        r = self.db.execute("SELECT * FROM filehash WHERE disk=? AND path=?",
                            (disk, path)).fetchone()
        if r and (r["size"], r["mtime_ns"], r["ino"]) == (size, mtime_ns, ino):
            return str(r["sha"])
        return None

    def put_sha(self, disk: str, path: str, size: int, mtime_ns: int, ino: int,
                sha: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO filehash (disk,path,size,mtime_ns,ino,sha) "
                        "VALUES (?,?,?,?,?,?)",
                        (disk, path, size, mtime_ns, ino, sha))
        self.db.commit()

    # --- log and budget --------------------------------------------------------------

    def log(self, level: str, msg: str) -> None:
        self.db.execute("INSERT INTO log (at,level,msg) VALUES (?,?,?)", (time.time(), level, msg))
        self.db.execute("DELETE FROM log WHERE rowid IN (SELECT rowid FROM log "
                        "ORDER BY at DESC LIMIT -1 OFFSET 2000)")
        self.db.commit()

    def recent_log(self, n: int = 20) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM log ORDER BY at DESC LIMIT ?", (n,)))

    def size(self) -> int:
        pc = self.db.execute("PRAGMA page_count").fetchone()[0]
        ps = self.db.execute("PRAGMA page_size").fetchone()[0]
        fl = self.db.execute("PRAGMA freelist_count").fetchone()[0]
        return int((pc - fl) * ps)

    def enforce_budget(self, budget: int) -> list[str]:
        """Drop listings, cheapest-to-lose first and biggest first, until under budget."""
        dropped: list[str] = []
        if self.size() <= budget:
            return dropped
        for risk in TRIM_ORDER:
            rows = list(self.db.execute(
                "SELECT disk, path, length(listing) AS n FROM units WHERE risk=? AND "
                "listing IS NOT NULL ORDER BY n DESC", (risk,)))
            for r in rows:
                self.db.execute("UPDATE units SET listing=NULL, detail=json_set(detail, "
                                "'$.listing_dropped', 1) WHERE disk=? AND path=?",
                                (r["disk"], r["path"]))
                dropped.append(f"{r['path'] or '/'} ({r['n']} B)")
                self.db.commit()
                if self.size() <= budget:
                    self.db.execute("VACUUM")
                    return dropped
        self.db.execute("VACUUM")
        return dropped

    def vacuum(self) -> None:
        self.db.execute("VACUUM")


def _like_prefix(p: str) -> str:
    esc = p.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{esc}/%"


def _row(r: sqlite3.Row) -> UnitRow:
    return UnitRow(
        disk=r["disk"], path=r["path"], kind=r["kind"], risk=r["risk"], files=r["files"],
        bytes=r["bytes"], source=json.loads(r["source"]) if r["source"] else None,
        rebuild=json.loads(r["rebuild"] or "[]"), detail=json.loads(r["detail"] or "{}"),
        note=r["note"] or "", listing_n=r["listing_n"] or 0,
        has_listing=bool(r["has_listing"]), scanned_at=r["scanned_at"],
        notified_at=r["notified_at"], notified_to=r["notified_to"],
        at_risk_since=r["at_risk_since"],
    )


def sha256_file(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hf_hub_entries(hub_root: str) -> Iterator[tuple[str, str, str, str]]:
    """(sha256, repo_id, revision, file) for every LFS blob in an HF hub cache dir."""
    try:
        names = os.listdir(hub_root)
    except OSError:
        return
    for name in names:
        parts = name.split("--")
        if len(parts) < 3 or parts[0] not in ("models", "datasets", "spaces"):
            continue
        if parts[0] != "models":
            continue  # weights we try to identify are model files
        repo = parts[1] + "/" + "--".join(parts[2:])
        snaps = os.path.join(hub_root, name, "snapshots")
        try:
            revs = os.listdir(snaps)
        except OSError:
            continue
        for rev in revs:
            base = os.path.join(snaps, rev)
            for dp, _dn, fn in os.walk(base):
                for f in fn:
                    p = os.path.join(dp, f)
                    tgt = os.path.basename(os.readlink(p)) if os.path.islink(p) else ""
                    if len(tgt) == 64:
                        yield (tgt, repo, rev, os.path.relpath(p, base))
