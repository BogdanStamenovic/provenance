from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from provenance.classify import AT_RISK, RECREATABLE, Unit
from provenance.config import Config, ConfigError, load
from provenance.engine import Disk, Engine, affected_prefix
from provenance.store import Store, pack_listing, unpack_listing


def test_listing_roundtrip() -> None:
    rows = [("b/2", 5, ""), ("a/1", 3, "ab" * 32)]
    blob, n = pack_listing(rows)
    assert n == 2 and unpack_listing(blob) == sorted(rows)


def test_replace_keeps_notification_only_while_still_at_risk(tmp_path: Path) -> None:
    st = Store(tmp_path / "p.db")
    st.replace_units("d", "", [Unit("x", "data", AT_RISK, 1, 1)])
    st.mark_notified("d", "x", "agent-1")
    st.replace_units("d", "x", [Unit("x", "data", AT_RISK, 2, 2)])
    assert st.units("d")[0].notified_to == "agent-1"
    st.replace_units("d", "x", [Unit("x", "data", RECREATABLE, 2, 2)])
    st.replace_units("d", "x", [Unit("x", "data", AT_RISK, 2, 2)])
    assert st.units("d")[0].notified_to is None


def test_replace_prefix_does_not_touch_siblings(tmp_path: Path) -> None:
    st = Store(tmp_path / "p.db")
    st.replace_units("d", "", [Unit("a", "git", RECREATABLE), Unit("a_b", "git", RECREATABLE),
                               Unit("a/c", "git", RECREATABLE)])
    st.replace_units("d", "a", [])
    assert [u.path for u in st.units("d")] == ["a_b"]  # '_' is not a LIKE wildcard here


def test_budget_drops_cheapest_listings_first(tmp_path: Path) -> None:
    st = Store(tmp_path / "p.db")
    import random
    random.seed(0)
    noise = [(f"f{random.random()}", random.randint(0, 10**12), "") for _ in range(20000)]
    st.replace_units("d", "", [Unit("risky", "data", AT_RISK, listing=noise[:10000]),
                               Unit("safe", "hf-cache", RECREATABLE, listing=noise[10000:])])
    before = st.size()
    # sqlite page overhead: freeing one of two equal listings brings it to ~62%, not 50%.
    dropped = st.enforce_budget(int(before * 0.7))
    assert dropped and dropped[0].startswith("safe")
    assert st.listing("d", "risky"), "the at-risk listing is the last to go"
    assert st.size() < before


def test_reader_tolerates_unmigrated_db(tmp_path: Path) -> None:
    """A newer CLI must read a db an older daemon wrote (update order, ownbox rollback)."""
    p = tmp_path / "old.db"
    db = sqlite3.connect(p)
    db.executescript("""
      CREATE TABLE disks (uuid TEXT PRIMARY KEY, root TEXT, label TEXT, size INTEGER,
        first_seen REAL, last_seen REAL, last_full_scan REAL);
      CREATE TABLE units (disk TEXT, path TEXT, kind TEXT, risk TEXT, files INTEGER,
        bytes INTEGER, source TEXT, rebuild TEXT, detail TEXT, note TEXT, listing BLOB,
        listing_n INTEGER, scanned_at REAL, notified_at REAL, notified_to TEXT,
        PRIMARY KEY (disk, path));
      INSERT INTO units VALUES ('d','x','data','at_risk',1,1,NULL,'[]','{}','n',NULL,0,0,NULL,NULL);
    """)
    db.commit()
    db.close()
    st = Store(p, readonly=True)
    assert st.units("d")[0].at_risk_since is None
    st.close()
    rw = Store(p)  # the writer migrates
    rw.replace_units("d", "x", [Unit("x", "data", AT_RISK)])
    assert rw.units("d")[0].at_risk_since is not None


def test_affected_prefix() -> None:
    units = {"projects/app", "models/m"}
    containers = {"", "projects", "models"}
    assert affected_prefix("projects/app/src/x.py", units, containers) == ("projects/app", False)
    assert affected_prefix("projects/new/README", units, containers) == ("projects/new", False)
    assert affected_prefix("projects/loose.txt", units, containers) == ("projects", True)
    assert affected_prefix("top.bin", units, containers) == ("", True)
    assert affected_prefix("fresh/a/b", units, containers) == ("fresh", False)


def test_config_refuses_recipe_on_watched_disk(tmp_path: Path) -> None:
    c = tmp_path / "c.toml"
    c.write_text('state_dir = "/mnt/hdd/.prov"\ndisks = ["/mnt/hdd"]\n')
    with pytest.raises(ConfigError, match="trusted"):
        load(str(c))
    c.write_text('nope = 1\n')
    with pytest.raises(ConfigError, match="unknown key"):
        load(str(c))


def _engine(tmp_path: Path, **kw: object) -> tuple[Engine, Store, Disk]:
    cfg = Config(state_dir=str(tmp_path / "state"), disks=[str(tmp_path / "disk")],
                 notify_after_s=0, **kw)  # type: ignore[arg-type]
    st = Store(cfg.db_path)
    return Engine(cfg, st), st, Disk("u1", str(tmp_path / "disk"), "", 0)


def test_writer_of_at_risk_unit_is_told_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                             ) -> None:
    eng, st, d = _engine(tmp_path)
    (tmp_path / "disk" / "t").mkdir(parents=True)
    (tmp_path / "disk" / "t" / "w.safetensors").write_bytes(b"w")
    wid = st.writer_id("k", "{}", None, "sess", "agent-7", "bodas")
    st.record_writes("u1", {("t/w.safetensors", wid): (time.time(), time.time(), 1)})
    eng.scan(d)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(eng, "_send", lambda a, m: sent.append((a, m)) or True)
    eng.notify(d)
    eng.notify(d)
    assert len(sent) == 1 and sent[0][0] == "agent-7"
    assert "t/w.safetensors" in sent[0][1] and "--expendable" in sent[0][1]


def test_notify_waits_for_the_dust_to_settle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                             ) -> None:
    eng, st, d = _engine(tmp_path)
    eng.cfg.notify_after_s = 300
    (tmp_path / "disk" / "t").mkdir(parents=True)
    (tmp_path / "disk" / "t" / "w.safetensors").write_bytes(b"w")
    wid = st.writer_id("k", "{}", None, "sess", "agent-7", "bodas")
    st.record_writes("u1", {("t/w.safetensors", wid): (time.time(), time.time(), 1)})
    eng.scan(d)
    assert eng.pending_notices(d) == {}


def test_hf_index_survives_cache_deletion(tmp_path: Path) -> None:
    eng, st, _d = _engine(tmp_path)
    hub = tmp_path / "hub"
    rev, sha = "a" * 40, "b" * 64
    snap = hub / "models--o--n" / "snapshots" / rev
    snap.mkdir(parents=True)
    (hub / "models--o--n" / "blobs").mkdir()
    (hub / "models--o--n" / "blobs" / sha).write_bytes(b"w")
    (snap / "m.safetensors").symlink_to(f"../../blobs/{sha}")
    eng.cfg.hf_hubs = [str(hub)]
    eng.refresh_hf_index()
    import shutil
    shutil.rmtree(hub)
    eng.refresh_hf_index()
    assert st.hf_blob(sha) == {"repo": "o/n", "revision": rev, "file": "m.safetensors"}


def test_claim_via_spool_is_attributed_to_file_owner(tmp_path: Path) -> None:
    eng, st, d = _engine(tmp_path)
    (tmp_path / "disk" / "x").mkdir(parents=True)
    eng.cfg.spool.mkdir(parents=True)
    probe = {"u1": d}
    import provenance.engine as pe
    orig = pe.probe_disk
    pe.probe_disk = lambda root: probe["u1"]  # type: ignore[assignment]
    try:
        (eng.cfg.spool / "1.json").write_text(json.dumps(
            {"path": str(tmp_path / "disk" / "x"), "kind": "expendable", "value": "tmp",
             "agent": "agent-3"}))
        done = eng.ingest_spool()
    finally:
        pe.probe_disk = orig
    assert done and done[0][1] == "x"
    c = st.claims("u1")["x"]
    assert c["kind"] == "expendable" and c["by"].endswith("(agent-3)")
    assert not list(eng.cfg.spool.glob("*.json"))
