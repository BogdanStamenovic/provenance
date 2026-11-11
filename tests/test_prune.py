from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from conftest import ctx_for, git

import provenance.prune as pr
from provenance.classify import AT_RISK, EXPENDABLE, RECREATABLE, Unit, preserve_atimes, walk
from provenance.config import Config, ConfigError, load
from provenance.engine import Daemon, Disk, Engine
from provenance.store import Store

OLD = time.time() - 60 * 86400
REV = "a" * 40
SHA = "b" * 64


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Engine, Store, Disk, Path]:
    area = tmp_path / "area"
    area.mkdir()
    cfg = Config(state_dir=str(tmp_path / "state"), areas=[str(area)], hf_hubs=[],
                 prune_after_days=30, trash_grace_h=0)
    st = Store(cfg.db_path)
    eng = Engine(cfg, st)
    d = Disk(f"area:{area}", str(area), "test", 0)
    monkeypatch.setattr(pr, "mount_options", lambda p: ("/", "ext4", ["rw", "relatime"]))
    monkeypatch.setattr(pr, "paths_in_use", lambda: set())
    monkeypatch.setattr(pr, "team_repos", lambda: set())
    monkeypatch.setattr(pr, "verify", lambda c, a, listing, hf, timeout=30: "verified (stub)")
    return eng, st, d, area


def _store_units(st: Store, d: Disk, units: list[Unit]) -> None:
    st.replace_units(d.uuid, "", units)


def _unit(path: str, kind: str, risk: str = RECREATABLE, used: float = OLD) -> Unit:
    u = Unit(path, kind, risk, 1, 10, rebuild=[f'echo rebuild "$ROOT/{path}"'])
    u.detail["last_used"] = used
    return u


def test_only_unused_recreatable_prunable_kinds_qualify(env: tuple) -> None:
    eng, st, d, area = env
    for p in ("venv", "fresh", "risky", "data", "cache", "keepme", "store"):
        (area / p).mkdir()
    _store_units(st, d, [
        _unit("venv", "venv"),
        _unit("fresh", "venv", used=time.time() - 3600),
        _unit("risky", "git", AT_RISK),
        _unit("data", "data", AT_RISK),
        _unit("cache", "cache", EXPENDABLE),
        _unit("keepme", "venv"),
        _unit("store", "hf-store"),
    ])
    st.set_claim(d.uuid, "keepme", "keep", "mine", "bogdan")
    got = {c.unit.path: c for c in pr.Pruner(eng).candidates(d)}
    assert got["venv"].ok and got["cache"].ok
    assert "used 0.0 d ago" in got["fresh"].reasons
    assert "risk is at_risk" in got["risky"].reasons
    assert any("never pruned" in r for r in got["data"].reasons)
    assert any("kept by bogdan" in r for r in got["keepme"].reasons)
    assert any("never pruned" in r for r in got["store"].reasons)


def test_in_use_and_team_repos_are_never_pruned(env: tuple, monkeypatch: pytest.MonkeyPatch
                                                ) -> None:
    eng, st, d, area = env
    for p in ("a", "b"):
        (area / p).mkdir()
    _store_units(st, d, [_unit("a", "venv"), _unit("b", "git")])
    monkeypatch.setattr(pr, "paths_in_use", lambda: {f"{area}/a/bin/python"})
    monkeypatch.setattr(pr, "team_repos", lambda: {f"{area}/b"})
    got = {c.unit.path: c for c in pr.Pruner(eng).candidates(d)}
    assert any(r.startswith("in use:") for r in got["a"].reasons)
    assert any(r.startswith("hotline team repo") for r in got["b"].reasons)


def test_noatime_needs_watch_coverage(env: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, st, d, area = env
    monkeypatch.setattr(pr, "mount_options", lambda p: ("/m", "ntfs3", ["rw", "noatime"]))
    (area / "v").mkdir()
    (area / "w").mkdir()
    _store_units(st, d, [_unit("v", "venv"), _unit("w", "venv")])
    c = {x.unit.path: x for x in pr.Pruner(eng).candidates(d)}
    assert any("watched for 0.0 of 30 d" in r for r in c["v"].reasons), "no coverage yet"
    st.set_coverage(d.uuid, time.time() - 31 * 86400, time.time())
    st.record_usage(d.uuid, {"w": time.time() - 86400})
    c = {x.unit.path: x for x in pr.Pruner(eng).candidates(d)}
    assert c["v"].ok
    assert "opened 1.0 d ago" in c["w"].reasons


def test_plan_review_apply_trash_restore(env: tuple) -> None:
    eng, st, d, area = env
    (area / "env").mkdir()
    (area / "env" / "f").write_text("x")
    _store_units(st, d, [_unit("env", "venv")])
    p = pr.Pruner(eng)
    made = p.propose(d)
    assert made is not None
    pid, _ = made
    assert p.propose(d) is None, "one pending plan per area"
    assert "Approve: provenance prune --approve " + pid in p.summary(pid)
    eng.decide_plan(pid, True, "hotline-80")
    res = p.apply(d, ["env"], pid)
    assert [x["path"] for x in res["done"]] == ["env"]
    assert not (area / "env").exists()
    assert (area / ".provenance-trash" / pid / "env" / "f").read_text() == "x"
    assert st.plan(pid)["status"] == "applied"
    assert p.restore(str(area / "env")).startswith("moved back")
    assert (area / "env" / "f").read_text() == "x"


def test_apply_rechecks_and_skips_what_changed(env: tuple) -> None:
    eng, st, d, area = env
    (area / "env").mkdir()
    _store_units(st, d, [_unit("env", "venv")])
    p = pr.Pruner(eng)
    pid, _ = p.propose(d) or ("", [])
    _store_units(st, d, [_unit("env", "venv", used=time.time())])  # used since the plan
    res = p.apply(d, ["env"], pid)
    assert not res["done"] and "used" in res["skipped"][0]["why"]
    assert (area / "env").exists()


def test_rejected_items_are_not_reproposed(env: tuple) -> None:
    eng, st, d, area = env
    (area / "env").mkdir()
    _store_units(st, d, [_unit("env", "venv")])
    p = pr.Pruner(eng)
    pid, _ = p.propose(d) or ("", [])
    eng.decide_plan(pid, False, "hotline-80")
    assert p.propose(d) is None


def test_purge_then_restore_from_recipe(env: tuple, tmp_path: Path) -> None:
    eng, st, d, area = env
    (area / "env").mkdir()
    u = _unit("env", "venv")
    u.rebuild = ['mkdir -p "$ROOT/env" && echo rebuilt > "$ROOT/env/f"']
    _store_units(st, d, [u])
    p = pr.Pruner(eng)
    p.apply(d, ["env"], None)
    assert p.purge_trash(0)
    assert not (area / ".provenance-trash").exists()
    assert "rebuilt from the recipe" in p.restore(str(area / "env"))
    assert (area / "env" / "f").read_text() == "rebuilt\n"


def test_areas_never_notify_writers(env: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, st, d, area = env
    (area / "x").mkdir()
    _store_units(st, d, [_unit("x", "data", AT_RISK)])
    monkeypatch.setattr(eng, "_send", lambda a, m: pytest.fail("must not notify"))
    assert eng.notify(d) == []


def test_config_guards(tmp_path: Path) -> None:
    c = tmp_path / "c.toml"
    c.write_text(f'state_dir = "{tmp_path}/a/state"\nareas = ["{tmp_path}/a"]\n')
    with pytest.raises(ConfigError, match="delete itself"):
        load(str(c))
    c.write_text('areas = ["/home"]\n')
    with pytest.raises(ConfigError, match="too broad"):
        load(str(c))


def test_preserve_atimes_undoes_our_reads_but_not_edits(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("1")
    b.write_text("2")
    for f in (a, b):
        os.utime(f, (OLD, OLD))
    with preserve_atimes(str(tmp_path)):
        os.utime(a, (time.time(), OLD))  # a read (atime only)
        b.write_text("changed")  # a real edit
    assert abs(a.stat().st_atime - OLD) < 1
    assert b.stat().st_mtime > OLD + 1000


def test_git_scan_does_not_make_repo_look_used(disk: Path, pushed_repo: Path) -> None:
    (pushed_repo / "README").write_text("hi\n")  # same content, new mtime: stale index
    for dp, _dn, fn in os.walk(pushed_repo):
        for f in fn:
            os.utime(os.path.join(dp, f), (OLD, OLD))
    u = {x.path: x for x in walk(ctx_for(disk))}["projects/app"]
    assert abs(float(u.detail["last_used"]) - OLD) < 1  # type: ignore[arg-type]
    walk(ctx_for(disk))  # git status rehashes the file; the next scan must not see it
    u = {x.path: x for x in walk(ctx_for(disk))}["projects/app"]
    assert abs(float(u.detail["last_used"]) - OLD) < 1  # type: ignore[arg-type]
    assert git("status", "--porcelain", cwd=pushed_repo) == ""


def test_hf_cache_listing_uses_etag_not_shared_store_hash(disk: Path) -> None:
    hub = disk / "hub"
    repo = hub / "models--o--n"
    store = hub / "blobs" / "cd"
    store.mkdir(parents=True)
    (hub / "blobs" / ".huggingface-shared-blobs").write_text("1")
    xet = "cd" + "e" * 62
    (store / xet).write_bytes(b"w")
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / SHA).symlink_to(f"../../blobs/cd/{xet}")
    (repo / "snapshots" / REV).mkdir(parents=True)
    (repo / "snapshots" / REV / "m.safetensors").symlink_to(f"../../blobs/{SHA}")
    units = {u.path: u for u in walk(ctx_for(disk))}
    assert units["hub/blobs"].kind == "hf-store"
    hfu = units["hub/models--o--n"]
    assert (f"{REV[:12]}/m.safetensors", 1, SHA) in hfu.listing


def test_coverage_survives_restart_not_long_gap(env: tuple, monkeypatch: pytest.MonkeyPatch
                                                ) -> None:
    eng, st, d, _area = env
    dmn = Daemon(eng)
    now = time.time()
    monkeypatch.setattr("provenance.engine._boot_time", lambda: now - 86400 * 5)
    st.set_coverage(d.uuid, now - 20 * 86400, now - 600)  # restarted 10 min ago
    dmn._start_coverage(d, now)
    assert st.coverage(d.uuid)[0] == pytest.approx(now - 20 * 86400)
    st.set_coverage(d.uuid, now - 20 * 86400, now - 2 * 86400)  # stopped for 2 days
    dmn._start_coverage(d, now)
    assert st.coverage(d.uuid)[0] == pytest.approx(now)
    monkeypatch.setattr("provenance.engine._boot_time", lambda: now - 300)
    st.set_coverage(d.uuid, now - 20 * 86400, now - 2 * 86400)  # box was off, just booted
    dmn._start_coverage(d, now)
    assert st.coverage(d.uuid)[0] == pytest.approx(now - 20 * 86400)
