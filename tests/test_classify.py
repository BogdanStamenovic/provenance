from __future__ import annotations

import os
import re
from pathlib import Path

from conftest import ctx_for, git

from provenance import classify
from provenance.classify import AT_RISK, EXPENDABLE, RECREATABLE, REDUNDANT, Unit, walk

REV = "71034c5d8bde858ff824298bdedc65515b97d2b9"
SHA = "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500"


def by_path(units: list[Unit]) -> dict[str, Unit]:
    return {u.path: u for u in units}


def test_clean_pushed_repo_is_recreatable(disk: Path, pushed_repo: Path) -> None:
    u = by_path(walk(ctx_for(disk)))["projects/app"]
    assert (u.kind, u.risk) == ("git", RECREATABLE)
    assert u.rebuild[0].startswith("git clone ")


def test_unpushed_commit_is_at_risk_and_restore_targets_pushed_base(
        disk: Path, pushed_repo: Path) -> None:
    base = git("rev-parse", "HEAD", cwd=pushed_repo).strip()
    git("commit", "-q", "--allow-empty", "-m", "local", cwd=pushed_repo)
    u = by_path(walk(ctx_for(disk)))["projects/app"]
    assert u.risk == AT_RISK
    assert "1 commit(s) on no remote branch" in u.note
    assert base in u.rebuild[1], "a clone cannot reach the unpushed commit"
    assert any("lost" in c for c in u.rebuild)


def test_gitignored_weights_are_found_inside_a_clean_repo(disk: Path, pushed_repo: Path) -> None:
    (pushed_repo / ".git" / "info" / "exclude").write_text("ckpt/\n")
    (pushed_repo / "ckpt").mkdir()
    (pushed_repo / "ckpt" / "step.safetensors").write_bytes(b"x" * 100)
    u = by_path(walk(ctx_for(disk)))["projects/app"]
    assert u.risk == AT_RISK
    assert "ckpt/step.safetensors [weights]" in u.note


def test_hf_local_dir_needs_attribution_for_repo_id(disk: Path) -> None:
    d = disk / "models" / "m"
    meta = d / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"w")
    (meta / "model.safetensors.metadata").write_text(f"{REV}\n{SHA}\n1.0\n")
    u = by_path(walk(ctx_for(disk)))["models/m"]
    assert u.risk == AT_RISK and "repo id is unknown" in u.note
    known = ctx_for(disk, source_for=lambda rel: {"kind": "hf", "ref": "org/m"}
                    if rel == "models/m" else None)
    u = by_path(walk(known))["models/m"]
    assert u.risk == RECREATABLE
    assert u.rebuild == [f'hf download org/m --revision {REV} --local-dir "$ROOT/models/m"']
    assert ("model.safetensors", 1, SHA) in u.listing


def test_hf_cache_repo(disk: Path) -> None:
    repo = disk / "hub" / "models--org--name"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / REV).mkdir(parents=True)
    (repo / "blobs" / SHA).write_bytes(b"w")
    (repo / "snapshots" / REV / "model.safetensors").symlink_to(f"../../blobs/{SHA}")
    u = by_path(walk(ctx_for(disk)))["hub/models--org--name"]
    assert (u.kind, u.risk) == ("hf-cache", RECREATABLE)
    assert u.rebuild == [f'hf download org/name --revision {REV} --cache-dir "$ROOT/hub"']


def test_weights_matched_by_hash(disk: Path) -> None:
    (disk / "loose").mkdir()
    (disk / "loose" / "renamed.safetensors").write_bytes(b"w")
    ctx = ctx_for(disk, file_sha=lambda *a: SHA,
                  hf_blob=lambda sha: {"repo": "org/n", "revision": REV, "file": "m.safetensors"}
                  if sha == SHA else None)
    u = by_path(walk(ctx))["loose/renamed.safetensors"]
    assert u.risk == RECREATABLE and "identical (sha256)" in u.note


def test_unknown_weights_are_at_risk(disk: Path) -> None:
    (disk / "train").mkdir()
    (disk / "train" / "lora.safetensors").write_bytes(b"x")
    u = by_path(walk(ctx_for(disk)))["train/lora.safetensors"]
    assert (u.kind, u.risk) == ("weights", AT_RISK)


def test_sibling_does_not_inherit_a_files_source(disk: Path) -> None:
    (disk / "dl").mkdir()
    (disk / "dl" / "a.tar.gz").write_bytes(b"a")
    (disk / "dl" / "b.tar.gz").write_bytes(b"b")
    url = {"kind": "url", "ref": "https://e/a.tar.gz"}
    ctx = ctx_for(disk, source_for=lambda rel: url if rel == "dl/a.tar.gz" else None)
    units = by_path(walk(ctx))
    assert units["dl/a.tar.gz"].risk == RECREATABLE
    assert units["dl/b.tar.gz"].risk == AT_RISK


def test_small_bin_is_not_weights(disk: Path) -> None:
    (disk / "fw").mkdir()
    (disk / "fw" / "firmware.bin").write_bytes(b"\0" * 10)
    assert by_path(walk(ctx_for(disk)))["fw"].kind == "data"


def test_venv_freeze(disk: Path) -> None:
    v = disk / "envs" / "e"
    sp = v / "lib" / "python3.13" / "site-packages" / "six-1.17.0.dist-info"
    sp.mkdir(parents=True)
    (v / "pyvenv.cfg").write_text("version = 3.13.5\n")
    (sp / "METADATA").write_text("Metadata-Version: 2.1\nName: six\nVersion: 1.17.0\n")
    u = by_path(walk(ctx_for(disk)))["envs/e"]
    assert u.risk == RECREATABLE
    assert u.detail["requirements"] == ["six==1.17.0"]


def test_caches_and_timeshift(disk: Path) -> None:
    (disk / "a" / "__pycache__").mkdir(parents=True)
    (disk / "a" / "__pycache__" / "x.pyc").write_bytes(b"x")
    snap = disk / "timeshift" / "snapshots" / "2026-11-03_20-00-01"
    snap.mkdir(parents=True)
    (snap / "info.json").write_text("{}")
    units = by_path(walk(ctx_for(disk)))
    assert units["a/__pycache__"].risk == EXPENDABLE
    assert units["timeshift"].risk == REDUNDANT


def test_pure_data_subtree_is_one_unit(disk: Path) -> None:
    for c in range(5):
        d = disk / "dataset" / f"class{c}"
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{i}.jpg").write_bytes(b"j")
    units = walk(ctx_for(disk))
    data = [u for u in units if u.path.startswith("dataset")]
    assert len(data) == 1 and data[0].path == "dataset" and data[0].files == 15
    assert ("class3/2.jpg", 1, "") in data[0].listing


def test_claims_override(disk: Path) -> None:
    (disk / "x").mkdir()
    (disk / "x" / "w.safetensors").write_bytes(b"w")
    claims = {"x/w.safetensors": {"kind": "expendable", "value": "scratch", "by": "a"}}
    u = by_path(walk(ctx_for(disk, claim_for=claims.get)))["x/w.safetensors"]
    assert u.risk == EXPENDABLE
    claims = {"x/w.safetensors": {"kind": "source", "value": "hf:org/n@main", "by": "a"}}
    u = by_path(walk(ctx_for(disk, claim_for=claims.get)))["x/w.safetensors"]
    assert u.risk == RECREATABLE and u.rebuild[0].startswith("hf download org/n w.safetensors")


def test_parse_source() -> None:
    assert classify.parse_source("hf:a/b@v1") == {"kind": "hf", "ref": "a/b", "revision": "v1"}
    assert classify.parse_source("https://huggingface.co/a/b")["kind"] == "hf"
    assert classify.parse_source("https://github.com/a/b")["kind"] == "git"
    assert classify.parse_source("https://e.com/f.zip")["kind"] == "url"
    assert classify.parse_source("trained with train.py on data/x")["kind"] == "manual"


SAFE_CMD = re.compile(r"^(git|hf|uv|curl|mkdir|mv|rm -rf|rsync|OLLAMA_MODELS=|\(cd )")


def test_every_rebuild_command_writes_only_under_root(disk: Path, pushed_repo: Path) -> None:
    """A restore once ran `sudo timeshift --create` against the live root. Never again."""
    test_caches_and_timeshift(disk)
    test_hf_cache_repo(disk)
    test_venv_freeze(disk)
    (disk / "dl").mkdir(exist_ok=True)
    (disk / "dl" / "f.tgz").write_bytes(b"f")
    ctx = ctx_for(disk, source_for=lambda rel: {"kind": "url", "ref": "https://e/f.tgz"}
                  if rel == "dl/f.tgz" else None)
    for u in walk(ctx):
        for cmd in u.rebuild:
            if cmd.startswith("#"):
                continue
            first = cmd.splitlines()[0]
            assert "sudo" not in cmd, cmd
            assert "$ROOT" in first, cmd
            assert SAFE_CMD.match(first), cmd


def test_human() -> None:
    assert classify.human(0) == "0 B"
    assert classify.human(1536) == "1.5 KB"
    assert os.sep == "/"
