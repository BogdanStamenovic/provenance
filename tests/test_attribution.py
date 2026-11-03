from __future__ import annotations

import os
from pathlib import Path

import pytest

from provenance.attribution import (
    Proc,
    Writer,
    capture,
    hf_blobs_named_in,
    hf_snapshot_path,
    source_from_argv,
    source_of,
)

REV = "71034c5d8bde858ff824298bdedc65515b97d2b9"
SHA = "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500"


@pytest.mark.parametrize(
    ("argv", "kind", "ref", "rev"),
    [
        (["hf", "download", "org/model", "--local-dir", "/d"], "hf", "org/model", None),
        (["huggingface-cli", "download", "--revision", "abc", "org/m"], "hf", "org/m", "abc"),
        (["/v/bin/python3", "-m", "huggingface_hub.commands.huggingface_cli"], None, None, None),
        (["uvx", "hf", "download", "a/b", "--repo-type", "dataset"], "hf", "datasets/a/b", None),
        (["git", "clone", "--depth", "1", "https://x/y.git", "dst"], "git", "https://x/y.git", None),
        (["git", "clone", "-b", "dev", "git@h:a/b"], "git", "git@h:a/b", "dev"),
        (["wget", "-q", "-O", "f.tgz", "https://e.com/f.tgz"], "url", "https://e.com/f.tgz", None),
        (["curl", "-fL", "-o", "x", "https://e.com/x"], "url", "https://e.com/x", None),
        (["ollama", "pull", "llama3:8b"], "ollama", "llama3:8b", None),
        (["rsync", "-a", "/home/u/proj", "/hdd/"], "copy", "/home/u/proj", None),
        (["ls", "-la"], None, None, None),
    ],
)
def test_source_from_argv(argv: list[str], kind: str | None, ref: str | None,
                          rev: str | None) -> None:
    s = source_from_argv(argv)
    if kind is None:
        assert s is None
        return
    assert s is not None
    assert (s.kind, s.ref, s.revision) == (kind, ref, rev)


def test_copy_out_of_hf_snapshot_names_repo_and_file() -> None:
    src = f"/c/hub/models--org--name/snapshots/{REV}/sub/w.safetensors"
    s = source_from_argv(["cp", "-L", src, "/hdd/x.safetensors"])
    assert s is not None
    assert (s.kind, s.ref, s.revision, s.file) == ("hf", "org/name", REV, "sub/w.safetensors")
    assert hf_snapshot_path(src) == ("org/name", REV, "sub/w.safetensors")
    assert hf_snapshot_path("/somewhere/else/w.safetensors") is None


def test_blob_learned_from_cmdline_while_cache_exists(tmp_path: Path) -> None:
    snap = tmp_path / "hub" / "models--org--name" / "snapshots" / REV
    blobs = tmp_path / "hub" / "models--org--name" / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    (blobs / SHA).write_bytes(b"w")
    (snap / "model.safetensors").symlink_to(f"../../blobs/{SHA}")
    w = Writer(chain=[Proc(1, ["cp", "-L", "model.safetensors", "/hdd/x"], str(snap))])
    assert hf_blobs_named_in(w) == [(SHA, "org/name", REV, "model.safetensors")]


def test_real_fetch_beats_copy_further_up_the_chain() -> None:
    w = Writer(chain=[Proc(3, ["rsync", "-a", "/tmp/x", "/hdd/x"], "/"),
                      Proc(2, ["hf", "download", "a/b"], "/")])
    s = source_of(w)
    assert s is not None and s.kind == "hf"


def test_capture_own_process_finds_session(monkeypatch: pytest.MonkeyPatch) -> None:
    w = capture(os.getpid())
    assert w.chain and w.chain[0].pid == os.getpid()
    assert w.user
