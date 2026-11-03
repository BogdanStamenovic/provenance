from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from provenance.classify import AT_RISK, RECREATABLE, Unit
from provenance.cli import main
from provenance.store import Store


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "provenance" in capsys.readouterr().out


def test_usage_error_is_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["claim", "/x"]) == 2
    assert "error" in capsys.readouterr().err


@pytest.fixture
def cfg(tmp_path: Path) -> str:
    state = tmp_path / "state"
    st = Store(state / "provenance.db")
    st.touch_disk("u1", str(tmp_path / "disk"), "", 0)
    st.replace_units("u1", "", [
        Unit("models/m", "hf-local", RECREATABLE, 1, 5, source={"kind": "hf", "ref": "o/m"},
             rebuild=['hf download o/m --local-dir "$ROOT/models/m"'],
             listing=[("model.safetensors", 5, "")]),
        Unit("train/w.safetensors", "weights", AT_RISK, 1, 9, note="no source"),
    ])
    st.close()
    c = tmp_path / "c.toml"
    c.write_text(f'state_dir = "{state}"\ndisks = ["{tmp_path / "disk"}"]\n')
    return str(c)


def test_status_and_strict(cfg: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", cfg, "status"]) == 0
    out = capsys.readouterr().out
    assert "would be lost" in out and "train/w.safetensors" in out
    assert main(["--config", cfg, "status", "--strict"]) == 1
    capsys.readouterr()
    assert main(["--config", cfg, "status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["disks"][0]["risk"]["at_risk"]["units"] == 1


def test_rebuild_script_runs_and_lists_losses(cfg: str, tmp_path: Path,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", cfg, "rebuild", "--into", str(tmp_path / "new")]) == 0
    script = capsys.readouterr().out
    assert "ROOT='" + str(tmp_path / "new") + "'" in script
    assert "# train/w.safetensors" in script
    r = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr


def test_verify_restore(cfg: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    new = tmp_path / "new" / "models" / "m"
    new.mkdir(parents=True)
    assert main(["--config", cfg, "verify-restore", str(tmp_path / "new")]) == 1
    (new / "model.safetensors").write_bytes(b"12345")
    assert main(["--config", cfg, "verify-restore", str(tmp_path / "new")]) == 0


def test_claim_unprivileged_goes_to_spool(cfg: str, tmp_path: Path) -> None:
    (tmp_path / "state" / "spool").mkdir()
    (tmp_path / "disk" / "train").mkdir(parents=True)
    (tmp_path / "disk" / "train" / "w.safetensors").write_bytes(b"w")
    assert main(["--config", cfg, "-q", "claim", str(tmp_path / "disk" / "train"
                                                     / "w.safetensors"), "--expendable"]) == 0
    req = json.loads(next((tmp_path / "state" / "spool").glob("*.json")).read_text())
    assert req["kind"] == "expendable"


def test_show(cfg: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", cfg, "show", str(tmp_path / "disk" / "models" / "m" / "x")]) == 0
    out = capsys.readouterr().out
    assert "(models/m/x is inside this unit)" in out and "hf download o/m" in out
