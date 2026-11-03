from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from provenance.classify import Context

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "GIT_CONFIG_GLOBAL": "/dev/null"}


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def disk(tmp_path: Path) -> Path:
    d = tmp_path / "disk"
    d.mkdir()
    return d


@pytest.fixture
def pushed_repo(tmp_path: Path, disk: Path) -> Path:
    """A clone on the disk whose every commit is on its (local, bare) remote."""
    remote = tmp_path / "remote.git"
    git("init", "-q", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "seed"
    git("clone", "-q", str(remote), str(seed))
    (seed / "README").write_text("hi\n")
    git("add", "README", cwd=seed)
    git("commit", "-q", "-m", "first", cwd=seed)
    git("push", "-q", "origin", "main", cwd=seed)
    repo = disk / "projects" / "app"
    repo.parent.mkdir(parents=True)
    git("clone", "-q", str(remote), str(repo))
    return repo


def ctx_for(disk: Path, **kw: object) -> Context:
    return Context(root=str(disk), **kw)  # type: ignore[arg-type]
