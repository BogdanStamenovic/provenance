"""Configuration: /etc/provenance.toml (path overridable with --config or $PROVENANCE_CONFIG)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = "/etc/provenance.toml"


class ConfigError(Exception):
    pass


@dataclass
class Config:
    # Where the recipe lives. Must be on a trusted disk, never on a watched one.
    state_dir: str = "/var/lib/provenance"
    disks: list[str] = field(default_factory=list)
    budget_mb: float = 50.0
    # HF hub caches to index by blob sha256, so loose weights can be matched to a repo.
    hf_hubs: list[str] = field(default_factory=lambda: [
        "/home/bodas/.cache/huggingface/hub", "/mnt/offload/hf/hub"])
    # Notifications to the agent that wrote an at-risk unit run as this user.
    notify_user: str = "bodas"
    # hotline agent to tell when the writer was not an agent (a human shell, a service).
    # Empty: nobody is told; it still shows up in `provenance status`.
    notify_fallback: str = ""
    notify_cmd: str = "/home/bodas/.claude/bin/hotline"
    debounce_s: float = 60.0
    # Ask the writer only once a unit has been at risk this long, so a burst of writes
    # becomes one message and a file that is claimed or deleted quickly is never asked about.
    notify_after_s: float = 300.0
    full_scan_every_h: float = 24.0
    hash_weights: bool = True
    # --- single-disk mode ----------------------------------------------------------------
    # Directories on the trusted disk(s) whose unused, recreatable units may be deleted
    # once their recipe is saved. Not mount points: any directory. Never the state_dir.
    areas: list[str] = field(default_factory=list)
    # Untouched this long (atime on relatime mounts; observed opens on noatime ones).
    prune_after_days: float = 30.0
    # Soft delete: units sit in <area>/.provenance-trash this long before real deletion.
    # HF caches are the exception (huggingface_hub deletes them; see prune.py).
    trash_grace_h: float = 24.0
    # The daemon proposes plans only when true. Each plan goes to prune_approver and
    # nothing is deleted until that plan id is approved.
    auto_prune: bool = False
    prune_approver: str = "hotline-80"
    prune_report_to: str = ""
    plan_ttl_h: float = 48.0
    path: str = ""

    @property
    def db_path(self) -> Path:
        return Path(self.state_dir) / "provenance.db"

    @property
    def spool(self) -> Path:
        return Path(self.state_dir) / "spool"

    @property
    def budget(self) -> int:
        return int(self.budget_mb * 1024 * 1024)


def load(path: str | None = None) -> Config:
    p = path or os.environ.get("PROVENANCE_CONFIG") or DEFAULT_PATH
    cfg = Config(path=p)
    if not os.path.exists(p):
        if path:
            raise ConfigError(f"config file {p} does not exist")
        return cfg
    try:
        data = tomllib.loads(Path(p).read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    for k, v in data.items():
        if not hasattr(cfg, k) or k == "path":
            raise ConfigError(f"{p}: unknown key {k!r}")
        setattr(cfg, k, v)
    cfg.areas = [os.path.realpath(a) for a in cfg.areas]
    state = os.path.realpath(cfg.state_dir)
    for a in cfg.areas:
        if state == a or state.startswith(a.rstrip("/") + "/"):
            raise ConfigError(f"state_dir {cfg.state_dir} is inside prune area {a}: the "
                              "recipe could delete itself")
        if a in ("/", "/home", "/usr", "/etc", "/var", "/boot") or a == os.path.expanduser("~"):
            raise ConfigError(f"area {a} is too broad; name the directories to prune")
    for d in cfg.disks:
        if os.path.realpath(cfg.state_dir).startswith(os.path.realpath(d).rstrip("/") + "/"):
            raise ConfigError(f"state_dir {cfg.state_dir} is on watched disk {d}: the recipe "
                              "must live on a trusted disk")
    return cfg
