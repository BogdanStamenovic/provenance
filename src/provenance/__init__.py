"""provenance: keep a recipe on the SSD for recreating everything on an untrusted disk."""

from __future__ import annotations

__version__ = "0.1.1"

from .classify import AT_RISK, EXPENDABLE, RECREATABLE, REDUNDANT, ClassifyError, Unit
from .cli import main
from .store import Store, StoreError

__all__ = [
    "AT_RISK",
    "EXPENDABLE",
    "RECREATABLE",
    "REDUNDANT",
    "ClassifyError",
    "Store",
    "StoreError",
    "Unit",
    "__version__",
    "main",
]
