"""HF cache sizes and deletion, done by huggingface_hub itself.

huggingface_hub >= 2.4 keeps a cache-wide shared Xet blob store (`<cache>/blobs/xx/<hash>`)
that repos' `blobs/<etag>` symlinks point into, refcounted by `.refs` manifests. Deleting
a `models--X` directory by hand frees nothing and leaves the store's bookkeeping stale, so
provenance never does that: it asks huggingface_hub what deleting a repo would free and
lets huggingface_hub delete it.

Runs as a subprocess (`python -m provenance.hfcache plan|delete CACHE REPO_DIR...`) under
the cache owner's uid, so nothing in the user's cache becomes root-owned. Prints JSON.
"""

from __future__ import annotations

import json
import os
import pwd
import subprocess
import sys
from typing import Any


class HFCacheError(Exception):
    pass


def _strategy(cache: str, repo_dirs: list[str]) -> tuple[Any, dict[str, Any]]:
    from huggingface_hub import scan_cache_dir  # type: ignore[import-not-found,unused-ignore]

    info = scan_cache_dir(cache)
    by_dir = {r.repo_path.name: r for r in info.repos}
    revs: list[str] = []
    per: dict[str, Any] = {}
    for d in repo_dirs:
        r = by_dir.get(d)
        if r is None:
            per[d] = {"found": False}
            continue
        hashes = [x.commit_hash for x in r.revisions]
        revs += hashes
        alone = info.delete_revisions(*hashes).expected_freed_size if hashes else 0
        per[d] = {"found": True, "repo_id": r.repo_id, "repo_type": r.repo_type,
                  "revisions": hashes, "freed_alone": alone, "size_on_disk": r.size_on_disk}
    return info.delete_revisions(*revs), per


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] not in ("plan", "delete"):
        print("usage: python -m provenance.hfcache plan|delete CACHE_DIR REPO_DIR...",
              file=sys.stderr)
        return 2
    cmd, cache, dirs = argv[0], argv[1], argv[2:]
    strategy, per = _strategy(cache, dirs)
    out = {"freed_together": strategy.expected_freed_size, "repos": per}
    if cmd == "delete":
        strategy.execute()
        out["deleted"] = True
    print(json.dumps(out))
    return 0


def run(cmd: str, cache: str, repo_dirs: list[str], python: str | None = None,
        timeout: float = 1800) -> dict[str, Any]:
    """Run plan/delete as the cache's owner and return its JSON result."""
    py = python or sys.executable
    args = [py, "-m", "provenance.hfcache", cmd, cache, *repo_dirs]
    owner = os.stat(cache).st_uid
    if os.geteuid() == 0 and owner != 0:
        args = ["runuser", "-u", pwd.getpwuid(owner).pw_name, "--", *args]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HFCacheError(f"hfcache {cmd}: {exc}") from exc
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-3:]
        raise HFCacheError(f"hfcache {cmd} failed ({r.returncode}): {' | '.join(tail)}")
    result: dict[str, Any] = json.loads(r.stdout)
    return result


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
