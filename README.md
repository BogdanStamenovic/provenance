# provenance

provenance keeps a recipe, on the SSD, for recreating everything on a disk you do not
trust. It watches the disk, works out where each thing on it came from, and records how
to get it back: `hf download org/name --revision SHA`, `git clone URL && checkout SHA`, a
download URL, a pip freeze. Anything with no known source is marked **AT RISK**, and the
agent that wrote it is asked to name a source or to say losing it is fine.

The point is the ratio. Model weights, cloned repos, downloads, venvs and backups make up
most of a big data disk, and almost all of it can be fetched again. The recipe stores the
*where from* rather than the bytes, so a multi-terabyte disk fits in tens of megabytes on
the trusted disk. If the HDD dies, `provenance rebuild` prints the script that brings
back everything recreatable, and the same recipe lists exactly what was lost.

Nothing is ever written to the watched disk. The recipe exists to survive that disk.

With no second disk, **single-disk mode** turns the same recipe into free space: it deletes
units nobody has used for a while and that can be rebuilt from their source, after an
approver has reviewed the list, and brings any of them back on request. See
[Single-disk mode](#single-disk-mode).

## How it works

```
 untrusted HDD                         trusted SSD (/var/lib/provenance/provenance.db)
 ─────────────                         ─────────────────────────────────────────────
 every write ──fanotify──► who wrote it: process chain, cwd, Claude session → agent
                           what it is:   git / HF / ollama / venv / cache / timeshift /
                                         weights / archive / image / data
                           how to get it back, and the risk state ─────────► recipe
 at-risk data ───────────► hotline message to the agent that wrote it
```

**Watching.** One fanotify mark (`FAN_MARK_FILESYSTEM`, `FAN_REPORT_DFID_NAME`) covers the
whole filesystem, however many directories it has; there is no per-directory watch to run
out of. Events carry the writer's pid. The daemon snapshots that pid's ancestry from
`/proc` straight away, keeping each argv and cwd. Every process a Claude Code session starts
inherits `CLAUDE_CODE_SESSION_ID`, which hotline's `agents.json` maps to an agent name: that
is who gets asked when something lands without a source.

**Recognising sources.** A fetch command anywhere in the writer's ancestry is the source:

| Command | Recorded as |
|---|---|
| `hf download` / `huggingface-cli download` (also via `python -m`, `uv run`, `uvx`) | HF repo, revision, repo type |
| `git clone` | remote URL; the repo itself is then read for HEAD, branch, unpushed work |
| `wget`, `curl`, `aria2c`, `axel`, `yt-dlp`, `gdown` | the URL |
| `ollama pull` | the model tag |
| `cp`/`rsync`/`mv`/`ln` out of an HF cache snapshot | HF repo, revision and file (taken from the path) |
| `cp`/`rsync`/`mv` from elsewhere | the source path; **redundant** while it still exists |

**Classifying.** A daily full scan, plus a rescan of whatever changed (60 s after it stops
changing), walks the disk. A directory a classifier recognises becomes a *unit*, and the
walk does not go inside it:

| Unit | Recognised by | Recreated with |
|---|---|---|
| `git` | `.git` | clone + checkout of the newest commit a remote has; unpushed commits, uncommitted changes, untracked files and gitignored content (walked: weights inside `ckpt/` are found) make it AT RISK |
| `hf-cache` | `models--org--name/snapshots/` | `hf download --revision R --cache-dir` per revision |
| `hf-local` | `.cache/huggingface/download/*.metadata` | `hf download --revision R --local-dir`. The metadata holds only the commit, so the repo id must come from the writer, a known blob hash, or a claim |
| `ollama` | `manifests/` + `blobs/` | `ollama pull` per tag |
| `venv` | `pyvenv.cfg` | `uv venv` + install from the freeze read out of `*.dist-info` |
| `cache` | `__pycache__`, `uv-cache`, `.cache/pip`, ... | nothing: refills on use (expendable) |
| `timeshift` | `timeshift/snapshots/*/info.json` | nothing to fetch: a backup of the live SSD (redundant) |
| `weights`, `archive`, `disk-image` | extension (`.bin`/`.npz` only from 50 MB) | from its own writer's URL/HF source, or by sha256 match |
| `data` | everything else | AT RISK unless copied from a path that still exists |

A subtree holding nothing but unclassified files is one `data` unit at its top folder, so
a dataset of 10 000 class folders is one entry, not 10 000.

**Finding the source of loose weights.** A `.safetensors` copied onto the disk under a new
name is hashed (once; cached by size, mtime and inode) and matched against:

1. every HF LFS blob ever seen in the configured HF caches. HF names blobs by their sha256,
   and the index is part of the recipe, so it outlives the cache: wiping
   `~/.cache/huggingface` forgets nothing already learned. It is refreshed every 60 s
   (both caches here index in under 10 ms);
2. blobs named by the copy command itself, resolved at write time, while the cache still
   exists;
3. hourly, online: the HF tree API's `lfs.oid` (sha256) for every file of every HF repo the
   box has ever touched, for weights still unidentified.

**Risk states.**

| State | Meaning |
|---|---|
| `recreatable` | known source and a command that brings it back |
| `redundant` | a copy of something that lives elsewhere (timeshift, a still-existing SSD path) |
| `at_risk` | unique, no known source: gone if the disk dies |
| `expendable` | someone said losing it is fine (or it is a cache) |

**Asking the writer.** Once a unit has been at risk for 5 minutes (so a burst of writes
is one message), the agent that wrote it gets one hotline message listing its at-risk
units and the two ways to answer. It is told once; the unit stays in `provenance status`
until someone answers.

**Keeping sources honest.** A weekly timer runs `check-sources`: HF revision still served,
git remote still up, URL still answering, ollama manifest still there. A recreatable unit
whose source has vanished is listed in `status` with what would be lost.

## Single-disk mode

When there is no second disk, provenance can make room instead: it deletes units that
are **unused** and **recreatable**, after their recipe is saved, and can bring any of them
back. Name the directories it may prune as `areas` in `/etc/provenance.toml`. The recipe
directory can never be inside one, and `/`, `/home`, `/usr`, `/etc`, `/var`, `/boot` and
`~` are refused as too broad.

**What "unused" means.** It is measured, never guessed:

| Mount | Last use comes from | Prunable after |
|---|---|---|
| `relatime` / `strictatime` | newest atime/mtime of the unit's files (atime is accurate to about a day) | `prune_after_days` (30) without use |
| `noatime` | every `open`/`exec` the daemon sees (fanotify mount mark; works on ntfs3) | the daemon has watched the area **without a gap** for `prune_after_days`, and seen no open in that time |

On a noatime mount there is no history, so nothing there is prunable until the daemon
has watched it for the full period. A restart (< 1 h gap) or a power-off (the machine
booted after the last heartbeat) keeps the clock; any other gap, or an overflowed event
queue, starts it over. provenance's own reads never count: it opens files with
`O_NOATIME`, skips `.git/`, and puts back the atimes its `git` calls bump.

**The bar for deleting**, checked when the plan is made and again at the moment of
deletion:

- kind is one of `hf-cache`, `hf-local`, `git`, `venv`, `cache`, `ollama`, or HF-sourced
  `weights`, and risk is `recreatable` (caches: `expendable`);
- the source is re-verified online right then: the HF tree has the same sha256s; `git
  fetch` and HEAD is on a remote branch and the tree is clean; every pinned venv package
  exists on PyPI; the ollama manifest is still served;
- no process has anything inside it open, mapped, or as its cwd; it is not a hotline team's
  repo; nobody claimed it `--keep`; it was not in a plan rejected in the last 30 days.

Never pruned: AT RISK units, plain data, timeshift, URL-only downloads (URLs rot), venvs
with packages from local paths or private indexes, huggingface_hub's shared blob store.

**Review before every automatic delete.** With `auto_prune = true` the daemon makes at most
one plan per area per day and sends a concise list (path, kind, size, last use, the proof
it can be rebuilt) to `prune_approver` (hotline-80). Nothing is deleted until that plan id
is approved with `provenance prune --approve ID`; `--reject ID` drops it, and unapproved
plans expire after `plan_ttl_h` (48). The result goes back to the approver and to
`prune_report_to`. All sessions here run as one uid, so "approved by hotline-80" is a
recorded label, not a lock: it is a review step, not a security boundary.

**Soft delete.** Pruned units are renamed into `<area>/.provenance-trash/<plan>/` and only
removed after `trash_grace_h` (24); `provenance restore PATH` moves one straight back.
After the grace period, `restore` rebuilds it from the recipe as the unit's owner and
checks it against the stored listing.

**HF caches are the exception.** huggingface_hub >= 2.4 keeps a cache-wide shared blob store
(`<cache>/blobs/xx/<xet hash>`, refcounted by `.refs` manifests) that each repo's `blobs/`
symlinks point into. Deleting or renaming a `models--X` directory frees nothing there and
leaves the store's bookkeeping stale, so provenance asks huggingface_hub what deleting a
repo frees and lets huggingface_hub delete it (`scan_cache_dir().delete_revisions()`), run
as the cache owner. Those have no trash period; anything using the model re-downloads it,
or `provenance restore` does.

## Install

```bash
ownbox install provenance
```

That runs `deploy/install.sh` (as root, through sudo): a root-owned release in
`/opt/provenance/releases/<rev>` with `/opt/provenance/current` pointing at it, the
`provenance` and weekly `provenance-check` systemd units, the claims drop box
`/var/lib/provenance/spool` (mode 1733), and `/etc/provenance.toml` if there is none.
`ownbox update provenance` reruns the same script: it never touches the config or the
recipe, keeps the previous release, and restarts the daemon. `ownbox uninstall` asks
before deleting the recipe (without a terminal it keeps it).

Then name the disk in `/etc/provenance.toml`; the daemon notices the edit and restarts:

```toml
disks = ["/mnt/hdd"]
```

The disk must be its own filesystem mounted exactly there (an fstab line with `nofail`),
with kernel file-handle support for the fanotify write mark: ext4, xfs, btrfs and the
in-kernel ntfs3 driver work (tested 2026-11-11); FUSE filesystems such as ntfs-3g do not.
On a filesystem without it, an area still works but has no write attribution.

For single-disk mode, add `areas` instead (or as well) and leave `auto_prune = false` until
`provenance prune` shows a plan you agree with.

Manual install, for development:

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## Usage

```
provenance status [--json] [--strict]          what is on each disk, by risk; what would be lost
provenance ls [PATH] [--risk STATE] [--json]   units at or under PATH
provenance show PATH [--files] [--json]        one unit: source, writers, rebuild commands
provenance claim PATH --source SRC             SRC: hf:org/name[@rev], git:URL, https://URL, or text
provenance claim PATH --expendable [REASON]
provenance claim PATH --clear
provenance rebuild [PATH] [--into ROOT] [--run [--yes]]   print (or run) the restore script
provenance verify-restore ROOT [--hash]        compare a restored tree with the recipe
provenance check-sources                       are the recorded sources still online?
provenance identify                            look up unknown weights via HF repo trees (root)
provenance scan [PATH] [--dry-run]             scan now (root; --dry-run stores nothing)
provenance notify [--dry-run]                  ask writers about at-risk data now (root)
provenance export -o FILE.json.xz              the whole recipe as one file
provenance prune [AREA] [--all] [--days N]     single-disk mode: what would be deleted (dry run)
provenance prune --propose                     save that as a plan and send it for review (root)
provenance prune --approve|--reject PLAN       decide a pending plan (any user; via the spool)
provenance prune --apply [--yes]               delete now without the review step (root; asks)
provenance plans [--show PLAN]                 plans and their status
provenance pruned [--json]                     what was deleted, and where it can come back from
provenance restore PATH [--dry-run]            bring one back: from trash, else rebuild it
provenance claim PATH --keep [REASON]          never prune this
provenance watch                               the daemon (systemd runs this)
```

Reading the recipe needs no privileges. `claim` works for any user: run unprivileged, it
drops the request into the spool and the daemon applies it within a minute, recording the
file owner and the agent as who made it.

Exit codes: `0` success; `1` an operation failed, `verify-restore` found differences,
`check-sources` found a source gone, or `status --strict` found something at risk; `2`
usage error or aborted. Output meant for reading or piping goes to stdout, everything else
to stderr.

### Examples

```console
$ provenance status
recipe: /var/lib/provenance/provenance.db  320.0 KB of 50 MB budget

/mnt/provtest  uuid 20d7a810-...  mounted  last full scan 2026-11-03 22:10
  AT RISK           4 unit(s)      5.5 MB
  recreatable       5 unit(s)     50.6 MB
  redundant         1 unit(s)         0 B
  expendable        1 unit(s)         2 B

  would be lost if this disk died:
       3.0 MB  projects/spoon  [git] 2 commit(s) on no remote branch; ignored ckpt/step1000.safetensors [weights]: ...  (asked manager-8)
       2.0 MB  scratch-train/my_lora.safetensors  [weights] weights with no known source ...

$ provenance ls /mnt/provtest/loose
recreatable    38.5 MB  weights    loose/mystery.safetensors  identical (sha256) to jaredpalmer/kev-0.6b:adapter_model.safetensors

$ provenance claim /mnt/hdd/scratch-train/my_lora.safetensors --source 'trained by train.py on data/x, 2 h on the 4060'
$ provenance rebuild --into /mnt/newdisk > restore.sh && bash restore.sh
$ provenance verify-restore /mnt/newdisk --hash
checked 12 file(s): 0 missing, 0 differ; 0 unit(s) had no listing
```

## Measured

On this box (Ryzen 5 5600, ext4 loop image on NVMe, warm page cache), 2026-11-03:

| What | Number |
|---|---|
| Compressed listing, dataset-shaped names (`train/nXXXXXXXX/img_NNNNNNNN.JPEG` + size) | 7.7 B/file |
| Compressed listing, messy user data (varied names, depths, sizes) | 11.4 B/file |
| So a 50 MB recipe keeps full listings for | ~4-6 million files |
| Packing a 1 000 000-file listing (xz -9e) | 18 s |
| Full scan, 150 000 empty files in 300 folders | 2.5 s |
| Indexing both HF caches (43 blobs) | < 10 ms |
| End-to-end restore of the test disk into an empty dir | 12/12 files back, 0 missing, 0 differ (sha256) |

Git repos, venvs and timeshift snapshots store no per-file listing (the source knows the
files), which is most of why the budget holds.

## Limitations

**Attribution races short processes.** The writer's `/proc` entry is read when the event
is processed. A process that writes and exits within a millisecond (`mv`, `touch`, a quick
`cp` of a small file) may be gone by then, and that write has no writer. Long-running
fetchers (`hf download`, `git clone`, `wget`) are reliably caught. What is missed falls back
to classification by content, and to AT RISK if that does not identify it.

**Writes while the daemon is down have no writer.** The daily full scan (and the scan on
start) still classifies them by content: repos, HF caches, venvs and hash-matched weights
come out right. An `hf download --local-dir` folder made while the daemon was down, from a
repo whose blobs were never seen, is AT RISK until claimed.

**Some weights cannot be traced.** Weights from an HF repo this box has never touched in
any way, copied in by a command that does not name it, are not found by any of the three
lookups. They are AT RISK and the writer is asked, which is the intended outcome.

**"Pushed" means what the local remote-tracking refs say.** A repo counts as fully pushed
if its commits are on `refs/remotes/*`. A ref that was deleted on the server is caught by the
weekly `check-sources`, not at scan time.

**Venv rebuilds assume PyPI.** The freeze comes from `*.dist-info`. Packages installed from
a local path are listed in the unit's detail and need that path to rebuild; packages from
a private index need that index.

**Rebuilt is not identical for git.** A clone restores tracked files at the recorded commit.
Ignored content is restored only where it was itself recognised (a venv, HF weights, a
lockfile-backed `node_modules`); the rest is reported as lost, with a byte count.

**HF repos can change.** Rebuilds pin the recorded revision, which HF keeps serving unless
the repo is deleted or the commit is purged; `check-sources` reports that.

**Listings are dropped, not stored partially, when over budget.** Order: expendable, then
redundant, then recreatable, then at-risk units, largest first. A unit without a listing can
still be rebuilt; `verify-restore` just cannot check it file by file.

**One filesystem per watched path, Linux 5.9+, root.** fanotify needs `CAP_SYS_ADMIN`.
Subdirectories on other filesystems (bind mounts, nested mounts) are skipped by the walk.

**Single-disk mode trusts the source to stay up.** A pruned unit comes back only if its
source still serves it: an HF repo can be deleted, a GitHub repo made private, a PyPI
release yanked. Verification at delete time says the source was there then, not that it
will be. The weekly `check-sources` covers the live recipe, not pruned units.

**Restores cost bandwidth and time.** Rebuilding a 19 GB model means downloading 19 GB.
`prune_after_days` and the review step are the only things standing between a model you
use monthly and a monthly re-download.

**Restored venvs are equivalent, not identical.** The freeze pins versions, not builds or
hashes, and `uv` may pick a different Python patch release.

**Uses the open-tracker only on noatime mounts.** On relatime mounts last use comes from
atime alone, so a file that is only `stat`ed or listed (never read) does not count as used.
A Windows boot that reads files on a shared NTFS partition is not seen by a Linux daemon;
the power-off rule above treats such a gap as harmless.

**Not a backup tool.** It never copies anything. AT RISK data stays at risk until a person
or agent backs it up, gives it a source, or calls it expendable.
