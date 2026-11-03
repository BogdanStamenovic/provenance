#!/usr/bin/env bash
# Install or upgrade provenance as a root service. Idempotent: safe to rerun, and it is
# both ownbox's setup and update step.
#
# Keeps: /etc/provenance.toml (the config) and /var/lib/provenance (the recipe).
# Replaces: /opt/provenance/current (code; the previous release is kept) and the units.
#
# The daemon runs as root, so it runs a root-owned copy in /opt rather than the ownbox
# checkout: a user-writable checkout would let any process of that user edit code root runs.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "install.sh: run as root (sudo)" >&2; exit 1; }
SRC="$(cd "$(dirname "$0")/.." && pwd)"
OPT=/opt/provenance
CONF=/etc/provenance.toml
STATE=/var/lib/provenance

PY=$(command -v python3)
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || { echo "install.sh: needs Python >= 3.11 ($PY is older)" >&2; exit 1; }

# One directory per release, `current` points at the live one. venvs hardcode their own
# path, so each is built where it stays; the switch is an atomic symlink rename, and a
# failed build never touches the running version.
REV=$(git -C "$SRC" rev-parse --short=12 HEAD 2>/dev/null || date +%s)
[ -z "$(git -C "$SRC" status --porcelain 2>/dev/null)" ] || REV="$REV-dirty-$(date +%s)"
REL="$OPT/releases/$REV"
if [ ! -x "$REL/.venv/bin/provenance" ]; then
  rm -rf "$REL"
  mkdir -p "$REL"
  "$PY" -m venv "$REL/.venv"
  "$REL/.venv/bin/pip" install --quiet --disable-pip-version-check "$SRC"
  "$REL/.venv/bin/provenance" --version >/dev/null
fi
ln -sfn "$REL" "$OPT/current.tmp"
mv -T "$OPT/current.tmp" "$OPT/current"
# Keep the live release and the one before it; drop older ones.
ls -1dt "$OPT"/releases/* | tail -n +3 | xargs -r rm -rf

if [ ! -e "$CONF" ]; then
  install -m 0644 "$SRC/deploy/provenance.toml.example" "$CONF"
  echo "install.sh: wrote $CONF -- add the HDD mount point to 'disks'" >&2
else
  echo "install.sh: kept existing $CONF" >&2
fi

mkdir -p "$STATE" "$STATE/spool"
chmod 0755 "$STATE"
chmod 1733 "$STATE/spool"   # drop box: anyone may leave a claim, nobody may read others'

install -m 0644 "$SRC/deploy/provenance.service" /etc/systemd/system/provenance.service
install -m 0644 "$SRC/deploy/provenance-check.service" /etc/systemd/system/provenance-check.service
install -m 0644 "$SRC/deploy/provenance-check.timer" /etc/systemd/system/provenance-check.timer
systemctl daemon-reload
systemctl enable --quiet provenance.service provenance-check.timer
systemctl restart provenance.service
systemctl start provenance-check.timer
sleep 2
systemctl is-active --quiet provenance.service \
  || { journalctl -u provenance -n 30 --no-pager >&2; echo "install.sh: service failed" >&2; exit 1; }
echo "install.sh: provenance $("$OPT/current/.venv/bin/provenance" --version | cut -d' ' -f2) ($REV) running" >&2
