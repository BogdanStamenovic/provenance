#!/usr/bin/env bash
# Undo install.sh. The recipe (/var/lib/provenance) and config are user data: removed only
# with --purge, or when asked at a terminal. Without a terminal they are kept.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "uninstall.sh: run as root (sudo)" >&2; exit 1; }
PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1
systemctl disable --now --quiet provenance.service provenance-check.timer 2>/dev/null || true
rm -f /etc/systemd/system/provenance.service /etc/systemd/system/provenance-check.service \
      /etc/systemd/system/provenance-check.timer
systemctl daemon-reload
rm -rf /opt/provenance
if [ "$PURGE" = 0 ] && [ -t 0 ] && [ -e /var/lib/provenance ]; then
  printf 'Delete the recipe in /var/lib/provenance and /etc/provenance.toml? This is the only record of how to rebuild the watched disks. [y/N] ' >&2
  read -r a
  [ "$a" = y ] || [ "$a" = Y ] && PURGE=1
fi
if [ "$PURGE" = 1 ]; then
  rm -rf /var/lib/provenance /etc/provenance.toml
  echo "uninstall.sh: removed the recipe and config" >&2
else
  echo "uninstall.sh: kept /var/lib/provenance and /etc/provenance.toml (rerun with --purge to delete)" >&2
fi
