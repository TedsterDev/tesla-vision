#!/bin/bash
# install_ingest.sh - install the TeslaCam ingest service.
#
#     sudo ./scripts/install_ingest.sh
#     sudo ./scripts/install_ingest.sh --uninstall
#
# Installs deploy/ingest.service, enables and starts it, then verifies three
# things separately and says which failed - the same discipline as
# install_gps.sh, because "the inbox is empty" is what a healthy pipeline with
# no new footage, a broken ingest, and an uninstalled ingest all look like.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$REPO_ROOT/deploy/ingest.service"
UNIT_DST="/etc/systemd/system/ingest.service"

[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0" >&2; exit 1; }

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl disable --now ingest.service 2>/dev/null || true
  rm -f "$UNIT_DST"
  systemctl daemon-reload
  echo "ingest.service removed. The inbox and heartbeat were left in place."
  exit 0
fi

echo "==> systemd unit"
install -m 644 "$UNIT_SRC" "$UNIT_DST"

# systemd applies EnvironmentFile= AFTER Environment=, so a BASE_DIR line in
# .env would win over the unit. Compute the effective value the way systemd
# will, rather than trusting the unit alone.
BASE_DIR_FROM_ENV=""
if [[ -f "$REPO_ROOT/.env" ]]; then
  BASE_DIR_FROM_ENV="$(sed -n 's/^BASE_DIR=//p' "$REPO_ROOT/.env" | tail -1)"
fi
BASE_DIR_FROM_UNIT="$(sed -n 's/^Environment=BASE_DIR=//p' "$UNIT_DST" | tail -1)"
SERVICE_BASE_DIR="${BASE_DIR_FROM_ENV:-$BASE_DIR_FROM_UNIT}"
if [[ -n "$BASE_DIR_FROM_ENV" && "$BASE_DIR_FROM_ENV" != "$BASE_DIR_FROM_UNIT" ]]; then
  echo "    NOTE .env sets BASE_DIR=$BASE_DIR_FROM_ENV, which overrides the unit's $BASE_DIR_FROM_UNIT"
fi

echo "==> state directories under $SERVICE_BASE_DIR"
install -d -m 755 "$SERVICE_BASE_DIR/inbox" "$SERVICE_BASE_DIR/logs"

echo "==> enabling and starting"
systemctl daemon-reload
systemctl enable --now ingest.service

sleep 3
echo
echo "==> checking the service"
if systemctl is-active --quiet ingest.service; then
  echo "    OK   ingest.service is active"
else
  echo "    FAIL ingest.service is not active - journalctl -u ingest -n 50" >&2
  exit 1
fi

HEARTBEAT="$SERVICE_BASE_DIR/logs/ingest.json"
echo "==> checking the heartbeat ($HEARTBEAT)"
for _ in $(seq 1 10); do [[ -s "$HEARTBEAT" ]] && break; sleep 1; done
if [[ -s "$HEARTBEAT" ]]; then
  echo "    OK   heartbeat written"
  python3 - "$HEARTBEAT" <<'PY'
import json, sys
beat = json.load(open(sys.argv[1]))
for key in ("state", "reason", "copied_total", "last_error"):
    print(f"         {key:14} {beat.get(key)}")
PY
else
  echo "    FAIL no heartbeat after 10s - journalctl -u ingest -n 50" >&2
  exit 1
fi

echo
echo "==> checking the source volume"
if findmnt -no SOURCE /mnt/teslacam >/dev/null 2>&1; then
  echo "    OK   /mnt/teslacam is mounted"
else
  echo "    INFO /mnt/teslacam is not mounted. The service idles until it is."
  echo "         That is the expected state while the volume is offered to the car;"
  echo "         see README 'Feeding the inbox' for the export/mount cycle."
fi

echo
echo "Done. Ingest runs on boot and copies stable clips into $SERVICE_BASE_DIR/inbox"
echo "whenever /mnt/teslacam is mounted and NOT exported to the car."
