#!/bin/bash
# restart_ingest.sh - restart the host ingest service so it runs the current
# src/ingest.py, then show that it came back and is writing its heartbeat.
#
# Run it as YOURSELF; it asks for sudo for the one step that needs it:
#
#     ./scripts/restart_ingest.sh
#
# ingest.service runs straight from this checkout (deploy/ingest.service), so
# a code change reaches it only on restart. Needed after the change that
# writes a <clip>.mp4.source sidecar - until then, new clips reach the queue
# without their SentryClips/SavedClips tag and are queued as recent footage.
set -euo pipefail

HEARTBEAT=/mnt/jetsondata/tesla-alerts/logs/ingest.json

die() { echo "restart_ingest: $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] || die "run this as yourself, without sudo; it asks for sudo when it needs it"
systemctl cat ingest.service >/dev/null 2>&1 || die "ingest.service is not installed (sudo ./scripts/install_ingest.sh)"

before=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)
sudo systemctl restart ingest.service
echo "restarted; waiting for a fresh heartbeat..."

for _ in $(seq 1 20); do
  sleep 1
  now=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)
  if [[ $now -gt $before ]]; then
    systemctl is-active ingest.service
    python3 -c "
import json; b = json.load(open('$HEARTBEAT'))
print('state:', b.get('state'), '-', b.get('reason'))"
    echo "ok - ingest is running the current code"
    exit 0
  fi
done
systemctl --no-pager --lines=10 status ingest.service || true
die "no fresh heartbeat after 20 s; see the status above"
