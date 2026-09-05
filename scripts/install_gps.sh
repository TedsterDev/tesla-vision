#!/bin/bash
# install_gps.sh - install the GNSS location reader as a system service.
#
# Run once, as root:
#     sudo ./scripts/install_gps.sh
#
# What it does, in order, all of it reversible with --uninstall:
#   1. installs deploy/gps.service into /etc/systemd/system
#   2. creates BASE_DIR/logs, because the unit's ReadWritePaths= must exist
#   3. daemon-reload, enable, restart
#   4. VERIFIES three separate things and says which one failed:
#        - the receiver is actually attached
#        - systemd says the service is active
#        - the service has written a fresh heartbeat
#
# Step 4 is the whole reason this is not three `systemctl` calls typed by hand.
# A GPS service with no receiver, a GPS service that crashed on import, and a
# GPS service quietly working in a garage all produce exactly the same thing in
# the polls table: nothing. Checking them separately, right now, while someone
# is standing at the machine, is far cheaper than deducing it from an empty
# table next week.
#
# Unlike install_natix.sh, a failed check here does NOT leave the service
# disabled. An unplugged receiver is a recoverable state the service is built
# to sit in - it reopens the by-id path on its own within RECONNECT_DELAY_SECONDS
# - so refusing to install over it would trade a self-healing system for one
# that needs a human every time a connector vibrates loose.
#
# It does NOT write to .env, does not touch the database, and does not touch
# the containers. gps.py is stdlib-only and creates its own tables on connect.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$REPO_ROOT/deploy/gps.service"
UNIT_DST="/etc/systemd/system/gps.service"
DEFAULT_DEVICE="/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00"

[[ $EUID -eq 0 ]] || { echo "install_gps: must run as root (sudo $0)" >&2; exit 1; }

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl disable --now gps.service 2>/dev/null || true
  rm -f "$UNIT_DST"
  systemctl daemon-reload
  echo "gps.service removed. Its logs and heartbeat file were left in place."
  exit 0
fi

# Written as an `if` rather than `[[ -f x ]] && . x` on purpose: under `set -e`
# that one-liner *exits the installer* when .env is absent, because a failed &&
# list is a failed command. Silently aborting an install because an optional
# file is missing is a miserable way to spend an afternoon.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; . "$REPO_ROOT/.env"; set +a
fi
GPS_DEVICE="${GPS_DEVICE:-$DEFAULT_DEVICE}"

echo "==> systemd unit"
# The unit hardcodes this checkout's path; rewrite it if the repo moved.
sed "s|/home/tedster/projects/tesla-alerts|$REPO_ROOT|g" "$UNIT_SRC" > "$UNIT_DST"
chmod 644 "$UNIT_DST"

# Read BASE_DIR back out of the INSTALLED unit rather than from .env. systemd
# applies EnvironmentFile= and Environment= in file order and the unit sets
# BASE_DIR after loading .env, so the unit's value is the one that wins - and
# it is where the heartbeat will actually land. Guessing from .env here would
# make this script check the wrong directory and report a healthy service as
# broken.
SERVICE_BASE_DIR="$(sed -n 's/^Environment=BASE_DIR=//p' "$UNIT_DST" | tail -1)"
SERVICE_BASE_DIR="${SERVICE_BASE_DIR:-/mnt/jetsondata/tesla-alerts}"
HEARTBEAT="$SERVICE_BASE_DIR/logs/gps.json"

echo "==> state directory $SERVICE_BASE_DIR/logs"
# ProtectSystem=strict with ReadWritePaths= refuses to start the unit if the
# path does not exist, and the resulting failure names systemd, not the missing
# directory. mkdir -p is idempotent and creates nothing that was not going to
# be created on the first heartbeat anyway.
mkdir -p "$SERVICE_BASE_DIR/logs"

echo "==> enabling and starting"
systemctl daemon-reload
systemctl enable gps.service
systemctl restart gps.service

echo
echo "==> checking the receiver"
if [[ -e "$GPS_DEVICE" ]]; then
  echo "    OK   $GPS_DEVICE"
else
  echo "    WARN $GPS_DEVICE is not present."
  echo "         The service is installed and running anyway - it reopens that"
  echo "         path by itself within a few seconds of the receiver appearing,"
  echo "         so plugging it in is the whole fix. Compare against:"
  echo "           ls -l /dev/serial/by-id/"
fi

echo
echo "==> checking the service"
if systemctl is-active --quiet gps.service; then
  echo "    OK   gps.service is active"
else
  echo "    FAIL gps.service is not active." >&2
  systemctl --no-pager --lines=20 status gps.service || true
  exit 1
fi

echo
echo "==> waiting for the first heartbeat ($HEARTBEAT)"
# Up to 30s. The service writes one before it reads a single sentence, so this
# is bounded by process startup and not by whether there is a fix - which is
# the point: this check must pass indoors, with no receiver, in a garage.
heartbeat_seen=0
for _ in $(seq 1 30); do
  if [[ -s "$HEARTBEAT" ]]; then heartbeat_seen=1; break; fi
  sleep 1
done

if [[ $heartbeat_seen -eq 1 ]]; then
  echo "    OK   heartbeat written"
  python3 - "$HEARTBEAT" <<'SUMMARISE' || true
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

# Deliberately no coordinates: this file has none, and printing any would put
# the car's position into an installer's scrollback and anyone's paste of it.
for field in ("schema", "pid", "device_present", "port_open",
              "sentences_total", "fix_valid", "satellites_used",
              "satellites_in_view", "last_error"):
    print(f"         {field:<20} {payload.get(field)!r}")
SUMMARISE
else
  echo "    FAIL no heartbeat after 30s." >&2
  echo "         The service is running but has not written $HEARTBEAT." >&2
  echo "         Most likely BASE_DIR in the unit and ReadWritePaths= disagree," >&2
  echo "         or the directory is not writable. Check:" >&2
  echo "           journalctl -u gps -n 50" >&2
  echo "           systemd-analyze security gps.service" >&2
  exit 1
fi

cat <<SUMMARY

Done. Location logging is running and will restart on boot.

NEXT STEP: open the GPS page and confirm what the receiver can actually see.

  http://100.98.227.42:8080/gps        on the tailnet
  http://localhost:8080/gps            on this machine

Indoors it should read "receiver healthy and searching" with satellites in
view and none used. That is the correct answer, not a fault - a fix needs sky.
If it reads 0 in view as well, the antenna or its cable is the problem, and
that distinction is the entire reason the page exists.

  journalctl -u gps -f                              watch it work
  sudo systemctl stop gps                           pause logging
  sudo python3 $REPO_ROOT/src/gps.py --status       one-shot report, writes nothing
  sudo python3 $REPO_ROOT/src/gps.py --raw          raw NMEA, writes nothing
  sudo $0 --uninstall                               undo everything

Both diagnostic modes are safe to run while the service is up: neither opens
the database and neither writes the heartbeat file.
SUMMARY
