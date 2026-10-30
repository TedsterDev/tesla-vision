#!/bin/bash
# install_carstore.sh - install the car storage service (src/carstore.py).
#
#     sudo ./scripts/install_carstore.sh
#     sudo ./scripts/install_carstore.sh --uninstall
#
# What it does, in order, saying what it changed at each step:
#   1. installs btrfs-progs if mkfs.btrfs is missing (needs internet once)
#   2. loads btrfs now and at every boot (/etc/modules-load.d/carstore.conf)
#   3. makes the old /mnt/teslacam and /mnt/jetsondata fstab entries `noauto`.
#      This MUST happen before the old TESLACAM drive is plugged back in: its
#      jetsondata partition would otherwise be mounted over
#      /mnt/jetsondata at the next boot, hiding the live data on the card
#      (scout.db, media/) behind whatever old data that partition holds - and
#      the containers would start on it. /etc/fstab is backed up first.
#   4. installs and starts deploy/carstore.service
#   5. checks the service is active and its heartbeat is being written
#
# After this, the dashboard's Storage page lists USB drives and can format one
# as the car's drive. Nothing is formatted by this script.
#
# --uninstall stops the service, stops offering the car's image over USB and
# removes the unit. Drives, images and fstab are left as they are.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$REPO_ROOT/deploy/carstore.service"
UNIT_DST="/etc/systemd/system/carstore.service"
GADGET=/sys/kernel/config/usb_gadget/g1

[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0" >&2; exit 1; }

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl disable --now carstore.service 2>/dev/null || true
  # The service is gone; nothing would keep the export consistent.
  if [[ -f "$GADGET/UDC" && -n "$(cat "$GADGET/UDC")" ]]; then
    echo "" > "$GADGET/UDC"
    echo "stopped offering the car's image over USB"
  fi
  rm -f "$UNIT_DST"
  systemctl daemon-reload
  echo "carstore.service removed. Drives, the car's image and /etc/fstab were left as they are."
  exit 0
fi

echo "==> 1. tools"
if ! command -v mkfs.btrfs >/dev/null; then
  echo "    installing btrfs-progs"
  apt-get install -y btrfs-progs
fi
for tool in mkfs.btrfs mkfs.vfat fsck.vfat e2fsck losetup sfdisk wipefs fallocate blkid findmnt lsblk; do
  command -v "$tool" >/dev/null || { echo "    FAIL $tool is missing" >&2; exit 1; }
done
echo "    OK   all tools present"

echo "==> 2. btrfs kernel module"
modprobe btrfs
echo btrfs > /etc/modules-load.d/carstore.conf
grep -qw btrfs /proc/filesystems && echo "    OK   btrfs loaded, and loaded at boot"

echo "==> 3. fstab: old TESLACAM / jetsondata entries -> noauto"
needs_change=$(awk '!/^[[:space:]]*#/ && ($2 == "/mnt/teslacam" || $2 == "/mnt/jetsondata") && $4 !~ /(^|,)noauto(,|$)/' /etc/fstab)
if [[ -n "$needs_change" ]]; then
  backup="/etc/fstab.carstore-$(date +%Y%m%d-%H%M%S)"
  cp -p /etc/fstab "$backup"
  awk 'BEGIN { OFS = "\t" }
       !/^[[:space:]]*#/ && ($2 == "/mnt/teslacam" || $2 == "/mnt/jetsondata") && $4 !~ /(^|,)noauto(,|$)/ {
         $4 = $4 ",noauto"
       } { print }' "$backup" > /etc/fstab
  systemctl daemon-reload
  echo "    changed (backup: $backup):"
  diff "$backup" /etc/fstab | sed 's/^/      /' || true
else
  echo "    OK   nothing to change"
fi
if findmnt -n /mnt/jetsondata >/dev/null; then
  echo "    WARN /mnt/jetsondata is mounted from $(findmnt -no SOURCE /mnt/jetsondata) right now -"
  echo "         the containers may be using that drive, not the card. Check before going on."
fi

echo "==> 4. systemd unit"
install -m 644 "$UNIT_SRC" "$UNIT_DST"
systemctl daemon-reload
systemctl enable --now carstore.service

BASE_DIR_FROM_ENV=""
if [[ -f "$REPO_ROOT/.env" ]]; then
  BASE_DIR_FROM_ENV="$(sed -n 's/^BASE_DIR=//p' "$REPO_ROOT/.env" | tail -1)"
fi
SERVICE_BASE_DIR="${BASE_DIR_FROM_ENV:-$(sed -n 's/^Environment=BASE_DIR=//p' "$UNIT_DST" | tail -1)}"

echo "==> 5. checking the service"
sleep 5
if systemctl is-active --quiet carstore.service; then
  echo "    OK   carstore.service is active"
else
  echo "    FAIL carstore.service is not active - journalctl -u carstore -n 50" >&2
  exit 1
fi
HEARTBEAT="$SERVICE_BASE_DIR/logs/carstore.json"
for _ in $(seq 1 15); do [[ -s "$HEARTBEAT" ]] && break; sleep 1; done
if [[ -s "$HEARTBEAT" ]]; then
  python3 - "$HEARTBEAT" <<'PY'
import json, sys
beat = json.load(open(sys.argv[1]))
print(f"    OK   heartbeat written: state={beat.get('state')}")
print(f"         {beat.get('reason') or ''}")
for drive in beat.get("drives", []):
    print(f"         drive {drive['path']}: {drive['model']}, {drive['size_bytes'] / 1e9:.0f} GB, "
          f"{drive['role']}{' (not offered: ' + drive['blocked'] + ')' if drive.get('blocked') else ''}")
PY
else
  echo "    FAIL no heartbeat at $HEARTBEAT - journalctl -u carstore -n 50" >&2
  exit 1
fi
echo
echo "Done. Open the dashboard's Storage page to format a drive as the car's drive."
