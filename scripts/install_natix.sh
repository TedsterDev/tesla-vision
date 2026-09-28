#!/bin/bash
# install_natix.sh - install the NATIX VX360 mirror as a system service.
#
# Run once, as root:
#     sudo ./scripts/install_natix.sh
#
# What it does, in order, all of it reversible with --uninstall:
#   1. creates the mountpoint /mnt/natixv360
#   2. installs deploy/natix-mirror.service into /etc/systemd/system
#   3. installs deploy/99-natix-vx360.rules into /etc/udev/rules.d
#   4. identifies the stick and refuses to go further if it cannot
#   5. runs the FIRST FULL MIRROR in the foreground, so you watch it happen
#   6. verifies independently that the clips really landed, byte-sampled
#   7. only then enables the service for continuous mirroring
#
# Steps 4-6 are why this is not just three `install` calls. A background
# service that fails to mount looks exactly like a background service with
# nothing to do, and the difference would not surface until you went looking
# for footage that was never uploaded. Any failure in 4-6 aborts without
# enabling the service, so there is no half-installed state that quietly
# retries a broken configuration forever.
#
# It does NOT touch the USB gadget, the containers, or the database schema -
# the worker creates its own tables on first connect.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$REPO_ROOT/deploy/natix-mirror.service"
UNIT_DST="/etc/systemd/system/natix-mirror.service"
RULES_SRC="$REPO_ROOT/deploy/99-natix-vx360.rules"
RULES_DST="/etc/udev/rules.d/99-natix-vx360.rules"
MOUNTPOINT="${NATIX_MOUNTPOINT:-/mnt/natixv360}"

[[ $EUID -eq 0 ]] || { echo "install_natix: must run as root (sudo $0)" >&2; exit 1; }

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl disable --now natix-mirror.service 2>/dev/null || true
  rm -f "$UNIT_DST" "$RULES_DST"
  systemctl daemon-reload
  udevadm control --reload-rules || true
  findmnt -n "$MOUNTPOINT" >/dev/null 2>&1 && umount "$MOUNTPOINT" || true
  echo "natix-mirror removed. $MOUNTPOINT left in place (empty)."
  exit 0
fi

echo "==> mountpoint $MOUNTPOINT"
mkdir -p "$MOUNTPOINT"

echo "==> systemd unit"
# The unit hardcodes this checkout's path; rewrite it if the repo moved.
sed "s|/home/tedster/projects/tesla-alerts|$REPO_ROOT|g" "$UNIT_SRC" > "$UNIT_DST"
chmod 644 "$UNIT_DST"

# Everything below runs before the service is enabled, deliberately. A first
# mirror is several gigabytes over USB 2.0, and watching it happen in the
# foreground - then verifying it independently - is worth far more than
# discovering in the journal an hour later that it silently failed to mount.
export BASE_DIR="${BASE_DIR:-/mnt/jetsondata/tesla-alerts}"
export PYTHONPATH="$REPO_ROOT"
# Written as an `if` rather than `[[ -f x ]] && . x` on purpose: under
# `set -e` that one-liner *exits the installer* when .env is absent, because a
# failed && list is a failed command. Silently aborting an install because an
# optional file is missing is a miserable way to spend an afternoon.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; . "$REPO_ROOT/.env"; set +a
fi

echo
echo "==> identifying the stick"
if ! python3 "$REPO_ROOT/scripts/natix_probe.py"; then
  echo
  echo "No usable VX360 found. The service is NOT enabled." >&2
  echo "Plug the stick into a USB-A port and re-run this script." >&2
  exit 1
fi

echo
echo "==> first mirror pass (this copies the whole archive; minutes, not seconds)"
# --limit 0 means "no cap": get the entire archive across in one go rather than
# NATIX_BATCH clips at a time, so the verification below is meaningful.
if ! python3 -u "$REPO_ROOT/src/natix_worker.py" --once --limit 0; then
  echo
  echo "First mirror pass failed. The service is NOT enabled, so nothing will" >&2
  echo "retry in the background until you have looked at the error above." >&2
  exit 1
fi

echo
echo "==> verifying what actually landed on the stick"
if ! python3 -u "$REPO_ROOT/scripts/natix_verify.py" --sample 5; then
  echo
  echo "Verification failed. The service is NOT enabled - fix the above first." >&2
  echo "Re-run just the check with:" >&2
  echo "  sudo BASE_DIR=$BASE_DIR python3 $REPO_ROOT/scripts/natix_verify.py" >&2
  exit 1
fi

echo
echo "==> udev rule (installed only now, deliberately)"
# This rule restarts natix-mirror on hotplug. Installing it earlier and running
# `udevadm trigger` would have fired it against the stick that is already
# plugged in, starting the service while the foreground first pass was halfway
# through mounting the same device. Two writers, one exFAT volume, no locking
# between them - so the rule goes in only once the foreground work is done.
install -m 644 "$RULES_SRC" "$RULES_DST"
udevadm control --reload-rules

echo
echo "==> enabling the service for continuous mirroring"
systemctl daemon-reload
systemctl enable natix-mirror.service
systemctl restart natix-mirror.service
sleep 3
systemctl --no-pager --lines=10 status natix-mirror.service || true

cat <<SUMMARY

Done. The archive is on the stick and the service will keep it current.

  journalctl -u natix-mirror -f                       watch it work
  sudo systemctl stop natix-mirror                    pause mirroring
  sudo BASE_DIR=$BASE_DIR \
       python3 $REPO_ROOT/src/natix_worker.py --status    one-shot report
  sudo BASE_DIR=$BASE_DIR \
       python3 $REPO_ROOT/scripts/natix_verify.py         re-verify the stick
  sudo $0 --uninstall                                 undo everything

Dashboard: the NATIX tab at http://localhost:8080/natix

One thing this script cannot check: whether the VX360's own firmware notices
files that appeared while it was exporting its flash to us, or whether it only
rescans after a physical replug. We unmount and flush after every pass, which
is the strongest hint we can give it. Confirm in the NATIX app that the clips
queue for upload - if they do not, unplug and replug the stick once and check
again, and that tells us a periodic detach is needed rather than an unmount.
SUMMARY
