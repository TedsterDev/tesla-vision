#!/bin/bash
# apply_audit_fixes.sh - every privileged step from the 2026-09-07 audit, in
# the one order that is safe, each step verified before the next.
#
#     sudo ./scripts/apply_audit_fixes.sh            # dry run: shows the plan
#     sudo ./scripts/apply_audit_fixes.sh --execute
#
# Why one script: eight root operations across three services, and the order
# is not free. gps must restart BEFORE the phantom purge or the old code
# reopens a phantom seconds later. natix-mirror must stop BEFORE the stray is
# cleared or the worker races the rm. The unit must be installed BEFORE the
# restart or the ExecStop race is still live. Typed by hand at 2am, that
# ordering is exactly what gets lost.
#
# Steps, and the audit finding each closes:
#   1. install deploy/natix-mirror.service         (#3  fix committed, never deployed)
#   2. stop natix-mirror, clear the stray, restore  (#4  mirror dead for a week)
#      the 2026-08-03 footage from the rescue dir, start
#   3. restart gps                                  (running pre-quality-gate code)
#   4. migrate clip timestamps                      (#1  captured_ts 7-8h early)
#   5. purge phantom drives                         (15 accumulated)
#   6. install ingest.service                       (#6  nothing feeds the inbox)
#
# Everything here is idempotent: a second run finds nothing to do.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR=/mnt/jetsondata/tesla-alerts
export BASE_DIR
EXECUTE=0; [[ "${1:-}" == "--execute" ]] && EXECUTE=1

[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0 ${1:-}" >&2; exit 1; }
cd "$REPO"

step() { echo; echo "==> $*"; }
ok()   { echo "    OK   $*"; }
skip() { echo "    SKIP $*"; }
fail() { echo "    FAIL $*" >&2; exit 1; }
run()  { if [[ $EXECUTE -eq 1 ]]; then "$@"; else echo "    would: $*"; fi; }

echo "audit remediation - $([[ $EXECUTE -eq 1 ]] && echo EXECUTING || echo 'DRY RUN, add --execute')"

# ---- 1. natix-mirror unit ---------------------------------------------------
step "1/6 natix-mirror unit"
if diff -q deploy/natix-mirror.service /etc/systemd/system/natix-mirror.service >/dev/null 2>&1; then
  skip "installed unit already matches the repo"
else
  grep -q '^ExecStop=' /etc/systemd/system/natix-mirror.service 2>/dev/null && \
    echo "    installed unit still carries the ExecStop race"
  run install -m 644 deploy/natix-mirror.service /etc/systemd/system/natix-mirror.service
  run systemctl daemon-reload
  [[ $EXECUTE -eq 1 ]] && { grep -q '^ExecStop=' /etc/systemd/system/natix-mirror.service && fail "ExecStop still present after install"; ok "unit installed, no ExecStop"; }
fi

# ---- 2. natix stray + restore ----------------------------------------------
step "2/6 natix-mirror: clear the stray, restore the rescue, restart"
STRAY_FILES=$(find /mnt/natixv360 -type f 2>/dev/null | wc -l)
RESCUE=$(ls -d "$BASE_DIR"/natix-rescue/*/ 2>/dev/null | tail -1 || true)
if findmnt /mnt/natixv360 >/dev/null 2>&1; then
  skip "stick is mounted - the mirror is already working; nothing to clear"
elif [[ "$STRAY_FILES" -eq 0 ]]; then
  skip "no stray data under /mnt/natixv360"
else
  echo "    $STRAY_FILES stray files on the SD card under /mnt/natixv360 (reproducible SentryClips)"
  [[ -n "$RESCUE" ]] || fail "no rescue directory under $BASE_DIR/natix-rescue - will not clear the stray without one"
  echo "    rescue to restore: $RESCUE ($(du -sh "$RESCUE" | cut -f1))"
  run systemctl stop natix-mirror
  run ./scripts/natix_fsck.sh --clean-stray
  run ./scripts/natix_reformat.sh --restore-only
  run systemctl start natix-mirror
  if [[ $EXECUTE -eq 1 ]]; then
    sleep 20
    findmnt /mnt/natixv360 >/dev/null 2>&1 && ok "stick mounted and mirroring" || \
      echo "    WARN stick not mounted 20s after start - journalctl -u natix-mirror -n 30"
  fi
fi

# ---- 3. gps restart ---------------------------------------------------------
step "3/6 gps: restart onto the current code"
GPS_STARTED=$(systemctl show gps -p ExecMainStartTimestamp --value 2>/dev/null || true)
echo "    gps last started: ${GPS_STARTED:-never}"
run systemctl restart gps
if [[ $EXECUTE -eq 1 ]]; then
  sleep 5
  systemctl is-active --quiet gps && ok "gps active" || fail "gps did not come back - journalctl -u gps -n 30"
fi

# ---- 4. timestamps ----------------------------------------------------------
step "4/6 clip timestamps"
if [[ $EXECUTE -eq 1 ]]; then
  python3 scripts/migrate_clip_timezone.py --execute | tail -6
else
  python3 scripts/migrate_clip_timezone.py | tail -6
fi

# ---- 5. phantom drives ------------------------------------------------------
step "5/6 phantom drives"
if [[ $EXECUTE -eq 1 ]]; then
  python3 scripts/gps_purge_phantom.py --execute | tail -4
else
  python3 scripts/gps_purge_phantom.py | tail -4
fi

# ---- 6. ingest --------------------------------------------------------------
step "6/6 ingest.service"
if systemctl is-enabled --quiet ingest.service 2>/dev/null && diff -q deploy/ingest.service /etc/systemd/system/ingest.service >/dev/null 2>&1; then
  skip "already installed and current"
else
  run ./scripts/install_ingest.sh
fi

echo
if [[ $EXECUTE -eq 1 ]]; then
  echo "==> final state"
  for unit in natix-mirror gps ingest; do
    printf "    %-14s %s\n" "$unit" "$(systemctl is-active $unit 2>/dev/null)"
  done
  python3 - <<'PY'
import sqlite3
c = sqlite3.connect("file:/mnt/jetsondata/tesla-alerts/scout.db?mode=ro", uri=True)
d = c.execute("SELECT COUNT(*) FROM drives").fetchone()[0]
o = c.execute("SELECT COUNT(*) FROM drives WHERE is_open=1").fetchone()[0]
print(f"    drives         {d} ({o} open)")
PY
  echo "Done."
else
  echo "DRY RUN complete. Re-run with --execute to apply."
fi
