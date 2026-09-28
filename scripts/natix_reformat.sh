#!/bin/bash
# natix_reformat.sh - recover the VX360's leaked clusters by reformatting it,
# without losing the one thing on it that exists nowhere else.
#
# Why this exists
# ---------------
# 28.49GB of this stick is allocated to no file at all. That was measured, not
# guessed: `statvfs` calls 32.94GB used, while all 116 files on the volume
# occupy 4.46GB and contain 4.45GB. The ~10MB between those two totals is
# cluster rounding (116 files x 128KB cluster = 14.5MB maximum), and its being
# non-zero is what proves the FUSE driver reports real allocation rather than
# echoing st_size back - so the numbers can be believed. The walk's file count
# matches fsck's exactly, so nothing is hidden from it.
#
# `fsck.exfat --repair` does not recover it. exfatprogs 1.1.3 reports `clean`
# in both -n and -y modes on this volume, because it does not cross-check the
# allocation bitmap; it has nothing to find and so nothing to fix. That is not
# a bug in this stick, it is the limit of that tool version.
#
# So a reformat is the remaining remedy. Two things make doing it by hand
# dangerous, and both are the entire reason this script exists:
#
# 1. TeslaCam/EncryptedClips is NATIX's own upload spool, and it holds clips
#    dated 2026-08-03. The Jetson archive holds 132 clips spanning
#    2026-02-16 12:49-13:42 and NOTHING else - zero clips from August. That
#    spool is therefore the only copy of that footage in existence. `mkfs`
#    destroys it permanently. This script copies every foreign file off the
#    stick and verifies the copy by sha256 BEFORE it will format anything, and
#    puts it back afterwards.
#
# 2. mkfs.exfat 1.1.3 has no option to set the volume UUID, so a reformat
#    mints a new random one. .env pins the stick by NATIX_VOLUME_UUID, so
#    after a naive reformat that pin silently stops matching and identity
#    quietly degrades to matching on the volume label alone. This script reads
#    the new UUID and updates .env, keeping a backup.
#
# Clips this mirror wrote are deliberately NOT restored: they are reproducible
# from the archive for free, and letting the mirror rebuild its window onto an
# empty 35GB stick is the whole point of doing this.
#
# Usage:
#     sudo ./scripts/natix_reformat.sh              # dry run: inventory only
#     sudo ./scripts/natix_reformat.sh --execute    # actually do it
#
# The dry run touches nothing and is the default. Read it before using
# --execute.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNTPOINT="${NATIX_MOUNTPOINT:-/mnt/natixv360}"
BASE_DIR="${BASE_DIR:-/mnt/jetsondata/tesla-alerts}"
RESCUE_ROOT="$BASE_DIR/natix-rescue"
ENV_FILE="$REPO_ROOT/.env"
LABEL="VX360"

MODE="${1:---dry-run}"

die() { echo "natix_reformat: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

[[ $EUID -eq 0 ]] || die "must run as root (sudo $0 ${1:-})"
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" || "$MODE" == "--restore-only" ]] \
  || die "usage: $0 [--dry-run|--execute|--restore-only [rescue-dir]]"
command -v mkfs.exfat >/dev/null || die "mkfs.exfat not found (apt install exfatprogs)"
command -v sha256sum  >/dev/null || die "sha256sum not found"

# ---------------------------------------------------------------------------
# Identify the stick the same way everything else does
# ---------------------------------------------------------------------------
# Never resolve the device by guessing at /dev/sd*. This asks the same code the
# mirror uses, which refuses non-USB, non-removable and system disks - so the
# thing about to be formatted is the thing the mirror writes to, or nothing.
step "identifying the stick"
DEVICE="$(
  BASE_DIR="$BASE_DIR" PYTHONPATH="$REPO_ROOT" \
  python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from src import natix
found = natix.find_device()
print(found.volume.path if found else '')
"
)"
[[ -n "$DEVICE" ]] || die "no usable NATIX device found - run scripts/natix_probe.py"
[[ -b "$DEVICE" ]] || die "$DEVICE is not a block device"

OLD_UUID="$(blkid -s UUID -o value "$DEVICE" 2>/dev/null || true)"
DEV_SIZE="$(lsblk -bno SIZE "$DEVICE" | head -1)"
echo "  device   $DEVICE"
echo "  uuid     ${OLD_UUID:-(none)}"
echo "  size     $(( DEV_SIZE / 1024 / 1024 / 1024 )) GB"

# A mounted filesystem must never be formatted, and fsck/mkfs on one corrupts
# it. Refuse rather than unmount silently - if something is using it, the user
# should know why before this proceeds.
if findmnt -n -S "$DEVICE" >/dev/null 2>&1; then
  die "$DEVICE is mounted at $(findmnt -n -o TARGET -S "$DEVICE"). Stop natix-mirror first:
    sudo systemctl stop natix-mirror"
fi

# ---------------------------------------------------------------------------
# Restore-only: put a rescue back onto an already-formatted stick
# ---------------------------------------------------------------------------
# Needed because the restore can fail *after* the format has already happened,
# leaving a verified rescue on disk and an empty stick. Re-running --execute
# would be wrong: there is nothing left to inventory, and it would format a
# stick that is already blank. This resumes from the rescue instead, and is
# safe to run repeatedly - it overwrites and re-verifies every file.
if [[ "$MODE" == "--restore-only" ]]; then
  RESCUE="${2:-}"
  if [[ -z "$RESCUE" ]]; then
    RESCUE="$(find "$RESCUE_ROOT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null \
              | sort | tail -1)"
  fi
  [[ -n "$RESCUE" && -d "$RESCUE" ]] \
    || die "no rescue directory found under $RESCUE_ROOT"
  [[ -f "$RESCUE/.manifest" ]] \
    || die "$RESCUE has no .manifest - cannot tell what should be restored"

  COUNT="$(tr -cd '\0' < "$RESCUE/.manifest" | wc -c)"
  step "restoring $COUNT files from $RESCUE"

  "$REPO_ROOT/scripts/natix_mount.sh" mount "$DEVICE" "$MOUNTPOINT"
  trap '"$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true' EXIT

  while IFS= read -r -d '' relative; do
    mkdir -p "$MOUNTPOINT/$(dirname "$relative")"
    cp --preserve=timestamps "$RESCUE/$relative" "$MOUNTPOINT/$relative"
  done < "$RESCUE/.manifest"
  sync

  step "verifying the restore by sha256"
  mismatches=0
  while IFS= read -r -d '' relative; do
    want="$(sha256sum < "$RESCUE/$relative" | cut -d' ' -f1)"
    got="$(sha256sum < "$MOUNTPOINT/$relative" | cut -d' ' -f1)"
    [[ "$want" == "$got" ]] || { echo "  MISMATCH $relative"; mismatches=$(( mismatches + 1 )); }
  done < "$RESCUE/.manifest"

  if [[ "$mismatches" -eq 0 ]]; then
    echo "  all $COUNT files verified byte-identical"
  else
    echo "  $mismatches files did not restore. The rescue at $RESCUE is intact"
    echo "  - do not delete it."
  fi
  df -h "$MOUNTPOINT" | tail -1 | sed 's/^/  /'
  "$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true
  trap - EXIT
  [[ "$mismatches" -eq 0 ]] || exit 1
  echo
  echo "Done. Restart mirroring with:  sudo systemctl start natix-mirror"
  exit 0
fi

# ---------------------------------------------------------------------------
# Inventory: what is ours (reproducible) and what is not (irreplaceable)
# ---------------------------------------------------------------------------
step "inventory (mounted read-only, nothing is modified)"
"$REPO_ROOT/scripts/natix_mount.sh" mount-ro "$DEVICE" "$MOUNTPOINT"
cleanup() {
  "$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true
}
trap cleanup EXIT

MANIFEST="$(mktemp)"

# Everything below reaches Python through the ENVIRONMENT, and the script is a
# quoted heredoc so the shell expands nothing inside it.
#
# The earlier version interpolated $MOUNTPOINT and friends into a double-quoted
# `python3 -c "..."`, which means the shell was also parsing every other line of
# that Python. A prose comment containing `while read` in backticks became a
# command substitution: bash ran it, `while read` is an incomplete compound
# command, and it failed with "syntax error: unexpected end of file". That one
# happened to land in a comment, so the substitution emptied and Python never
# noticed - the script produced correct output while printing an error. The
# next backtick or $(...) to appear in this block would not be so lucky, and
# this block is the code that decides which files get destroyed. So the shell
# does not get to read it at all.
MANIFEST_PATH="$MANIFEST" \
STICK_MOUNTPOINT="$MOUNTPOINT" \
STICK_OLD_UUID="$OLD_UUID" \
BASE_DIR="$BASE_DIR" \
PYTHONPATH="$REPO_ROOT" \
python3 - <<'PYEOF'
import os, sqlite3, sys
from pathlib import Path

MANIFEST = os.environ['MANIFEST_PATH']
MOUNTPOINT = os.path.normpath(os.environ['STICK_MOUNTPOINT'])
OLD_UUID = os.environ['STICK_OLD_UUID']
BASE_DIR = os.environ['BASE_DIR']
root = Path(MOUNTPOINT)

# Everything this mirror recorded writing to THIS device. Anything else on the
# volume was put there by someone or something else and is not ours to lose.
ours = set()
try:
    connection = sqlite3.connect(os.path.join(BASE_DIR, 'scout.db'))
    connection.row_factory = sqlite3.Row
    device_id = 'uuid:' + OLD_UUID
    for row in connection.execute(
        'SELECT dest_path FROM natix_mirror WHERE device_id=?', (device_id,)
    ):
        if not row['dest_path']:
            continue
        # Resolve each recorded path to a mountpoint-relative one so the
        # comparison below can be exact. Suffix matching was tried here and is
        # wrong at any anchoring: a recorded 'TeslaCam/SentryClips/ev1/a.mp4'
        # ends with '/a.mp4', so a DIFFERENT foreign file named 'a.mp4' at the
        # volume root matches it, gets judged 'ours', is not rescued, and is
        # destroyed by the format. Silently and permanently. Anything that does
        # not resolve to a path under this mountpoint is left out of the set
        # entirely, so it counts as foreign and gets rescued - every ambiguity
        # here has to fall the same way, toward copying more.
        dest = os.path.normpath(row['dest_path'])
        if os.path.isabs(dest):
            try:
                dest = os.path.relpath(dest, MOUNTPOINT)
            except ValueError:
                continue
            if dest.startswith(os.pardir):
                continue
        ours.add(dest)
    connection.close()
except Exception as error:
    print(f'  note: could not read the mirror database ({error}).', file=sys.stderr)
    print('  Treating every file as foreign, which errs toward rescuing more.',
          file=sys.stderr)

foreign_bytes = ours_bytes = 0
foreign_count = ours_count = 0
lines = []
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.is_symlink():
        continue
    relative = str(path.relative_to(root))
    size = path.stat().st_size
    # Match on the tail, because dest_path may be absolute or mountpoint-relative
    # depending on when the row was written.
    is_ours = relative in ours
    if is_ours:
        ours_count += 1; ours_bytes += size
    else:
        foreign_count += 1; foreign_bytes += size
        lines.append(relative)

# NUL-separated, and every entry terminated. Joining on newlines left the last
# entry without one, and `while read` silently drops an unterminated final
# line - so the last foreign file would not have been rescued, and would then
# have been destroyed by the format. NUL also makes filenames containing
# spaces or newlines safe. A script that decides what to destroy must not be
# guessing at field boundaries.
with open(MANIFEST, 'wb') as handle:
    for line in lines:
        handle.write(line.encode() + b'\0')

gb = lambda n: n / 1024 ** 3
print(f'  ours (this mirror wrote it, reproducible): {ours_count} files, {gb(ours_bytes):.2f} GB')
print(f'  foreign (exists only here, must survive): {foreign_count} files, {gb(foreign_bytes):.2f} GB')
print()
if foreign_count:
    print('  foreign top-level directories:')
    tops = {}
    for line in lines:
        top = line.split(os.sep)[0] if os.sep in line else '(root)'
        tops[top] = tops.get(top, 0) + 1
    for top, count in sorted(tops.items()):
        print(f'    {count:5d} files  {top}')
PYEOF

FOREIGN_COUNT="$(tr -cd '\0' < "$MANIFEST" | wc -c)"

if [[ "$MODE" == "--dry-run" ]]; then
  cat <<SUMMARY

=========================================================================
DRY RUN - nothing has been changed.

With --execute this would:
  1. copy $FOREIGN_COUNT foreign files to $RESCUE_ROOT/<timestamp>/
  2. verify every one of them by sha256, and STOP if any differ
  3. mkfs.exfat -L $LABEL $DEVICE          <-- destroys everything
  4. read the new volume UUID and update NATIX_VOLUME_UUID in .env
     (mkfs.exfat cannot preserve the old one; .env is backed up first)
  5. copy the foreign files back and verify them again
  6. leave the stick unmounted, ready for install_natix.sh

Clips this mirror wrote are NOT restored - they are rebuilt from the
archive by the next mirror pass onto a now-empty 35GB stick.

Run it for real with:
    sudo $0 --execute
=========================================================================
SUMMARY
  exit 0
fi

# ---------------------------------------------------------------------------
# Rescue, and prove the rescue is good BEFORE destroying the original
# ---------------------------------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
RESCUE="$RESCUE_ROOT/$STAMP"
step "rescuing $FOREIGN_COUNT foreign files to $RESCUE"

free_bytes="$(df -B1 --output=avail "$BASE_DIR" | tail -1)"
need_bytes="$(
  while IFS= read -r -d '' relative; do
    stat -c %s "$MOUNTPOINT/$relative"
  done < "$MANIFEST" | awk '{total += $1} END {printf "%.0f\n", total}'
)"
# "%.0f", and neither of the two obvious alternatives:
#
#   print    - awk holds totals as doubles and prints with a default format of
#              %.6g, so any sum past a million becomes "2.94015e+09". Bash
#              arithmetic cannot parse that, and the failure is not safe: an
#              unparseable value inside [[ x -gt $((...)) ]] collapses the
#              expansion to nothing, so the test reads as `-gt` with no operand
#              and evaluates TRUE. The guard does not abort, it PASSES. That is
#              how a run reached mkfs past a check that had already errored.
#   %d       - awk here is mawk, whose %d clamps at INT32_MAX. The real total
#              2940208143 prints as 2147483647, understating the rescue by 27%,
#              and a 35GB figure by 94%. Silently wrong in the lenient
#              direction, which is the worst direction for a free-space guard.
#
# %.0f goes through the double, which holds every byte count this script will
# ever see exactly, and prints it in full.
[[ "$need_bytes" =~ ^[0-9]+$ ]] \
  || die "could not total the rescue size (got '$need_bytes'). Refusing to
continue: this figure is the free-space check that stands between here and
mkfs, and a check that cannot be evaluated is not a check."
echo "  need $(( need_bytes / 1024 / 1024 )) MB, have $(( free_bytes / 1024 / 1024 )) MB free in $BASE_DIR"
[[ "$free_bytes" -gt $(( need_bytes + 1024 * 1024 * 1024 )) ]] \
  || die "not enough free space in $BASE_DIR to rescue the stick safely"

mkdir -p "$RESCUE"
while IFS= read -r -d '' relative; do
  mkdir -p "$RESCUE/$(dirname "$relative")"
  cp -a "$MOUNTPOINT/$relative" "$RESCUE/$relative"
done < "$MANIFEST"
sync

step "verifying the rescue by sha256 (before anything is destroyed)"
mismatches=0
while IFS= read -r -d '' relative; do
  source_hash="$(sha256sum < "$MOUNTPOINT/$relative" | cut -d' ' -f1)"
  rescue_hash="$(sha256sum < "$RESCUE/$relative" | cut -d' ' -f1)"
  if [[ "$source_hash" != "$rescue_hash" ]]; then
    echo "  MISMATCH $relative"
    mismatches=$(( mismatches + 1 ))
  fi
done < "$MANIFEST"
[[ "$mismatches" -eq 0 ]] \
  || die "$mismatches files did not copy correctly. NOTHING has been formatted.
The stick is untouched; investigate before retrying."
echo "  all $FOREIGN_COUNT files verified byte-identical"

cp -a "$MANIFEST" "$RESCUE/.manifest"

# ---------------------------------------------------------------------------
# The destructive step
# ---------------------------------------------------------------------------
cleanup
trap - EXIT

step "formatting $DEVICE (this destroys everything on it)"
echo "  a verified copy of every foreign file is in $RESCUE"
mkfs.exfat -L "$LABEL" "$DEVICE"
sync
partprobe "$DEVICE" 2>/dev/null || true
udevadm settle 2>/dev/null || true

NEW_UUID="$(blkid -s UUID -o value "$DEVICE" 2>/dev/null || true)"
[[ -n "$NEW_UUID" ]] || die "could not read the new volume UUID from $DEVICE.
The rescue is intact at $RESCUE - restore it by hand before doing anything else."
echo "  new volume UUID: $NEW_UUID  (was ${OLD_UUID:-none})"

# ---------------------------------------------------------------------------
# Re-pin, or identity silently degrades to matching on the label alone
# ---------------------------------------------------------------------------
step "updating the pin in .env"
if [[ -f "$ENV_FILE" ]] && grep -q '^NATIX_VOLUME_UUID=' "$ENV_FILE"; then
  cp -a "$ENV_FILE" "$ENV_FILE.bak.$STAMP"
  sed -i "s/^NATIX_VOLUME_UUID=.*/NATIX_VOLUME_UUID=$NEW_UUID/" "$ENV_FILE"
  echo "  NATIX_VOLUME_UUID=$NEW_UUID  (backup: .env.bak.$STAMP)"
else
  echo "  WARNING: no NATIX_VOLUME_UUID line found in $ENV_FILE."
  echo "  Add this by hand, or the stick will be identified by label alone:"
  echo "      NATIX_VOLUME_UUID=$NEW_UUID"
fi

# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------
step "restoring $FOREIGN_COUNT foreign files"
"$REPO_ROOT/scripts/natix_mount.sh" mount "$DEVICE" "$MOUNTPOINT"
trap cleanup EXIT

while IFS= read -r -d '' relative; do
  mkdir -p "$MOUNTPOINT/$(dirname "$relative")"
  # --preserve=timestamps, NOT -a. exFAT has no concept of ownership, so `cp -a`
  # tries to chown, fails with "Operation not permitted", and returns non-zero -
  # which under `set -e` killed the restore after its very first file. Mtimes
  # are the only attribute worth carrying here and the only one the filesystem
  # can hold.
  cp --preserve=timestamps "$RESCUE/$relative" "$MOUNTPOINT/$relative"
done < "$MANIFEST"
sync

step "verifying the restore"
mismatches=0
while IFS= read -r -d '' relative; do
  rescue_hash="$(sha256sum < "$RESCUE/$relative" | cut -d' ' -f1)"
  stick_hash="$(sha256sum < "$MOUNTPOINT/$relative" | cut -d' ' -f1)"
  [[ "$rescue_hash" == "$stick_hash" ]] || { echo "  MISMATCH $relative"; mismatches=$(( mismatches + 1 )); }
done < "$MANIFEST"
if [[ "$mismatches" -ne 0 ]]; then
  echo
  echo "  $mismatches files did not restore correctly."
  echo "  The rescue copy is still intact at $RESCUE - do not delete it."
else
  echo "  all $FOREIGN_COUNT files verified byte-identical"
fi

df -h "$MOUNTPOINT" | tail -1 | sed 's/^/  /'
cleanup
trap - EXIT
rm -f "$MANIFEST"

cat <<DONE

=========================================================================
Done. The stick is empty apart from the rescued foreign files.

The rescue copy is kept at:
    $RESCUE
Delete it only once you have confirmed the stick is healthy - it is the
only copy of that footage.

Next, rebuild the mirror onto the now-empty stick:
    sudo $REPO_ROOT/scripts/install_natix.sh

Two things worth knowing:
  - The mirror database still holds rows for the old device id
    (uuid:$OLD_UUID). They are harmless - rows are keyed by device, and
    the stick is a new device now - but they will never be reused.
  - With ~35GB free the rolling window should cover the whole archive
    rather than the 40 of 132 clips it managed before.
=========================================================================
DONE
