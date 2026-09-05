#!/bin/bash
# natix_fsck.sh - check the VX360's filesystem for space that is marked used
# but belongs to nothing.
#
# Why this exists
# ---------------
# The stick came back reporting 34.65GB of 35GB in use while a full directory
# walk could only find 6.14GB of actual files. ~28.5GB was allocated to
# nothing. That is the signature of orphaned clusters: the exFAT allocation
# bitmap says those clusters are taken, but no directory entry references them
# any more.
#
# It is an entirely expected state for this device. The stick lives in a car
# and loses power mid-write every time the car sleeps, and it has also been
# mounted on a Mac (.Trashes, .Spotlight-V100 and .fseventsd are all present).
# Neither of those ever gets a clean unmount.
#
# No amount of deleting files recovers that space, because there are no files
# to delete - which is why the mirror's own reclaim logic correctly found
# nothing eligible. Only a filesystem check can hand those clusters back.
#
# Usage:
#     sudo ./scripts/natix_fsck.sh              # fsck + read-only content walk
#     sudo ./scripts/natix_fsck.sh --list-trash # what desktop OS leftovers hold
#     sudo ./scripts/natix_fsck.sh --empty-trash# reclaim that space
#     sudo ./scripts/natix_fsck.sh --write-test # is the free-space figure true?
#     sudo ./scripts/natix_fsck.sh --repair     # fix what fsck found
#     sudo ./scripts/natix_fsck.sh --clean-stray# recover from a silent mount failure
#
# Only --empty-trash and --repair modify anything.
#
# The device must NOT be mounted. The script refuses if it is.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNTPOINT="${NATIX_MOUNTPOINT:-/mnt/natixv360}"

die() { echo "natix_fsck: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root (sudo $0 ${1:-})"
command -v fsck.exfat >/dev/null || die "fsck.exfat not found (apt install exfatprogs)"

# Find the stick the same way everything else does, so we cannot possibly fsck
# a different disk than the one the mirror writes to.
DEVICE="$(
  BASE_DIR="${BASE_DIR:-/mnt/jetsondata/tesla-alerts}" \
  PYTHONPATH="$REPO_ROOT" \
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

# fsck on a mounted filesystem corrupts it. Refuse rather than warn.
if findmnt -n -S "$DEVICE" >/dev/null 2>&1; then
  die "$DEVICE is mounted. Unmount first:  $REPO_ROOT/scripts/natix_mount.sh umount $MOUNTPOINT"
fi
if findmnt -n "$MOUNTPOINT" >/dev/null 2>&1; then
  die "$MOUNTPOINT is mounted. Unmount first:  $REPO_ROOT/scripts/natix_mount.sh umount $MOUNTPOINT"
fi

echo "device: $DEVICE"
lsblk -no SIZE,FSTYPE,LABEL,UUID "$DEVICE" | sed 's/^/  /'
echo

if [[ "${1:-}" == "--clean-stray" ]]; then
  # Recover from a silent mount failure.
  #
  # If a FUSE mount fails after forking, mount(8) still exits 0. The mirror
  # then wrote clips into the bare mountpoint DIRECTORY on the Jetson's own SD
  # card, which succeeds silently because writing to a directory always does.
  # This removes that stranded copy and resets the database rows that describe
  # it, so the next pass mirrors those clips onto the actual stick.
  #
  # It refuses to run while anything is mounted at the mountpoint - that would
  # mean deleting from the stick instead of from the host disk.
  if findmnt -n "$MOUNTPOINT" >/dev/null 2>&1; then
    die "$MOUNTPOINT is currently a mountpoint. Unmount first, or there is nothing stray to clean."
  fi
  [[ -d "$MOUNTPOINT" ]] || die "$MOUNTPOINT does not exist - nothing to clean"

  bytes="$(du -sb "$MOUNTPOINT" 2>/dev/null | cut -f1)"
  files="$(find "$MOUNTPOINT" -type f 2>/dev/null | wc -l)"
  echo "==> stray data on the HOST disk at $MOUNTPOINT"
  echo "    $files files, $(( ${bytes:-0} / 1024 / 1024 )) MB"
  echo
  find "$MOUNTPOINT" -maxdepth 3 2>/dev/null | head -20 | sed 's/^/    /'
  if [[ "${bytes:-0}" -eq 0 ]]; then
    echo
    echo "    Nothing here. The mountpoint is clean."
    exit 0
  fi

  echo
  echo "==> removing (these are copies; the originals are untouched in"
  echo "    /mnt/jetsondata/tesla-alerts/processed)"
  find "$MOUNTPOINT" -mindepth 1 -delete
  echo "    removed"

  echo
  echo "==> resetting mirror bookkeeping so these clips are re-copied to the stick"
  BASE_DIR="${BASE_DIR:-/mnt/jetsondata/tesla-alerts}" python3 -c "
import sqlite3
connection = sqlite3.connect('${BASE_DIR:-/mnt/jetsondata/tesla-alerts}/scout.db')
removed = connection.execute('DELETE FROM natix_mirror').rowcount
connection.execute('UPDATE natix_devices SET mirrored_count=0, mirrored_bytes=0')
connection.commit()
print(f'    cleared {removed} mirror rows')
"
  df -h "$(dirname "$MOUNTPOINT")" | tail -1 | sed 's/^/    /'
  echo
  echo "Now re-run:  sudo $REPO_ROOT/scripts/install_natix.sh"
  exit 0
fi

if [[ "${1:-}" == "--empty-trash" || "${1:-}" == "--list-trash" ]]; then
  # Desktop operating systems leave large droppings on removable media that
  # their owners never see. .Trashes in particular is where macOS puts files
  # you "deleted" - they occupy the volume until the trash is emptied, and a
  # stick that has been in a Mac can be almost entirely trash.
  #
  # --list-trash only reports. --empty-trash removes. Both print the full
  # inventory first, because this is somebody's deleted footage and they should
  # see what goes before it goes.
  ACTION="${1}"
  "$REPO_ROOT/scripts/natix_mount.sh" \
    "$([[ "$ACTION" == "--list-trash" ]] && echo mount-ro || echo mount)" \
    "$DEVICE" "$MOUNTPOINT"
  trap '"$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true' EXIT

  TRASH_DIRS=(".Trashes" ".Spotlight-V100" ".fseventsd" "System Volume Information" "\$RECYCLE.BIN")
  total=0
  echo "==> desktop OS leftovers on the stick"
  echo
  for name in "${TRASH_DIRS[@]}"; do
    target="$MOUNTPOINT/$name"
    [[ -e "$target" ]] || continue
    bytes="$(du -sb "$target" 2>/dev/null | cut -f1)"
    files="$(find "$target" -type f 2>/dev/null | wc -l)"
    echo "  $(( ${bytes:-0} / 1024 / 1024 )) MB  ${files} files  $name"
    total=$(( total + ${bytes:-0} ))
  done
  echo
  echo "  total: $(( total / 1024 / 1024 )) MB"

  # Cross-check against df, in BOTH size dimensions. This distinction is the
  # whole diagnostic, and getting it wrong sent this script's earlier verdict
  # into the weeds:
  #
  #   apparent size  - how many bytes the file *contains*  (du -sb, st_size)
  #   allocated size - how many bytes the file *occupies*  (du -s, st_blocks)
  #
  # `du -sb` implies --apparent-size. So did the Python walk. Measuring the
  # same dimension twice and calling the second one an independent check is
  # how "the space is in no file at all" got asserted without ever testing it.
  # On exFAT the two diverge for real: a file can carry a valid-data-length
  # far below its allocated cluster chain, so a spool of preallocated clips
  # occupies the volume while appearing to contain almost nothing.
  #
  # Which of the two matches df tells you what is actually wrong, and the two
  # faults have completely different remedies:
  #
  #   allocated ~= df used   -> the space IS in files, just not in their
  #                             contents. Deleting those files recovers it.
  #                             fsck is right to call the volume clean.
  #   both  <<  df used      -> clusters the bitmap marks taken that no
  #                             directory entry references. Only --repair or
  #                             a reformat recovers that.
  echo
  echo "==> whole-volume accounting (du vs df, no Python involved)"
  du_apparent="$(du -sb "$MOUNTPOINT" 2>/dev/null | cut -f1)"
  du_allocated="$(du -s --block-size=1 "$MOUNTPOINT" 2>/dev/null | cut -f1)"
  file_count="$(find "$MOUNTPOINT" -type f 2>/dev/null | wc -l)"
  df_used="$(df -B1 --output=used "$MOUNTPOINT" | tail -1)"
  df_avail="$(df -B1 --output=avail "$MOUNTPOINT" | tail -1)"
  echo "  files on the volume:            ${file_count}"
  echo "  bytes those files contain:      $(( du_apparent / 1024 / 1024 )) MB   (apparent, st_size)"
  echo "  bytes those files occupy:       $(( du_allocated / 1024 / 1024 )) MB   (allocated, st_blocks)"
  echo "  bytes df calls used:            $(( df_used / 1024 / 1024 )) MB"
  echo "  bytes df calls free:            $(( df_avail / 1024 / 1024 )) MB"

  slack=$(( du_allocated - du_apparent ))
  unaccounted=$(( df_used - du_allocated ))
  THRESHOLD=268435456   # 256MB - below this it is cluster rounding, not a fault

  # Guard against believing a number the driver may not be reporting. exFAT
  # here is FUSE, not a kernel driver, and a FUSE filesystem is free to derive
  # st_blocks from st_size rather than from the real cluster chain. If it does,
  # the two totals are identical by construction and their agreement means
  # nothing - so say so, instead of letting a degraded check read as a passing
  # one. Some rounding difference is expected on a healthy volume; exact
  # equality across hundreds of files is the tell.
  if [[ $file_count -gt 20 && $du_allocated -eq $du_apparent ]]; then
    echo
    echo "  NOTE: allocated and apparent totals are byte-for-byte identical"
    echo "  across ${file_count} files. That is not what a real filesystem looks like;"
    echo "  it means this FUSE mount is deriving st_blocks from st_size and does"
    echo "  not know the true allocation. Treat the two numbers below as ONE"
    echo "  measurement, not two - this run cannot tell preallocated files apart"
    echo "  from leaked clusters, and any verdict it gives you rests on that."
  fi

  echo
  if [[ $unaccounted -gt $THRESHOLD ]]; then
    echo "  VERDICT: $(( unaccounted / 1024 / 1024 )) MB is allocated to no file at all."
    echo "  Those are leaked clusters - the bitmap marks them used but nothing"
    echo "  points at them. Deleting files cannot recover it, because there are"
    echo "  no files. Only --repair or a reformat will. See the README."
  elif [[ $slack -gt $THRESHOLD ]]; then
    echo "  VERDICT: the space IS in files - they just do not contain it."
    echo "  $(( slack / 1024 / 1024 )) MB is allocated to files beyond the bytes they hold."
    echo "  That is preallocated or sparsely-written data, most likely a spool"
    echo "  whose clips were reserved at full size and never filled in. fsck is"
    echo "  right to call this volume clean; nothing is corrupt. Deleting the"
    echo "  files that own those clusters recovers the space - a repair will not."
    echo
    echo "  The heaviest offenders, by what they OCCUPY rather than contain:"
    du -a --block-size=1 "$MOUNTPOINT" 2>/dev/null | sort -rn | head -12 |
      while read -r occupied path; do
        contained="$(du -sb "$path" 2>/dev/null | cut -f1)"
        printf "    %8s MB occupied  %8s MB contained  %s\n" \
          "$(( occupied / 1024 / 1024 ))" "$(( ${contained:-0} / 1024 / 1024 ))" \
          "${path#$MOUNTPOINT/}"
      done
  else
    echo "  VERDICT: accounting reconciles. Every byte df calls used belongs to"
    echo "  a file this walk can see. If the volume is full, it is full of data."
  fi

  if [[ "$ACTION" == "--list-trash" ]]; then
    echo
    echo "  Nothing was removed. To reclaim this space:"
    echo "    sudo $0 --empty-trash"
    echo "  Note that .Trashes holds files somebody deleted on a Mac. They are"
    echo "  recoverable until this runs, and not afterwards."
  else
    echo
    echo "==> removing"
    for name in "${TRASH_DIRS[@]}"; do
      target="$MOUNTPOINT/$name"
      [[ -e "$target" ]] || continue
      rm -rf "$target" && echo "  removed $name"
    done
    sync
    echo
    df -h "$MOUNTPOINT" | tail -1 | sed 's/^/  /'
  fi

  "$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true
  trap - EXIT
  exit 0
fi

if [[ "${1:-}" == "--write-test" ]]; then
  # The decisive experiment when fsck says "clean" but the volume claims to be
  # full: stop arguing with statvfs and try to write.
  #
  # Two implementations are in play - the stick is mounted by relan's
  # exfat-fuse 1.3.0, while fsck comes from exfatprogs 1.1.3. If they disagree
  # about how many clusters are allocated, the mount's free-space figure can be
  # wrong in either direction, and no amount of reading settles it. A real
  # write does.
  "$REPO_ROOT/scripts/natix_mount.sh" mount "$DEVICE" "$MOUNTPOINT"
  trap '"$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true' EXIT

  TESTFILE="$MOUNTPOINT/.natix-write-test"
  before="$(df -B1 --output=avail "$MOUNTPOINT" | tail -1)"
  before_mb=$((before / 1024 / 1024))

  # The size has to EXCEED the reported free space, or the test proves nothing.
  # The first version of this defaulted to a flat 256MB, wrote it into 362MB of
  # reported free space, and announced that free space was under-reported -
  # having demonstrated only that 256 is less than 362. The whole question is
  # whether we can write PAST the reported limit, so the target is derived from
  # that limit rather than fixed.
  SIZE_MB="${2:-$((before_mb + 512))}"

  echo "==> WRITE TEST"
  echo "    reported free: ${before_mb} MB"
  echo "    attempting:    ${SIZE_MB} MB  (deliberately more than reported free)"
  echo "    Writes one temporary file and deletes it again. Nothing else on the"
  echo "    stick is touched."
  echo

  # dd's status, not tail's. `if dd ... | tail -1` tests the LAST command in the
  # pipeline, so an out-of-space dd used to be reported as a success.
  dd_status=0
  dd if=/dev/zero of="$TESTFILE" bs=1M count="$SIZE_MB" conv=fsync 2>"$MOUNTPOINT/../.ddlog" || dd_status=$?
  tail -1 "$MOUNTPOINT/../.ddlog" 2>/dev/null || true
  rm -f "$MOUNTPOINT/../.ddlog"

  actual="$(stat -c %s "$TESTFILE" 2>/dev/null || echo 0)"
  actual_mb=$((actual / 1024 / 1024))
  echo
  echo "    dd exit status: $dd_status"
  echo "    actually written: ${actual_mb} MB"

  if [[ $actual_mb -gt $before_mb ]]; then
    echo
    echo "    VERDICT: wrote PAST the reported limit (${actual_mb}MB > ${before_mb}MB)."
    echo "    exfat-fuse is under-reporting free space. Set"
    echo "    NATIX_IGNORE_FREE_SPACE=1 so the mirror stops trusting it."
  else
    echo
    echo "    VERDICT: stopped at the reported limit. The free-space figure is"
    echo "    ACCURATE - the stick genuinely holds ~$((35 * 1024 - before_mb))MB"
    echo "    of real data. NATIX_IGNORE_FREE_SPACE will not help."
    echo "    Find the files with the content walk:  sudo $0"
  fi

  after="$(df -B1 --output=avail "$MOUNTPOINT" | tail -1)"
  echo "    free after (df): $((after / 1024 / 1024)) MB"
  rm -f "$TESTFILE"
  sync
  echo "    test file removed"

  "$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true
  trap - EXIT
  exit 0
fi

if [[ "${1:-}" == "--repair" ]]; then
  echo "==> REPAIRING (fsck.exfat -y). This modifies the filesystem."
  echo "    Orphaned clusters are returned to free space; nothing that a"
  echo "    directory entry still points at is touched."
  echo
  # -y answers yes to every repair. -v so the log says what it did.
  fsck.exfat -y -v "$DEVICE" || true
  echo
  echo "==> re-checking"
  fsck.exfat -n -v "$DEVICE" || true
else
  echo "==> READ-ONLY CHECK (fsck.exfat -n). Nothing will be modified."
  echo
  # Tee it, so the file count can be compared against the walk automatically
  # rather than asking a human to eyeball two numbers printed 30 lines apart.
  fsck_output="$(fsck.exfat -n -v "$DEVICE" 2>&1 || true)"
  echo "$fsck_output"
  fsck_files="$(sed -n 's/.*files \([0-9]\+\).*/\1/p' <<<"$fsck_output" | tail -1)"
  fsck_clean="no"
  grep -q "clean\." <<<"$fsck_output" && fsck_clean="yes"

  # fsck reports how many files it can see. If that count disagrees with what
  # a directory walk can see, the walk is the thing that is broken - which is
  # exactly the bug this section exists to catch. Mount read-only and compare.
  echo
  echo "==> READ-ONLY CONTENT WALK (compare the file count against fsck's)"
  echo
  "$REPO_ROOT/scripts/natix_mount.sh" mount-ro "$DEVICE" "$MOUNTPOINT"
  trap '"$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true' EXIT

  # The walk's numbers are needed by the verdict below, so it writes them to a
  # file for the shell to read rather than only printing them for a human. An
  # earlier version of that verdict referenced variables set in a different
  # branch of this script; they were empty here, and every comparison silently
  # compared against nothing.
  STATS="$(mktemp)"
  BASE_DIR="${BASE_DIR:-/mnt/jetsondata/tesla-alerts}" \
  PYTHONPATH="$REPO_ROOT" \
  python3 -c "
import os, sys
sys.path.insert(0, '$REPO_ROOT')
from pathlib import Path
from src import natix

root = Path('$MOUNTPOINT')
for line in natix.describe_space(root):
    print('  ' + line)

walk = natix.walk_tree(root, keep_largest=0)
stats = os.statvfs(str(root))
used = (stats.f_blocks - stats.f_bavail) * stats.f_frsize
print()
print(f'  walk totals: {walk.files} files, {walk.directories} directories, '
      f'{len(walk.errors)} unreadable paths')

with open('$STATS', 'w') as handle:
    handle.write(f'WALK_FILES={walk.files}\n')
    handle.write(f'WALK_CONTAINED={walk.bytes_seen}\n')
    handle.write(f'WALK_OCCUPIED={walk.bytes_allocated}\n')
    handle.write(f'WALK_ERRORS={len(walk.errors)}\n')
    handle.write(f'DF_USED={used}\n')
"

  # Defaults, so a failed python run cannot make the verdict below read as a
  # reconciled volume. Absent numbers must not look like agreeing numbers.
  WALK_FILES=-1; WALK_CONTAINED=0; WALK_OCCUPIED=0; WALK_ERRORS=0; DF_USED=0
  # shellcheck disable=SC1090
  [[ -s "$STATS" ]] && source "$STATS"
  rm -f "$STATS"
  THRESHOLD=268435456
  unaccounted=$(( DF_USED - WALK_OCCUPIED ))

  "$REPO_ROOT/scripts/natix_mount.sh" umount "$MOUNTPOINT" >/dev/null 2>&1 || true
  trap - EXIT

  echo
  echo "==> WHAT TO DO ABOUT IT"
  echo
  # This used to say: "If fsck said 'clean', the space is genuinely occupied by
  # real files and there is nothing to repair." That is the opposite of the
  # truth on this device, and it was printed directly beneath a walk that had
  # just proved otherwise. exfatprogs 1.1.3 in no-modify mode reports `clean`
  # on a volume with tens of gigabytes of clusters belonging to no directory
  # entry, so a clean verdict from `-n` says nothing about the allocation
  # bitmap. Trust the measurement, not fsck's headline.
  if [[ "$unaccounted" -gt "$THRESHOLD" ]]; then
    echo "  The walk found $(( unaccounted / 1024 / 1024 )) MB that belongs to no file, in either"
    echo "  size dimension - neither the bytes the files hold nor the clusters"
    echo "  they occupy account for it."
    if [[ "$fsck_clean" == "yes" ]]; then
      echo
      echo "  fsck called this volume clean anyway. That is not a contradiction"
      echo "  to resolve in fsck's favour: exfatprogs 1.1.3 in no-modify mode"
      echo "  reports 'clean' without cross-checking the allocation bitmap, so"
      echo "  its verdict is silent on exactly the fault measured above."
    fi
    echo
    echo "  Try the non-destructive repair first:"
    echo "      sudo $0 --repair"
    echo
    echo "  If that also reports clean and frees nothing, the only remaining"
    echo "  remedy is a reformat. Before doing that, know what you would lose:"
    echo "  everything this walk can see. Check the bucket listing above -"
    echo "  clips this mirror wrote are reproducible from the archive and cost"
    echo "  nothing to lose, but anything the device put there itself (an"
    echo "  upload spool, for instance) exists only on the stick."
  else
    echo "  The accounting reconciles: the space is in files the walk can see."
    echo "  There is nothing for a repair to do. Read the bucket listing above"
    echo "  to decide what to remove."
  fi

  # The walk-vs-fsck cross-check, done here rather than left to the reader.
  if [[ -n "$fsck_files" ]]; then
    echo
    if [[ "$WALK_FILES" -ge $(( fsck_files - 2 )) ]]; then
      echo "  Cross-check: the walk saw ${WALK_FILES} files, fsck saw ${fsck_files}. They agree,"
      echo "  so nothing is hidden from the walk and its totals can be trusted."
    else
      echo "  WARNING: the walk saw only ${WALK_FILES} files but fsck saw ${fsck_files}."
      echo "  The walk is missing files, so every total above understates reality"
      echo "  and the verdict rests on numbers that are too low. Fix that before"
      echo "  acting on anything printed here."
    fi
  fi
fi

echo
echo "Then re-run the mirror:"
echo "    sudo $REPO_ROOT/scripts/install_natix.sh"
