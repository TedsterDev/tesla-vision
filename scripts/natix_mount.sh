#!/bin/bash
# natix_mount.sh - the only privileged step in the NATIX path.
#
# src/natix.py runs unprivileged. Mounting a block device is the one thing it
# cannot do itself, so it shells out to this script through `sudo -n`. That
# means this script ends up in a NOPASSWD sudoers rule, which makes it a
# potential privilege-escalation footgun: anything it will mount, any user in
# the allowed group can mount.
#
# So it validates rather than trusts:
#   - the device must be a USB-transport block device (never the boot card)
#   - the device must not be the LUN currently exported to the car
#   - the mountpoint must be inside the allowlist below, nowhere else
#   - nosuid,nodev,noexec are forced, so nothing on the stick can be executed
#
# Usage:  natix_mount.sh mount  /dev/sda1 /mnt/natixv360
#         natix_mount.sh umount /mnt/natixv360
set -euo pipefail

ALLOWED_MOUNT_PREFIXES=("/mnt/natixv360" "/mnt/natix" "/media/natix")

die() { echo "natix_mount: $*" >&2; exit 1; }

mountpoint_allowed() {
  local target="$1"
  for prefix in "${ALLOWED_MOUNT_PREFIXES[@]}"; do
    [[ "$target" == "$prefix" || "$target" == "$prefix"/* ]] && return 0
  done
  return 1
}

gadget_backing_files() {
  shopt -s nullglob
  for lun in /sys/kernel/config/usb_gadget/*/functions/mass_storage.*/lun.*/file; do
    cat "$lun" 2>/dev/null
  done
  shopt -u nullglob
}

action="${1:-}"

case "$action" in
  mount|mount-ro)
    device="${2:-}"; target="${3:-}"
    [[ -n "$device" && -n "$target" ]] || die "usage: $0 mount <device> <mountpoint>"
    [[ -b "$device" ]] || die "$device is not a block device"
    mountpoint_allowed "$target" || die "refusing to mount outside ${ALLOWED_MOUNT_PREFIXES[*]}: $target"

    # Transport check: lsblk reports the parent disk's bus. Anything that is
    # not USB is, by definition, not a stick somebody just plugged in.
    transport="$(lsblk -no TRAN "$(lsblk -no PKNAME "$device" | head -1 | sed 's|^|/dev/|')" 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -z "$transport" ]]; then
      transport="$(lsblk -no TRAN "$device" 2>/dev/null | head -1 | tr -d ' ')"
    fi
    [[ "$transport" == "usb" ]] || die "$device is on transport '${transport:-unknown}', not usb"

    # Never mount the image we are currently handing to the car.
    while read -r backing; do
      [[ -z "$backing" ]] && continue
      if [[ "$(readlink -f "$backing")" == "$(readlink -f "$device")" ]]; then
        die "$device is the LUN currently exported to the car"
      fi
    done < <(gadget_backing_files)

    fstype="$(lsblk -no FSTYPE "$device" | head -1 | tr -d ' ')"
    [[ -n "$fstype" ]] || die "$device has no detectable filesystem"

    mkdir -p "$target"
    if findmnt -n "$target" >/dev/null 2>&1; then
      echo "already mounted: $target"
      exit 0
    fi

    # exFAT has no kernel driver on this L4T 5.15 kernel, only exfat-fuse, so
    # it needs a different invocation from everything else.
    # Inspection must never be able to modify the stick. A read-only mount is
    # the difference between "look at what is on it" and "hope nothing writes".
    if [[ "$action" == "mount-ro" ]]; then
      case "$fstype" in
        exfat)
          mount -t exfat-fuse -o ro,nosuid,nodev,noexec "$device" "$target" \
          || mount.exfat-fuse -o ro "$device" "$target"
          ;;
        ntfs|ntfs3)
          mount -t ntfs-3g -o ro,nosuid,nodev,noexec "$device" "$target"
          ;;
        *)
          mount -t "$fstype" -o ro,nosuid,nodev,noexec "$device" "$target"
          ;;
      esac
      echo "mounted $device ($fstype) READ-ONLY at $target"
      exit 0
    fi

    case "$fstype" in
      exfat)
        # This L4T 5.15 kernel has no exfat driver, and Ubuntu ships no
        # /sbin/mount.exfat - only mount.exfat-fuse. So `mount -t exfat` fails
        # and `mount -t exfat-fuse` is the working form: mount(8) applies the
        # generic flags (nosuid/nodev/noexec/noatime) itself and hands only the
        # exfat-specific ones to the FUSE helper, which would reject the
        # generic ones if we called it directly.
        mount -t exfat-fuse -o rw,noatime,nosuid,nodev,noexec,umask=0002,uid=1000,gid=1000 \
          "$device" "$target" \
        || mount.exfat-fuse -o rw,umask=0002,uid=1000,gid=1000 "$device" "$target"
        ;;
      vfat)
        mount -t vfat -o rw,noatime,nosuid,nodev,noexec,umask=0002,uid=1000,gid=1000,flush \
          "$device" "$target"
        ;;
      ntfs|ntfs3)
        mount -t ntfs-3g -o rw,noatime,nosuid,nodev,noexec,uid=1000,gid=1000 \
          "$device" "$target"
        ;;
      ext2|ext3|ext4)
        mount -t "$fstype" -o rw,noatime,nosuid,nodev,noexec "$device" "$target"
        # ext* carries real ownership; make sure the service user can write.
        chown 1000:1000 "$target" 2>/dev/null || true
        ;;
      *)
        die "unsupported filesystem '$fstype' on $device"
        ;;
    esac
    echo "mounted $device ($fstype) at $target"
    ;;

  umount)
    target="${2:-}"
    [[ -n "$target" ]] || die "usage: $0 umount <mountpoint>"
    mountpoint_allowed "$target" || die "refusing to unmount outside ${ALLOWED_MOUNT_PREFIXES[*]}: $target"
    findmnt -n "$target" >/dev/null 2>&1 || { echo "not mounted: $target"; exit 0; }
    sync
    umount "$target" || { sleep 2; umount -l "$target"; }
    echo "unmounted $target"
    ;;

  *)
    die "usage: $0 {mount|mount-ro <device> <mountpoint>|umount <mountpoint>}"
    ;;
esac
