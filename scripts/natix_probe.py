#!/usr/bin/env python3
"""
natix_probe.py - identify a NATIX VX360 the first time you plug it in.

Discovery in src/natix.py works off a profile: a serial, a USB vendor/product
id, a volume UUID, a name pattern. Until the physical stick has been seen once,
none of those values are known, and the only thing left is shape matching -
"a removable USB volume of about the right size" - which is deliberately not
trusted enough to write to.

This script closes that gap. Plug the stick into any USB-A port on the Jetson,
run it, and it prints:

    1. every attached volume with the verdict discovery reached and why
    2. the full USB descriptor chain for the candidate
    3. the filesystem layout, if it can mount it read-only
    4. the exact .env lines that will pin it

It never writes to the device. The read-only mount is a mount, not a copy, and
it is unmounted before exit.

    python3 scripts/natix_probe.py            # judge what is attached
    python3 scripts/natix_probe.py --all      # include volumes already rejected
    python3 scripts/natix_probe.py --tree     # mount read-only and list contents
"""
import argparse
import os
import re
import subprocess
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import natix   # noqa: E402


EVENT_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr or "").rstrip()


def directory_size(path: Path) -> int:
    """Bytes held below `path`. Read-only, symlink-safe, tolerant of races."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        try:
            total += entry.stat().st_size
        except OSError:
            continue
    return total


def heading(text: str) -> None:
    print()
    print(text)
    print("=" * len(text))


def describe_usb_chain(volume: natix.Volume) -> None:
    """Walk sysfs from the block device up to the USB device that owns it."""
    disk_name = Path(volume.parent_path).name
    node = Path(f"/sys/block/{disk_name}")
    try:
        current = node.resolve()
    except OSError:
        print(f"  (no sysfs path for {disk_name})")
        return

    for _ in range(12):
        if (current / "idVendor").exists():
            attributes = [
                "idVendor", "idProduct", "manufacturer", "product", "serial",
                "bcdDevice", "speed", "bMaxPower", "version",
            ]
            print(f"  USB device: {current}")
            for attribute in attributes:
                path = current / attribute
                if path.exists():
                    try:
                        print(f"    {attribute:<14} {path.read_text().strip()}")
                    except OSError:
                        pass
            return
        if current.parent == current:
            break
        current = current.parent
    print("  (device is not behind a USB controller)")


def print_candidate(candidate: natix.Candidate, index: int) -> None:
    volume = candidate.volume
    marker = {
        "pinned": "PINNED",
        "strong": "STRONG",
        "likely": "likely",
        "weak": "  no  ",
    }[candidate.confidence]

    print(f"\n[{index}] {volume.path}   {marker}   {'USABLE' if candidate.usable else 'not usable'}")
    print(f"     size        {volume.size_gb:.1f} GB ({volume.size_bytes} bytes)")
    print(f"     filesystem  {volume.fstype or '(none detected)'}")
    print(f"     label       {volume.label or '-'}")
    print(f"     uuid        {volume.volume_uuid or '-'}")
    print(f"     transport   {volume.transport or '-'}   removable={volume.removable}")
    print(f"     vendor      {volume.vendor or '-'}")
    print(f"     model       {volume.model or '-'}")
    print(f"     serial      {volume.serial or '-'}")
    print(f"     usb id      {volume.usb_vendor_id or '?'}:{volume.usb_product_id or '?'}")
    print(f"     mounted at  {volume.mountpoint or '(not mounted)'}")
    print(f"     device key  {candidate.device_id}")
    for reason in candidate.reasons:
        print(f"     + {reason}")
    for problem in candidate.disqualifiers:
        print(f"     ! {problem}")


def suggest_env(candidate: natix.Candidate) -> None:
    """
    Print the .env lines that will pin this exact stick.

    Ordering matters more than it looks. The VX360 is a small Linux computer
    exporting its flash with the kernel's file-backed storage gadget, so it
    reports the stock gadget identifiers - USB 0525:a4a5, serial
    "Linux_File-Stor_Gadget-0:0". Those are shared by every Linux storage
    gadget on earth, this Jetson included when it is presenting a drive to the
    car. Pinning on them would eventually match the wrong device, so they are
    listed last and clearly marked, and src/natix.py refuses to honour them as
    identities even if you set them.
    """
    volume = candidate.volume
    heading("Pin this device (paste into .env)")

    strong: list[str] = []
    weak: list[tuple[str, str]] = []

    if volume.volume_uuid:
        strong.append(f"NATIX_VOLUME_UUID={volume.volume_uuid}")
    if volume.serial:
        if natix.is_generic_serial(volume.serial):
            weak.append((f"NATIX_SERIAL={volume.serial}",
                         "stock Linux-gadget serial, not unique - ignored by natix.py"))
        else:
            strong.append(f"NATIX_SERIAL={volume.serial}")
    if volume.usb_vendor_id and volume.usb_product_id:
        pair = f"{volume.usb_vendor_id}:{volume.usb_product_id}"
        if natix.is_generic_usb_id(pair):
            weak.append((f"NATIX_USB_ID={pair}",
                         "stock Linux-gadget USB id, not unique - ignored by natix.py"))
        else:
            strong.append(f"NATIX_USB_ID={pair}")

    if strong:
        print("  Use this - it is unique to this stick:\n")
        for line in strong:
            print(f"    {line}")
        print("\n  With a pin in place discovery reports confidence 'pinned', and the")
        print("  worker mirrors without needing NATIX_MIN_CONFIDENCE lowered.")
    else:
        print("  This volume exposes no unique identifier at all.")
        print("  Falling back to the label pattern is the best available option;")
        print(f"  its label is {volume.label!r}.")

    if weak:
        print("\n  Do NOT rely on these - every Linux storage gadget reports them,")
        print("  including this Jetson when it presents a drive to the car:\n")
        for line, why in weak:
            print(f"    # {line}    ({why})")


def show_tree(candidate: natix.Candidate, depth: int) -> None:
    """Mount read-only and print what is on the stick."""
    volume = candidate.volume
    heading("Filesystem contents")

    already_mounted = bool(volume.mountpoint)
    mountpoint = Path(volume.mountpoint) if already_mounted else Path("/mnt/natixv360")

    if not already_mounted:
        prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
        helper = REPO_ROOT / "scripts" / "natix_mount.sh"
        result = subprocess.run(
            prefix + [str(helper), "mount-ro", volume.path, str(mountpoint)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            print(f"  could not mount: {(result.stderr or result.stdout).strip()}")
            print(f"  try:  sudo {helper} mount {volume.path} {mountpoint}")
            return

    try:
        print(f"  mounted READ-ONLY at {mountpoint}")
        used = natix.free_bytes(mountpoint)
        import os as _os
        stats = _os.statvfs(str(mountpoint))
        total = stats.f_blocks * stats.f_frsize
        print(f"  capacity   {total / (1000 ** 3):.1f} GB")
        print(f"  free space {used / (1000 ** 3):.2f} GB "
              f"({100 * used / total:.1f}%)")
        print()
        print(run(["find", str(mountpoint), "-maxdepth", str(depth)])[:6000])
        print()
        markers = [m for m in natix.NATIX_MARKER_PATHS if (mountpoint / m).exists()]
        print(f"  NATIX markers found: {markers or 'none'}")

        # --- who is actually using the space -------------------------------
        # The interesting question on a full stick is not "what is on it" but
        # "what is holding the space, and is any of it ours". Answer both, in
        # bytes, without touching anything.
        heading("Space accounting (read-only)")
        top_level: list[tuple[int, str]] = []
        for entry in sorted(mountpoint.iterdir()):
            if entry.is_symlink():
                continue
            size = directory_size(entry) if entry.is_dir() else entry.stat().st_size
            top_level.append((size, entry.name))
        for size, name in sorted(top_level, reverse=True):
            print(f"  {size / 1024 ** 3:8.2f} GB  {name}")

        teslacam = mountpoint / "TeslaCam"
        print(f"\n  TeslaCam directory: {'present' if teslacam.is_dir() else 'absent'}")
        if teslacam.is_dir():
            for bucket in sorted(p.name for p in teslacam.iterdir() if p.is_dir()):
                bucket_dir = teslacam / bucket
                entries = sorted(p.name for p in bucket_dir.iterdir())
                size = directory_size(bucket_dir)
                stamps = [n for n in entries if EVENT_STAMP_RE.match(n)]
                print(f"    {bucket:<13} {size / 1024 ** 3:6.2f} GB  "
                      f"{len(entries)} entries")
                if stamps:
                    print(f"      oldest event {stamps[0]}   newest {stamps[-1]}")

        # Anything on the stick that a Tesla did not write is worth flagging
        # explicitly, because it is the part no automated cleanup should ever
        # be allowed to reason about.
        others = [name for _, name in top_level
                  if name not in ("TeslaCam",) and not name.startswith(".")]
        print(f"\n  Non-TeslaCam top-level entries: {others or 'none'}")
    finally:
        if not already_mounted:
            try:
                natix.unmount(mountpoint)
                print(f"\n  unmounted {mountpoint}")
            except natix.MountError as error:
                print(f"\n  WARNING: could not unmount: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="include volumes discovery already ruled out")
    parser.add_argument("--tree", action="store_true",
                        help="mount the best candidate read-only and list its contents")
    parser.add_argument("--depth", type=int, default=3,
                        help="how deep to list with --tree (default 3)")
    arguments = parser.parse_args()

    heading("Attached volumes")
    candidates = natix.discover(include_all=arguments.all)
    if not candidates:
        print("\n  Nothing plausible is attached.")
        print("  Every volume on this machine is either the boot card or the USB")
        print("  gadget image the car writes to.")
        print("\n  Plug the VX360 into a USB-A port and run this again. If it still")
        print("  does not appear, check `lsblk` and `dmesg | tail` - a stick that")
        print("  enumerates but exposes no filesystem shows up in dmesg only.")
        heading("Current profile")
        print(f"  NATIX_SERIAL         {natix.NATIX_SERIAL or '(unset)'}")
        print(f"  NATIX_USB_ID         {natix.NATIX_USB_ID or '(unset)'}")
        print(f"  NATIX_VOLUME_UUID    {natix.NATIX_VOLUME_UUID or '(unset)'}")
        print(f"  NATIX_NAME_PATTERNS  {natix.NATIX_NAME_PATTERNS}")
        print(f"  NATIX_MIN_CONFIDENCE {natix.NATIX_MIN_CONFIDENCE}")
        print(f"  size window          {natix.NATIX_MIN_SIZE_GB}-{natix.NATIX_MAX_SIZE_GB} GB")
        return 1

    for index, candidate in enumerate(candidates, start=1):
        print_candidate(candidate, index)

    best = candidates[0]
    heading(f"USB descriptor chain for {best.volume.path}")
    describe_usb_chain(best.volume)

    suggest_env(best)

    if arguments.tree:
        show_tree(best, arguments.depth)

    heading("Verdict")
    if best.usable:
        print(f"  {best.volume.path} is usable now (confidence '{best.confidence}').")
        print("  Start mirroring with:  python3 src/natix_worker.py --once")
    else:
        print(f"  {best.volume.path} is NOT usable: confidence '{best.confidence}'")
        print(f"  but NATIX_MIN_CONFIDENCE is '{natix.NATIX_MIN_CONFIDENCE}'.")
        if best.disqualifiers:
            print("  It is also disqualified outright:")
            for problem in best.disqualifiers:
                print(f"    - {problem}")
        else:
            print("  Pin it with the .env lines above (preferred), or approve it")
            print("  from the dashboard's NATIX page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
