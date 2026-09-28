"""
natix.py

NATIX VX360 integration.

What the device is
------------------
The VX360 is not a camera. It is a 128-256GB USB stick with an ARM Cortex-A7,
Wi-Fi and Bluetooth inside it. You normally plug it into a Tesla's glovebox USB
port *instead of* a dashcam drive: the car writes TeslaCam footage onto it, and
when the car is parked near a known Wi-Fi network the stick's own firmware
uploads that footage to NATIX's cloud. To a host computer it enumerates as
ordinary USB mass storage - "same as a USB stick" is the vendor's own wording.

Why this module exists
----------------------
In our topology the car does NOT write to the VX360. The car's USB-C line goes
to the Jetson, which presents a mass-storage gadget backed by an image file on
its own card (scripts/teslacam_image.sh). That is the whole point: one drive,
under our control, that the Scout pipeline can read whenever it likes.

But that also means the VX360 never sees a single frame - the car isn't talking
to it any more. So the Jetson has to become the writer. This module puts the
footage on the stick, in the exact directory layout Tesla would have written,
so the stick's firmware finds it and uploads it as if nothing had changed.

    Tesla ──USB-C──► Jetson gadget (ext4 image)
                        │
                        ├─► Scout pipeline (processor, correlator, dashboard)
                        └─► USB-A host port ──► VX360  ◄── this module

Design constraints that shaped the code
---------------------------------------
1. **Standard library only.** The mirror runs on the *host*, not in a
   container, because it mounts and unmounts real block devices in the host's
   mount namespace. Host Python is 3.10 with no venv, so no third-party
   imports are allowed here. (src.common and src.db are themselves stdlib.)

2. **Never guess which stick to write to.** A wrong match means we format-fill
   somebody's photo backup with dashcam clips. Discovery therefore returns
   *every* candidate with a confidence level and the reasons behind it, and
   refuses to act on anything below `NATIX_MIN_CONFIDENCE`. The root disk and
   the gadget backing store are hard-excluded regardless of what they look
   like.

3. **Crash-safe copies.** The car cuts power without warning. Every file is
   written to a temp name on the destination filesystem and then `os.replace`d
   into place, so a half-copied clip can never appear under its real name.

4. **Loop recording.** The stick is smaller than our archive will eventually
   be, so mirroring prunes its oldest complete events to make room, exactly
   like a dashcam does. We only ever prune events *we* put there.
"""
import json
import os
import re
import shutil
import subprocess
import time
import uuid

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from src.common import (
    BASE_DIR,
    INBOX_DIR,
    PROCESSED_DIR,
    env_flag,
    env_int,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Pin the stick once you have physically identified it. `scripts/natix_probe.py`
# prints the exact lines to paste into .env. Until then discovery falls back to
# name matching and then to shape matching, and refuses to write unless you
# have raised NATIX_MIN_CONFIDENCE's floor or approved the device in the UI.
NATIX_SERIAL = os.environ.get("NATIX_SERIAL", "").strip()
NATIX_USB_ID = os.environ.get("NATIX_USB_ID", "").strip().lower()   # "1234:abcd"
NATIX_VOLUME_UUID = os.environ.get("NATIX_VOLUME_UUID", "").strip()

# Substrings we accept in a volume label / USB vendor / USB model string. The
# retail stick reports itself differently across firmware revisions, so this is
# a list rather than one magic string.
NATIX_NAME_PATTERNS = [
    pattern.strip().lower()
    for pattern in os.environ.get("NATIX_NAME_PATTERNS", "natix,vx360,vx-360,vx_360").split(",")
    if pattern.strip()
]

# Files/directories that, if present on the volume, identify it as the stick.
# Checked only once the volume is mounted, so it upgrades confidence on the
# second pass rather than the first.
NATIX_MARKER_PATHS = [
    marker.strip()
    for marker in os.environ.get(
        "NATIX_MARKER_PATHS", ".natix,natix.json,.vx360,NATIX,.natix_device"
    ).split(",")
    if marker.strip()
]

# Shape matching, used only when nothing above hit. The retail device is 128GB
# or 256GB; the floor keeps us away from small utility sticks.
NATIX_MIN_SIZE_GB = env_int("NATIX_MIN_SIZE_GB", 64)
NATIX_MAX_SIZE_GB = env_int("NATIX_MAX_SIZE_GB", 2048)

# Lowest confidence we will actually mount and write to.
#   pinned  - serial / USB id / volume UUID matched an explicit setting
#   strong  - the name or an on-disk marker said NATIX
#   likely  - it is simply a removable USB volume of plausible size
NATIX_MIN_CONFIDENCE = os.environ.get("NATIX_MIN_CONFIDENCE", "strong").strip().lower()
CONFIDENCE_ORDER = {"weak": 0, "likely": 1, "strong": 2, "pinned": 3}

NATIX_MOUNTPOINT = Path(os.environ.get("NATIX_MOUNTPOINT", "/mnt/natixv360"))

# Leave this much free on the stick so its own firmware has room to work.
NATIX_RESERVE_MB = env_int("NATIX_RESERVE_MB", 2048)

# Skip the free-space pre-check and copy until the filesystem itself refuses.
#
# Needed because this stick is mounted by relan's exfat-fuse while fsck comes
# from exfatprogs, and the two can disagree about how many clusters are
# allocated. When fsck calls the volume clean but the mount insists it is 99%
# full, the mount is the one that is wrong - and a pre-check that believes it
# blocks every copy forever, for a condition that does not exist.
#
# This is safe to turn on because a genuine ENOSPC is already handled: the copy
# is atomic, so a failed write removes its own temp file, leaves no partial
# clip under a real name, records the failure, and moves on. The pre-check is a
# courtesy that avoids pointless IO, not a correctness guarantee.
NATIX_IGNORE_FREE_SPACE = env_flag("NATIX_IGNORE_FREE_SPACE", False)

# Which end of the archive the stick holds when the archive is bigger than the
# stick. 'newest' (default) or 'oldest'. See build_plan for why newest is both
# the more useful answer and the only one that terminates.
NATIX_MIRROR_ORDER = os.environ.get("NATIX_MIRROR_ORDER", "newest").strip().lower()

# The only bucket names Tesla's layout defines. Anything else under TeslaCam/
# was not put there by a car and is none of our business.
TESLA_BUCKETS = ("RecentClips", "SentryClips", "SavedClips")

# Buckets we may reclaim space from when the stick arrives already full of
# footage we did not write. Empty - the default - means never touch anything of
# theirs.
#
# Why this exists: the VX360 loop-records only while the *car* is writing to
# it. Here the car writes to the Jetson instead, so the stick's own
# loop-deletion never runs again and it stays frozen at whatever state the car
# left it in - permanently full, with nothing we send it ever fitting. Taking
# over the loop is not a nicety in this topology, it is the only way the
# arrangement works at all.
#
# It still defaults to off, because the footage is the owner's and only they
# know whether it has been uploaded yet. 'SavedClips' should stay out of any
# value you set: those are the clips somebody deliberately pressed the button
# to keep.
NATIX_RECLAIM_BUCKETS = [
    bucket.strip()
    for bucket in os.environ.get("NATIX_RECLAIM_BUCKETS", "").split(",")
    if bucket.strip()
]

# Which TeslaCam bucket we file clips under when the clip row doesn't say.
# SentryClips is the right default: it is what the car produces unattended,
# and it is the bucket NATIX's uploader is documented to consume.
NATIX_DEFAULT_BUCKET = os.environ.get("NATIX_DEFAULT_BUCKET", "SentryClips").strip()

# Tesla writes an event.json beside each event's clips. Synthesising a minimal
# one costs nothing and means the stick sees a well-formed event folder.
NATIX_WRITE_EVENT_JSON = env_flag("NATIX_WRITE_EVENT_JSON", True)

# Unmount after each mirror pass. Safer in a car (power can vanish), at the
# cost of a mount/unmount per cycle.
NATIX_UNMOUNT_AFTER = env_flag("NATIX_UNMOUNT_AFTER", True)

# Paths we must never treat as a NATIX stick, no matter how they present.
PROTECTED_MOUNTPOINTS = ("/", "/boot", "/boot/efi", "/home", "/var", "/usr")

REPO_ROOT = Path(__file__).resolve().parents[1]
MOUNT_HELPER = Path(
    os.environ.get("NATIX_MOUNT_HELPER", str(REPO_ROOT / "scripts" / "natix_mount.sh"))
)

# Identifiers that are NOT identities.
#
# The VX360 is itself a small Linux box (an Allwinner sunxi SoC running
# Armbian) exporting part of its flash with the kernel's file-backed storage
# gadget. That gadget ships with fixed defaults, so the stick reports the
# generic Linux ids - vendor 0525:a4a5 "Netchip", serial
# "Linux_File-Stor_Gadget-0:0", model "File-Stor Gadget". Every Linux storage
# gadget on earth reports the same ones, including this very Jetson when it is
# presenting itself to the car. Pinning on them would therefore match the wrong
# device, so we treat them as hints and fall through to the volume UUID, which
# really is unique.
GENERIC_USB_IDS = {"0525:a4a5", "1d6b:0104", "1d6b:0100", "0525:a4a0"}
GENERIC_SERIALS = {
    "linux_file-stor_gadget-0:0",
    "linux_file-stor_gadget",
    "0:0",
    "0000",
    "1234567890",
}
GENERIC_MODELS = {"file-stor gadget", "mass storage gadget", "usb disk"}


def is_generic_serial(serial: str | None) -> bool:
    if not serial:
        return True
    return serial.strip().lower() in GENERIC_SERIALS


def is_generic_usb_id(pair: str | None) -> bool:
    return not pair or pair.strip().lower() in GENERIC_USB_IDS


TESLA_CLIP_RE = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-(?P<camera>[a-z_]+)\.mp4$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Block device discovery
# ---------------------------------------------------------------------------
@dataclass
class Volume:
    """One mountable filesystem on an attached block device."""
    path: str                     # /dev/sda1
    parent_path: str              # /dev/sda   (== path when unpartitioned)
    size_bytes: int = 0
    fstype: str | None = None
    label: str | None = None
    volume_uuid: str | None = None
    mountpoint: str | None = None
    removable: bool = False
    read_only: bool = False
    transport: str | None = None  # usb | mmc | nvme | ...
    vendor: str | None = None
    model: str | None = None
    serial: str | None = None
    usb_vendor_id: str | None = None
    usb_product_id: str | None = None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1000 ** 3)


@dataclass
class Candidate:
    """A volume, judged against the NATIX profile."""
    volume: Volume
    confidence: str = "weak"
    reasons: list[str] = field(default_factory=list)
    disqualifiers: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        if self.disqualifiers:
            return False
        floor = CONFIDENCE_ORDER.get(NATIX_MIN_CONFIDENCE, 2)
        return CONFIDENCE_ORDER.get(self.confidence, 0) >= floor

    @property
    def device_id(self) -> str:
        return device_key(self.volume)


def device_key(volume: Volume) -> str:
    """
    A stable identity for a stick across replugs.

    A *real* serial is best - it survives reformatting. But the VX360 reports
    the stock Linux-gadget serial, which is shared by every Linux storage
    gadget (see GENERIC_SERIALS), so we skip it and use the volume UUID, which
    is genuinely unique to this stick's filesystem. Label is the last resort,
    and the device node after that - the 'dev:' prefix marks that as an
    identity that will not survive a replug.
    """
    if volume.serial and not is_generic_serial(volume.serial):
        return f"serial:{volume.serial}"
    if volume.volume_uuid:
        return f"uuid:{volume.volume_uuid}"
    if volume.label:
        return f"label:{volume.label}"
    return f"dev:{volume.path}"


def _run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )


def _lsblk_json() -> dict[str, Any]:
    """
    Ask lsblk for the whole tree in JSON.

    lsblk reads udev's database, so it gives us FSTYPE/LABEL/UUID/SERIAL
    without root - which matters, because discovery runs unprivileged and only
    the mount step escalates.
    """
    columns = (
        "NAME,PATH,SIZE,TYPE,FSTYPE,LABEL,UUID,MOUNTPOINT,RM,RO,"
        "TRAN,VENDOR,MODEL,SERIAL,PKNAME"
    )
    # Not "no devices" - "cannot see devices". The difference matters: the
    # first is a fact about the car, the second is a fact about where this code
    # is running, and conflating them makes the dashboard say "no stick
    # attached" when the truth is "I am in a container and cannot look".
    if shutil.which("lsblk") is None:
        raise FileNotFoundError("lsblk is not available in this environment")
    if not Path("/run/udev").exists():
        # lsblk is present in our containers and happily lists the host's
        # /sys/block, but with every fstype, label and UUID null - because
        # those come from udev's database, which containers do not get, and
        # /dev carries no block nodes to fall back on. Everything would score
        # 'weak' and the page would report an empty drive bay. Refuse instead.
        raise FileNotFoundError(
            "no udev database at /run/udev - block device metadata is not "
            "visible from here (this is normal inside a container)"
        )
    result = _run(["lsblk", "-J", "-b", "-o", columns])
    if result.returncode != 0 or not result.stdout.strip():
        return {"blockdevices": []}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"blockdevices": []}


def _usb_ids_for(device_name: str) -> tuple[str | None, str | None]:
    """
    Walk /sys upward from a block device to the USB device that owns it and
    read its idVendor/idProduct.

    lsblk reports TRAN=usb but not the USB ids, and the ids are the only thing
    that survives a reformat *and* a different filesystem - so they're worth
    the sysfs walk.
    """
    node = Path(f"/sys/block/{device_name}")
    try:
        current = node.resolve()
    except OSError:
        return (None, None)

    for _ in range(12):
        vendor_file = current / "idVendor"
        product_file = current / "idProduct"
        if vendor_file.exists() and product_file.exists():
            try:
                return (
                    vendor_file.read_text().strip().lower(),
                    product_file.read_text().strip().lower(),
                )
            except OSError:
                return (None, None)
        if current.parent == current:
            break
        current = current.parent
    return (None, None)


def _root_disk_names() -> set[str]:
    """
    Names of the disks backing anything we must not touch.

    We resolve the *parent* of each protected mountpoint's source, so that a
    second partition on the boot card is excluded too - the OS card in this
    machine has fifteen partitions and only one of them is '/'.
    """
    protected: set[str] = set()
    tree = _lsblk_json()

    def walk(node: dict[str, Any], top_name: str | None) -> None:
        name = node.get("name") or ""
        root_name = top_name or name
        mountpoint = node.get("mountpoint")
        if mountpoint in PROTECTED_MOUNTPOINTS or (
            mountpoint and mountpoint.startswith("/boot")
        ):
            protected.add(root_name)
        for child in node.get("children", []) or []:
            walk(child, root_name)

    for disk in tree.get("blockdevices", []):
        walk(disk, None)
    return protected


def list_volumes() -> list[Volume]:
    """Every filesystem-bearing volume currently attached, root disk included."""
    tree = _lsblk_json()
    volumes: list[Volume] = []

    def emit(node: dict[str, Any], disk: dict[str, Any]) -> None:
        node_type = node.get("type")
        if node_type not in ("disk", "part"):
            return
        # A disk with partitions is a container, not a volume - its children
        # are the real filesystems. A disk with no children may itself be
        # formatted (superfloppy), which is how many USB sticks ship.
        if node_type == "disk" and (node.get("children") or []):
            return

        disk_name = disk.get("name") or ""
        usb_vendor, usb_product = _usb_ids_for(disk_name)
        volumes.append(
            Volume(
                path=node.get("path") or f"/dev/{node.get('name')}",
                parent_path=disk.get("path") or f"/dev/{disk_name}",
                size_bytes=int(node.get("size") or 0),
                fstype=node.get("fstype"),
                label=node.get("label"),
                volume_uuid=node.get("uuid"),
                mountpoint=node.get("mountpoint"),
                removable=bool(disk.get("rm")),
                read_only=bool(node.get("ro")),
                transport=disk.get("tran"),
                vendor=(disk.get("vendor") or "").strip() or None,
                model=(disk.get("model") or "").strip() or None,
                serial=(disk.get("serial") or "").strip() or None,
                usb_vendor_id=usb_vendor,
                usb_product_id=usb_product,
            )
        )

    for disk in tree.get("blockdevices", []):
        if disk.get("type") != "disk":
            continue
        emit(disk, disk)
        for child in disk.get("children", []) or []:
            emit(child, disk)

    return volumes


def _text_matches_natix(volume: Volume) -> str | None:
    """Return the field that matched a NATIX name pattern, or None."""
    for field_name in ("label", "model", "vendor", "serial"):
        value = (getattr(volume, field_name) or "").lower()
        if not value:
            continue
        for pattern in NATIX_NAME_PATTERNS:
            if pattern in value:
                return f"{field_name}={getattr(volume, field_name)!r} contains {pattern!r}"
    return None


def _marker_on_volume(volume: Volume) -> str | None:
    """If the volume is already mounted, look for a NATIX marker on it."""
    if not volume.mountpoint:
        return None
    root = Path(volume.mountpoint)
    for marker in NATIX_MARKER_PATHS:
        if (root / marker).exists():
            return f"marker {marker!r} present on {volume.mountpoint}"
    return None


def _gadget_backing_paths() -> set[str]:
    """
    Whatever the USB gadget is currently exporting to the car.

    Writing to the LUN's backing store while the car has it mounted corrupts
    it, so this is a hard exclusion rather than a warning.
    """
    backing: set[str] = set()
    gadget_root = Path("/sys/kernel/config/usb_gadget")
    if not gadget_root.exists():
        return backing
    for lun_file in gadget_root.glob("*/functions/mass_storage.*/lun.*/file"):
        try:
            value = lun_file.read_text().strip()
        except OSError:
            continue
        if value:
            backing.add(value)
            # A loop-backed LUN points at an image file; exclude the loop
            # device standing in front of it as well.
            backing.add(os.path.realpath(value))
    return backing


def evaluate(volume: Volume, protected_disks: set[str] | None = None) -> Candidate:
    """Judge one volume against the NATIX profile, explaining the verdict."""
    if protected_disks is None:
        protected_disks = _root_disk_names()

    candidate = Candidate(volume=volume)
    disk_name = Path(volume.parent_path).name

    # --- hard exclusions ---------------------------------------------------
    if disk_name in protected_disks:
        candidate.disqualifiers.append(
            f"{volume.parent_path} carries the operating system"
        )
    if volume.mountpoint in PROTECTED_MOUNTPOINTS:
        candidate.disqualifiers.append(f"mounted at protected path {volume.mountpoint}")
    if volume.mountpoint and Path(volume.mountpoint) == Path(BASE_DIR):
        candidate.disqualifiers.append("this is the Scout data volume")
    if volume.path in _gadget_backing_paths():
        candidate.disqualifiers.append("currently exported to the car as the USB gadget LUN")
    if volume.read_only:
        candidate.disqualifiers.append("volume is read-only")

    # --- identity: pinned --------------------------------------------------
    if NATIX_SERIAL and volume.serial and volume.serial.strip() == NATIX_SERIAL:
        if is_generic_serial(volume.serial):
            candidate.reasons.append(
                f"serial {volume.serial!r} matches NATIX_SERIAL but is the stock "
                f"Linux-gadget serial, shared by every Linux storage gadget - "
                f"not treated as an identity"
            )
        else:
            candidate.confidence = "pinned"
            candidate.reasons.append(f"serial matches NATIX_SERIAL ({NATIX_SERIAL})")

    if NATIX_VOLUME_UUID and volume.volume_uuid == NATIX_VOLUME_UUID:
        candidate.confidence = "pinned"
        candidate.reasons.append(
            f"volume UUID matches NATIX_VOLUME_UUID ({NATIX_VOLUME_UUID})"
        )

    if NATIX_USB_ID and volume.usb_vendor_id and volume.usb_product_id:
        pair = f"{volume.usb_vendor_id}:{volume.usb_product_id}"
        if pair == NATIX_USB_ID:
            if is_generic_usb_id(pair):
                candidate.reasons.append(
                    f"USB id {pair} matches NATIX_USB_ID but is the stock "
                    f"Linux-gadget id - not treated as an identity"
                )
            else:
                candidate.confidence = "pinned"
                candidate.reasons.append(f"USB id matches NATIX_USB_ID ({pair})")

    # --- identity: strong --------------------------------------------------
    if candidate.confidence != "pinned":
        name_hit = _text_matches_natix(volume)
        marker_hit = _marker_on_volume(volume)
        if name_hit:
            candidate.confidence = "strong"
            candidate.reasons.append(name_hit)
        if marker_hit:
            candidate.confidence = "strong"
            candidate.reasons.append(marker_hit)

    # --- identity: likely (shape only) -------------------------------------
    if candidate.confidence == "weak":
        looks_right = (
            volume.transport == "usb"
            and volume.fstype in ("exfat", "vfat", "ext4", "ext3", "ntfs")
            and NATIX_MIN_SIZE_GB <= volume.size_gb <= NATIX_MAX_SIZE_GB
        )
        if looks_right:
            candidate.confidence = "likely"
            candidate.reasons.append(
                f"removable USB {volume.fstype} volume of {volume.size_gb:.0f}GB "
                f"- shape matches a VX360 but nothing identifies it as one"
            )
        else:
            if volume.transport != "usb":
                candidate.reasons.append(f"transport is {volume.transport!r}, not usb")
            elif not volume.fstype:
                candidate.reasons.append("no filesystem detected")
            elif volume.fstype not in ("exfat", "vfat", "ext4", "ext3", "ntfs"):
                candidate.reasons.append(f"filesystem {volume.fstype!r} is not one a VX360 ships with")
            else:
                candidate.reasons.append(
                    f"size {volume.size_gb:.1f}GB outside "
                    f"{NATIX_MIN_SIZE_GB}-{NATIX_MAX_SIZE_GB}GB"
                )

    return candidate


def discover(include_all: bool = False) -> list[Candidate]:
    """
    Every attached volume, judged, best first.

    `include_all=False` drops volumes that are neither plausible nor
    interesting, which is most of them on a machine with fifteen boot
    partitions. Pass True from the probe script, where seeing the rejects is
    the entire point.
    """
    protected = _root_disk_names()
    candidates = [evaluate(volume, protected) for volume in list_volumes()]
    if not include_all:
        # Worth reporting: anything plausible, plus anything on USB that we
        # rejected - that second group is the "why isn't my stick showing up"
        # case. A rejected internal boot partition is neither, and this machine
        # has fifteen of them.
        candidates = [
            candidate
            for candidate in candidates
            if candidate.confidence != "weak"
            or (candidate.disqualifiers and candidate.volume.transport == "usb")
        ]
    candidates.sort(
        key=lambda candidate: (
            CONFIDENCE_ORDER.get(candidate.confidence, 0),
            candidate.volume.size_bytes,
        ),
        reverse=True,
    )
    return candidates


def find_device() -> Candidate | None:
    """The one stick we are willing to write to, or None."""
    for candidate in discover():
        if candidate.usable:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Mount / unmount
# ---------------------------------------------------------------------------
class MountError(RuntimeError):
    """Raised when we could not get the stick mounted read-write."""


def _sudo_prefix() -> list[str]:
    """Empty when we are already root; a non-interactive sudo otherwise."""
    return [] if os.geteuid() == 0 else ["sudo", "-n"]


def mount_source(path: Path) -> str | None:
    """The device backing `path`, or None when `path` is not a mountpoint."""
    result = _run(["findmnt", "-n", "-o", "SOURCE", str(path)])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def mount(candidate: Candidate, mountpoint: Path | None = None) -> Path:
    """
    Mount the stick read-write and return where it landed.

    The exit code of the mount helper is NOT sufficient evidence that a mount
    happened, and believing it caused the worst bug in this module's history:
    a FUSE mount failed while `mount(8)` still returned 0, the mirror carried
    on, and 1.1GB of clips went into the bare mountpoint directory on the
    Jetson's own SD card instead of onto the stick. Nothing complained, because
    writing to a directory always works.

    So every path out of here is verified against `findmnt`, and the device it
    reports has to be the device we asked for.
    """
    volume = candidate.volume
    if not candidate.usable:
        raise MountError(
            f"refusing to mount {volume.path}: confidence {candidate.confidence!r} "
            f"is below NATIX_MIN_CONFIDENCE={NATIX_MIN_CONFIDENCE!r}"
            + (f"; {'; '.join(candidate.disqualifiers)}" if candidate.disqualifiers else "")
        )

    if volume.mountpoint:
        adopted = Path(volume.mountpoint)
        _verify_capacity(adopted, volume)
        return adopted

    target = Path(mountpoint or NATIX_MOUNTPOINT)

    # Already mounted? Only adopt it if it is *our* device down there - and put
    # the adopted mount through the same capacity check as a fresh one. Skipping
    # the check on this path would have let exactly the failure we are guarding
    # against walk straight through.
    existing = mount_source(target)
    if existing:
        if os.path.realpath(existing) != os.path.realpath(volume.path):
            raise MountError(
                f"{target} already has {existing} mounted, not {volume.path}"
            )
        _verify_capacity(target, volume)
        return target

    # Not mounted, but not empty either. That content is the fingerprint of a
    # previous silent mount failure - files we wrote to the host disk thinking
    # they were going to the stick. Mounting over it would hide them forever
    # while they quietly consume the boot card, so refuse and say where to look.
    if target.is_dir():
        try:
            stray = sorted(entry.name for entry in target.iterdir())
        except OSError:
            stray = []
        if stray:
            raise MountError(
                f"{target} is not a mountpoint but is not empty either "
                f"(contains {stray[:4]}). That data is on the host disk, not on "
                f"the stick - almost certainly written during an earlier mount "
                f"that failed without saying so. Inspect it, then clear it with: "
                f"sudo {REPO_ROOT}/scripts/natix_fsck.sh --clean-stray"
            )

    command = _sudo_prefix() + [str(MOUNT_HELPER), "mount", volume.path, str(target)]
    if not MOUNT_HELPER.exists():
        raise MountError(f"mount helper missing: {MOUNT_HELPER}")

    result = _run(command, timeout=60)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if "password is required" in detail or "no tty" in detail:
            raise MountError(
                "mounting needs root and passwordless sudo is not configured. "
                f"Run: sudo {REPO_ROOT}/scripts/install_natix.sh"
            )
        raise MountError(f"mount of {volume.path} failed: {detail}")

    # The helper said it worked. Check.
    source = mount_source(target)
    if source is None:
        raise MountError(
            f"the mount helper reported success but nothing is mounted at "
            f"{target}. FUSE helpers can fail after forking, so a zero exit "
            f"status does not mean the filesystem attached. Refusing to write "
            f"to a bare directory."
        )
    if os.path.realpath(source) != os.path.realpath(volume.path):
        raise MountError(
            f"{target} ended up backed by {source}, not {volume.path}"
        )

    _verify_capacity(target, volume)
    return target


def _verify_capacity(target: Path, volume: Volume) -> None:
    """
    Cross-check the mounted filesystem's size against the block device's.

    '35GB stick, 73.5GB free' was the line in the journal that gave the whole
    incident away - 73.5GB was the Jetson's own SD card, because nothing was
    mounted and we were looking at the host filesystem. Device identity is the
    primary check; this catches the same class of mistake by a second route
    that does not depend on findmnt being right.
    """
    if not volume.size_bytes:
        return
    stats = os.statvfs(str(target))
    mounted_total = stats.f_blocks * stats.f_frsize
    if mounted_total > volume.size_bytes * 1.5:
        raise MountError(
            f"{target} reports {mounted_total / 1024 ** 3:.1f}GB but "
            f"{volume.path} is only {volume.size_bytes / 1024 ** 3:.1f}GB - "
            f"this is not the stick"
        )


def unmount(mountpoint: Path) -> None:
    """Flush and unmount, tolerating an already-unmounted path."""
    if not is_mounted(mountpoint):
        return
    command = _sudo_prefix() + [str(MOUNT_HELPER), "umount", str(mountpoint)]
    result = _run(command, timeout=60)
    if result.returncode != 0 and is_mounted(mountpoint):
        raise MountError(
            f"unmount of {mountpoint} failed: {(result.stderr or result.stdout).strip()}"
        )


def is_mounted(path: Path) -> bool:
    return _run(["findmnt", "-n", str(path)]).returncode == 0


def free_bytes(path: Path) -> int:
    stats = os.statvfs(str(path))
    return stats.f_bavail * stats.f_frsize


# ---------------------------------------------------------------------------
# Mirroring
# ---------------------------------------------------------------------------
@dataclass
class MirrorPlan:
    """One clip's journey from our archive onto the stick."""
    clip_id: str
    filename: str
    source: Path
    bucket: str
    event: str                 # 2026-02-16_20-49-20
    relative_dest: str         # TeslaCam/SentryClips/2026-02-16_20-49-20/....mp4
    size_bytes: int


@dataclass
class MirrorResult:
    copied: int = 0
    copied_bytes: int = 0
    skipped: int = 0
    missing: int = 0
    pruned: int = 0
    pruned_bytes: int = 0
    reclaimed: int = 0
    reclaimed_bytes: int = 0
    reclaimed_detail: list[str] = field(default_factory=list)
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    stopped_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def event_name_for(filename: str, captured_ts: int | None) -> str:
    """
    The event folder a clip belongs to.

    Tesla groups the six cameras of one moment into a folder named for that
    moment, and every one of our filenames carries that stamp already - so the
    filename is the authority, with captured_ts only as a fallback for clips
    whose names we didn't parse.
    """
    match = TESLA_CLIP_RE.match(filename)
    if match:
        return match.group("stamp")
    if captured_ts:
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(captured_ts))
    return "unsorted"


def resolve_source(filename: str, extra_roots: Iterable[Path] = ()) -> Path | None:
    """Find a clip's bytes. Processed first - that is where they end up."""
    for root in (PROCESSED_DIR, INBOX_DIR, *extra_roots):
        candidate_path = Path(root) / filename
        if candidate_path.is_file():
            return candidate_path
    return None


def build_plan(
    connection,
    device_id: str,
    limit: int = 0,
    extra_roots: Iterable[Path] = (),
    root: Path | None = None,
) -> tuple[list[MirrorPlan], int]:
    """
    Clips that belong on this stick but are not on it yet, oldest first.

    Oldest-first matters: the stick is a rolling window, and filling it in
    chronological order means a partial mirror is still a contiguous stretch of
    history rather than a scatter.
    """
    # 'pruned' has to be excluded alongside 'done', and leaving it out was a
    # genuine bug: the stick holds ~44 of 132 clips, so the mirror filled up,
    # pruned its own oldest to make room, and - because a pruned clip looked
    # un-mirrored again - immediately recopied it, evicting something else. The
    # observed result was 60 done / 72 pruned oscillating forever, rewriting
    # flash at 9MB/s and never converging. A clip that has been evicted from
    # this stick has had its turn.
    # A 'done' row is a claim that the clip is on the stick. When `root` is
    # given, verify the claim instead of trusting it, because it can be false
    # in ways nothing else notices:
    #
    #   - the stick was reformatted, or its files cleared by hand
    #   - a mount silently failed and the "copy" went to the host disk, which
    #     was then cleaned up underneath these rows
    #
    # Without this the clip is excluded forever and the stick is permanently
    # short, with the database insisting everything is fine. Re-planning is
    # cheap and self-correcting: mirror() skips any destination that already
    # exists at the right size, so a clip that IS present costs one stat.
    #
    # 'pruned' is never re-checked. Those files are absent on purpose - the
    # rolling window evicted them - and re-planning them is precisely the
    # rewrite loop this exclusion was added to stop.
    already = set()
    for row in connection.execute(
        "SELECT clip_id, state, dest_path FROM natix_mirror "
        "WHERE device_id=? AND state IN ('done','pruned')",
        (device_id,),
    ):
        if root is not None and row["state"] == "done" and row["dest_path"]:
            destination = Path(row["dest_path"])
            if not destination.is_absolute():
                destination = root / destination
            if not destination.exists():
                continue
        already.add(row["clip_id"])

    # Newest first by default. The archive is bigger than the stick, so this is
    # a rolling window and the question is which end of history it holds. Newest
    # is both the more useful answer for a surveillance tool and the one that
    # converges: fill with the most recent clips, then stop. Oldest-first fills
    # with footage that the next pass then evicts to make room for newer, which
    # is how the rewrite loop above got started.
    order = "DESC" if NATIX_MIRROR_ORDER == "newest" else "ASC"
    rows = connection.execute(
        f"SELECT id, filename, camera, captured_ts, clip_source "
        f"FROM clips ORDER BY captured_ts {order}, filename {order}"
    ).fetchall()

    plans: list[MirrorPlan] = []
    missing = 0
    for row in rows:
        if row["id"] in already:
            continue
        source = resolve_source(row["filename"], extra_roots)
        if source is None:
            missing += 1
            continue
        event = event_name_for(row["filename"], row["captured_ts"])
        bucket = (row["clip_source"] or NATIX_DEFAULT_BUCKET).strip() or NATIX_DEFAULT_BUCKET
        plans.append(
            MirrorPlan(
                clip_id=row["id"],
                filename=row["filename"],
                source=source,
                bucket=bucket,
                event=event,
                relative_dest=f"TeslaCam/{bucket}/{event}/{row['filename']}",
                size_bytes=source.stat().st_size,
            )
        )
        if limit and len(plans) >= limit:
            break

    return plans, missing


def _write_event_json(event_dir: Path, event: str, plans: list[MirrorPlan]) -> None:
    """
    Drop a minimal Tesla-shaped event.json if the folder doesn't have one.

    Tesla writes this file and downstream consumers - including the VX360's
    uploader - use it to decide an event's timestamp and reason. We only ever
    create it, never overwrite: a real one from the car is always better than
    ours.
    """
    target = event_dir / "event.json"
    if target.exists():
        return
    try:
        readable = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.strptime(event, "%Y-%m-%d_%H-%M-%S")
        )
    except ValueError:
        readable = event
    payload = {
        "timestamp": readable,
        "city": "",
        "est_lat": "",
        "est_lon": "",
        "reason": "sentry_aware_object_detection",
        "camera": "0",
    }
    _atomic_write(target, json.dumps(payload, indent=2).encode("utf-8"))


def _atomic_write(target: Path, payload: bytes) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _copy_atomic(source: Path, target: Path) -> int:
    """
    Copy one clip so that the destination name never names a partial file.

    The temp file is created in the destination directory, so `os.replace` is a
    same-filesystem rename and therefore atomic. fsync before the rename is
    what makes that true across a power cut, which is the failure mode this
    device actually has.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        with open(source, "rb") as reader, open(temporary, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        size = temporary.stat().st_size
        os.replace(temporary, target)
        return size
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prune_oldest(connection, device_id: str, root: Path, need_bytes: int) -> tuple[int, int]:
    """
    Delete our oldest mirrored events until `need_bytes` are free.

    Two safety properties matter here. We only delete files that `natix_mirror`
    says *we* wrote, so anything the user or the stick's own firmware put there
    survives. And we delete a whole event at a time, because half an event on a
    dashcam is worse than none.
    """
    freed_files = 0
    freed_bytes = 0

    events = connection.execute(
        "SELECT event_folder, SUM(size_bytes) AS total, MIN(copied_ts) AS oldest "
        "FROM natix_mirror WHERE device_id=? AND state='done' "
        "GROUP BY event_folder",
        (device_id,),
    ).fetchall()

    # Order by the event's own timestamp, not by the full path. event_folder is
    # 'TeslaCam/<bucket>/<stamp>', so sorting the whole string sorts by bucket
    # first - which would drain every SavedClips event before touching a
    # SentryClips one, regardless of age. SavedClips are the clips the driver
    # deliberately kept; they are the last thing that should go.
    ordered = sorted(events, key=lambda row: row["event_folder"].rsplit("/", 1)[-1])

    for event_row in ordered:
        if free_bytes(root) >= need_bytes:
            break
        event_folder = event_row["event_folder"]
        rows = connection.execute(
            "SELECT id, dest_path, size_bytes FROM natix_mirror "
            "WHERE device_id=? AND event_folder=? AND state='done'",
            (device_id, event_folder),
        ).fetchall()
        for row in rows:
            file_path = root / row["dest_path"]
            try:
                if file_path.exists():
                    file_path.unlink()
                    freed_bytes += row["size_bytes"] or 0
                freed_files += 1
            except OSError:
                continue
            connection.execute(
                "UPDATE natix_mirror SET state='pruned' WHERE id=?", (row["id"],)
            )
        # Take the now-empty event folder with it, event.json included.
        folder = root / event_folder
        if folder.is_dir():
            for leftover in folder.iterdir():
                if leftover.name in ("event.json", "thumb.png"):
                    leftover.unlink(missing_ok=True)
            try:
                folder.rmdir()
            except OSError:
                pass
        connection.commit()

    return freed_files, freed_bytes


def _event_stamp_of(name: str) -> str | None:
    """The Tesla event timestamp in a folder or clip name, or None."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", name):
        return name
    match = TESLA_CLIP_RE.match(name)
    return match.group("stamp") if match else None


@dataclass
class WalkResult:
    """What a directory walk found, including what it could not read."""
    # Two different questions, and on exFAT they get two different answers:
    #   bytes_seen      - what the files CONTAIN (st_size)
    #   bytes_allocated - what the files OCCUPY  (st_blocks * 512)
    # A file may occupy a full cluster chain while its valid-data-length says
    # it contains almost nothing, which is what a preallocated spool looks
    # like. Totalling only st_size and then declaring the difference against
    # statvfs to be "space in no file at all" is a conclusion this walk had no
    # evidence for; it had only ever measured one of the two.
    bytes_seen: int = 0
    bytes_allocated: int = 0
    files: int = 0
    directories: int = 0
    errors: list[str] = field(default_factory=list)
    largest: list[tuple[int, Path]] = field(default_factory=list)


def walk_tree(path: Path, keep_largest: int = 0) -> WalkResult:
    """
    Total the bytes below `path`, *reporting* whatever it could not read.

    The reporting is the point. The first version of this silently `continue`d
    past every OSError, which meant a volume whose contents it could not stat
    came back looking nearly empty - and a diagnostic that under-counts without
    saying so is worse than no diagnostic, because it gets believed. Anything
    unreadable now comes back in `errors` and the caller cross-checks the total
    against statvfs.
    """
    result = WalkResult()
    if path.is_symlink():
        return result
    try:
        if path.is_file():
            info = path.stat()
            result.bytes_seen = info.st_size
            result.bytes_allocated = info.st_blocks * 512
            result.files = 1
            if keep_largest:
                result.largest = [(result.bytes_seen, path)]
            return result
    except OSError as error:
        result.errors.append(f"{path}: {error}")
        return result

    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError as error:
            result.errors.append(f"{current}/: {error}")
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    result.directories += 1
                    stack.append(entry)
                elif entry.is_file():
                    info = entry.stat()
                    size = info.st_size
                    result.files += 1
                    result.bytes_seen += size
                    result.bytes_allocated += info.st_blocks * 512
                    if keep_largest:
                        result.largest.append((size, entry))
            except OSError as error:
                result.errors.append(f"{entry}: {error}")

    if keep_largest:
        result.largest.sort(key=lambda item: item[0], reverse=True)
        del result.largest[keep_largest:]
    return result


def _entry_size(path: Path) -> int:
    """Bytes held by a file or below a directory. Symlink-safe."""
    return walk_tree(path).bytes_seen


def plan_reclaim(
    connection,
    device_id: str,
    root: Path,
    need_bytes: int,
    buckets: list[str] | None = None,
) -> list[tuple[str, Path, int]]:
    """
    Decide - without touching anything - which pre-existing events would have
    to go to free `need_bytes`. Returns (stamp, path, size) oldest first.

    Separated from the deletion so the decision can be inspected, tested and
    printed on its own. Every guard lives here; `reclaim_foreign` only carries
    out what this returns.

    The rules, all of which must hold for an entry to be eligible:
      * it is under TeslaCam/<bucket>/ and <bucket> is one of Tesla's three
        real bucket names AND is listed in NATIX_RECLAIM_BUCKETS
      * its name parses as a Tesla event stamp or clip name, so a stray
        directory is skipped rather than deleted
      * it is not a folder holding clips *we* mirrored - those are ours and
        prune_oldest owns them
      * it is not a symlink and not a dotfile
    """
    allowed = [
        bucket
        for bucket in (buckets if buckets is not None else NATIX_RECLAIM_BUCKETS)
        if bucket in TESLA_BUCKETS
    ]
    if not allowed:
        return []

    teslacam = root / "TeslaCam"
    if not teslacam.is_dir():
        return []

    ours = {
        row["event_folder"]
        for row in connection.execute(
            "SELECT DISTINCT event_folder FROM natix_mirror WHERE device_id=?",
            (device_id,),
        )
    }

    candidates: list[tuple[str, Path, int]] = []
    for bucket in allowed:
        bucket_dir = teslacam / bucket
        if not bucket_dir.is_dir():
            continue
        for entry in sorted(bucket_dir.iterdir()):
            if entry.is_symlink() or entry.name.startswith("."):
                continue
            stamp = _event_stamp_of(entry.name)
            if stamp is None:
                continue
            if entry.is_dir():
                if f"TeslaCam/{bucket}/{entry.name}" in ours:
                    continue
            elif not (entry.is_file() and entry.name.lower().endswith(".mp4")):
                # RecentClips is flat .mp4 files rather than event folders;
                # anything else in there is not ours to judge.
                continue
            candidates.append((stamp, entry, _entry_size(entry)))

    # Oldest recording first - sorting on the stamp, not on the path, so
    # "oldest" cannot come out meaning "first bucket alphabetically".
    candidates.sort(key=lambda item: item[0])

    shortfall = need_bytes - free_bytes(root)
    chosen: list[tuple[str, Path, int]] = []
    running = 0
    for stamp, path, size in candidates:
        if running >= shortfall:
            break
        chosen.append((stamp, path, size))
        running += size
    return chosen


def reclaim_foreign(
    connection,
    device_id: str,
    root: Path,
    need_bytes: int,
    buckets: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Carry out the plan from `plan_reclaim`: take over the loop-deletion the
    stick can no longer do for itself.

    Returns (entries removed, bytes freed, a line per entry for the log).
    Every deletion is logged by name and size, because this is the one place in
    the system that destroys somebody else's data and a silent version of it
    would be indefensible.
    """
    chosen = plan_reclaim(connection, device_id, root, need_bytes, buckets)
    removed = 0
    freed = 0
    log: list[str] = []

    for _stamp, path, size in chosen:
        label = f"{path.parent.name}/{path.name} ({size / 1024 ** 2:.0f}MB)"
        if dry_run:
            removed += 1
            freed += size
            log.append(f"would reclaim {label}")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as error:
            log.append(f"could not reclaim {label}: {error}")
            continue
        removed += 1
        freed += size
        log.append(f"reclaimed {label}")
        if free_bytes(root) >= need_bytes:
            break

    return (removed, freed, log)


def describe_space(root: Path) -> list[str]:
    """
    A read-only account of who is holding the stick's space.

    Called on the "no room" path, while the volume is still mounted, so a
    single run tells you both that it is full and what is filling it. Touches
    nothing.
    """
    lines: list[str] = []
    stats = os.statvfs(str(root))
    total = stats.f_blocks * stats.f_frsize
    free = stats.f_bavail * stats.f_frsize
    used = total - free
    lines.append(
        f"capacity {total / 1024 ** 3:.1f}GB, free {free / 1024 ** 3:.2f}GB "
        f"({100 * free / total:.1f}%), so {used / 1024 ** 3:.2f}GB is in use"
    )

    walked = 0
    occupied = 0
    all_errors: list[str] = []
    biggest: list[tuple[int, Path]] = []
    heavy: list[tuple[int, int, str]] = []   # (allocated, contained, name)
    for entry in sorted(root.iterdir()):
        if entry.is_symlink():
            continue
        found = walk_tree(entry, keep_largest=5)
        walked += found.bytes_seen
        occupied += found.bytes_allocated
        all_errors.extend(found.errors)
        biggest.extend(found.largest)
        heavy.append((found.bytes_allocated, found.bytes_seen, entry.name))
        slack_here = found.bytes_allocated - found.bytes_seen
        lines.append(
            f"  {found.bytes_seen / 1024 ** 3:7.2f}GB  {entry.name}"
            f"  ({found.files} files)"
            + (f"  [occupies {found.bytes_allocated / 1024 ** 3:.2f}GB]"
               if slack_here > 256 * 1024 ** 2 else "")
            + (f"  [{len(found.errors)} unreadable]" if found.errors else "")
        )

    # Reconcile - in both size dimensions, because they answer different
    # questions and only comparing both can tell the two faults apart:
    #
    #   contained (st_size)          what the files hold
    #   occupied  (st_blocks * 512)  what the files take up
    #
    # If `occupied` accounts for what statvfs calls used, the space is in real
    # files that simply do not contain it - a preallocated spool - and deleting
    # those files recovers it. If BOTH fall short, the clusters belong to no
    # directory entry at all and only a repair or reformat recovers them.
    #
    # This used to total st_size alone and then announce orphaned clusters as
    # the "most likely" cause. It could not have known that: it had measured
    # one dimension and drawn a conclusion that needed two.
    #
    # On exFAT specifically, expect the slack branch to stay quiet, and do not
    # read that as confirmation of anything. exFAT has no preallocation beyond
    # the file size the way ext4 fallocate does: the stream-extension entry's
    # DataLength *is* st_size, and the cluster chain covers it rounded up to
    # one cluster, so slack here is bounded by (files x cluster size) - tens of
    # megabytes on this volume, not tens of gigabytes. The branch is kept
    # because it costs one stat field and because being unable to tell the two
    # faults apart is precisely how the earlier version went wrong; it is not
    # kept because it is expected to fire. Note too that this is a FUSE mount,
    # and a FUSE driver may derive st_blocks from st_size rather than from the
    # real chain - in which case the two totals agree by construction and their
    # agreement is not evidence. natix_fsck.sh checks for exactly that and says
    # so. If both totals fall far short of statvfs on this device, leaked
    # clusters really are the remaining explanation - but now that is what the
    # measurement showed, rather than what the report assumed.
    threshold = 256 * 1024 * 1024
    lines.append(f"  {walked / 1024 ** 3:7.2f}GB  <- total the files contain")
    lines.append(f"  {occupied / 1024 ** 3:7.2f}GB  <- total the files occupy")
    slack = occupied - walked
    unaccounted = used - occupied

    if unaccounted > threshold:
        lines.append("")
        lines.append(
            f"  DISCREPANCY: {unaccounted / 1024 ** 3:.2f}GB is allocated to no file at all."
        )
        lines.append("  Not the files' contents, and not their cluster chains either.")
        lines.append("  That is leaked clusters: the allocation bitmap marks them used")
        lines.append("  but no directory entry points at them any more. This device")
        lines.append("  loses power mid-write every time the car sleeps and never gets")
        lines.append("  a clean unmount, which is exactly how that happens.")
        lines.append("  Deleting files cannot recover it, because there are no files.")
        lines.append("  Check it (read-only, changes nothing):")
        lines.append("    sudo ./scripts/natix_fsck.sh")
        lines.append("  then recover the space with:  sudo ./scripts/natix_fsck.sh --repair")
    elif slack > threshold:
        lines.append("")
        lines.append(
            f"  The space IS in files - they just do not contain it. "
            f"{slack / 1024 ** 3:.2f}GB is"
        )
        lines.append("  allocated to files beyond the bytes they actually hold, which is")
        lines.append("  what preallocated or sparsely-written clips look like: a spool")
        lines.append("  that reserved each clip at full size and never filled it in.")
        lines.append("  Nothing is corrupt, and fsck is right to call this volume clean.")
        lines.append("  Deleting the files that own those clusters recovers the space.")
        lines.append("  A repair will not - there is nothing for it to fix.")
        lines.append("")
        lines.append("  by what they occupy rather than contain:")
        for allocated, contained, name in sorted(heavy, reverse=True)[:6]:
            if allocated - contained <= threshold:
                continue
            lines.append(
                f"    {allocated / 1024 ** 3:6.2f}GB occupied "
                f"{contained / 1024 ** 3:6.2f}GB contained   {name}"
            )

    # Desktop operating systems leave sizeable droppings on removable media and
    # people rarely know they are there. Call them out with sizes rather than
    # deleting them: .Trashes in particular is data somebody deleted on a Mac
    # and may still expect to be able to recover.
    leftovers = [
        (name, size)
        for size, name in (
            (walk_tree(root / candidate).bytes_seen, candidate)
            for candidate in (".Trashes", ".Spotlight-V100", ".fseventsd",
                              "System Volume Information", "$RECYCLE.BIN")
            if (root / candidate).exists()
        )
    ]
    if leftovers:
        total_leftover = sum(size for _name, size in leftovers)
        lines.append("")
        lines.append(
            f"  desktop OS leftovers holding {total_leftover / 1024 ** 3:.2f}GB "
            f"(safe to remove, but they are yours to remove):"
        )
        for name, size in leftovers:
            lines.append(f"    {size / 1024 ** 3:6.2f}GB  {name}")

    if all_errors:
        lines.append("")
        lines.append(f"  {len(all_errors)} paths could not be read:")
        for error in all_errors[:10]:
            lines.append(f"    {error}")

    if biggest:
        biggest.sort(key=lambda item: item[0], reverse=True)
        lines.append("")
        lines.append("  largest files we can see:")
        for size, path in biggest[:10]:
            lines.append(f"    {size / 1024 ** 2:8.0f}MB  {path.relative_to(root)}")

    teslacam = root / "TeslaCam"
    if not teslacam.is_dir():
        lines.append("  no TeslaCam directory - this stick holds something else")
        return lines

    for bucket in sorted(p.name for p in teslacam.iterdir() if p.is_dir()):
        bucket_dir = teslacam / bucket
        entries = sorted(p.name for p in bucket_dir.iterdir())
        stamps = [name for name in entries if _event_stamp_of(name)]
        known = " (a Tesla bucket)" if bucket in TESLA_BUCKETS else " (NOT a Tesla bucket)"
        lines.append(
            f"  TeslaCam/{bucket}{known}: {_entry_size(bucket_dir) / 1024 ** 3:.2f}GB, "
            f"{len(entries)} entries, {len(stamps)} of them Tesla-shaped"
        )
        if stamps:
            lines.append(f"    oldest {stamps[0]}   newest {stamps[-1]}")
    return lines


def describe_reclaim_plan(
    connection, device_id: str, root: Path, need_bytes: int, buckets: list[str]
) -> list[str]:
    """
    Exactly what reclaiming would delete, without deleting it.

    The point is to let somebody approve a specific list of folders rather than
    a policy in the abstract. Same code path as the real thing - it calls
    plan_reclaim, so what it prints is what would go.
    """
    plan = plan_reclaim(connection, device_id, root, need_bytes, buckets)
    if not plan:
        return [
            f"nothing in {buckets} is eligible - either those buckets are empty, "
            f"hold nothing Tesla-shaped, or hold only clips we wrote"
        ]
    total = sum(size for _stamp, _path, size in plan)
    lines = [
        f"would delete {len(plan)} pre-existing events from {buckets}, "
        f"freeing {total / 1024 ** 3:.2f}GB:"
    ]
    for stamp, path, size in plan[:20]:
        lines.append(f"    {stamp}  {path.parent.name}/{path.name}  "
                     f"{size / 1024 ** 2:.0f}MB")
    if len(plan) > 20:
        lines.append(f"    ... and {len(plan) - 20} more")
    return lines


def mirror(
    connection,
    device_id: str,
    root: Path,
    limit: int = 0,
    dry_run: bool = False,
    extra_roots: Iterable[Path] = (),
    on_progress=None,
    should_stop=None,
) -> MirrorResult:
    """
    Put every not-yet-mirrored clip onto the stick at `root`.

    `root` is the stick's mountpoint. Everything below it is written under
    TeslaCam/<bucket>/<event>/ - byte-identical files, in the layout the car
    would have produced, because the stick's firmware is looking for exactly
    that and has no idea a Jetson is standing in for a Tesla.

    `on_progress(done, total, plan)` is called after each file. A full first
    mirror is several gigabytes over USB 2.0, which is minutes of apparent
    silence; the installer uses this to show that something is happening.
    """
    result = MirrorResult()
    plans, missing = build_plan(
        connection, device_id, limit=limit, extra_roots=extra_roots, root=root
    )
    result.missing = missing

    if not plans:
        return result

    reserve = NATIX_RESERVE_MB * 1024 * 1024
    events_touched: dict[str, list[MirrorPlan]] = {}
    total_planned = len(plans)
    done = 0

    for plan in plans:
        # Stop between files, not between passes. A pass is a whole batch, and
        # at the speed of USB 2.0 that is minutes - long enough that a
        # `systemctl stop` would hit TimeoutStopSec and be answered with
        # SIGKILL while a copy was still in flight. Between files the stick is
        # consistent and the unmount is clean, so this is where a stop belongs.
        if should_stop is not None and should_stop():
            result.stopped_reason = (
                f"asked to stop; {done} of {total_planned} clips copied this pass"
            )
            break

        target = root / plan.relative_dest

        if target.exists() and target.stat().st_size == plan.size_bytes:
            # Already there - most likely a previous run that died between the
            # rename and the database write. Record it and move on.
            result.skipped += 1
            if not dry_run:
                _record(connection, device_id, plan, state="done")
            continue

        required = plan.size_bytes + reserve
        if NATIX_IGNORE_FREE_SPACE:
            # Trust the filesystem, not its free-space accounting. A real
            # ENOSPC lands in the OSError handler below, which cleans up after
            # itself; a phantom one no longer blocks anything.
            pass
        elif free_bytes(root) < required:
            # Evict only to make room for something NEWER. Without this a
            # rolling window thrashes: it drops a recent clip to fit an older
            # one, then drops that to fit the recent one again. A window that
            # only ever moves forwards terminates; one that trades in both
            # directions does not.
            oldest_row = connection.execute(
                "SELECT MIN(event_folder) AS oldest FROM natix_mirror "
                "WHERE device_id=? AND state='done'",
                (device_id,),
            ).fetchone()
            oldest_on_stick = (oldest_row["oldest"] or "").rsplit("/", 1)[-1]
            if oldest_on_stick and plan.event <= oldest_on_stick:
                result.stopped_reason = (
                    f"stick is full and {plan.filename} is older than everything "
                    f"already on it (oldest there is {oldest_on_stick}); evicting "
                    f"newer footage to fit older would never settle. "
                    f"{result.copied} clips copied this pass"
                )
                break

            freed_files, freed_bytes_total = (0, 0)
            if not dry_run:
                freed_files, freed_bytes_total = prune_oldest(
                    connection, device_id, root, required
                )
            result.pruned += freed_files
            result.pruned_bytes += freed_bytes_total

            # Only once our own footage is exhausted do we consider theirs, and
            # only for buckets explicitly listed. On a stick that arrived full
            # from the car this is the step that makes any of it possible; with
            # the default empty list it is a no-op.
            if free_bytes(root) < required:
                taken, taken_bytes, taken_log = reclaim_foreign(
                    connection, device_id, root, required, dry_run=dry_run
                )
                result.reclaimed += taken
                result.reclaimed_bytes += taken_bytes
                result.reclaimed_detail.extend(taken_log)

            if free_bytes(root) < required:
                shortfall = required - free_bytes(root)
                result.stopped_reason = (
                    f"only {free_bytes(root) // (1024 * 1024)}MB free on the stick; "
                    f"need {required // (1024 * 1024)}MB for {plan.filename} "
                    f"(reserve is {NATIX_RESERVE_MB}MB, short by "
                    f"{shortfall // (1024 * 1024)}MB)"
                )
                if not NATIX_RECLAIM_BUCKETS:
                    result.stopped_reason += (
                        ". The stick holds footage we did not write and "
                        "NATIX_RECLAIM_BUCKETS is empty, so nothing was removed. "
                        "The VX360 only loop-deletes while the car writes to it, "
                        "which no longer happens here - so it will not free space "
                        "on its own. Inspect it with "
                        "'natix_probe.py --tree', then set e.g. "
                        "NATIX_RECLAIM_BUCKETS=SentryClips,RecentClips to let the "
                        "mirror take over the loop"
                    )
                break

        if dry_run:
            result.copied += 1
            result.copied_bytes += plan.size_bytes
            continue

        try:
            written = _copy_atomic(plan.source, target)
        except OSError as error:
            result.failed += 1
            result.errors.append(f"{plan.filename}: {error}")
            _record(connection, device_id, plan, state="failed", error=str(error))
            continue

        result.copied += 1
        result.copied_bytes += written
        events_touched.setdefault(plan.event, []).append(plan)
        _record(connection, device_id, plan, state="done")

        # Commit per file rather than per pass. A power cut halfway through a
        # 5GB first mirror should cost us the current file, not the record of
        # the two hundred that already landed.
        connection.commit()
        done += 1
        if on_progress is not None:
            on_progress(done, total_planned, plan)

    if NATIX_WRITE_EVENT_JSON and not dry_run:
        for event, event_plans in events_touched.items():
            event_dir = root / f"TeslaCam/{event_plans[0].bucket}/{event}"
            if event_dir.is_dir():
                try:
                    _write_event_json(event_dir, event, event_plans)
                except OSError as error:
                    result.errors.append(f"event.json for {event}: {error}")

    if not dry_run:
        _sync(root)
        connection.commit()
    return result


def _sync(root: Path) -> None:
    """Push the filesystem's cache to the flash before we walk away."""
    _run(["sync", "-f", str(root)], timeout=120)


def _record(connection, device_id: str, plan: MirrorPlan, state: str, error: str | None = None) -> None:
    connection.execute(
        "INSERT INTO natix_mirror "
        "(id, device_id, clip_id, filename, event_folder, bucket, dest_path, "
        " size_bytes, source_mtime, state, error, copied_ts) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET state=excluded.state, error=excluded.error, "
        "  dest_path=excluded.dest_path, size_bytes=excluded.size_bytes, "
        "  copied_ts=excluded.copied_ts",
        (
            f"{device_id}:{plan.clip_id}",
            device_id,
            plan.clip_id,
            plan.filename,
            f"TeslaCam/{plan.bucket}/{plan.event}",
            plan.bucket,
            plan.relative_dest,
            plan.size_bytes,
            int(plan.source.stat().st_mtime) if plan.source.exists() else None,
            state,
            error,
            int(time.time()),
        ),
    )


# ---------------------------------------------------------------------------
# Device bookkeeping
# ---------------------------------------------------------------------------
def register_device(connection, candidate: Candidate, mountpoint: Path | None = None) -> str:
    """Record that we have seen this stick, and return its stable id."""
    volume = candidate.volume
    device_id = candidate.device_id
    now = int(time.time())

    existing = connection.execute(
        "SELECT id, first_seen_ts, is_approved FROM natix_devices WHERE id=?", (device_id,)
    ).fetchone()

    free = None
    if mountpoint and is_mounted(Path(mountpoint)):
        free = free_bytes(Path(mountpoint))

    connection.execute(
        "INSERT INTO natix_devices "
        "(id, serial, volume_uuid, label, vendor, model, usb_vendor_id, usb_product_id, "
        " size_bytes, fstype, confidence, first_seen_ts, last_seen_ts, last_mount, free_bytes) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "  serial=excluded.serial, volume_uuid=excluded.volume_uuid, label=excluded.label, "
        "  vendor=excluded.vendor, model=excluded.model, usb_vendor_id=excluded.usb_vendor_id, "
        "  usb_product_id=excluded.usb_product_id, size_bytes=excluded.size_bytes, "
        "  fstype=excluded.fstype, confidence=excluded.confidence, "
        "  last_seen_ts=excluded.last_seen_ts, last_mount=excluded.last_mount, "
        "  free_bytes=excluded.free_bytes",
        (
            device_id,
            volume.serial,
            volume.volume_uuid,
            volume.label,
            volume.vendor,
            volume.model,
            volume.usb_vendor_id,
            volume.usb_product_id,
            volume.size_bytes,
            volume.fstype,
            candidate.confidence,
            existing["first_seen_ts"] if existing else now,
            now,
            str(mountpoint) if mountpoint else None,
            free,
        ),
    )
    connection.execute(
        "UPDATE natix_devices SET "
        "  mirrored_count = (SELECT COUNT(*) FROM natix_mirror "
        "                    WHERE device_id=? AND state='done'), "
        "  mirrored_bytes = (SELECT COALESCE(SUM(size_bytes),0) FROM natix_mirror "
        "                    WHERE device_id=? AND state='done') "
        "WHERE id=?",
        (device_id, device_id, device_id),
    )
    connection.commit()
    return device_id


def status(connection) -> dict[str, Any]:
    """
    Everything the dashboard needs to render the NATIX panel.

    The dashboard runs in a container and the worker runs on the host, so live
    block-device discovery may not be possible from where this is called - a
    container has no lsblk and no udev database even though it can see /sys.
    That is fine: the database is the interface between the two. When discovery
    is unavailable we report that honestly and fall back to what the worker
    last recorded, rather than showing an empty list that reads as "no stick
    attached".
    """
    discovery_available = True
    try:
        candidates = discover(include_all=False)
    except (OSError, FileNotFoundError):
        discovery_available = False
        candidates = []
    attached = [
        {
            "device_id": candidate.device_id,
            "path": candidate.volume.path,
            "confidence": candidate.confidence,
            "usable": candidate.usable,
            "size_gb": round(candidate.volume.size_gb, 1),
            "fstype": candidate.volume.fstype,
            "label": candidate.volume.label,
            "model": candidate.volume.model,
            "serial": candidate.volume.serial,
            "usb_id": (
                f"{candidate.volume.usb_vendor_id}:{candidate.volume.usb_product_id}"
                if candidate.volume.usb_vendor_id
                else None
            ),
            "mountpoint": candidate.volume.mountpoint,
            "reasons": candidate.reasons,
            "disqualifiers": candidate.disqualifiers,
        }
        for candidate in candidates
    ]

    known = [dict(row) for row in connection.execute(
        "SELECT * FROM natix_devices ORDER BY last_seen_ts DESC"
    )]
    total_clips = connection.execute("SELECT COUNT(*) AS n FROM clips").fetchone()["n"]

    for device in known:
        device["pending"] = max(0, total_clips - (device.get("mirrored_count") or 0))

    last_pass_raw = connection.execute(
        "SELECT value FROM settings WHERE key='natix_last_pass'"
    ).fetchone()
    try:
        last_pass = json.loads(last_pass_raw["value"]) if last_pass_raw else None
    except (json.JSONDecodeError, TypeError):
        last_pass = None

    return {
        "attached": attached,
        "known": known,
        "total_clips": total_clips,
        "last_pass": last_pass,
        "discovery_available": discovery_available,
        "min_confidence": NATIX_MIN_CONFIDENCE,
        "mountpoint": str(NATIX_MOUNTPOINT),
        "reserve_mb": NATIX_RESERVE_MB,
        "default_bucket": NATIX_DEFAULT_BUCKET,
    }
