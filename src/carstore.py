"""
carstore.py

Near-real-time footage from the car, without ever taking its drive away.

The old design exported the TESLACAM partition itself to the car as a USB
mass-storage LUN, which made the drive all-or-nothing: while the car had it the
Jetson could not read it, and footage arrived only when someone stopped the
export by hand. This is teslausb's answer instead
(github.com/marcone/teslausb, run/make_snapshot.sh):

  - The car records into an IMAGE FILE, cam_disk.img, that lives on a btrfs
    filesystem on a USB drive (the "backing store"). The USB gadget offers
    that file to the car, which sees an ordinary FAT32 stick.
  - Every CARSTORE_SNAPSHOT_SECONDS the image is copied with
    `cp --reflink=always`. On btrfs that copy shares every block with the
    original, so it takes a moment and almost no space, and the car keeps
    recording throughout - its writes simply stop being shared.
  - The copy is ours, so it can be repaired (fsck) and mounted read-only
    without the car noticing. Finished clips are copied into inbox/, newest
    first, and the copy is deleted.

teslausb formats its backing partition XFS; this kernel (5.15-tegra) has no
XFS and no dm-snapshot, but it has btrfs, which supports reflink copies the
same way. The image's own filesystem is FAT32 because the kernel cannot mount
exFAT and Tesla accepts FAT32 (service.tesla.com: exFAT, MS-DOS FAT, ext3 or
ext4; at least 64 GB; at least 4 MB/s sustained writes).

The backing store is not tied to one drive. Any USB drive can become it: the
dashboard's Storage page lists attached drives and asks this service, through
jobs/carstore-request.json, to format one. Formatting is the only destructive
thing this service does, and it re-checks every condition itself rather than
trusting the request - see validate_request.

Runs on the host as root (deploy/carstore.service): it mounts filesystems,
attaches loop devices and writes configfs, none of which a container should
be able to do. Its heartbeat, logs/carstore.json, is what the dashboard shows.

Configuration (environment, normally .env):
    CARSTORE_SNAPSHOT_SECONDS   how often to snapshot the car's image, default 60
    CARSTORE_IMAGE_GB           image size; 0 (default) = as large as the store allows
    CARSTORE_RESERVE_GB         backing-store space kept free for snapshots, default 4
    CARSTORE_MAX_AGE_HOURS      RecentClips older than this are skipped, default 2
    CARSTORE_MIN_FREE_GB        stop copying below this much free space, default 5
    CARSTORE_COPY_BUDGET_SECONDS  longest one cycle spends copying, default 40
    CARSTORE_MOUNT              where the backing store is mounted, default /backingfiles
    CARSTORE_GADGET_ROOT        configfs gadget root; tests point it elsewhere
    CARSTORE_UDC_ROOT           /sys/class/udc; tests point it elsewhere
"""
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.clipmeta import parse_clip_filename
from src.clipqueue import CLIP_SECONDS
from src.common import (
    DB_PATH,
    INBOX_DIR,
    JOBS_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
    ensure_dirs,
    env_int,
)
from src.ingest import safe_copy_to_inbox

BACKING_LABEL = "teslacam-backing"
IMAGE_LABEL = "TESLACAM"
TESLA_MIN_IMAGE_GB = 64
GB = 1000 ** 3

SNAPSHOT_SECONDS = env_int("CARSTORE_SNAPSHOT_SECONDS", 60)
IMAGE_GB = env_int("CARSTORE_IMAGE_GB", 0)
RESERVE_GB = env_int("CARSTORE_RESERVE_GB", 4)
MAX_AGE_HOURS = env_int("CARSTORE_MAX_AGE_HOURS", 2)
MIN_FREE_GB = env_int("CARSTORE_MIN_FREE_GB", 5)
# Longest a snapshot cycle spends copying. A first snapshot can hold hours of
# footage; copying it all in one go would stall the heartbeat and the next
# snapshot. The rest follows next cycle - newest first, so fresh clips win.
COPY_BUDGET_SECONDS = env_int("CARSTORE_COPY_BUDGET_SECONDS", 40)
LOOP_SECONDS = 5
REQUEST_MAX_AGE_SECONDS = 300

BACKING_MOUNT = Path(os.environ.get("CARSTORE_MOUNT", "/backingfiles"))
GADGET_ROOT = Path(os.environ.get("CARSTORE_GADGET_ROOT", "/sys/kernel/config/usb_gadget"))
UDC_ROOT = Path(os.environ.get("CARSTORE_UDC_ROOT", "/sys/class/udc"))
WORK_ROOT = Path(os.environ.get("CARSTORE_WORK_ROOT", "/run/carstore"))
GADGET_NAME = "g1"

STATUS_PATH = LOGS_DIR / "carstore.json"
REQUEST_PATH = JOBS_DIR / "carstore-request.json"

# Filesystems a TESLACAM partition can be read from here before it is
# formatted (the kernel has no exFAT).
RESCUABLE_FSTYPES = ("vfat", "ext4", "ext3", "ext2")
# Only an identified NATIX stick is protected. natix.py's "likely" is shape
# alone - any 64-2048 GB USB volume in a common filesystem - which would
# block the old TESLACAM drive and most SSDs.
NATIX_CONFIDENCES = ("strong", "pinned")
# Smallest drive or partition worth formatting. Below Tesla's 64 GB the page
# warns but allows it.
MIN_FORMAT_GB = 8


class CarstoreError(RuntimeError):
    """A step failed; the message says which and why."""


# ---------------------------------------------------------------------------
# Commands - one seam, so tests can replace the system underneath
# ---------------------------------------------------------------------------
def _run(command: list[str], timeout: int = 120, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                          check=False, input=input_text)


RUN: Callable[..., subprocess.CompletedProcess] = _run
DISK_USAGE = shutil.disk_usage


def must(command: list[str], timeout: int = 120, input_text: str | None = None,
         ok_codes: tuple[int, ...] = (0,)) -> str:
    """Run a command that has to work; its stdout, or a CarstoreError naming it."""
    result = RUN(command, timeout=timeout, input_text=input_text)
    if result.returncode not in ok_codes:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise CarstoreError(f"{' '.join(command[:2])} failed ({result.returncode}): "
                            f"{detail[-1] if detail else 'no output'}")
    return result.stdout


def log(message: str) -> None:
    print(f"[💾 carstore] {message}", flush=True)


# ---------------------------------------------------------------------------
# Drives: what is attached, and what each one may be used for
# ---------------------------------------------------------------------------
@dataclass
class Partition:
    path: str
    size_bytes: int
    fstype: str | None
    label: str | None
    mountpoint: str | None


@dataclass
class Drive:
    id: str
    path: str
    model: str
    size_bytes: int
    partitions: list[Partition] = field(default_factory=list)
    role: str = "available"
    blocked: str | None = None     # why it cannot be formatted, if it cannot


def drive_id(disk: dict) -> str:
    """
    An opaque, stable name for a drive.

    Derived from the serial when there is one, so it survives a replug; never
    the serial itself, which this service does not publish.
    """
    basis = disk.get("serial") or f"{disk.get('path')}|{disk.get('model')}|{disk.get('size')}"
    return hashlib.sha256(str(basis).encode()).hexdigest()[:12]


def inventory(tree: dict, protected_disks: set[str], natix_disks: set[str]) -> list[Drive]:
    """
    Every USB drive, with its role and whether it may be formatted.

    The card the OS runs from is never listed (protected_disks); the NATIX
    stick is listed but blocked, so the page can say why it is not offered.
    """
    drives: list[Drive] = []
    for disk in tree.get("blockdevices", []) or []:
        if disk.get("type") != "disk" or disk.get("tran") != "usb":
            continue
        name = disk.get("name") or ""
        if name in protected_disks:
            continue
        children = [child for child in disk.get("children", []) or [] if child.get("type") == "part"]
        nodes = children or ([disk] if disk.get("fstype") else [])
        drive = Drive(
            id=drive_id(disk),
            path=disk.get("path") or f"/dev/{name}",
            model=" ".join(filter(None, [(disk.get("vendor") or "").strip(),
                                         (disk.get("model") or "").strip()])) or "USB drive",
            size_bytes=int(disk.get("size") or 0),
            partitions=[Partition(
                path=node.get("path") or f"/dev/{node.get('name')}",
                size_bytes=int(node.get("size") or 0),
                fstype=node.get("fstype"),
                label=node.get("label"),
                mountpoint=node.get("mountpoint"),
            ) for node in nodes],
        )
        labels = {partition.label for partition in drive.partitions}
        foreign_mounts = [partition.mountpoint for partition in drive.partitions
                          if partition.mountpoint and Path(partition.mountpoint) != BACKING_MOUNT]
        if name in natix_disks:
            drive.role, drive.blocked = "NATIX stick", "this is the NATIX VX360 stick"
        elif BACKING_LABEL in labels:
            drive.role = "car storage"
        elif IMAGE_LABEL in labels:
            drive.role = "old TESLACAM drive"
        if foreign_mounts and not drive.blocked:
            drive.blocked = f"mounted at {', '.join(foreign_mounts)} - in use"
        drives.append(drive)
    return drives


def scan_drives() -> tuple[dict, list[Drive]]:
    """lsblk, the protected disks and the NATIX verdicts, from natix.py's discovery."""
    from src import natix
    tree = natix._lsblk_json()
    protected = natix._root_disk_names()
    natix_disks = {
        Path(volume.parent_path).name for volume in natix.list_volumes()
        if natix.evaluate(volume, protected).confidence in NATIX_CONFIDENCES
    }
    return tree, inventory(tree, protected, natix_disks)


def backing_partition(drives: list[Drive]) -> Partition | None:
    """The first btrfs partition labelled as the backing store."""
    for drive in drives:
        for partition in drive.partitions:
            if partition.label == BACKING_LABEL and partition.fstype == "btrfs":
                return partition
    return None


# ---------------------------------------------------------------------------
# Format requests from the dashboard
# ---------------------------------------------------------------------------
class RequestError(ValueError):
    """A request this service refuses; the message is shown on the page."""


def read_request(path: Path = REQUEST_PATH) -> dict | None:
    """Take the pending request, if any. It is consumed even when invalid."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {"invalid": True}
    path.unlink(missing_ok=True)
    return payload if isinstance(payload, dict) else {"invalid": True}


def validate_request(request: dict, drives: list[Drive], nonce: str,
                     now: float) -> tuple[Drive, Partition | None]:
    """
    Everything that has to be true before anything is erased - checked here,
    on the host, against the drives as they are NOW, not as the page saw them.
    """
    if request.get("action") != "format":
        raise RequestError("unknown request")
    if request.get("confirm") != "ERASE":
        raise RequestError("not confirmed - type ERASE to confirm")
    if request.get("nonce") != nonce:
        raise RequestError("the drive list changed since the page loaded - reload and try again")
    if not isinstance(request.get("ts"), (int, float)) or now - request["ts"] > REQUEST_MAX_AGE_SECONDS:
        raise RequestError("request expired - try again")
    drive = next((d for d in drives if d.id == request.get("drive_id")), None)
    if drive is None:
        raise RequestError("that drive is no longer attached")
    if drive.blocked:
        raise RequestError(f"cannot format: {drive.blocked}")
    partition = None
    if request.get("partition"):
        partition = next((p for p in drive.partitions if p.path == request["partition"]), None)
        if partition is None:
            raise RequestError("that partition is no longer on the drive")
    size = partition.size_bytes if partition else drive.size_bytes
    if size < MIN_FORMAT_GB * GB:
        raise RequestError(f"too small: {size / GB:.1f} GB (at least {MIN_FORMAT_GB} GB)")
    return drive, partition


def format_backing(drive: Drive, partition: Partition | None,
                   progress: Callable[[str], None]) -> dict:
    """
    Make `partition` (or the whole drive) the backing store. Destructive.

    A TESLACAM partition is read first and its clips that were never processed
    are copied into the inbox, so formatting the old drive loses no footage
    that can be read. Returns what happened, for the status page.
    """
    result: dict[str, Any] = {"rescued": 0, "rescue": None}
    if partition is not None and partition.label == IMAGE_LABEL:
        if partition.fstype in RESCUABLE_FSTYPES:
            progress("copying unprocessed clips off the old TESLACAM partition")
            result["rescued"], result["rescue"] = rescue_clips(
                partition, on_copy=lambda count: progress(f"copying unprocessed clips: {count} so far"))
        else:
            result["rescue"] = f"{partition.fstype or 'unknown'} cannot be read here - nothing copied"

    target = partition.path if partition else drive.path
    progress(f"erasing {target}")
    must(["wipefs", "-a", target])
    if partition is None:
        # One GPT partition spanning the drive, Linux type.
        must(["sfdisk", "--quiet", target], input_text="label: gpt\n,,L\n")
        must(["udevadm", "settle"])
        target = first_partition(target)
    progress(f"creating btrfs on {target}")
    must(["mkfs.btrfs", "-f", "-L", BACKING_LABEL, target], timeout=600)
    must(["udevadm", "settle"])
    result["formatted"] = target
    return result


def first_partition(disk: str) -> str:
    """The first partition node of `disk`, after the kernel has seen it."""
    names = must(["lsblk", "-ln", "-o", "PATH,TYPE", disk]).split("\n")
    for line in names:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "part":
            return parts[0]
    raise CarstoreError(f"no partition appeared on {disk}")


def rescue_clips(partition: Partition, on_copy: Callable[[int], None] | None = None) -> tuple[int, str]:
    """Copy never-processed clips off a partition about to be erased."""
    mountpoint = WORK_ROOT / "rescue"
    mountpoint.mkdir(parents=True, exist_ok=True)
    # noload: a read-only ext4 mount otherwise replays the journal - a write.
    options = "ro,noload" if (partition.fstype or "").startswith("ext") else "ro"
    must(["mount", "-o", options, partition.path, str(mountpoint)])
    try:
        copied = copy_clips(sorted_clips(mountpoint / "TeslaCam", now=time.time(), max_age_hours=None),
                            known_clip_names(), on_copy=on_copy)
    finally:
        RUN(["umount", str(mountpoint)])
    note = f"{copied['copied']} copied"
    if copied["paused"]:
        note += f"; stopped early: {copied['paused']}"
    return copied["copied"], note


# ---------------------------------------------------------------------------
# The car's image on the backing store
# ---------------------------------------------------------------------------
def image_path() -> Path:
    return BACKING_MOUNT / "cam_disk.img"


def is_mounted(path: Path) -> bool:
    return RUN(["findmnt", "-n", str(path)]).returncode == 0


def ensure_backing_mounted(partition: Partition) -> None:
    if is_mounted(BACKING_MOUNT):
        return
    BACKING_MOUNT.mkdir(parents=True, exist_ok=True)
    must(["mount", "-o", "noatime", partition.path, str(BACKING_MOUNT)])
    log(f"backing store {partition.path} mounted at {BACKING_MOUNT}")


def choose_image_bytes(total_bytes: int, free_bytes: int) -> int:
    """As large as the store allows less the snapshot reserve, or CARSTORE_IMAGE_GB."""
    available = free_bytes - RESERVE_GB * GB
    wanted = IMAGE_GB * GB if IMAGE_GB else available
    size = min(wanted, available)
    if size < 1 * GB:
        raise CarstoreError(f"backing store too small for an image ({free_bytes / GB:.1f} GB free)")
    return size - size % (1024 * 1024)


def ensure_image() -> Path:
    """
    Create the car's image if there is none: one FAT32 partition, labelled
    TESLACAM, with the TeslaCam folder Tesla requires. Built under a temporary
    name and renamed, so a half-made image is never offered to the car.
    """
    image = image_path()
    if image.exists():
        return image
    usage = DISK_USAGE(BACKING_MOUNT)
    size = choose_image_bytes(usage.total, usage.free)
    building = image.with_name(".building-" + image.name)
    building.unlink(missing_ok=True)
    log(f"creating {image} ({size / GB:.0f} GB, FAT32)")
    must(["fallocate", "-l", str(size), str(building)], timeout=600)
    must(["sfdisk", "--quiet", str(building)], input_text="label: dos\n,,c\n")
    loop = attach(building)
    try:
        must(["mkfs.vfat", "-F", "32", "-n", IMAGE_LABEL, f"{loop}p1"], timeout=600)
        mountpoint = WORK_ROOT / "newimage"
        mountpoint.mkdir(parents=True, exist_ok=True)
        must(["mount", f"{loop}p1", str(mountpoint)])
        try:
            (mountpoint / "TeslaCam").mkdir(exist_ok=True)
        finally:
            must(["umount", str(mountpoint)])
    finally:
        RUN(["losetup", "-d", loop])
    building.rename(image)
    return image


def attach(image: Path) -> str:
    """A loop device for `image`, with its partitions scanned."""
    return must(["losetup", "-P", "--show", "-f", str(image)]).strip()


# ---------------------------------------------------------------------------
# The USB gadget
# ---------------------------------------------------------------------------
def udc_name() -> str | None:
    names = sorted(path.name for path in UDC_ROOT.glob("*")) if UDC_ROOT.exists() else []
    return names[0] if names else None


def udc_state(udc: str | None) -> str | None:
    """'configured' once a host - the car - has enumerated the gadget."""
    if not udc:
        return None
    try:
        return (UDC_ROOT / udc / "state").read_text().strip()
    except OSError:
        return None


def _write(path: Path, value: str) -> None:
    path.write_text(value + "\n")


def exported_file() -> str | None:
    lun = GADGET_ROOT / GADGET_NAME / "functions/mass_storage.0/lun.0/file"
    try:
        return lun.read_text().strip() or None
    except OSError:
        return None


def export_image(image: Path) -> bool:
    """
    Offer `image` to the car. Returns True when it changed anything.

    Same identity and settings as teslausb's enable_gadget.sh and the old
    usb_mode_tesla.sh. Whatever else is bound to the UDC - at boot, NVIDIA's
    l4t demo gadget - is unbound first; a UDC serves one gadget.
    """
    udc = udc_name()
    if udc is None:
        raise CarstoreError("no USB device controller (UDC) - is the USB-C port in device mode?")
    gadget = GADGET_ROOT / GADGET_NAME
    if exported_file() == str(image) and _read(gadget / "UDC") == udc:
        return False

    for other in GADGET_ROOT.glob("*"):
        if other.name != GADGET_NAME and _read(other / "UDC") == udc:
            _write(other / "UDC", "")
            log(f"unbound gadget {other.name} from {udc}")

    lun = gadget / "functions/mass_storage.0/lun.0"
    if not gadget.exists():
        gadget.mkdir(parents=True)
        for name, value in (("idVendor", "0x1d6b"), ("idProduct", "0x0104"),
                            ("bcdDevice", "0x0100"), ("bcdUSB", "0x0200")):
            _write(gadget / name, value)
        (gadget / "strings/0x409").mkdir(parents=True, exist_ok=True)
        _write(gadget / "strings/0x409/manufacturer", "tesla-vision")
        _write(gadget / "strings/0x409/product", "TeslaCam")
        (gadget / "configs/c.1/strings/0x409").mkdir(parents=True, exist_ok=True)
        _write(gadget / "configs/c.1/strings/0x409/configuration", "TeslaCam")
        lun.mkdir(parents=True, exist_ok=True)
    link = gadget / "configs/c.1/mass_storage.0"
    if not link.exists():
        link.symlink_to(gadget / "functions/mass_storage.0")

    if _read(gadget / "UDC"):
        _write(gadget / "UDC", "")
    _write(lun / "file", str(image))
    _write(lun / "inquiry_string", "TeslaCam")
    _write(gadget / "UDC", udc)
    log(f"exporting {image} to the car on {udc}")
    return True


def stop_export() -> None:
    gadget = GADGET_ROOT / GADGET_NAME
    if _read(gadget / "UDC"):
        _write(gadget / "UDC", "")
    lun_file = gadget / "functions/mass_storage.0/lun.0/file"
    if lun_file.exists():
        _write(lun_file, "")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Snapshots and clips
# ---------------------------------------------------------------------------
def mp4_is_complete(path: Path) -> bool:
    """
    True when the file is a whole MP4: top-level boxes that tile the file
    exactly, one of them a moov. A clip still being written - or cut short in
    a crash-consistent snapshot - fails this, and is left for the next pass.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            offset, saw_moov = 0, False
            while offset < size:
                handle.seek(offset)
                header = handle.read(16)
                if len(header) < 8:
                    return False
                box_size = int.from_bytes(header[:4], "big")
                box_type = header[4:8]
                if box_size == 1:
                    if len(header) < 16:
                        return False
                    box_size = int.from_bytes(header[8:16], "big")
                elif box_size == 0:
                    box_size = size - offset
                if box_size < 8:
                    return False
                saw_moov = saw_moov or box_type == b"moov"
                offset += box_size
            return saw_moov and offset == size
    except OSError:
        return False


def sorted_clips(root: Path, now: float, max_age_hours: float | None = MAX_AGE_HOURS) -> list[Path]:
    """
    Clips worth copying, newest first.

    The newest clip of each camera in each folder is the one the car may still
    be writing, so it waits until it is CLIP_SECONDS*2 old. RecentClips older
    than max_age_hours are not real time any more and are skipped; saved and
    sentry events are kept whatever their age.
    """
    found: list[tuple[int, tuple[str, str], Path]] = []
    newest: dict[tuple[str, str], int] = {}
    if not root.is_dir():
        return []
    for clip in root.rglob("*.mp4"):
        meta = parse_clip_filename(clip.name)
        captured = meta.get("captured_ts")
        if captured is None:
            continue
        if (max_age_hours is not None and "RecentClips" in clip.parts
                and now - captured > max_age_hours * 3600):
            continue
        key = (str(clip.parent), meta.get("camera") or "")
        newest[key] = max(newest.get(key, captured), captured)
        found.append((captured, key, clip))
    kept = [(captured, clip) for captured, key, clip in found
            if captured < newest[key] or now - captured >= CLIP_SECONDS * 2]
    return [clip for _, clip in sorted(kept, key=lambda pair: pair[0], reverse=True)]


def known_clip_names() -> Callable[[str], bool]:
    """Whether a clip name is already in the pipeline: inbox, processed/ or the clips table."""
    names: set[str] = set()
    try:
        connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
        try:
            names = {row[0] for row in connection.execute("SELECT filename FROM clips")}
        finally:
            connection.close()
    except sqlite3.Error:
        pass

    def known(name: str) -> bool:
        return (name in names or (INBOX_DIR / name).exists() or (PROCESSED_DIR / name).exists())
    return known


def copy_clips(clips: list[Path], known: Callable[[str], bool],
               free_bytes: Callable[[], int] | None = None, deadline: float | None = None,
               on_copy: Callable[[int], None] | None = None) -> dict:
    """
    Copy each unknown, complete clip into the inbox, until space runs low or
    the deadline passes. on_copy(count) runs after each copy, so the caller
    can keep its heartbeat fresh through a long copy.
    """
    free_bytes = free_bytes or (lambda: DISK_USAGE(INBOX_DIR).free)
    outcome: dict[str, Any] = {"copied": 0, "incomplete": 0, "newest_ts": None, "paused": None,
                               "deferred": False}
    for clip in clips:
        if known(clip.name):
            continue
        if deadline is not None and time.time() > deadline:
            outcome["deferred"] = True
            break
        if free_bytes() < MIN_FREE_GB * GB:
            outcome["paused"] = f"less than {MIN_FREE_GB} GB free on the Jetson"
            break
        if not mp4_is_complete(clip):
            outcome["incomplete"] += 1
            continue
        if safe_copy_to_inbox(clip, require_stable=False):
            outcome["copied"] += 1
            if on_copy:
                on_copy(outcome["copied"])
            captured = parse_clip_filename(clip.name).get("captured_ts")
            if captured and (outcome["newest_ts"] is None or captured > outcome["newest_ts"]):
                outcome["newest_ts"] = captured
    return outcome


def fsck_command(fstype: str | None, device: str) -> tuple[list[str], tuple[int, ...]] | None:
    """
    Repair the snapshot copy - ours, so automatic repair is safe - and the
    exit codes that mean "clean" or "errors corrected". fsck.vfat's 2 is a
    usage error; e2fsck's 2 is "corrected, reboot advised", which is fine here.
    """
    if fstype == "vfat":
        return ["fsck.vfat", "-a", device], (0, 1)
    if fstype in ("ext4", "ext3", "ext2"):
        return ["e2fsck", "-p", device], (0, 1, 2)
    return None


def snapshot_and_copy(image: Path, now: float, on_copy: Callable[[int], None] | None = None) -> dict:
    """One cycle: reflink snapshot, repair, mount read-only, copy, clean up."""
    snapshots = BACKING_MOUNT / "snapshots"
    snapshots.mkdir(exist_ok=True)
    snap = snapshots / "snap.img"
    mountpoint = WORK_ROOT / "snapshot"
    mountpoint.mkdir(parents=True, exist_ok=True)
    cleanup_snapshot(snap, mountpoint)

    started = time.time()
    must(["cp", "--reflink=always", str(image), str(snap)])
    loop = attach(snap)
    try:
        device = f"{loop}p1"
        fstype = must(["blkid", "-o", "value", "-s", "TYPE", device], ok_codes=(0, 2)).strip() or None
        repair = fsck_command(fstype, device)
        if repair is None:
            raise CarstoreError(f"the car's image holds {fstype or 'no filesystem'}, which cannot be "
                                "read here - was it reformatted in the car? Format it again from Storage.")
        must(repair[0], timeout=300, ok_codes=repair[1])
        options = "ro,noload" if fstype.startswith("ext") else "ro"
        must(["mount", "-o", options, device, str(mountpoint)])
        try:
            outcome = copy_clips(sorted_clips(mountpoint / "TeslaCam", now=now), known_clip_names(),
                                 deadline=time.time() + COPY_BUDGET_SECONDS, on_copy=on_copy)
        finally:
            RUN(["umount", str(mountpoint)])
    finally:
        RUN(["losetup", "-d", loop])
        snap.unlink(missing_ok=True)
    outcome["fstype"] = fstype
    outcome["seconds"] = round(time.time() - started, 1)
    return outcome


def cleanup_snapshot(snap: Path, mountpoint: Path) -> None:
    """Undo whatever a crashed cycle left: the mount, its loop device, the copy."""
    if is_mounted(mountpoint):
        RUN(["umount", str(mountpoint)])
    if snap.exists():
        for line in must(["losetup", "-j", str(snap)], ok_codes=(0, 1)).splitlines():
            RUN(["losetup", "-d", line.split(":", 1)[0]])
        snap.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
def write_status(status: dict, path: Path = STATUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status["written_ts"] = int(time.time())
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status), encoding="utf-8")
    os.replace(temporary, path)


class Service:
    """The loop: drives, requests, the backing store, the export, snapshots."""

    def __init__(self) -> None:
        self.nonce = secrets.token_hex(8)
        self.drive_signature = ""
        self.last_snapshot = 0.0
        self.status: dict[str, Any] = {
            "schema": 1, "pid": os.getpid(), "started_ts": int(time.time()),
            "state": "starting", "reason": None, "nonce": self.nonce,
            "drives": [], "backing": None, "export": None, "job": None,
            "snapshot": {"last_ts": None, "seconds": None, "copied_last": 0, "copied_total": 0,
                         "incomplete_last": 0, "newest_clip_ts": None, "paused": None,
                         "deferred": False, "copying": None, "error": None, "fstype": None},
            "settings": {"snapshot_seconds": SNAPSHOT_SECONDS, "max_age_hours": MAX_AGE_HOURS,
                         "min_free_gb": MIN_FREE_GB, "tesla_min_image_gb": TESLA_MIN_IMAGE_GB},
        }

    def publish(self, drives: list[Drive]) -> None:
        signature = json.dumps([asdict(drive) for drive in drives], sort_keys=True)
        if signature != self.drive_signature:
            # A changed drive list invalidates any request made against the old one.
            self.drive_signature = signature
            self.nonce = secrets.token_hex(8)
        self.status["nonce"] = self.nonce
        self.status["drives"] = [asdict(drive) for drive in drives]

    def handle_request(self, request: dict, drives: list[Drive]) -> None:
        job = {"state": "running", "started_ts": int(time.time()), "finished_ts": None,
               "drive_id": request.get("drive_id"), "partition": request.get("partition"),
               "step": "checking the request", "message": None}
        self.status["job"] = job
        write_status(self.status)
        try:
            drive, partition = validate_request(request, drives, self.nonce, time.time())
        except RequestError as error:
            job.update(state="refused", message=str(error), finished_ts=int(time.time()))
            log(f"format request refused: {error}")
            return

        def progress(step: str) -> None:
            job["step"] = step
            log(step)
            write_status(self.status)

        try:
            if any(p.label == BACKING_LABEL for p in drive.partitions) or (
                    partition is not None and partition.mountpoint):
                progress("stopping the car's export of this drive")
                stop_export()
                RUN(["umount", str(BACKING_MOUNT)])
            result = format_backing(drive, partition, progress)
            job.update(state="done", step="formatted", finished_ts=int(time.time()),
                       message=f"formatted {result['formatted']} as car storage"
                               + (f"; old clips: {result['rescue']}" if result["rescue"] else ""))
        except (CarstoreError, OSError, subprocess.SubprocessError) as error:
            job.update(state="failed", message=str(error), finished_ts=int(time.time()))
            log(f"format failed: {error}")

    def tick(self, now: float) -> None:
        tree, drives = scan_drives()
        self.publish(drives)
        request = read_request()
        if request is not None:
            self.handle_request(request, drives)
            tree, drives = scan_drives()
            self.publish(drives)

        partition = backing_partition(drives)
        if partition is None:
            if exported_file() == str(image_path()):
                # The drive went away under the car: stop offering a dead file.
                stop_export()
                RUN(["umount", "-l", str(BACKING_MOUNT)])
            self.status.update(state="no_storage", backing=None,
                               reason="no car storage drive - format one on the Storage page")
            self.status["export"] = {"active": False, "udc_state": udc_state(udc_name())}
            return

        ensure_backing_mounted(partition)
        image = ensure_image()
        export_image(image)
        usage = DISK_USAGE(BACKING_MOUNT)
        image_bytes = image.stat().st_size
        udc = udc_name()
        self.status["backing"] = {
            "device": partition.path, "size_bytes": usage.total, "free_bytes": usage.free,
            "image_bytes": image_bytes,
            "below_tesla_minimum": image_bytes < TESLA_MIN_IMAGE_GB * GB,
        }
        self.status["export"] = {"active": exported_file() == str(image), "udc": udc,
                                 "udc_state": udc_state(udc),
                                 "car_connected": udc_state(udc) == "configured"}
        self.status.update(state="exporting", reason=None)

        if now - self.last_snapshot >= SNAPSHOT_SECONDS:
            self.last_snapshot = now
            snap = self.status["snapshot"]
            try:
                def heartbeat(count: int) -> None:
                    snap["copying"] = count
                    write_status(self.status)

                outcome = snapshot_and_copy(image, now, on_copy=heartbeat)
                snap.update(last_ts=int(now), seconds=outcome["seconds"], copied_last=outcome["copied"],
                            incomplete_last=outcome["incomplete"], paused=outcome["paused"],
                            deferred=outcome["deferred"], fstype=outcome["fstype"], error=None,
                            copying=None)
                snap["copied_total"] += outcome["copied"]
                if outcome["newest_ts"] and (snap["newest_clip_ts"] or 0) < outcome["newest_ts"]:
                    snap["newest_clip_ts"] = outcome["newest_ts"]
                if outcome["copied"]:
                    log(f"snapshot: {outcome['copied']} clip(s) -> inbox in {outcome['seconds']}s")
            except (CarstoreError, OSError, subprocess.SubprocessError) as error:
                snap.update(last_ts=int(now), error=str(error))
                log(f"snapshot failed: {error}")


def main() -> int:  # pragma: no cover - the loop around tick(), exercised on the Jetson
    ensure_dirs()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"backing store {BACKING_MOUNT}, snapshot every {SNAPSHOT_SECONDS}s, inbox {INBOX_DIR}")
    service = Service()
    while True:
        try:
            service.tick(time.time())
        except (CarstoreError, OSError, subprocess.SubprocessError) as error:
            service.status.update(state="error", reason=str(error))
            log(f"error: {error}")
        write_status(service.status)
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
