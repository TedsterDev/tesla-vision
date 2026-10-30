"""
test_carstore.py

carstore.py turns a USB drive into the car's recording drive and reads the
car's footage from snapshots of it. Two failures matter far more than any
other:

  - erasing the wrong thing. Every refusal in validate_request and every
    exclusion in inventory is pinned here, because a format that runs when it
    should not is not recoverable.
  - footage that never arrives, silently. A clip the car is still writing must
    wait; a finished one must arrive exactly once, newest first.

The commands underneath (mount, losetup, cp --reflink, fsck, mkfs) are
replaced by FakeSystem, which records them and simulates their effects inside
a sandbox: mounting a snapshot "shows" a TeslaCam tree, configfs is a plain
directory.

Run:  python3 tests/test_carstore.py
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SANDBOX = Path(tempfile.mkdtemp(prefix="carstore_test_"))
os.environ["BASE_DIR"] = str(SANDBOX / "data")
os.environ["DB_PATH"] = str(SANDBOX / "data" / "scout.db")
os.environ["CARSTORE_MOUNT"] = str(SANDBOX / "backingfiles")
os.environ["CARSTORE_GADGET_ROOT"] = str(SANDBOX / "configfs")
os.environ["CARSTORE_UDC_ROOT"] = str(SANDBOX / "udc")
os.environ["CARSTORE_WORK_ROOT"] = str(SANDBOX / "run")

from src import carstore                                   # noqa: E402
from src.clipmeta import parse_clip_filename                # noqa: E402
from src.common import INBOX_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402

ensure_dirs()
GB = 1000 ** 3
Usage = namedtuple("Usage", "total used free")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def completed(code=0, out="", err=""):
    return subprocess.CompletedProcess([], code, out, err)


class FakeSystem:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.mounted: dict[str, str] = {}
        self.fail: dict[tuple, tuple[int, str]] = {}
        self.image_tree: dict[str, bytes] = {}
        self.fstype: str | None = "vfat"
        self.partitions_listing = "/dev/sdb disk\n/dev/sdb1 part\n"
        self.loops_for_snapshot = ""

    def ran(self, *prefix) -> list[list[str]]:
        return [call for call in self.calls if tuple(call[:len(prefix)]) == prefix]

    def __call__(self, command, timeout=120, input_text=None):
        self.calls.append(list(command))
        for prefix, (code, err) in self.fail.items():
            if tuple(command[:len(prefix)]) == prefix:
                return completed(code, "", err)
        name = command[0]
        if name == "findmnt":
            return completed(0 if command[-1] in self.mounted else 1)
        if name == "mount":
            device, mountpoint = command[-2], command[-1]
            self.mounted[mountpoint] = device
            Path(mountpoint).mkdir(parents=True, exist_ok=True)
            if mountpoint.endswith(("snapshot", "rescue")):
                for relative, data in self.image_tree.items():
                    target = Path(mountpoint) / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            return completed()
        if name == "umount":
            mountpoint = command[-1]
            self.mounted.pop(mountpoint, None)
            if mountpoint.endswith(("snapshot", "rescue", "newimage")):
                shutil.rmtree(mountpoint, ignore_errors=True)
                Path(mountpoint).mkdir(parents=True, exist_ok=True)
            return completed()
        if name == "cp":
            shutil.copyfile(command[-2], command[-1])
            return completed()
        if name == "losetup":
            if "--show" in command:
                return completed(0, "/dev/loop7\n")
            if "-j" in command:
                return completed(0 if self.loops_for_snapshot else 1, self.loops_for_snapshot)
            return completed()
        if name == "blkid":
            return completed(0 if self.fstype else 2, (self.fstype or "") + "\n")
        if name == "fallocate":
            Path(command[-1]).write_bytes(b"\0" * 64)
            return completed()
        if name == "lsblk":
            return completed(0, self.partitions_listing)
        return completed()


def install(fake: FakeSystem):
    carstore.RUN = fake
    return fake


def box(kind: bytes, payload: bytes) -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


COMPLETE_MP4 = box(b"ftyp", b"isom0000") + box(b"moov", b"m" * 24) + box(b"mdat", b"d" * 40)
UNFINISHED_MP4 = box(b"ftyp", b"isom0000") + (1000).to_bytes(4, "big") + b"mdat" + b"d" * 40


def ts(name: str) -> int:
    return parse_clip_filename(name)["captured_ts"]


def reset_dirs():
    for directory in (INBOX_DIR, PROCESSED_DIR, carstore.BACKING_MOUNT, carstore.GADGET_ROOT,
                      carstore.UDC_ROOT, carstore.WORK_ROOT):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
    Path(os.environ["DB_PATH"]).unlink(missing_ok=True)
    carstore.REQUEST_PATH.unlink(missing_ok=True)


def disk(name, size_gb, children=(), tran="usb", serial="SER123", fstype=None, model="Extreme"):
    return {"name": name, "path": f"/dev/{name}", "type": "disk", "tran": tran, "size": int(size_gb * GB),
            "serial": serial, "vendor": "SanDisk ", "model": model, "fstype": fstype,
            "children": list(children)}


def part(name, size_gb, fstype=None, label=None, mountpoint=None):
    return {"name": name, "path": f"/dev/{name}", "type": "part", "size": int(size_gb * GB),
            "fstype": fstype, "label": label, "mountpoint": mountpoint}


def drives_from(*disks, protected=(), natix=()):
    return carstore.inventory({"blockdevices": list(disks)}, set(protected), set(natix))


# ---------------------------------------------------------------------------
# Which drives may be formatted
# ---------------------------------------------------------------------------
def test_the_os_card_and_internal_disks_are_never_listed():
    drives = drives_from(disk("mmcblk0", 119, tran=None), disk("sda", 128),
                         disk("sdc", 128, serial="C"), protected=("sdc",))
    assert [d.path for d in drives] == ["/dev/sda"], drives


def test_the_natix_stick_is_listed_but_never_offered():
    drives = drives_from(disk("sda", 35, [part("sda1", 35, "exfat", "VX360")]), natix=("sda",))
    assert drives[0].role == "NATIX stick" and "NATIX" in drives[0].blocked


def test_roles_follow_the_labels():
    drives = drives_from(
        disk("sda", 128, [part("sda1", 128, "btrfs", "teslacam-backing", str(carstore.BACKING_MOUNT))]),
        disk("sdb", 250, [part("sdb1", 120, "ext4", "TESLACAM"), part("sdb2", 130, "ext4", "jetsondata")],
             serial="OTHER"),
        disk("sdc", 64, serial=None))
    roles = {d.path: (d.role, d.blocked) for d in drives}
    # Mounted at the backing store's own mountpoint is not "in use" elsewhere.
    assert roles["/dev/sda"] == ("car storage", None), roles
    assert roles["/dev/sdb"] == ("old TESLACAM drive", None), roles
    assert roles["/dev/sdc"] == ("available", None), roles
    assert drives[2].partitions == []


def test_a_drive_mounted_somewhere_else_is_in_use():
    drives = drives_from(disk("sda", 128, [part("sda1", 128, "ext4", "data", "/mnt/jetsondata")]))
    assert "in use" in drives[0].blocked and "/mnt/jetsondata" in drives[0].blocked


def test_a_formatted_stick_without_partitions_is_its_own_partition():
    drives = drives_from(disk("sda", 64, fstype="vfat"))
    assert [p.path for p in drives[0].partitions] == ["/dev/sda"]


def test_a_drive_id_is_stable_and_never_the_serial():
    first = carstore.drive_id({"serial": "SER123"})
    assert first == carstore.drive_id({"serial": "SER123"}) and len(first) == 12
    assert "SER123" not in first
    assert carstore.drive_id({"path": "/dev/sda", "model": "x", "size": 1}) != first
    drives = drives_from(disk("sda", 128, serial="SER123"))
    assert "SER123" not in json.dumps([d.__dict__ for d in drives], default=str)


def test_scan_drives_uses_natix_discovery_and_only_trusts_an_identified_stick():
    from src import natix
    tree = {"blockdevices": [disk("sda", 35, [part("sda1", 35, "exfat", "VX360")]),
                             disk("sdb", 128, [part("sdb1", 128, "ext4", "TESLACAM")], serial="B")]}
    volumes = [natix.Volume(path="/dev/sda1", parent_path="/dev/sda"),
               natix.Volume(path="/dev/sdb1", parent_path="/dev/sdb")]
    verdicts = {"/dev/sda1": "strong", "/dev/sdb1": "likely"}
    saved = (natix._lsblk_json, natix._root_disk_names, natix.list_volumes, natix.evaluate)
    natix._lsblk_json = lambda: tree
    natix._root_disk_names = lambda: {"mmcblk0"}
    natix.list_volumes = lambda: volumes
    natix.evaluate = lambda volume, protected: natix.Candidate(volume=volume, confidence=verdicts[volume.path])
    try:
        _, drives = carstore.scan_drives()
    finally:
        natix._lsblk_json, natix._root_disk_names, natix.list_volumes, natix.evaluate = saved
    by_path = {d.path: d for d in drives}
    assert by_path["/dev/sda"].role == "NATIX stick"
    # "likely" is shape alone: it must not block the old TESLACAM drive.
    assert by_path["/dev/sdb"].blocked is None


def test_the_backing_store_is_found_only_as_btrfs():
    drives = drives_from(disk("sda", 128, [part("sda1", 128, "ext4", "teslacam-backing")]),
                         disk("sdb", 128, [part("sdb1", 128, "btrfs", "teslacam-backing")], serial="B"))
    assert carstore.backing_partition(drives).path == "/dev/sdb1"
    assert carstore.backing_partition(drives[:1]) is None


# ---------------------------------------------------------------------------
# Format requests
# ---------------------------------------------------------------------------
def request_for(drive, partition=None, **changes):
    request = {"action": "format", "confirm": "ERASE", "nonce": "n1", "ts": time.time(),
               "drive_id": drive.id, "partition": partition}
    request.update(changes)
    return request


def test_requests_are_consumed_even_when_unreadable():
    assert carstore.read_request() is None
    carstore.REQUEST_PATH.write_text("{not json")
    assert carstore.read_request() == {"invalid": True}
    assert not carstore.REQUEST_PATH.exists()
    carstore.REQUEST_PATH.write_text("[1, 2]")
    assert carstore.read_request() == {"invalid": True}
    carstore.REQUEST_PATH.write_text(json.dumps({"action": "format"}))
    assert carstore.read_request() == {"action": "format"}


def test_every_unsafe_request_is_refused_with_a_reason():
    drives = drives_from(
        disk("sda", 128, [part("sda1", 128, "ext4", "TESLACAM")]),
        disk("sdb", 35, [part("sdb1", 35, "exfat", "VX360")], serial="N"),
        disk("sdc", 4, serial="S"), natix=("sdb",))
    good, natix_stick, tiny = drives
    cases = [
        (request_for(good, action="wipe"), "unknown request"),
        (request_for(good, confirm="erase"), "type ERASE"),
        (request_for(good, nonce="old"), "drive list changed"),
        (request_for(good, ts=None), "expired"),
        (request_for(good, ts=time.time() - 3600), "expired"),
        (request_for(good, drive_id="000000000000"), "no longer attached"),
        (request_for(natix_stick), "NATIX"),
        (request_for(good, partition="/dev/sda9"), "partition is no longer"),
        (request_for(tiny), "too small"),
    ]
    for request, expected in cases:
        try:
            carstore.validate_request(request, drives, "n1", time.time())
            raise AssertionError(f"accepted: {request}")
        except carstore.RequestError as error:
            assert expected in str(error), (expected, str(error))


def test_a_good_request_names_exactly_what_will_be_erased():
    drives = drives_from(disk("sda", 128, [part("sda1", 64, "ext4", "TESLACAM"), part("sda2", 64, "ext4")]))
    drive, partition = carstore.validate_request(request_for(drives[0], "/dev/sda1"), drives, "n1", time.time())
    assert partition.path == "/dev/sda1"
    drive, partition = carstore.validate_request(request_for(drives[0]), drives, "n1", time.time())
    assert drive.path == "/dev/sda" and partition is None


def test_formatting_a_teslacam_partition_copies_its_unprocessed_clips_first():
    reset_dirs()
    fake = install(FakeSystem())
    name = "2026-10-29_08-00-00-front.mp4"
    fake.image_tree = {f"TeslaCam/SavedClips/2026-10-29_08-05-00/{name}": COMPLETE_MP4}
    drives = drives_from(disk("sda", 128, [part("sda1", 128, "ext4", "TESLACAM"), part("sda2", 10, "ext4")]))
    steps = []
    result = carstore.format_backing(drives[0], drives[0].partitions[0], steps.append)
    assert (INBOX_DIR / name).exists(), list(INBOX_DIR.iterdir())
    assert result["rescued"] == 1 and result["formatted"] == "/dev/sda1"
    # ext4 is mounted without journal replay: read-only must mean no writes.
    assert ["mount", "-o", "ro,noload", "/dev/sda1", str(carstore.WORK_ROOT / "rescue")] in fake.calls
    assert fake.ran("wipefs", "-a", "/dev/sda1") and fake.ran("mkfs.btrfs", "-f", "-L", "teslacam-backing", "/dev/sda1")
    assert not fake.ran("wipefs", "-a", "/dev/sda"), "the other partition must survive"
    assert steps[0].startswith("copying unprocessed clips")
    assert "copying unprocessed clips: 1 so far" in steps, steps


def test_an_unreadable_teslacam_partition_is_said_so_not_skipped_silently():
    reset_dirs()
    fake = install(FakeSystem())
    drives = drives_from(disk("sda", 128, [part("sda1", 128, "exfat", "TESLACAM")]))
    result = carstore.format_backing(drives[0], drives[0].partitions[0], lambda step: None)
    assert "cannot be read" in result["rescue"] and result["rescued"] == 0
    assert not fake.ran("mount")
    drives = drives_from(disk("sda", 128, [part("sda1", 128, None, "TESLACAM")]))
    assert "unknown" in carstore.format_backing(drives[0], drives[0].partitions[0], lambda s: None)["rescue"]


def test_formatting_a_whole_drive_partitions_it_first():
    reset_dirs()
    fake = install(FakeSystem())
    drives = drives_from(disk("sdb", 256, [part("sdb1", 256, "ext4", "stuff")]))
    result = carstore.format_backing(drives[0], None, lambda step: None)
    assert result == {"rescued": 0, "rescue": None, "formatted": "/dev/sdb1"}
    assert fake.ran("sfdisk", "--quiet", "/dev/sdb") and fake.ran("wipefs", "-a", "/dev/sdb")


def test_a_partition_that_never_appears_fails_the_format():
    install(FakeSystem()).partitions_listing = "/dev/sdb disk\n"
    try:
        carstore.first_partition("/dev/sdb")
        raise AssertionError("should fail")
    except carstore.CarstoreError as error:
        assert "no partition" in str(error)


def test_a_failed_command_names_itself_and_its_last_error_line():
    fake = install(FakeSystem())
    fake.fail[("wipefs",)] = (1, "first\nwipefs: /dev/sdb: probing initialization failed")
    fake.fail[("mkfs.btrfs",)] = (1, "")
    for command, expected in ((["wipefs", "-a", "/dev/sdb"], "probing initialization failed"),
                              (["mkfs.btrfs", "-f"], "no output")):
        try:
            carstore.must(command)
            raise AssertionError("should fail")
        except carstore.CarstoreError as error:
            assert expected in str(error) and command[0] in str(error), str(error)


def test_a_rescue_stopped_by_low_disk_says_so():
    reset_dirs()
    fake = install(FakeSystem())
    fake.image_tree = {"TeslaCam/RecentClips/2026-10-29_08-00-00-front.mp4": COMPLETE_MP4}
    carstore.DISK_USAGE = lambda path: Usage(100 * GB, 99 * GB, 1 * GB)
    try:
        copied, note = carstore.rescue_clips(carstore.Partition("/dev/sda1", 1, "vfat", "TESLACAM", None))
    finally:
        carstore.DISK_USAGE = shutil.disk_usage
    assert copied == 0 and "stopped early" in note, note
    assert ["mount", "-o", "ro", "/dev/sda1", str(carstore.WORK_ROOT / "rescue")] in fake.calls


# ---------------------------------------------------------------------------
# The backing store and the car's image
# ---------------------------------------------------------------------------
def test_the_backing_store_is_mounted_once():
    reset_dirs()
    fake = install(FakeSystem())
    partition = carstore.Partition("/dev/sda1", 128 * GB, "btrfs", "teslacam-backing", None)
    carstore.ensure_backing_mounted(partition)
    carstore.ensure_backing_mounted(partition)
    assert len(fake.ran("mount")) == 1


def test_the_image_is_as_large_as_the_store_allows_less_the_reserve():
    size = carstore.choose_image_bytes(128 * GB, 120 * GB)
    assert size == (116 * GB) - (116 * GB) % (1024 * 1024)
    carstore.IMAGE_GB = 64
    try:
        assert carstore.choose_image_bytes(128 * GB, 120 * GB) == 64 * GB - (64 * GB) % (1024 * 1024)
    finally:
        carstore.IMAGE_GB = 0
    try:
        carstore.choose_image_bytes(8 * GB, 4 * GB)
        raise AssertionError("should refuse")
    except carstore.CarstoreError as error:
        assert "too small" in str(error)


def test_a_new_image_gets_one_fat32_partition_and_the_teslacam_folder():
    reset_dirs()
    fake = install(FakeSystem())
    seen_folder = []
    original_umount = FakeSystem.__call__

    def watching(self, command, timeout=120, input_text=None):
        if command[0] == "umount" and command[-1].endswith("newimage"):
            seen_folder.append((Path(command[-1]) / "TeslaCam").is_dir())
        return original_umount(self, command, timeout, input_text)

    FakeSystem.__call__ = watching
    carstore.DISK_USAGE = lambda path: Usage(128 * GB, 1 * GB, 127 * GB)
    try:
        image = carstore.ensure_image()
    finally:
        FakeSystem.__call__ = original_umount
        carstore.DISK_USAGE = shutil.disk_usage
    assert image == carstore.image_path() and image.exists()
    assert not list(carstore.BACKING_MOUNT.glob(".building-*")), "the half-built name must be gone"
    assert ["mkfs.vfat", "-F", "32", "-n", "TESLACAM", "/dev/loop7p1"] in fake.calls
    assert fake.ran("sfdisk")[0][-1].endswith(".building-cam_disk.img")
    assert seen_folder == [True]
    assert fake.ran("losetup", "-d", "/dev/loop7")
    # Already there: nothing is rebuilt.
    calls = len(fake.calls)
    assert carstore.ensure_image() == image and len(fake.calls) == calls


# ---------------------------------------------------------------------------
# The USB gadget
# ---------------------------------------------------------------------------
def test_no_udc_means_no_export_and_says_why():
    reset_dirs()
    shutil.rmtree(carstore.UDC_ROOT)
    assert carstore.udc_name() is None and carstore.udc_state(None) is None
    try:
        carstore.export_image(carstore.image_path())
        raise AssertionError("should refuse")
    except carstore.CarstoreError as error:
        assert "UDC" in str(error)


def test_exporting_takes_the_udc_from_the_l4t_demo_gadget():
    reset_dirs()
    (carstore.UDC_ROOT / "3550000.usb").mkdir()
    (carstore.UDC_ROOT / "3550000.usb" / "state").write_text("configured\n")
    l4t = carstore.GADGET_ROOT / "l4t"
    l4t.mkdir()
    (l4t / "UDC").write_text("3550000.usb\n")
    image = carstore.image_path()
    assert carstore.export_image(image) is True
    gadget = carstore.GADGET_ROOT / "g1"
    assert (l4t / "UDC").read_text().strip() == ""
    assert (gadget / "UDC").read_text().strip() == "3550000.usb"
    assert carstore.exported_file() == str(image)
    assert (gadget / "idVendor").read_text().strip() == "0x1d6b"
    assert (gadget / "configs/c.1/mass_storage.0").is_symlink()
    assert carstore.udc_state("3550000.usb") == "configured"
    # Already offered: nothing to do.
    assert carstore.export_image(image) is False
    # A different file (the old block-device export) is replaced.
    (gadget / "functions/mass_storage.0/lun.0/file").write_text("/dev/disk/by-label/TESLACAM\n")
    assert carstore.export_image(image) is True and carstore.exported_file() == str(image)
    carstore.stop_export()
    assert carstore.exported_file() is None and (gadget / "UDC").read_text().strip() == ""
    carstore.stop_export()
    assert carstore.udc_state("missing") is None


# ---------------------------------------------------------------------------
# Which clips are finished, and copying them
# ---------------------------------------------------------------------------
def test_an_mp4_is_complete_only_when_its_boxes_tile_the_file_and_one_is_moov():
    reset_dirs()
    cases = {
        "whole.mp4": (COMPLETE_MP4, True),
        "unfinished.mp4": (UNFINISHED_MP4, False),
        "nomoov.mp4": (box(b"ftyp", b"isom0000") + box(b"mdat", b"d" * 8), False),
        "large.mp4": (box(b"ftyp", b"isom0000") + box(b"moov", b"m") +
                      (1).to_bytes(4, "big") + b"mdat" + (24).to_bytes(8, "big") + b"d" * 8, True),
        "toend.mp4": (box(b"moov", b"m") + (0).to_bytes(4, "big") + b"mdat" + b"d" * 30, True),
        "tiny_box.mp4": (box(b"moov", b"m") + (4).to_bytes(4, "big") + b"mdat", False),
        "short_header.mp4": (box(b"moov", b"m") + b"\0\0", False),
        "short_large.mp4": (box(b"moov", b"m") + (1).to_bytes(4, "big") + b"mdat" + b"\0\0", False),
    }
    for name, (data, expected) in cases.items():
        (INBOX_DIR / name).write_bytes(data)
        assert carstore.mp4_is_complete(INBOX_DIR / name) is expected, name
    assert carstore.mp4_is_complete(INBOX_DIR / "missing.mp4") is False


def test_the_clip_the_car_may_still_be_writing_waits_and_the_rest_go_newest_first():
    reset_dirs()
    root = SANDBOX / "tree" / "TeslaCam"
    shutil.rmtree(root.parent, ignore_errors=True)
    recent = root / "RecentClips"
    recent.mkdir(parents=True)
    names = ["2026-10-29_10-00-00-front.mp4", "2026-10-29_10-01-00-front.mp4",
             "2026-10-29_10-02-00-front.mp4", "2026-10-29_10-02-00-back.mp4",
             "2026-10-29_07-00-00-front.mp4", "notes.mp4"]
    for name in names:
        (recent / name).write_bytes(b"x")
    event = root / "SentryClips" / "2026-10-29_07-30-00"
    event.mkdir(parents=True)
    (event / "2026-10-29_07-29-00-left_repeater.mp4").write_bytes(b"x")
    now = ts("2026-10-29_10-02-00-front.mp4") + 30
    order = [clip.name for clip in carstore.sorted_clips(root, now=now)]
    # 10:02 is each camera's newest and only 30 s old: still being written.
    # 07:00 RecentClips is past the age limit; the 07:29 sentry clip is not.
    assert order == ["2026-10-29_10-01-00-front.mp4", "2026-10-29_10-00-00-front.mp4",
                     "2026-10-29_07-29-00-left_repeater.mp4"], order
    later = [clip.name for clip in carstore.sorted_clips(root, now=now + 300, max_age_hours=None)]
    assert later[:2] == ["2026-10-29_10-02-00-front.mp4", "2026-10-29_10-02-00-back.mp4"] or \
        later[:2] == ["2026-10-29_10-02-00-back.mp4", "2026-10-29_10-02-00-front.mp4"], later
    assert "2026-10-29_07-00-00-front.mp4" in later
    assert carstore.sorted_clips(SANDBOX / "nowhere", now=now) == []


def test_a_clip_already_in_the_pipeline_is_never_copied_twice():
    reset_dirs()
    connection = sqlite3.connect(os.environ["DB_PATH"])
    connection.execute("CREATE TABLE clips (filename TEXT)")
    connection.execute("INSERT INTO clips VALUES ('in_db.mp4')")
    connection.commit()
    connection.close()
    (INBOX_DIR / "in_inbox.mp4").write_bytes(b"x")
    (PROCESSED_DIR / "processed.mp4").write_bytes(b"x")
    known = carstore.known_clip_names()
    assert known("in_db.mp4") and known("in_inbox.mp4") and known("processed.mp4")
    assert not known("new.mp4")
    Path(os.environ["DB_PATH"]).unlink()
    assert not carstore.known_clip_names()("in_db.mp4"), "no database: nothing known from it"


def test_copying_skips_known_and_unfinished_clips_and_stops_when_disk_runs_low():
    reset_dirs()
    source = SANDBOX / "src"
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir()
    clips = {}
    for name, data in (("2026-10-29_10-03-00-front.mp4", COMPLETE_MP4),
                       ("2026-10-29_10-02-00-front.mp4", UNFINISHED_MP4),
                       ("2026-10-29_10-01-00-front.mp4", COMPLETE_MP4),
                       ("2026-10-29_10-00-00-front.mp4", COMPLETE_MP4)):
        (source / name).write_bytes(data)
        clips[name] = source / name
    ordered = list(clips.values())
    (INBOX_DIR / "2026-10-29_10-01-00-front.mp4").write_bytes(b"already here")
    outcome = carstore.copy_clips(ordered, known=lambda name: name == "2026-10-29_10-00-00-front.mp4",
                                  free_bytes=lambda: 100 * GB)
    assert outcome["copied"] == 1 and outcome["incomplete"] == 1 and outcome["paused"] is None, outcome
    assert outcome["newest_ts"] == ts("2026-10-29_10-03-00-front.mp4")
    assert (INBOX_DIR / "2026-10-29_10-03-00-front.mp4").read_bytes() == COMPLETE_MP4
    reset_dirs()
    outcome = carstore.copy_clips(ordered, known=lambda name: False, free_bytes=lambda: 1 * GB)
    assert outcome["copied"] == 0 and "GB free" in outcome["paused"]
    carstore.DISK_USAGE = lambda path: Usage(100 * GB, 0, 100 * GB)
    try:
        assert carstore.copy_clips(ordered[:1], known=lambda name: False)["copied"] == 1
    finally:
        carstore.DISK_USAGE = shutil.disk_usage


def test_the_newest_clip_received_is_the_newest_copied_not_the_last():
    reset_dirs()
    source = SANDBOX / "src2"
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir()
    names = ["2026-10-29_10-05-00-front.mp4", "2026-10-29_10-04-00-front.mp4", "event.mp4"]
    for name in names:
        (source / name).write_bytes(COMPLETE_MP4)
    outcome = carstore.copy_clips([source / name for name in names], known=lambda name: False,
                                  free_bytes=lambda: 100 * GB)
    assert outcome["copied"] == 3 and outcome["newest_ts"] == ts(names[0]), outcome


def test_a_cycle_out_of_time_defers_the_rest_and_reports_each_copy():
    reset_dirs()
    source = SANDBOX / "src3"
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir()
    clips = []
    for minute in range(3):
        clips.append(source / f"2026-10-29_10-0{minute}-00-front.mp4")
        clips[-1].write_bytes(COMPLETE_MP4)
    seen = []
    outcome = carstore.copy_clips(clips, known=lambda name: False, free_bytes=lambda: 100 * GB,
                                  deadline=time.time() + 60, on_copy=seen.append)
    assert outcome["copied"] == 3 and seen == [1, 2, 3] and outcome["deferred"] is False
    reset_dirs()
    outcome = carstore.copy_clips(clips, known=lambda name: False, free_bytes=lambda: 100 * GB,
                                  deadline=time.time() - 1)
    assert outcome["copied"] == 0 and outcome["deferred"] is True


def test_commands_really_run_through_subprocess():
    result = carstore._run([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
                           input_text="ok")
    assert result.returncode == 0 and result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# One snapshot cycle
# ---------------------------------------------------------------------------
def snapshot_ready(fstype="vfat"):
    reset_dirs()
    fake = install(FakeSystem())
    fake.fstype = fstype
    image = carstore.image_path()
    image.write_bytes(b"image")
    name = "2026-10-29_10-00-00-front.mp4"
    fake.image_tree = {f"TeslaCam/RecentClips/{name}": COMPLETE_MP4,
                       "TeslaCam/RecentClips/2026-10-29_10-01-00-front.mp4": COMPLETE_MP4}
    return fake, image, ts(name) + 60 + 10


def test_a_snapshot_cycle_copies_finished_clips_and_leaves_nothing_behind():
    fake, image, now = snapshot_ready()
    outcome = carstore.snapshot_and_copy(image, now)
    assert outcome["copied"] == 1 and outcome["fstype"] == "vfat", outcome
    assert (INBOX_DIR / "2026-10-29_10-00-00-front.mp4").exists()
    assert fake.ran("cp", "--reflink=always", str(image))
    assert ["fsck.vfat", "-a", "/dev/loop7p1"] in fake.calls
    assert ["mount", "-o", "ro", "/dev/loop7p1", str(carstore.WORK_ROOT / "snapshot")] in fake.calls
    assert fake.ran("losetup", "-d", "/dev/loop7")
    assert not (carstore.BACKING_MOUNT / "snapshots" / "snap.img").exists()
    assert not fake.mounted


def test_an_ext4_image_is_repaired_with_e2fsck_and_mounted_without_journal_replay():
    fake, image, now = snapshot_ready("ext4")
    carstore.snapshot_and_copy(image, now)
    assert ["e2fsck", "-p", "/dev/loop7p1"] in fake.calls
    assert ["mount", "-o", "ro,noload", "/dev/loop7p1", str(carstore.WORK_ROOT / "snapshot")] in fake.calls
    assert carstore.fsck_command("ntfs", "/dev/x") is None


def test_an_image_the_car_reformatted_is_reported_and_cleaned_up():
    fake, image, now = snapshot_ready(None)
    try:
        carstore.snapshot_and_copy(image, now)
        raise AssertionError("should fail")
    except carstore.CarstoreError as error:
        assert "reformatted in the car" in str(error)
    assert fake.ran("losetup", "-d", "/dev/loop7")
    assert not (carstore.BACKING_MOUNT / "snapshots" / "snap.img").exists()


def test_a_crashed_cycle_is_cleaned_up_before_the_next():
    fake, image, now = snapshot_ready()
    snap = carstore.BACKING_MOUNT / "snapshots" / "snap.img"
    snap.parent.mkdir(exist_ok=True)
    snap.write_bytes(b"old")
    fake.mounted[str(carstore.WORK_ROOT / "snapshot")] = "/dev/loop3p1"
    fake.loops_for_snapshot = f"/dev/loop3: []: ({snap})\n"
    carstore.snapshot_and_copy(image, now)
    assert fake.ran("umount", str(carstore.WORK_ROOT / "snapshot"))
    assert ["losetup", "-d", "/dev/loop3"] in fake.calls


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
def test_the_status_file_is_written_atomically():
    reset_dirs()
    carstore.write_status({"state": "x"})
    written = json.loads(carstore.STATUS_PATH.read_text())
    assert written["state"] == "x" and written["written_ts"] > 0


def test_a_changed_drive_list_invalidates_requests_made_against_the_old_one():
    service = carstore.Service()
    drives = drives_from(disk("sda", 128))
    service.publish(drives)
    first = service.nonce
    service.publish(drives_from(disk("sda", 128)))
    assert service.nonce == first, "same drives, same nonce"
    service.publish(drives_from(disk("sda", 128), disk("sdb", 64, serial="B")))
    assert service.nonce != first
    assert service.status["nonce"] == service.nonce and len(service.status["drives"]) == 2


def run_request(service, drives, request, format_effect=None):
    original = carstore.format_backing
    carstore.format_backing = format_effect or (lambda d, p, progress: (progress("erasing"), {
        "rescued": 0, "rescue": None, "formatted": p.path if p else d.path})[1])
    try:
        service.handle_request(request, drives)
    finally:
        carstore.format_backing = original
    return service.status["job"]


def test_a_refused_request_is_reported_on_the_page():
    reset_dirs()
    service = carstore.Service()
    drives = drives_from(disk("sda", 128))
    service.publish(drives)
    job = run_request(service, drives, request_for(drives[0], nonce="stale"))
    assert job["state"] == "refused" and "changed" in job["message"]


def test_formatting_the_current_car_storage_stops_the_export_first():
    reset_dirs()
    fake = install(FakeSystem())
    service = carstore.Service()
    drives = drives_from(disk("sda", 128, [part("sda1", 128, "btrfs", "teslacam-backing",
                                                str(carstore.BACKING_MOUNT))]))
    service.publish(drives)
    job = run_request(service, drives, request_for(drives[0], "/dev/sda1", nonce=service.nonce))
    assert job["state"] == "done" and "formatted /dev/sda1" in job["message"], job
    assert fake.ran("umount", str(carstore.BACKING_MOUNT))


def test_a_rescue_note_is_part_of_the_done_message_and_failures_are_reported():
    reset_dirs()
    install(FakeSystem())
    service = carstore.Service()
    drives = drives_from(disk("sda", 128, [part("sda1", 128, "ext4", "TESLACAM")]))
    service.publish(drives)
    job = run_request(service, drives, request_for(drives[0], "/dev/sda1", nonce=service.nonce),
                      lambda d, p, progress: {"rescued": 3, "rescue": "3 copied", "formatted": p.path})
    assert "old clips: 3 copied" in job["message"]

    def broken(d, p, progress):
        raise carstore.CarstoreError("mkfs.btrfs failed (1): device busy")
    job = run_request(service, drives, request_for(drives[0], "/dev/sda1", nonce=service.nonce), broken)
    assert job["state"] == "failed" and "device busy" in job["message"]


def service_with(drives_sequence, fake):
    service = carstore.Service()
    sequence = list(drives_sequence)
    carstore.scan_drives = lambda: ({}, sequence.pop(0) if len(sequence) > 1 else sequence[0])
    return service


SCAN_DRIVES = carstore.scan_drives


def test_with_no_car_storage_the_page_says_how_to_get_one():
    reset_dirs()
    fake = install(FakeSystem())
    service = service_with([drives_from(disk("sda", 128))], fake)
    try:
        service.tick(time.time())
    finally:
        carstore.scan_drives = SCAN_DRIVES
    assert service.status["state"] == "no_storage" and "Storage page" in service.status["reason"]
    assert not fake.ran("umount")


def test_a_drive_pulled_out_from_under_the_car_stops_the_export():
    reset_dirs()
    fake = install(FakeSystem())
    lun = carstore.GADGET_ROOT / "g1" / "functions/mass_storage.0/lun.0"
    lun.mkdir(parents=True)
    (lun / "file").write_text(str(carstore.image_path()))
    service = service_with([[]], fake)
    try:
        service.tick(time.time())
    finally:
        carstore.scan_drives = SCAN_DRIVES
    assert carstore.exported_file() is None
    assert fake.ran("umount", "-l", str(carstore.BACKING_MOUNT))


def test_a_tick_mounts_exports_snapshots_and_reports_how_far_behind_it_is():
    fake, image, now = snapshot_ready()
    (carstore.UDC_ROOT / "3550000.usb").mkdir()
    (carstore.UDC_ROOT / "3550000.usb" / "state").write_text("configured\n")
    drives = drives_from(disk("sda", 128, [part("sda1", 128, "btrfs", "teslacam-backing")]))
    service = service_with([drives], fake)
    carstore.DISK_USAGE = lambda path: Usage(128 * GB, 70 * GB, 58 * GB)
    try:
        service.tick(now)
        snap = service.status["snapshot"]
        assert service.status["state"] == "exporting"
        assert service.status["export"]["car_connected"] is True
        assert service.status["backing"]["below_tesla_minimum"] is True   # the test image is tiny
        assert snap["copied_last"] == 1 and snap["copied_total"] == 1 and snap["error"] is None
        assert snap["newest_clip_ts"] == ts("2026-10-29_10-00-00-front.mp4")
        assert snap["copying"] is None and snap["deferred"] is False
        assert json.loads(carstore.STATUS_PATH.read_text())["snapshot"]["copying"] == 1, \
            "the heartbeat is written while copying"
        # Within the interval: no second snapshot.
        copies = len(fake.ran("cp"))
        service.tick(now + 5)
        assert len(fake.ran("cp")) == copies
        # Next interval, nothing new: the counters hold.
        service.tick(now + carstore.SNAPSHOT_SECONDS)
        assert snap["copied_last"] == 0 and snap["copied_total"] == 1
        # A failing cycle is reported, not fatal.
        fake.fstype = None
        service.tick(now + 2 * carstore.SNAPSHOT_SECONDS)
        assert "reformatted" in snap["error"]
    finally:
        carstore.scan_drives = SCAN_DRIVES
        carstore.DISK_USAGE = shutil.disk_usage


def test_a_request_is_handled_in_the_tick_that_finds_it():
    reset_dirs()
    fake = install(FakeSystem())
    drives = drives_from(disk("sda", 128))
    service = service_with([drives, []], fake)
    service.publish(drives)
    carstore.REQUEST_PATH.write_text(json.dumps(request_for(drives[0], nonce=service.nonce)))
    original = carstore.format_backing
    carstore.format_backing = lambda d, p, progress: {"rescued": 0, "rescue": None, "formatted": "/dev/sda1"}
    try:
        service.tick(time.time())
    finally:
        carstore.format_backing = original
        carstore.scan_drives = SCAN_DRIVES
    assert service.status["job"]["state"] == "done"
    assert not carstore.REQUEST_PATH.exists()


# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {test.__name__}: {error}")
        except Exception as error:                        # noqa: BLE001
            failures += 1
            print(f"  ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    shutil.rmtree(SANDBOX, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
