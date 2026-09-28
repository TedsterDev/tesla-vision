"""
test_natix.py

Behavioural tests for the NATIX VX360 mirror.

The dangerous failure here is not "a clip didn't copy". It is "we mounted the
wrong device and filled somebody's backup drive with dashcam footage", or "we
pruned files we didn't put there", or "the car lost power mid-copy and the
stick now has a truncated clip under a real filename". So most of these tests
are about refusal and recovery rather than about the happy path.

Run:  python3 tests/test_natix.py       (or: python3 -m pytest tests/ -v)
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import natix  # noqa: E402


# ---------------------------------------------------------------------------
# A known-neutral profile
# ---------------------------------------------------------------------------
# natix.py reads its matching profile from the environment at import time, and
# the deployed .env pins the real stick by volume UUID. Run these tests with
# that .env loaded - which is exactly what happens inside the container - and
# every fixture volume matches the pin, so tests asserting "this only reaches
# 'strong'" or "this is refused" get 'pinned' and pass or fail for reasons that
# have nothing to do with the code. Pin the profile to a known baseline here so
# the suite means the same thing on the host, in the container, and in CI.
natix.NATIX_SERIAL = ""
natix.NATIX_USB_ID = ""
natix.NATIX_VOLUME_UUID = ""
natix.NATIX_MIN_CONFIDENCE = "strong"
natix.NATIX_NAME_PATTERNS = ["natix", "vx360", "vx-360", "vx_360"]
natix.NATIX_MIN_SIZE_GB = 16
natix.NATIX_MAX_SIZE_GB = 2048
natix.NATIX_RESERVE_MB = 2048
natix.NATIX_DEFAULT_BUCKET = "SentryClips"
natix.NATIX_WRITE_EVENT_JSON = True
# Off, so any test that exercises reclaiming has to ask for it by name.
natix.NATIX_RECLAIM_BUCKETS = []
natix.NATIX_IGNORE_FREE_SPACE = False
natix.NATIX_MIRROR_ORDER = "newest"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_volume(**overrides) -> natix.Volume:
    """A plausible VX360 volume, overridable field by field."""
    defaults = dict(
        path="/dev/sda1",
        parent_path="/dev/sda",
        size_bytes=37_579_915_264,
        fstype="exfat",
        label="VX360",
        volume_uuid="FFFB-6B2F",
        mountpoint=None,
        removable=True,
        read_only=False,
        transport="usb",
        vendor="Linux",
        model="File-Stor Gadget",
        serial="Linux_File-Stor_Gadget-0:0",
        usb_vendor_id="0525",
        usb_product_id="a4a5",
    )
    defaults.update(overrides)
    return natix.Volume(**defaults)


def make_db(path: Path) -> sqlite3.Connection:
    """A database with just the tables this module touches."""
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE clips (
            id TEXT PRIMARY KEY, filename TEXT UNIQUE, camera TEXT,
            captured_ts INTEGER, clip_source TEXT
        );
        CREATE TABLE natix_mirror (
            id TEXT PRIMARY KEY, device_id TEXT, clip_id TEXT, filename TEXT,
            event_folder TEXT, bucket TEXT, dest_path TEXT, size_bytes INTEGER,
            source_mtime INTEGER, state TEXT, error TEXT, copied_ts INTEGER,
            UNIQUE(device_id, clip_id)
        );
        CREATE TABLE natix_devices (
            id TEXT PRIMARY KEY, serial TEXT, volume_uuid TEXT, label TEXT,
            vendor TEXT, model TEXT, usb_vendor_id TEXT, usb_product_id TEXT,
            size_bytes INTEGER, fstype TEXT, confidence TEXT,
            first_seen_ts INTEGER, last_seen_ts INTEGER, last_mount TEXT,
            free_bytes INTEGER, mirrored_count INTEGER DEFAULT 0,
            mirrored_bytes INTEGER DEFAULT 0, is_approved INTEGER DEFAULT 0,
            note TEXT
        );
        """
    )
    return connection


def seed_clips(connection, archive: Path, events: dict[str, list[str]], size: int = 4096) -> None:
    """Create fake clip files and matching `clips` rows."""
    archive.mkdir(parents=True, exist_ok=True)
    for event, cameras in events.items():
        for index, camera in enumerate(cameras):
            filename = f"{event}-{camera}.mp4"
            (archive / filename).write_bytes(os.urandom(size))
            connection.execute(
                "INSERT INTO clips (id, filename, camera, captured_ts, clip_source) "
                "VALUES (?,?,?,?,?)",
                (f"clip-{event}-{camera}", filename, camera,
                 1771274960 + index, None),
            )
    connection.commit()


class Sandbox:
    """A temp archive + temp 'stick' with natix pointed at the archive."""

    def __enter__(self):
        self.root = Path(tempfile.mkdtemp(prefix="natix-test-"))
        self.archive = self.root / "processed"
        self.stick = self.root / "stick"
        self.archive.mkdir()
        self.stick.mkdir()
        self.connection = make_db(self.root / "test.db")
        self._saved_processed = natix.PROCESSED_DIR
        self._saved_inbox = natix.INBOX_DIR
        natix.PROCESSED_DIR = self.archive
        natix.INBOX_DIR = self.archive
        return self

    def __exit__(self, *_):
        natix.PROCESSED_DIR = self._saved_processed
        natix.INBOX_DIR = self._saved_inbox
        self.connection.close()
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Identity and refusal
# ---------------------------------------------------------------------------
def test_label_alone_identifies_the_stick():
    """The retail stick's only honest self-description is its volume label."""
    candidate = natix.evaluate(make_volume(), protected_disks=set())
    assert candidate.confidence == "strong", candidate.confidence
    assert any("vx360" in reason.lower() for reason in candidate.reasons)
    assert candidate.usable


def test_generic_gadget_serial_is_not_an_identity():
    """
    The VX360 is a Linux box running the kernel's file-backed storage gadget,
    so it reports the stock serial 'Linux_File-Stor_Gadget-0:0'. So does every
    other Linux storage gadget - including this Jetson when it is pretending to
    be a dashcam drive for the car. Keying on it would eventually make us
    mirror onto the wrong machine.
    """
    candidate = natix.evaluate(make_volume(), protected_disks=set())
    assert candidate.device_id == "uuid:FFFB-6B2F", candidate.device_id
    assert natix.is_generic_serial("Linux_File-Stor_Gadget-0:0")
    assert natix.is_generic_usb_id("0525:a4a5")
    assert not natix.is_generic_serial("A1B2C3D4")


def test_a_pinned_volume_uuid_beats_everything_else():
    """
    The counterpart to the neutral profile above: with a pin set, an otherwise
    anonymous stick becomes 'pinned'. This is how the deployed .env identifies
    the real unit, whose only unique handle is its filesystem UUID.
    """
    saved = natix.NATIX_VOLUME_UUID
    natix.NATIX_VOLUME_UUID = "FFFB-6B2F"
    try:
        anonymous = make_volume(label=None, model="Generic Flash", serial="4C530001")
        candidate = natix.evaluate(anonymous, protected_disks=set())
        assert candidate.confidence == "pinned", candidate.confidence
        assert candidate.usable
    finally:
        natix.NATIX_VOLUME_UUID = saved


def test_two_gadget_sticks_get_different_ids():
    """Same generic serial, different filesystems - must not collide."""
    first = natix.evaluate(make_volume(volume_uuid="FFFB-6B2F"), protected_disks=set())
    second = natix.evaluate(make_volume(volume_uuid="AAAA-1111", path="/dev/sdb1",
                                        parent_path="/dev/sdb"), protected_disks=set())
    assert first.device_id != second.device_id


def test_boot_disk_is_refused_even_if_it_looks_right():
    volume = make_volume(path="/dev/mmcblk0p1", parent_path="/dev/mmcblk0",
                         fstype="ext4", label="VX360")
    candidate = natix.evaluate(volume, protected_disks={"mmcblk0"})
    assert not candidate.usable
    assert any("operating system" in problem for problem in candidate.disqualifiers)


def test_read_only_volume_is_refused():
    candidate = natix.evaluate(make_volume(read_only=True), protected_disks=set())
    assert not candidate.usable
    assert any("read-only" in problem for problem in candidate.disqualifiers)


def test_unidentified_usb_stick_is_only_likely_not_usable():
    """
    A nameless 128GB USB stick has the right *shape*. That is not enough to
    start writing gigabytes to it, so it stops at 'likely' and the default
    confidence floor of 'strong' keeps our hands off it.
    """
    volume = make_volume(label=None, model="Cruzer Glide", vendor="SanDisk",
                         serial="4C530001", size_bytes=128_000_000_000)
    candidate = natix.evaluate(volume, protected_disks=set())
    assert candidate.confidence == "likely", candidate.confidence
    assert not candidate.usable, "shape matching alone must not authorise writes"


# ---------------------------------------------------------------------------
# Never write to a directory believing it is the stick
# ---------------------------------------------------------------------------
# The real incident: a FUSE mount failed while mount(8) returned 0, the mirror
# carried on, and 1.1GB of clips went into the bare mountpoint directory on the
# Jetson's SD card. Writing to a directory always succeeds, so nothing
# complained. Every one of these tests exists because of that.
class FakeMount:
    """Drive natix's mount plumbing without touching real block devices."""

    def __init__(self, helper_rc=0, findmnt_source=None, statvfs_total=None):
        self.helper_rc = helper_rc
        self.findmnt_source = findmnt_source
        self.statvfs_total = statvfs_total
        self.saved_run = natix._run
        self.saved_statvfs = natix.os.statvfs
        self.saved_helper = natix.MOUNT_HELPER

    def __enter__(self):
        class Result:
            def __init__(self, rc, out=""):
                self.returncode, self.stdout, self.stderr = rc, out, ""

        def fake_run(command, **_kwargs):
            if command and command[0] == "findmnt":
                if self.findmnt_source is None:
                    return Result(1)
                return Result(0, self.findmnt_source + "\n")
            return Result(self.helper_rc, "mounted")

        natix._run = fake_run
        natix.MOUNT_HELPER = Path(__file__)          # exists, so the check passes
        if self.statvfs_total is not None:
            total = self.statvfs_total

            class Stat:
                f_frsize = 4096
                f_blocks = total // 4096
                f_bavail = total // 4096
            natix.os.statvfs = lambda _p: Stat()
        return self

    def __exit__(self, *_):
        natix._run = self.saved_run
        natix.os.statvfs = self.saved_statvfs
        natix.MOUNT_HELPER = self.saved_helper


def test_mount_refuses_when_helper_succeeds_but_nothing_mounted():
    """The exact incident: exit code 0, no filesystem attached."""
    candidate = natix.evaluate(make_volume(), protected_disks=set())
    with Sandbox() as box, FakeMount(helper_rc=0, findmnt_source=None):
        try:
            natix.mount(candidate, mountpoint=box.stick / "mp")
        except natix.MountError as error:
            assert "nothing is mounted" in str(error), error
        else:
            raise AssertionError("accepted a mount that did not happen")


def test_mount_refuses_a_nonempty_directory_that_is_not_a_mountpoint():
    """
    Leftover files at the mountpoint mean an earlier mount failed silently and
    we wrote to the host disk. Mounting over them hides them forever while they
    keep consuming the boot card.
    """
    candidate = natix.evaluate(make_volume(), protected_disks=set())
    with Sandbox() as box:
        target = box.stick / "mp"
        (target / "TeslaCam" / "SentryClips").mkdir(parents=True)
        (target / "TeslaCam" / "SentryClips" / "stray.mp4").write_bytes(b"x")
        with FakeMount(helper_rc=0, findmnt_source=None):
            try:
                natix.mount(candidate, mountpoint=target)
            except natix.MountError as error:
                assert "not empty" in str(error) and "--clean-stray" in str(error)
            else:
                raise AssertionError("mounted over stray host-disk data")


def test_mount_refuses_when_a_different_device_is_mounted_there():
    candidate = natix.evaluate(make_volume(), protected_disks=set())
    with Sandbox() as box, FakeMount(findmnt_source="/dev/sdz9"):
        try:
            natix.mount(candidate, mountpoint=box.stick / "mp")
        except natix.MountError as error:
            assert "not /dev/sda1" in str(error), error
        else:
            raise AssertionError("adopted somebody else's mount")


def test_mount_refuses_a_volume_far_bigger_than_the_device():
    """
    '35GB stick reports 73.5GB free' was the tell that we were looking at the
    host SD card. Catch it on size as well as on device identity.
    """
    candidate = natix.evaluate(make_volume(), protected_disks=set())
    with Sandbox() as box, FakeMount(
        findmnt_source="/dev/sda1", statvfs_total=116 * 1024 ** 3
    ):
        try:
            natix.mount(candidate, mountpoint=box.stick / "mp")
        except natix.MountError as error:
            assert "is not the stick" in str(error), error
        else:
            raise AssertionError("accepted a 116GB filesystem as a 37.6GB stick")


def test_mount_accepts_a_real_mount_of_the_right_device():
    candidate = natix.evaluate(make_volume(), protected_disks=set())
    with Sandbox() as box, FakeMount(
        findmnt_source="/dev/sda1", statvfs_total=35 * 1024 ** 3
    ):
        where = natix.mount(candidate, mountpoint=box.stick / "mp")
        assert where == box.stick / "mp"


def test_mount_refuses_an_unusable_candidate():
    volume = make_volume(label=None, model="Cruzer Glide", serial="4C530001")
    candidate = natix.evaluate(volume, protected_disks=set())
    try:
        natix.mount(candidate)
    except natix.MountError as error:
        assert "below NATIX_MIN_CONFIDENCE" in str(error)
    else:
        raise AssertionError("mount() accepted a candidate below the floor")


def test_discovery_reports_blindness_rather_than_emptiness():
    """
    The dashboard runs in a container, where lsblk exists and cheerfully lists
    the host's /sys/block - with every label, UUID and fstype null, because
    udev's database isn't mounted in. Scoring that would rate everything 'weak'
    and render "no candidate device attached", which is a lie: the truth is "I
    cannot see". status() must say so instead.
    """
    saved = natix._lsblk_json

    def blind(*_args, **_kwargs):
        raise FileNotFoundError("no udev database at /run/udev")

    natix._lsblk_json = blind
    try:
        with Sandbox() as box:
            box.connection.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
            state = natix.status(box.connection)
    finally:
        natix._lsblk_json = saved

    assert state["discovery_available"] is False
    assert state["attached"] == []


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def test_event_folder_comes_from_the_filename():
    assert natix.event_name_for("2026-02-16_20-49-20-back.mp4", None) == "2026-02-16_20-49-20"
    assert natix.event_name_for("2026-02-16_20-49-20-left_repeater.mp4", None) == "2026-02-16_20-49-20"


def test_event_folder_falls_back_to_capture_time():
    name = natix.event_name_for("weird-name.mp4", 1771274960)
    assert name != "unsorted" and len(name) == len("2026-02-16_20-49-20")


def test_mirror_writes_tesla_layout():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-16_20-49-20": ["front", "back", "left_repeater"],
        })
        result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        assert result.copied == 3, result
        event_dir = box.stick / "TeslaCam/SentryClips/2026-02-16_20-49-20"
        assert event_dir.is_dir(), sorted(p.name for p in box.stick.rglob("*"))
        assert (event_dir / "2026-02-16_20-49-20-front.mp4").exists()
        # Tesla writes an event.json beside the clips; the stick's uploader
        # reads it, so a synthesised folder needs one too.
        assert (event_dir / "event.json").exists()
        payload = json.loads((event_dir / "event.json").read_text())
        assert payload["timestamp"].startswith("2026-02-16T20:49:20")


def test_copies_are_byte_identical():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_20-49-20": ["front"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)
        source = box.archive / "2026-02-16_20-49-20-front.mp4"
        target = box.stick / "TeslaCam/SentryClips/2026-02-16_20-49-20/2026-02-16_20-49-20-front.mp4"
        assert source.read_bytes() == target.read_bytes()


# ---------------------------------------------------------------------------
# Reclaiming space from footage we did not write
# ---------------------------------------------------------------------------
# The real stick arrived with 362MB free of 37.6GB, full of the car's own
# recordings. It will never free that space itself: a VX360 loop-deletes only
# while the car is writing to it, and in this topology the car writes to the
# Jetson. So the mirror has to take over the loop - which means deleting
# somebody else's footage, and these tests are the fence around that.
def make_foreign_event(stick: Path, bucket: str, stamp: str, cameras: int = 2,
                       size: int = 2048) -> Path:
    folder = stick / "TeslaCam" / bucket / stamp
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(cameras):
        (folder / f"{stamp}-cam{index}.mp4").write_bytes(os.urandom(size))
    (folder / "event.json").write_text("{}")
    return folder


def test_reclaim_is_off_by_default():
    """The whole feature must stay inert until somebody opts in by name."""
    with Sandbox() as box:
        old = make_foreign_event(box.stick, "SentryClips", "2026-01-01_08-00-00")
        removed, freed, _ = natix.reclaim_foreign(
            box.connection, "uuid:TEST", box.stick, need_bytes=2 ** 60
        )
        assert (removed, freed) == (0, 0)
        assert old.exists(), "reclaimed with an empty NATIX_RECLAIM_BUCKETS"


def test_reclaim_never_touches_savedclips():
    """SavedClips are the clips somebody pressed the button to keep."""
    with Sandbox() as box:
        saved = make_foreign_event(box.stick, "SavedClips", "2026-01-01_08-00-00")
        sentry = make_foreign_event(box.stick, "SentryClips", "2026-06-01_08-00-00")
        natix.reclaim_foreign(box.connection, "uuid:TEST", box.stick,
                              need_bytes=2 ** 60, buckets=["SentryClips"])
        assert saved.exists(), "deleted SavedClips"
        assert not sentry.exists()


def test_reclaim_refuses_a_bucket_tesla_never_defined():
    """A typo in NATIX_RECLAIM_BUCKETS must delete nothing, not everything."""
    with Sandbox() as box:
        mine = box.stick / "TeslaCam" / "MyBackups" / "2026-01-01_08-00-00"
        mine.mkdir(parents=True)
        (mine / "2026-01-01_08-00-00-cam0.mp4").write_bytes(b"precious")
        removed, _, _ = natix.reclaim_foreign(
            box.connection, "uuid:TEST", box.stick,
            need_bytes=2 ** 60, buckets=["MyBackups", "Sentryclips"],
        )
        assert removed == 0
        assert (mine / "2026-01-01_08-00-00-cam0.mp4").exists()


def test_reclaim_skips_entries_that_are_not_tesla_shaped():
    with Sandbox() as box:
        odd = box.stick / "TeslaCam" / "SentryClips" / "holiday-photos"
        odd.mkdir(parents=True)
        (odd / "beach.jpg").write_bytes(b"jpeg")
        natix.reclaim_foreign(box.connection, "uuid:TEST", box.stick,
                              need_bytes=2 ** 60, buckets=["SentryClips"])
        assert (odd / "beach.jpg").exists(), "deleted a folder that is not an event"


def test_reclaim_never_touches_our_own_mirrored_events():
    """Ours are prune_oldest's job; double ownership is how files vanish twice."""
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-03-01_08-00-00": ["front"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)
        ours = box.stick / "TeslaCam/SentryClips/2026-03-01_08-00-00"
        assert ours.is_dir()
        natix.reclaim_foreign(box.connection, "uuid:TEST", box.stick,
                              need_bytes=2 ** 60, buckets=["SentryClips"])
        assert ours.exists(), "reclaim ate an event the mirror owns"


def test_reclaim_takes_the_oldest_recording_first_across_buckets():
    with Sandbox() as box:
        oldest = make_foreign_event(box.stick, "RecentClips", "2026-01-01_08-00-00")
        newest = make_foreign_event(box.stick, "SentryClips", "2026-09-01_08-00-00")
        plan = natix.plan_reclaim(box.connection, "uuid:TEST", box.stick,
                                  need_bytes=natix.free_bytes(box.stick) + 1,
                                  buckets=["SentryClips", "RecentClips"])
        assert len(plan) == 1
        assert plan[0][1] == oldest, f"chose {plan[0][1]} before {oldest}"
        assert newest.exists()


def test_reclaim_stops_once_there_is_enough_room():
    """It is loop recording, not a wipe: free what is needed and no more."""
    with Sandbox() as box:
        events = [make_foreign_event(box.stick, "SentryClips", stamp)
                  for stamp in ("2026-01-01_08-00-00", "2026-02-01_08-00-00",
                                "2026-03-01_08-00-00")]
        one_event = natix._entry_size(events[0])
        plan = natix.plan_reclaim(box.connection, "uuid:TEST", box.stick,
                                  need_bytes=natix.free_bytes(box.stick) + one_event,
                                  buckets=["SentryClips"])
        assert len(plan) == 1, f"planned to delete {len(plan)} events to free one"


def test_reclaim_dry_run_deletes_nothing():
    with Sandbox() as box:
        event = make_foreign_event(box.stick, "SentryClips", "2026-01-01_08-00-00")
        removed, freed, log = natix.reclaim_foreign(
            box.connection, "uuid:TEST", box.stick, need_bytes=2 ** 60,
            buckets=["SentryClips"], dry_run=True,
        )
        assert removed == 1 and freed > 0
        assert event.exists(), "dry run deleted a folder"
        assert log and log[0].startswith("would reclaim")


def test_reclaim_logs_every_deletion_by_name_and_size():
    """This is the one place we destroy someone else's data. Never silently."""
    with Sandbox() as box:
        make_foreign_event(box.stick, "SentryClips", "2026-01-01_08-00-00")
        _, _, log = natix.reclaim_foreign(box.connection, "uuid:TEST", box.stick,
                                          need_bytes=2 ** 60, buckets=["SentryClips"])
        assert len(log) == 1
        assert "2026-01-01_08-00-00" in log[0] and "MB" in log[0]


def test_reclaim_never_escapes_the_teslacam_directory():
    with Sandbox() as box:
        outside = box.stick / "DCIM" / "2026-01-01_08-00-00"
        outside.mkdir(parents=True)
        (outside / "2026-01-01_08-00-00-cam0.mp4").write_bytes(b"someone's photos")
        (box.stick / "natix.json").write_bytes(b"{}")
        natix.reclaim_foreign(box.connection, "uuid:TEST", box.stick,
                              need_bytes=2 ** 60,
                              buckets=list(natix.TESLA_BUCKETS))
        assert (outside / "2026-01-01_08-00-00-cam0.mp4").exists()
        assert (box.stick / "natix.json").exists()


def test_full_stick_message_explains_why_it_will_not_self_heal():
    """
    The real failure we hit. The message has to say more than 'disk full',
    because the stick being full is not a transient condition here - it is
    permanent until somebody changes a setting.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front"]})
        saved = natix.NATIX_RESERVE_MB
        natix.NATIX_RESERVE_MB = 1024 * 1024 * 1024
        try:
            result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        finally:
            natix.NATIX_RESERVE_MB = saved
        assert result.copied == 0
        assert "NATIX_RECLAIM_BUCKETS" in (result.stopped_reason or "")
        assert "loop-delete" in (result.stopped_reason or "")


# ---------------------------------------------------------------------------
# Honest space accounting
# ---------------------------------------------------------------------------
# The real stick reported 34.65GB in use while the walk could only find 6.14GB
# of it. The first version of this code swallowed per-file errors and printed a
# tidy total, which is the worst possible behaviour: a wrong number that looks
# right gets believed and acted on.
def test_walk_reports_what_it_cannot_read():
    with Sandbox() as box:
        (box.stick / "readable.mp4").write_bytes(b"x" * 100)
        blocked = box.stick / "blocked"
        blocked.mkdir()
        (blocked / "hidden.mp4").write_bytes(b"y" * 500)
        os.chmod(blocked, 0o000)
        try:
            found = natix.walk_tree(box.stick)
            if os.geteuid() == 0:
                return   # root reads it anyway; nothing to assert
            assert found.errors, "unreadable directory reported no error"
            assert found.bytes_seen == 100, found.bytes_seen
        finally:
            os.chmod(blocked, 0o755)


def test_walk_finds_the_largest_files():
    with Sandbox() as box:
        (box.stick / "small.mp4").write_bytes(b"x" * 10)
        (box.stick / "big.mp4").write_bytes(b"x" * 5000)
        found = natix.walk_tree(box.stick, keep_largest=1)
        assert len(found.largest) == 1
        assert found.largest[0][1].name == "big.mp4"


def test_describe_space_flags_an_unexplained_gap():
    """
    Exactly the real case: the filesystem says most of the volume is in use,
    the walk can only account for a fraction, and the report has to lead with
    that rather than quietly printing the fraction.
    """
    with Sandbox() as box:
        (box.stick / "visible.mp4").write_bytes(b"x" * 1024)

        class FakeStat:
            f_blocks = 35 * 1024 ** 3 // 4096
            f_frsize = 4096
            f_bavail = 1 * 1024 ** 3 // 4096     # 1GB free of 35GB

        saved = natix.os.statvfs
        natix.os.statvfs = lambda _path: FakeStat()
        try:
            report = "\n".join(natix.describe_space(box.stick))
        finally:
            natix.os.statvfs = saved

        assert "DISCREPANCY" in report, report
        assert "allocated to no file at all" in report
        # It must point at the only thing that can actually recover the space.
        # Suggesting file deletion here would send someone hunting for files
        # that do not exist.
        assert "leaked clusters" in report
        assert "natix_fsck.sh" in report
        assert "Deleting files cannot recover it" in report
        # And it must NOT reach the preallocation verdict, whose remedy is the
        # exact opposite advice.
        assert "The space IS in files" not in report


def test_describe_space_tells_preallocation_apart_from_leaked_clusters():
    """
    The case the report could not previously see, and got wrong for it.

    A sparse file occupies far fewer bytes than it contains; a preallocated
    one occupies far more. Either way st_size alone cannot distinguish "the
    space is in files that do not contain it" from "the space is in no file
    at all" - and those two faults have opposite remedies. Deleting files
    fixes the first and does nothing for the second; a repair fixes the second
    and has nothing to do in the first.

    Here the volume is genuinely full of files whose cluster chains hold the
    space. The report must say so, and must NOT send anyone to --repair.
    """
    with Sandbox() as box:
        spool = box.stick / "EncryptedClips"
        spool.mkdir()
        # A file that CONTAINS little and OCCUPIES a lot. st_blocks is what
        # the filesystem allocated, so write real bytes and then report a
        # small st_size - which is what an exFAT valid-data-length shorter
        # than the cluster chain looks like from userspace.
        (spool / "reserved.mp4").write_bytes(b"x" * 4096)

        real_stat = natix.Path.stat

        class Occupying:
            """st_size tiny, st_blocks huge - a reserved-but-unfilled clip."""
            def __init__(self, info):
                self._info = info
                self.st_size = 1024
                self.st_blocks = (30 * 1024 ** 3) // 512
            def __getattr__(self, name):
                return getattr(self._info, name)

        def stat_override(self, *args, **kwargs):
            info = real_stat(self, *args, **kwargs)
            return Occupying(info) if self.name == "reserved.mp4" else info

        # 5GB free of 35GB, so statvfs calls 30GB used - exactly what the one
        # reserved clip occupies. The point of the fixture is that the volume
        # reconciles once you measure the right dimension: nothing is leaked,
        # the space is simply all inside a file that contains almost none of it.
        class FakeStat:
            f_blocks = 35 * 1024 ** 3 // 4096
            f_frsize = 4096
            f_bavail = 5 * 1024 ** 3 // 4096

        saved_statvfs = natix.os.statvfs
        natix.Path.stat = stat_override
        natix.os.statvfs = lambda _path: FakeStat()
        try:
            report = "\n".join(natix.describe_space(box.stick))
        finally:
            natix.Path.stat = real_stat
            natix.os.statvfs = saved_statvfs

        assert "The space IS in files" in report, report
        assert "preallocated" in report
        assert "Deleting the files that own those clusters recovers" in report
        # The wrong remedy must not appear: there is nothing for a repair to do.
        assert "DISCREPANCY" not in report
        assert "--repair" not in report
        # And it should name the directory actually holding the clusters.
        assert "EncryptedClips" in report


def test_describe_space_stays_quiet_when_it_adds_up():
    """
    The other half of the contract: no false alarm when the numbers agree.
    statvfs has to be faked here too - the sandbox 'stick' is a directory on
    the host filesystem, so a real statvfs describes the whole 116GB disk and
    every run would report a spurious 40GB gap.
    """
    with Sandbox() as box:
        payload = b"x" * (8 * 1024 * 1024)
        (box.stick / "visible.mp4").write_bytes(payload)

        class FakeStat:
            f_frsize = 4096
            f_blocks = (1024 ** 3) // 4096                       # 1GB volume
            f_bavail = (1024 ** 3 - len(payload)) // 4096        # all but the file

        saved = natix.os.statvfs
        natix.os.statvfs = lambda _path: FakeStat()
        try:
            report = "\n".join(natix.describe_space(box.stick))
        finally:
            natix.os.statvfs = saved

        assert "DISCREPANCY" not in report, report


# ---------------------------------------------------------------------------
# Idempotence and crash safety
# ---------------------------------------------------------------------------
def test_second_pass_copies_nothing():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_20-49-20": ["front", "back"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)
        again = natix.mirror(box.connection, "uuid:TEST", box.stick)
        assert again.copied == 0 and again.skipped == 0, again


def test_recovers_when_the_database_lost_the_record():
    """
    The rename lands, then power goes before the database write. On the next
    pass the file is already there and correct, so we must adopt it rather than
    copy 40MB again - and must not create a duplicate row.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_20-49-20": ["front"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)
        box.connection.execute("DELETE FROM natix_mirror")
        box.connection.commit()

        result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        assert result.skipped == 1 and result.copied == 0, result
        rows = box.connection.execute("SELECT COUNT(*) FROM natix_mirror").fetchone()[0]
        assert rows == 1


def test_a_truncated_destination_is_recopied():
    """Same size check, opposite outcome: a short file must not be adopted."""
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_20-49-20": ["front"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)
        target = box.stick / "TeslaCam/SentryClips/2026-02-16_20-49-20/2026-02-16_20-49-20-front.mp4"
        target.write_bytes(b"truncated")
        box.connection.execute("DELETE FROM natix_mirror")
        box.connection.commit()

        result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        assert result.copied == 1, result
        assert target.read_bytes() == (box.archive / target.name).read_bytes()


def test_no_partial_files_are_left_behind():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_20-49-20": ["front", "back"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)
        leftovers = [p.name for p in box.stick.rglob("*.part")]
        assert leftovers == [], leftovers


def test_a_failed_copy_leaves_no_file_under_the_real_name():
    """
    _copy_atomic writes to a temp name and renames. If the write dies the temp
    file goes, and the real name was never created - so a reader can never see
    a half-written clip.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_20-49-20": ["front"]})
        source = box.archive / "2026-02-16_20-49-20-front.mp4"
        target = box.stick / "TeslaCam/SentryClips/2026-02-16_20-49-20/2026-02-16_20-49-20-front.mp4"

        original_copy = natix.shutil.copyfileobj

        def explode(*_args, **_kwargs):
            raise OSError(28, "No space left on device")

        natix.shutil.copyfileobj = explode
        try:
            result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        finally:
            natix.shutil.copyfileobj = original_copy

        assert result.failed == 1 and result.copied == 0, result
        assert not target.exists(), "a failed copy created the real filename"
        assert [p.name for p in box.stick.rglob("*.part")] == []
        assert source.exists(), "the source archive must never be touched"


# ---------------------------------------------------------------------------
# Loop recording
# ---------------------------------------------------------------------------
def test_prune_removes_the_oldest_event_first():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-16_08-00-00": ["front"],
            "2026-02-17_08-00-00": ["front"],
            "2026-02-18_08-00-00": ["front"],
        })
        natix.mirror(box.connection, "uuid:TEST", box.stick)

        # Ask for more room than exists, so pruning runs until it gives up.
        freed_files, _ = natix.prune_oldest(
            box.connection, "uuid:TEST", box.stick, need_bytes=2 ** 60
        )
        assert freed_files == 3

        states = dict(box.connection.execute(
            "SELECT event_folder, state FROM natix_mirror"
        ).fetchall())
        assert all(state == "pruned" for state in states.values()), states
        assert not (box.stick / "TeslaCam/SentryClips/2026-02-16_08-00-00").exists()


def test_prune_orders_by_event_age_not_by_bucket():
    """
    event_folder is 'TeslaCam/<bucket>/<stamp>'. Sorting the whole string sorts
    by bucket first, so 'SavedClips' would always be pruned before
    'SentryClips' however new it was - and SavedClips are precisely the clips
    the driver chose to keep. Age has to win.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-10_08-00-00": ["front"],     # oldest, goes to SentryClips
            "2026-02-20_08-00-00": ["back"],      # newest, forced into SavedClips
        })
        box.connection.execute(
            "UPDATE clips SET clip_source='SavedClips' WHERE filename LIKE '2026-02-20%'"
        )
        box.connection.commit()
        natix.mirror(box.connection, "uuid:TEST", box.stick)

        saved = box.stick / "TeslaCam/SavedClips/2026-02-20_08-00-00"
        sentry = box.stick / "TeslaCam/SentryClips/2026-02-10_08-00-00"
        assert saved.is_dir() and sentry.is_dir()

        # Free just enough for one event: the older SentryClips one must go.
        one_clip = (box.archive / "2026-02-10_08-00-00-front.mp4").stat().st_size
        natix.prune_oldest(box.connection, "uuid:TEST", box.stick,
                           need_bytes=natix.free_bytes(box.stick) + one_clip)

        assert not sentry.exists(), "the older event should have been pruned"
        assert saved.exists(), "pruned a newer SavedClips event before an older one"


def test_prune_never_touches_files_we_did_not_write():
    """
    The stick is the user's, and its own firmware writes to it too. Pruning is
    driven entirely by our `natix_mirror` rows, so anything else survives.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)

        foreign = box.stick / "TeslaCam/SavedClips/someone-elses"
        foreign.mkdir(parents=True)
        (foreign / "holiday.mp4").write_bytes(b"not ours")
        (box.stick / "natix.json").write_bytes(b"{}")

        natix.prune_oldest(box.connection, "uuid:TEST", box.stick, need_bytes=2 ** 60)

        assert (foreign / "holiday.mp4").exists(), "pruned a file we did not write"
        assert (box.stick / "natix.json").exists(), "pruned the stick's own metadata"


def test_ignore_free_space_copies_anyway():
    """
    For the case where the mount's free-space figure is simply wrong: relan's
    exfat-fuse mounts the stick, exfatprogs' fsck calls the volume clean, and
    the two disagree about allocation. With the override on, the pre-check gets
    out of the way and the filesystem itself decides.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front"]})
        saved_reserve = natix.NATIX_RESERVE_MB
        saved_ignore = natix.NATIX_IGNORE_FREE_SPACE
        # A reserve larger than any filesystem: the pre-check can only refuse.
        natix.NATIX_RESERVE_MB = 1024 * 1024 * 1024
        natix.NATIX_IGNORE_FREE_SPACE = True
        try:
            result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        finally:
            natix.NATIX_RESERVE_MB = saved_reserve
            natix.NATIX_IGNORE_FREE_SPACE = saved_ignore

        assert result.copied == 1, result
        assert result.stopped_reason is None


def test_ignore_free_space_still_survives_a_real_enospc():
    """
    The override is only safe because a genuine out-of-space is still handled:
    no partial file under a real name, no stray .part, source untouched.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front"]})
        target = box.stick / "TeslaCam/SentryClips/2026-02-16_08-00-00/2026-02-16_08-00-00-front.mp4"

        saved_ignore = natix.NATIX_IGNORE_FREE_SPACE
        original_copy = natix.shutil.copyfileobj
        natix.NATIX_IGNORE_FREE_SPACE = True

        def out_of_space(*_args, **_kwargs):
            raise OSError(28, "No space left on device")

        natix.shutil.copyfileobj = out_of_space
        try:
            result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        finally:
            natix.shutil.copyfileobj = original_copy
            natix.NATIX_IGNORE_FREE_SPACE = saved_ignore

        assert result.failed == 1 and result.copied == 0, result
        assert not target.exists()
        assert [p.name for p in box.stick.rglob("*.part")] == []
        assert (box.archive / "2026-02-16_08-00-00-front.mp4").exists()


def test_mirror_stops_cleanly_when_the_stick_is_full():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front", "back"]})
        saved_reserve = natix.NATIX_RESERVE_MB
        # A reserve larger than the whole filesystem: nothing can ever fit.
        natix.NATIX_RESERVE_MB = 1024 * 1024 * 1024
        try:
            result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        finally:
            natix.NATIX_RESERVE_MB = saved_reserve

        assert result.copied == 0, result
        assert result.stopped_reason and "free on the stick" in result.stopped_reason
        assert list(box.stick.rglob("*.mp4")) == []


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def test_plan_is_newest_first_by_default():
    """
    The archive is bigger than the stick, so this is a rolling window and the
    only question is which end of history it keeps. Newest, for a tool whose
    job is telling you who is following you *now*.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-16_08-00-00": ["front"],
            "2026-02-18_08-00-00": ["front"],
        })
        plans, _ = natix.build_plan(box.connection, "uuid:TEST")
        assert [plan.event for plan in plans] == \
            sorted((plan.event for plan in plans), reverse=True)


def test_plan_order_can_be_flipped_to_oldest():
    saved = natix.NATIX_MIRROR_ORDER
    natix.NATIX_MIRROR_ORDER = "oldest"
    try:
        with Sandbox() as box:
            seed_clips(box.connection, box.archive, {
                "2026-02-18_08-00-00": ["front"],
                "2026-02-16_08-00-00": ["front"],
            })
            plans, _ = natix.build_plan(box.connection, "uuid:TEST")
            assert [p.event for p in plans] == sorted(p.event for p in plans)
    finally:
        natix.NATIX_MIRROR_ORDER = saved


def test_a_pruned_clip_is_not_recopied():
    """
    The bug the real stick exposed. A pruned clip used to look un-mirrored on
    the next pass, so it got recopied, evicting something else, forever.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)
        natix.prune_oldest(box.connection, "uuid:TEST", box.stick, need_bytes=2 ** 60)

        plans, _ = natix.build_plan(box.connection, "uuid:TEST")
        assert plans == [], "a pruned clip came back as a candidate"


def test_mirror_converges_on_a_stick_too_small_for_the_archive():
    """
    The regression test for the rewrite loop observed on real hardware:
    60 done / 72 pruned, oscillating forever at 9MB/s.

    Give the mirror an archive several times the stick's capacity and run it
    until it stops changing. It must reach a fixed point - and hold the NEWEST
    clips when it gets there.
    """
    with Sandbox() as box:
        events = {f"2026-02-{day:02d}_08-00-00": ["front", "back"]
                  for day in range(1, 13)}          # 24 clips
        seed_clips(box.connection, box.archive, events, size=100_000)

        capacity = 700_000                           # room for ~7 clips
        saved_free, saved_reserve = natix.free_bytes, natix.NATIX_RESERVE_MB
        natix.NATIX_RESERVE_MB = 0
        natix.free_bytes = lambda root: max(
            0, capacity - sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
        )
        try:
            history = []
            for _ in range(12):
                natix.mirror(box.connection, "uuid:TEST", box.stick)
                on_stick = sorted(p.name for p in box.stick.rglob("*.mp4"))
                history.append(tuple(on_stick))
                if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
                    break
            else:
                raise AssertionError(
                    f"never settled; last three states: {history[-3:]}"
                )
        finally:
            natix.free_bytes = saved_free
            natix.NATIX_RESERVE_MB = saved_reserve

        final = history[-1]
        assert final, "converged on an empty stick"
        # Newest-first means what survives is the recent end of the archive.
        assert any("2026-02-12" in name for name in final), final
        assert not any("2026-02-01" in name for name in final), final


def test_never_evicts_newer_footage_to_fit_older():
    """
    A window that trades in both directions never settles. If the next
    candidate is older than everything on the stick, stop rather than evict.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-20_08-00-00": ["front"],
            "2026-02-01_08-00-00": ["front"],
        }, size=100_000)

        saved_free, saved_reserve = natix.free_bytes, natix.NATIX_RESERVE_MB
        natix.NATIX_RESERVE_MB = 0
        capacity = 150_000                            # exactly one clip
        natix.free_bytes = lambda root: max(
            0, capacity - sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
        )
        try:
            result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        finally:
            natix.free_bytes = saved_free
            natix.NATIX_RESERVE_MB = saved_reserve

        names = [p.name for p in box.stick.rglob("*.mp4")]
        assert names == ["2026-02-20_08-00-00-front.mp4"], names
        assert "older than everything already on it" in (result.stopped_reason or "")


def test_plan_counts_clips_whose_files_are_gone():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front", "back"]})
        (box.archive / "2026-02-16_08-00-00-back.mp4").unlink()
        plans, missing = natix.build_plan(box.connection, "uuid:TEST")
        assert len(plans) == 1 and missing == 1


def test_limit_caps_a_pass():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-16_08-00-00": ["front", "back", "left_repeater", "right_repeater"],
        })
        result = natix.mirror(box.connection, "uuid:TEST", box.stick, limit=2)
        assert result.copied == 2, result
        assert len(list(box.stick.rglob("*.mp4"))) == 2


def test_a_done_clip_missing_from_the_stick_is_recopied():
    """
    A 'done' row is a claim, and the claim can be false.

    After the stick was reformatted, rows recorded against the new volume
    described files that had actually landed on the Jetson's SD card during a
    failed mount and were then cleaned up. build_plan excluded them as already
    mirrored, so those clips would never have been copied again and the stick
    would have stayed permanently short while the database insisted it was
    complete. Verify the file, do not trust the row.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-16_08-00-00": ["front", "back"],
        })
        result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        assert result.copied == 2

        # Second pass with everything in place copies nothing.
        again, _ = natix.build_plan(box.connection, "uuid:TEST", root=box.stick)
        assert again == [], "a clip that is present must not be re-planned"

        # Now the stick loses a file - reformat, manual delete, cleaned stray.
        victim = sorted(box.stick.rglob("*.mp4"))[0]
        victim.unlink()

        replanned, _ = natix.build_plan(box.connection, "uuid:TEST", root=box.stick)
        assert len(replanned) == 1, replanned
        assert replanned[0].filename == victim.name

        # And a full pass puts it back.
        healed = natix.mirror(box.connection, "uuid:TEST", box.stick)
        assert healed.copied == 1, healed
        assert victim.exists()


def test_a_pruned_clip_is_still_not_recopied_when_verifying():
    """
    The other half: pruned files are absent deliberately.

    Re-planning them is the rewrite loop - prune the oldest to make room, see
    it missing, copy it back, evict something else - so the existence check
    must apply to 'done' only.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front"]})
        natix.mirror(box.connection, "uuid:TEST", box.stick)
        box.connection.execute(
            "UPDATE natix_mirror SET state='pruned' WHERE device_id='uuid:TEST'"
        )
        box.connection.commit()
        for path in box.stick.rglob("*.mp4"):
            path.unlink()

        plans, _ = natix.build_plan(box.connection, "uuid:TEST", root=box.stick)
        assert plans == [], "a pruned clip must stay pruned even though it is gone"


def test_stop_lands_between_files_not_after_the_batch():
    """
    A `systemctl stop` has to take effect mid-pass.

    This is not a nicety. The worker only checked its stop flag between passes,
    so a stop waited for the whole batch - minutes over USB 2.0, past
    TimeoutStopSec, and systemd answers that with SIGKILL while a copy is in
    flight. The unit worked around it with an ExecStop that unmounted the
    stick, which systemd runs BEFORE signalling the process: the filesystem
    went away underneath a live copy ([Errno 103]) and the rest of that pass
    landed on the Jetson's own SD card. Stopping between files is what makes
    the unmount safe, so the flag has to be honoured inside the loop.
    """
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-16_08-00-00": ["front", "back", "left_repeater", "right_repeater"],
        })

        copied_before_stop = 2
        state = {"calls": 0}

        def should_stop():
            # Stop once two files are done, mimicking a signal arriving mid-pass.
            state["calls"] += 1
            return state["calls"] > copied_before_stop

        result = natix.mirror(
            box.connection, "uuid:TEST", box.stick, should_stop=should_stop
        )

        assert result.copied == copied_before_stop, result
        assert result.stopped_reason, "a stop must say why it stopped"
        assert "asked to stop" in result.stopped_reason
        # Whatever was copied is complete and consistent: no partial files, and
        # nothing half-renamed. That is what makes the unmount after it safe.
        assert list(box.stick.rglob("*.part")) == []
        assert len(list(box.stick.rglob("*.mp4"))) == copied_before_stop


def test_no_stop_hook_still_copies_everything():
    """The hook is optional; absent it, nothing changes."""
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {
            "2026-02-16_08-00-00": ["front", "back"],
        })
        result = natix.mirror(box.connection, "uuid:TEST", box.stick)
        assert result.copied == 2, result
        assert not result.stopped_reason


def test_dry_run_writes_nothing():
    with Sandbox() as box:
        seed_clips(box.connection, box.archive, {"2026-02-16_08-00-00": ["front"]})
        result = natix.mirror(box.connection, "uuid:TEST", box.stick, dry_run=True)
        assert result.copied == 1
        assert list(box.stick.rglob("*")) == []
        assert box.connection.execute("SELECT COUNT(*) FROM natix_mirror").fetchone()[0] == 0


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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
