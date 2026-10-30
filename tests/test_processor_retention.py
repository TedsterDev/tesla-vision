"""
test_processor_retention.py

processed/ keeps finished clips for the GIF worker, the box annotator, the
NATIX mirror and backfills. With live footage from carstore.py it grows by
roughly 10 GB per hour of driving, so the processor prunes the oldest. The
failure that matters is the wrong ones going: footage must leave oldest-first,
and only as much as needed.

Run:  python3 tests/test_processor_retention.py
"""
import os
import shutil
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SANDBOX = Path(tempfile.mkdtemp(prefix="processor_retention_test_"))
os.environ["BASE_DIR"] = str(SANDBOX)
os.environ["DB_PATH"] = str(SANDBOX / "scout.db")
os.environ.setdefault("MODELS_DIR", str(SANDBOX / "models"))

try:
    from src import processor
    from src.common import INBOX_DIR, PROCESSED_DIR, ensure_dirs
    from src.db import connect
except ImportError as import_error:                       # pragma: no cover
    print(f"  skip  (no processor dependencies here: {import_error})")
    sys.exit(0)

ensure_dirs()
GB = 1000 ** 3


def clip(directory: Path, name: str, size: int, mtime: int) -> Path:
    path = directory / name
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def fresh_processed() -> Path:
    shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
    PROCESSED_DIR.mkdir(parents=True)
    return PROCESSED_DIR


def test_the_oldest_footage_goes_first_and_only_until_under_the_cap():
    directory = fresh_processed()
    for index, mtime in enumerate((300, 100, 400, 200)):
        clip(directory, f"clip{index}.mp4", 100, mtime)
    removed = processor.prune_processed(directory, keep_bytes=250, min_free_bytes=0,
                                        free_bytes=lambda: 10 * GB)
    assert removed == 2, removed
    # mtimes 100 and 200 were the oldest recordings.
    assert sorted(path.name for path in directory.glob("*.mp4")) == ["clip0.mp4", "clip2.mp4"]


def test_low_disk_prunes_even_under_the_cap():
    directory = fresh_processed()
    for index in range(3):
        clip(directory, f"clip{index}.mp4", 10, 100 + index)
    free = iter([1, 1, 50])
    removed = processor.prune_processed(directory, keep_bytes=10 ** 9, min_free_bytes=10,
                                        free_bytes=lambda: next(free))
    assert removed == 2 and [p.name for p in directory.glob("*.mp4")] == ["clip2.mp4"]


def test_nothing_goes_when_there_is_room():
    directory = fresh_processed()
    clip(directory, "clip.mp4", 10, 100)
    assert processor.prune_processed(directory) == 0
    assert (directory / "clip.mp4").exists()


def test_a_clip_that_vanishes_mid_scan_is_skipped():
    directory = fresh_processed()
    clip(directory, "real.mp4", 10, 100)
    (directory / "dangling.mp4").symlink_to(directory / "gone.mp4")
    assert processor.prune_processed(directory, keep_bytes=0, min_free_bytes=0,
                                     free_bytes=lambda: GB) == 1


class NoDetections:
    face_engine = None

    def analyze(self, clip_path, progress=None):
        return processor.ClipAnalysis(), "cpu"


def test_a_processed_clip_is_moved_out_of_the_inbox_and_the_oldest_pruned():
    fresh_processed()
    clip(PROCESSED_DIR, "2026-10-29_07-00-00-front.mp4", 100, 100)
    incoming = clip(INBOX_DIR, "2026-10-29_10-00-00-front.mp4", 100, 200)
    saved = processor.PROCESSED_KEEP_GB
    processor.PROCESSED_KEEP_GB = 0          # everything over the cap: prune to the newest
    connection = connect()
    try:
        alerted, device = processor.process_clip(connection, NoDetections(), incoming)
    finally:
        processor.PROCESSED_KEEP_GB = saved
        connection.close()
    assert (alerted, device) == (False, "cpu")
    assert not incoming.exists()
    # Both went: with a 0 GB cap even the clip just processed is pruned.
    assert list(PROCESSED_DIR.glob("*.mp4")) == []


def test_a_processed_clip_with_room_to_spare_prunes_nothing():
    fresh_processed()
    incoming = clip(INBOX_DIR, "2026-10-29_11-00-00-front.mp4", 100, 300)
    connection = connect()
    try:
        processor.process_clip(connection, NoDetections(), incoming)
    finally:
        connection.close()
    assert [p.name for p in PROCESSED_DIR.glob("*.mp4")] == ["2026-10-29_11-00-00-front.mp4"]


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
