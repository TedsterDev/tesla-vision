"""
test_ingest.py

The one rule ingest enforces is that it never reads the car's volume while
the USB gadget is offering that volume to the car. Everything else here is
the copy being atomic and idempotent.

Run:  python3 tests/test_ingest.py
"""
import os
import shutil
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stability checks sleep; make them instant for tests.
os.environ.setdefault("BASE_DIR", tempfile.mkdtemp(prefix="ingest-test-"))
from src import common    # noqa: E402
from src import ingest    # noqa: E402


def fake_configfs(root: Path, backing: str | None) -> Path:
    """A gadget tree with one mass-storage LUN, or none if backing is None."""
    lun = root / "g1" / "functions" / "mass_storage.0" / "lun.0"
    lun.mkdir(parents=True)
    if backing is not None:
        (lun / "file").write_text(backing + "\n")
    return root


def test_the_volume_the_car_holds_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        configfs = fake_configfs(Path(tmp) / "cfg", "/dev/sdz1")
        state, reason = ingest.source_state(
            Path("/mnt/teslacam"), configfs, resolve_mount=lambda p: "/dev/sdz1")
        assert state == "exported", (state, reason)
        assert "offered to the car" in reason


def test_a_mounted_volume_the_gadget_is_not_offering_is_ready():
    with tempfile.TemporaryDirectory() as tmp:
        configfs = fake_configfs(Path(tmp) / "cfg", None)
        state, device = ingest.source_state(
            Path("/mnt/teslacam"), configfs, resolve_mount=lambda p: "/dev/sdz1")
        assert state == "ready", (state, device)
        assert device == "/dev/sdz1"


def test_a_different_exported_device_does_not_block_reading():
    # The NATIX stick being offered somewhere must not stop TeslaCam ingest.
    with tempfile.TemporaryDirectory() as tmp:
        configfs = fake_configfs(Path(tmp) / "cfg", "/dev/sdy1")
        state, _ = ingest.source_state(
            Path("/mnt/teslacam"), configfs, resolve_mount=lambda p: "/dev/sdz1")
        assert state == "ready", state


def test_an_unmounted_source_is_absent_not_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        configfs = fake_configfs(Path(tmp) / "cfg", None)
        state, reason = ingest.source_state(
            Path("/mnt/teslacam"), configfs, resolve_mount=lambda p: None)
        assert state == "absent", state
        assert "not a mountpoint" in reason


def test_a_gadget_tree_with_no_lun_file_is_handled():
    with tempfile.TemporaryDirectory() as tmp:
        configfs = Path(tmp) / "cfg"
        (configfs / "g1" / "functions" / "mass_storage.0" / "lun.0").mkdir(parents=True)
        assert ingest.exported_backing_devices(configfs) == set()


def _patched_stability(func):
    """Run func with file_is_stable made instant and always-true."""
    original = ingest.file_is_stable
    ingest.file_is_stable = lambda path, **_: True
    try:
        return func()
    finally:
        ingest.file_is_stable = original


def test_a_copy_lands_atomically_with_no_temp_file_left():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "TeslaCam" / "SentryClips" / "2026-02-16_20-49-20-back.mp4"
        src.parent.mkdir(parents=True); src.write_bytes(b"x" * 4096)
        inbox = Path(tmp) / "inbox"; inbox.mkdir()
        landed = _patched_stability(lambda: ingest.safe_copy_to_inbox(src, inbox))
        assert landed == inbox / src.name
        assert landed.read_bytes() == b"x" * 4096
        assert not list(inbox.glob(".tmp_*")), "temporary file left behind"


def test_a_clip_already_in_the_inbox_is_not_copied_again():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "a.mp4"; src.write_bytes(b"new")
        inbox = Path(tmp) / "inbox"; inbox.mkdir()
        (inbox / "a.mp4").write_bytes(b"old")
        result = _patched_stability(lambda: ingest.safe_copy_to_inbox(src, inbox))
        assert result is None
        assert (inbox / "a.mp4").read_bytes() == b"old", "overwrote an existing inbox clip"


def test_dotfiles_are_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / ".tmp_a.mp4"; src.write_bytes(b"partial")
        inbox = Path(tmp) / "inbox"; inbox.mkdir()
        assert _patched_stability(lambda: ingest.safe_copy_to_inbox(src, inbox)) is None
        assert not list(inbox.iterdir())


def test_a_mid_write_clip_is_skipped_this_pass():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "a.mp4"; src.write_bytes(b"growing")
        inbox = Path(tmp) / "inbox"; inbox.mkdir()
        original = ingest.file_is_stable
        ingest.file_is_stable = lambda path, **_: False
        try:
            assert ingest.safe_copy_to_inbox(src, inbox) is None
        finally:
            ingest.file_is_stable = original
        assert not list(inbox.iterdir())


def test_the_heartbeat_is_written_atomically():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "logs" / "ingest.json"
        ingest.write_heartbeat({"state": "absent"}, path)
        assert path.exists() and not path.with_suffix(".json.tmp").exists()


# ---------------------------------------------------------------------------
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test(); print(f"  PASS  {test.__name__}")
        except AssertionError as error:
            failures += 1; print(f"  FAIL  {test.__name__}: {error}")
        except Exception as error:                        # noqa: BLE001
            failures += 1; print(f"  ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    shutil.rmtree(os.environ["BASE_DIR"], ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
