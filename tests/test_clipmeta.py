"""
test_clipmeta.py

The timestamp a clip is stored under decides where every detection in it is
placed on the map. These tests exist because 132 clips were stored 7-8 hours
early and nothing failed: the parser was correct in the zone it happened to
run in, and that zone was wrong.

Run:  python3 tests/test_clipmeta.py
"""
import os
import sys
import time

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import clipmeta    # noqa: E402


def expected(name_date: str, name_time: str, zone: str) -> int:
    naive = datetime.strptime(f"{name_date} {name_time}", "%Y-%m-%d %H:%M:%S")
    return int(naive.replace(tzinfo=ZoneInfo(zone)).timestamp())


def test_the_process_timezone_does_not_change_the_answer():
    # The bug: a container running in UTC parsed "20:49:20" as 20:49 UTC.
    # Force this process into UTC and prove the parser no longer cares.
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        parsed = clipmeta.parse_clip_filename("2026-02-16_20-49-20-back.mp4")
        assert parsed["captured_ts"] == expected("2026-02-16", "20:49:20", clipmeta.CLIP_TIMEZONE), (
            parsed["captured_ts"])
        # And specifically NOT the UTC reading that the live database held.
        assert parsed["captured_ts"] != 1771274960, "still parsing in the process zone"
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def test_daylight_saving_is_applied_per_date():
    # February is UTC-8, August is UTC-7. A fixed offset would get one wrong.
    winter = clipmeta.parse_clip_filename("2026-02-16_12-00-00-front.mp4")["captured_ts"]
    summer = clipmeta.parse_clip_filename("2026-08-03_12-00-00-front.mp4")["captured_ts"]
    utc_winter = int(datetime(2026, 2, 16, 12, tzinfo=ZoneInfo("UTC")).timestamp())
    utc_summer = int(datetime(2026, 8, 3, 12, tzinfo=ZoneInfo("UTC")).timestamp())
    assert winter - utc_winter == 8 * 3600, (winter - utc_winter) / 3600
    assert summer - utc_summer == 7 * 3600, (summer - utc_summer) / 3600


def test_the_live_sample_corrects_by_eight_hours():
    # The exact row cited in the audit, and the delta the migration must apply.
    parsed = clipmeta.parse_clip_filename("2026-02-16_20-49-20-back.mp4")
    assert parsed["captured_ts"] - 1771274960 == 28800


def test_an_unparseable_name_still_yields_none_not_a_guess():
    parsed = clipmeta.parse_clip_filename("error_whatever.mp4")
    assert parsed["captured_ts"] is None


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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
