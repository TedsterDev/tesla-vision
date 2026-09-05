"""
test_poller.py

Behavioural tests for drive segmentation.

This is the layer where a wrong answer is invisible. A poll with the wrong
status is one bad row; a drive that will not close is a permanent distortion of
every question correlate.py asks, because distinct_drives is its heaviest
signal and detect_active_tail groups by drive_id. None of that raises, and
none of it looks wrong in the polls table.

Run:  python3 tests/test_poller.py
"""
import sqlite3
import sys
import uuid

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import poller                                           # noqa: E402
from src.db import SCHEMA_STATEMENTS                             # noqa: E402

METERS_PER_DEGREE_LAT = 111320.0
HOME_LAT, HOME_LON = 33.1696, -117.2259


def fresh_database():
    """An in-memory database with the real schema, so nothing touches disk."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    return connection


def write_poll(connection, ts, status, north_meters=0.0):
    """One poll, attached to a drive the way the services attach it."""
    poll = {
        "id": uuid.uuid4().hex[:12],
        "ts": ts,
        "lat": HOME_LAT + north_meters / METERS_PER_DEGREE_LAT,
        "lon": HOME_LON,
        "heading": 0.0,
        "speed": 25.0 if status == "D" else 1.2,
        "power": None,
        "shift_state": None,
        "status": status,
        "loc_available": 1,
        "odometer": None,
        "street": None,
        "city": None,
        "drive_id": None,
        "geocode_id": None,
    }
    poll["drive_id"] = poller.attach_drive(connection, poll)
    columns = ", ".join(poll)
    connection.execute(
        f"INSERT INTO polls ({columns}) VALUES ({', '.join('?' * len(poll))})",
        tuple(poll.values()),
    )
    connection.commit()
    return poll


def drives(connection):
    return list(connection.execute("SELECT * FROM drives ORDER BY start_ts"))


def dangling_count(connection):
    return connection.execute(
        "SELECT COUNT(*) FROM polls WHERE drive_id IS NOT NULL "
        "AND drive_id NOT IN (SELECT id FROM drives)"
    ).fetchone()[0]


def test_a_drive_closes_on_time_since_movement_not_time_since_poll():
    # The bug this replaced: the gap was measured to the last poll of ANY
    # status, and a parked poll is itself attached to the open drive. With a
    # receiver writing continuously the gap could never grow, so the close
    # branch was unreachable and one drive absorbed the whole database.
    connection = fresh_database()
    for offset, tick in enumerate(range(0, 100, 10)):
        write_poll(connection, tick, "D", north_meters=offset * 200)
    # Now parked, and reported often - which is what a GNSS does.
    for tick in range(100, 500, 5):
        write_poll(connection, tick, "P", north_meters=1800)

    open_drives = [row for row in drives(connection) if row["is_open"]]
    assert not open_drives, "the drive never closed despite 400s without movement"


def test_parked_noise_contributes_no_distance():
    # A stationary receiver's jitter has no net direction but plenty of length.
    # Summing it accrued 23.6 miles a day on a desk.
    connection = fresh_database()
    write_poll(connection, 0, "D", north_meters=0)
    write_poll(connection, 10, "D", north_meters=300)
    distance_after_driving = drives(connection)[0]["distance_miles"]

    # 40 parked polls wandering back and forth by 10m, well inside the timeout.
    for index, tick in enumerate(range(20, 220, 5)):
        write_poll(connection, tick, "P", north_meters=300 + (10 if index % 2 else -10))

    assert abs(drives(connection)[0]["distance_miles"] - distance_after_driving) < 1e-9, (
        "parked noise accumulated distance"
    )


def test_a_drive_that_went_nowhere_is_deleted():
    # The replacement for a guard that read `seconds_since_start < 60` and was
    # unreachable: reaching it already required a 300s gap, so the test could
    # never be true and no phantom drive was ever discarded.
    connection = fresh_database()
    write_poll(connection, 0, "D", north_meters=0)
    write_poll(connection, 10, "D", north_meters=20)     # 20m, then stops
    assert len(drives(connection)) == 1, "no drive opened to discard"
    write_poll(connection, 400, "P", north_meters=20)    # past the timeout

    assert drives(connection) == [], "a drive that moved 20m was kept"


def test_discarding_a_drive_relinks_its_polls_rather_than_orphaning_them():
    # Deleting the drives row while polls still point at it is strictly worse
    # than keeping it: location_at() stamps detections from polls, and
    # correlate.py counts distinct_drives off the detection's own column
    # without joining back - so the phantom would still be counted while the
    # operator's view of it was gone.
    connection = fresh_database()
    write_poll(connection, 0, "D", north_meters=0)
    write_poll(connection, 10, "D", north_meters=20)
    write_poll(connection, 400, "P", north_meters=20)

    assert dangling_count(connection) == 0, "polls point at a deleted drive"
    kept = connection.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
    assert kept == 3, f"polls were deleted along with the drive: {kept} of 3 left"


def test_a_short_real_drive_is_kept():
    # The failure mode of measuring by duration instead of displacement. A
    # 45-second hop to the end of the street is a real journey; ten minutes
    # spent stationary in a parking space is not.
    connection = fresh_database()
    write_poll(connection, 0, "D", north_meters=0)
    write_poll(connection, 20, "D", north_meters=250)
    write_poll(connection, 45, "D", north_meters=500)
    write_poll(connection, 400, "P", north_meters=500)

    remaining = drives(connection)
    assert len(remaining) == 1, "a genuine 500m drive was discarded"
    assert remaining[0]["is_open"] == 0


def test_a_drive_out_and_back_still_counts_as_a_drive():
    # Displacement is measured to the FURTHEST point reached, not to where the
    # drive ended. A trip to the shop and home again went somewhere.
    connection = fresh_database()
    write_poll(connection, 0, "D", north_meters=0)
    write_poll(connection, 30, "D", north_meters=800)
    write_poll(connection, 60, "D", north_meters=0)
    write_poll(connection, 400, "P", north_meters=0)

    assert len(drives(connection)) == 1, "a there-and-back drive was discarded"


def test_a_traffic_jam_does_not_split_one_journey_into_two():
    # The consequence that matters: detect_active_tail needs
    # TAIL_MIN_ENCOUNTERS_IN_DRIVE sightings within ONE drive. Splitting a
    # journey at the jam is how a real tail stops being detectable, in the
    # traffic where it is most observable.
    connection = fresh_database()
    position = 0.0
    tick = 0
    for _ in range(10):                       # driving normally
        write_poll(connection, tick, "D", north_meters=position)
        position += 300
        tick += 10
    # Crawling: the gate reads 'P' because 45s of displacement is small, but
    # movement resumes well inside DRIVE_IDLE_TIMEOUT_SECONDS.
    for _ in range(6):
        for _ in range(20):
            write_poll(connection, tick, "P", north_meters=position)
            tick += 5
        position += 40
        write_poll(connection, tick, "D", north_meters=position)
        tick += 5
    for _ in range(10):                       # moving freely again
        write_poll(connection, tick, "D", north_meters=position)
        position += 300
        tick += 10

    assert len(drives(connection)) == 1, (
        f"the jam split one journey into {len(drives(connection))} drives"
    )


def test_the_tesla_path_still_segments_the_same_way():
    # attach_drive is shared. The Fleet API reports a parked car every
    # PARKED_POLL_SECONDS with speed None, so its polls are sparse and 'P';
    # nothing above may change what that produces.
    connection = fresh_database()
    write_poll(connection, 0, "D", north_meters=0)
    write_poll(connection, 60, "D", north_meters=1600)
    write_poll(connection, 120, "D", north_meters=3200)
    for tick in (420, 720, 1020):             # parked, one poll per 300s
        write_poll(connection, tick, "P", north_meters=3200)

    recorded = drives(connection)
    assert len(recorded) == 1, "the Tesla path stopped producing one drive"
    assert recorded[0]["is_open"] == 0, "the Tesla path stopped closing drives"
    assert recorded[0]["distance_miles"] > 1.0, recorded[0]["distance_miles"]


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
