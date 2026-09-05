#!/usr/bin/env python3
"""
gps_purge_phantom.py - remove drives and polls invented by stationary GPS noise.

    python3 scripts/gps_purge_phantom.py              # dry run, the default
    python3 scripts/gps_purge_phantom.py --execute

Before the MotionGate landed, src/gps.py classified driving from instantaneous
speed alone. A stationary consumer receiver reports a random walk of roughly
0.5-3.5 mph forever, so a parked car was marked 'D' on 96% of samples, opened
drives it never took, and accumulated real distance_miles from noise.

That matters because distinct_drives is the heaviest signal correlate.py
scores on: left in place, every repeat sighting in the owner's own driveway
accumulates "seen on N separate drives" and starts scoring like a tail. The
rows are not merely untidy, they actively invert the answer the system exists
to give.

WHAT COUNTS AS PHANTOM, and why it is decided by measurement rather than by a
timestamp cutoff: a drive whose total NET displacement - start point to the
furthest point it ever reached - is under PHANTOM_DISPLACEMENT_METERS never
went anywhere, whatever its distance_miles says. A real drive ends somewhere
else, or at minimum passes somewhere else. This is the same quantity the gate
now uses live, applied backwards over what is already stored, so a genuine
short drive recorded before the fix is KEPT rather than assumed bad.

Polls are relinked, never deleted: a poll is still a true record that the car
was at that spot at that time, and location_at() reads them to stamp clips. It
is only the *drive* that was fiction, so drive_id is set to NULL and the row
stays.
"""
import argparse
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DB_PATH                                   # noqa: E402
from src.db import connect                                       # noqa: E402
from src.geo import haversine_miles                              # noqa: E402

# The gate opens at 50m. This is deliberately more generous: it is deciding
# whether to DELETE, so the benefit of the doubt goes to keeping a drive.
PHANTOM_DISPLACEMENT_METERS = float(os.environ.get("PHANTOM_DISPLACEMENT_METERS", 120.0))


def net_displacement_meters(points):
    """The furthest any point in the drive got from where it started."""
    if len(points) < 2:
        return 0.0
    start_lat, start_lon = points[0]
    return max(
        haversine_miles(start_lat, start_lon, lat, lon) * 1609.344
        for lat, lon in points[1:]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true",
                        help="actually delete. Without this, nothing is written.")
    arguments = parser.parse_args()

    # Printed before anything else, and not merely for tidiness. BASE_DIR
    # defaults to ~/tesla-alerts while the service runs with
    # BASE_DIR=/mnt/jetsondata/tesla-alerts, so a shell that has not exported
    # it opens a DIFFERENT database - and connect() calls ensure_dirs(), so it
    # cheerfully creates that whole tree and reports "no drives recorded".
    # A destructive script that silently addresses the wrong file, and whose
    # empty result reads as good news, is exactly the wrong failure to have.
    print(f"database: {DB_PATH}")
    if not Path(DB_PATH).exists():
        print("that file does not exist. Set BASE_DIR to the service's value:")
        print("  BASE_DIR=/mnt/jetsondata/tesla-alerts python3 scripts/gps_purge_phantom.py")
        return 1

    connection = connect()
    connection.row_factory = __import__("sqlite3").Row

    drives = list(connection.execute(
        "SELECT id, start_ts, end_ts, distance_miles, poll_count FROM drives ORDER BY start_ts"
    ))
    if not drives:
        print("no drives recorded - nothing to do")
        return 0

    phantom, genuine = [], []
    for drive in drives:
        points = [(row["lat"], row["lon"]) for row in connection.execute(
            "SELECT lat, lon FROM polls WHERE drive_id=? AND lat IS NOT NULL ORDER BY ts",
            (drive["id"],),
        )]
        displacement = net_displacement_meters(points)
        record = (drive, displacement, len(points))
        (phantom if displacement < PHANTOM_DISPLACEMENT_METERS else genuine).append(record)

    print(f"{len(drives)} drives: {len(genuine)} genuine, {len(phantom)} phantom "
          f"(net displacement under {PHANTOM_DISPLACEMENT_METERS:.0f}m)\n")

    for drive, displacement, count in phantom:
        span = (drive["end_ts"] or drive["start_ts"]) - drive["start_ts"]
        print(f"  PHANTOM {drive['id']}  {count:4} polls  {span:5}s  "
              f"claims {drive['distance_miles']:.3f}mi  actually went {displacement:.1f}m")
    for drive, displacement, count in genuine:
        print(f"  KEEP    {drive['id']}  {count:4} polls  "
              f"claims {drive['distance_miles']:.3f}mi  went {displacement:.0f}m")

    if not phantom:
        print("\nnothing to purge")
        return 0

    orphaned = sum(count for _, _, count in phantom)
    print(f"\n{len(phantom)} drives would be deleted; {orphaned} polls would be "
          f"kept with drive_id set to NULL")

    if not arguments.execute:
        print("\nDRY RUN - nothing written. Re-run with --execute to apply.")
        return 0

    for drive, _, _ in phantom:
        connection.execute("UPDATE polls SET drive_id=NULL WHERE drive_id=?", (drive["id"],))
        connection.execute("DELETE FROM drives WHERE id=?", (drive["id"],))
    connection.commit()
    print(f"\ndeleted {len(phantom)} phantom drives, relinked {orphaned} polls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
