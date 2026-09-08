#!/usr/bin/env python3
"""
migrate_clip_timezone.py - correct captured_ts values parsed in the wrong zone.

    python3 scripts/migrate_clip_timezone.py              # dry run
    python3 scripts/migrate_clip_timezone.py --execute

Tesla names each clip with the car's LOCAL wall-clock time. clipmeta.py used
to parse that with a naive datetime and .timestamp(), which interprets it in
the PROCESS's zone - and every container runs in UTC. So a clip shot at
20:49 Pacific was stored as 20:49 UTC, eight hours early (seven in summer).

That is not a display bug. location_at() takes the clip's timestamp and finds
the nearest GPS poll, so every detection was placed where the car had been
hours EARLIER - typically parked at home - which zeroes the drive signal and
fires the "anchored to one location" suppression on exactly the vehicles the
system exists to notice.

clipmeta.py now parses in CLIP_TIMEZONE. This script recomputes every stored
value from the filename with the corrected parser, which makes it idempotent:
a row that is already right recomputes to the same number and is not touched.

Rows corrected:
    clips.captured_ts    parsed directly from the filename
    alerts.timestamp     was context["ts"], i.e. the same wrong captured_ts;
                         recovered through alerts.source_file
Detection and correlation tables also carry this timestamp but hold no rows
on this deployment; they are reported, and corrected if any exist.
"""
import argparse
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clipmeta import CLIP_TIMEZONE, parse_clip_filename     # noqa: E402
from src.common import DB_PATH                                    # noqa: E402
from src.db import connect                                        # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="apply; default is a dry run")
    arguments = parser.parse_args()

    print(f"database: {DB_PATH}")
    print(f"zone:     {CLIP_TIMEZONE}")
    if not Path(DB_PATH).exists():
        print("that file does not exist. Set BASE_DIR to the service's value:")
        print("  sudo env BASE_DIR=/mnt/jetsondata/tesla-alerts python3 scripts/migrate_clip_timezone.py")
        return 1

    connection = connect()
    connection.row_factory = __import__("sqlite3").Row

    clip_updates = []
    for row in connection.execute("SELECT id, filename, captured_ts FROM clips"):
        corrected = parse_clip_filename(row["filename"])["captured_ts"]
        if corrected is None or corrected == row["captured_ts"]:
            continue
        clip_updates.append((corrected, row["id"], row["filename"], row["captured_ts"]))

    alert_updates = []
    for row in connection.execute("SELECT id, source_file, timestamp FROM alerts WHERE source_file IS NOT NULL"):
        corrected = parse_clip_filename(row["source_file"])["captured_ts"]
        if corrected is None or corrected == row["timestamp"]:
            continue
        alert_updates.append((corrected, row["id"], row["source_file"], row["timestamp"]))

    detection_updates = []
    for table in ("plate_detections", "face_detections"):
        for row in connection.execute(f"SELECT id, source_file, ts FROM {table} WHERE source_file IS NOT NULL"):
            corrected = parse_clip_filename(row["source_file"])["captured_ts"]
            if corrected is None or corrected == row["ts"]:
                continue
            detection_updates.append((table, corrected, row["id"], row["source_file"], row["ts"]))

    total = len(clip_updates) + len(alert_updates) + len(detection_updates)
    print(f"\nclips to correct:      {len(clip_updates)}")
    print(f"alerts to correct:     {len(alert_updates)}")
    print(f"detections to correct: {len(detection_updates)}")
    if clip_updates:
        corrected, _, filename, old = clip_updates[0]
        print(f"\nsample: {filename}")
        print(f"        stored {old}  ->  {corrected}   ({(corrected - old) / 3600:+.0f}h)")
    if total == 0:
        print("\nnothing to do - every stored value already matches the corrected parser")
        return 0
    if not arguments.execute:
        print("\nDRY RUN - nothing written. Re-run with --execute to apply.")
        return 0

    with connection:
        connection.executemany("UPDATE clips SET captured_ts=? WHERE id=?",
                               [(c, i) for c, i, _, _ in clip_updates])
        connection.executemany("UPDATE alerts SET timestamp=? WHERE id=?",
                               [(c, i) for c, i, _, _ in alert_updates])
        for table, corrected, row_id, _, _ in detection_updates:
            connection.execute(f"UPDATE {table} SET ts=? WHERE id=?", (corrected, row_id))
    print(f"\ncorrected {total} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
