"""
backfill.py

Re-run the Scout stages over clips that are already in PROCESSED_DIR.

The original pipeline processed clips with YOLO only, so the ~130 clips already
on disk have alerts but no plate reads, no face identities and no correlation
history. This walks them again through the full pipeline and populates those
tables, which means you get value out of existing footage instead of having to
wait for new drives.

Safe to re-run: a clip already marked processed in the `clips` table is skipped,
so an interrupted backfill resumes where it stopped.

Usage (on the host, through the running processor container):
    docker compose exec processor python -u scripts/backfill.py
    docker compose exec processor python -u scripts/backfill.py --limit 20
    docker compose exec processor python -u scripts/backfill.py --reset
"""
import argparse
import sys
import time

from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import PROCESSED_DIR, ensure_dirs           # noqa: E402
from src.clipmeta import describe_clip                       # noqa: E402
from src.db import connect, now_ts                           # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-analyse already-processed TeslaCam clips")
    parser.add_argument("--limit", type=int, default=0, help="stop after N clips (0 = all)")
    parser.add_argument("--reset", action="store_true",
                        help="clear plate/face detections first and start over")
    arguments = parser.parse_args()

    ensure_dirs()
    connection = connect()

    if arguments.reset:
        print("[backfill] clearing plate and face tables")
        for table in ("plate_detections", "plates", "face_detections", "faces",
                      "correlations", "clips"):
            connection.execute(f"DELETE FROM {table}")
        connection.commit()

    # Imported late: loading YOLO and the ONNX models takes a few seconds, and
    # doing it before argument parsing makes --help feel broken.
    from ultralytics import YOLO
    from src.alpr import PlateReader
    from src.faces import FaceEngine
    from src.processor import (
        analyze_clip, build_clip_context, record_face_detections,
        record_plate_detections, register_clip,
    )
    from src.correlate import run_correlation_pass

    model = YOLO("yolo26n.pt")
    plate_reader = PlateReader()
    face_engine = FaceEngine()
    print(f"[backfill] ALPR:  {plate_reader.status_text()}")
    print(f"[backfill] faces: {face_engine.status_text()}")

    clips = sorted(PROCESSED_DIR.glob("*.mp4"))
    print(f"[backfill] {len(clips)} clips in {PROCESSED_DIR}")

    processed = 0
    skipped = 0
    plates_found = 0
    faces_found = 0
    illegible_total = 0
    started_at = time.time()

    for clip_path in clips:
        if arguments.limit and processed >= arguments.limit:
            break

        already = connection.execute(
            "SELECT status FROM clips WHERE filename=?", (clip_path.name,)
        ).fetchone()
        if already and already["status"] == "processed":
            skipped += 1
            continue

        try:
            clip_id = register_clip(connection, clip_path)
            context = build_clip_context(connection, clip_id, clip_path)

            plate_reader.illegible_crop_count = 0
            analysis = analyze_clip(model, clip_path, plate_reader, face_engine)
            illegible_total += plate_reader.illegible_crop_count

            plate_text = record_plate_detections(connection, analysis, context)
            face_names = record_face_detections(connection, face_engine, analysis, context)

            if plate_text:
                plates_found += 1
            faces_found += len(face_names)

            connection.execute(
                "UPDATE clips SET status='processed', processed_ts=? WHERE id=?",
                (now_ts(), clip_id),
            )
            connection.commit()

            processed += 1
            detail = []
            if plate_text:
                detail.append(f"plate={plate_text}")
            if face_names:
                detail.append(f"faces={','.join(face_names)}")
            if plate_reader.illegible_crop_count:
                detail.append(f"illegible={plate_reader.illegible_crop_count}")

            print(f"[backfill] {processed}/{len(clips) - skipped} "
                  f"{describe_clip(clip_path.name)} "
                  f"veh={analysis.vehicle_count} ppl={analysis.person_count} "
                  f"{' '.join(detail)}")

        except Exception as exception_object:
            print(f"[backfill] ⧱❗️ ERROR {clip_path.name}: {exception_object}")

    print(f"\n[backfill] re-analysed {processed} clips ({skipped} already done) "
          f"in {time.time() - started_at:.0f}s")
    print(f"[backfill] clips yielding a plate read: {plates_found}")
    print(f"[backfill] face sightings recorded:     {faces_found}")
    print(f"[backfill] plate crops too dark to read: {illegible_total}")

    print("\n[backfill] running correlation pass…")
    newly_raised = run_correlation_pass(connection, verbose=True)
    print(f"[backfill] {len(newly_raised)} new findings")

    for table in ("plates", "plate_detections", "faces", "face_detections", "correlations"):
        count = connection.execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"]
        print(f"[backfill]   {table:20} {count}")

    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
