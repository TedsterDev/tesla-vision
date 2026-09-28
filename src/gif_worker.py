"""
gif_worker.py

Watches GIF_QUEUE_DIR for JSON job files created by processor.py
Each job is: {"video": "...", "alert_id": "..."},

Creates a 5-second GIF into MEDIA_DIR and updates the alert JSON status.
"""
import time
import json
import subprocess

from pathlib import Path

from src.common import (
    ensure_dirs,
    env_int,
    GIF_QUEUE_DIR,
    MEDIA_DIR,
    PROCESSED_DIR,
    ALERTS_DIR,
)
from src.db import connect

# GIF Generation Config (Scaled for dashboard)
#
# These are preview thumbnails for a dashboard that has to load over a phone
# hotspot, not archival footage - the source MP4 is still on disk if you need
# detail. At the original 640px/10fps a five-second clip came out at 6.8-10 MB
# each, and the Recent view renders many alerts at once, so a single page load
# pulled close to a gigabyte. 480px at 8fps is roughly a third of that and
# still perfectly legible for "what happened here".
#
# Raise them if you want richer previews and have the bandwidth.
GIF_SECONDS = env_int("GIF_SECONDS", 5)
FPS = env_int("GIF_FPS", 8)              # Set Frames per Second (Lower == Smaller GIF)
SCALE_WIDTH = env_int("GIF_SCALE_WIDTH", 480)

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def resolve_video_path(video_str: str) -> Path:
    """
    Processor currently enqueues the INBOX path, then moves the file into PROCESSED_DIR
    This resolver is robust:
        - If the path exists, use it
        - Else try PROCESSED_DIR/<basename>
    """
    video_path = Path(video_str)

    if video_path.exists():
        return video_path
    
    fallback = PROCESSED_DIR / video_path.name
    return fallback

def make_gif_ffmpeg(video_path: Path, out_gif: Path) -> None:
    """
    Create GIFs using ffmpeg.
    - Writes to a temp file then renames (atomic within the same filesystem).

    Two things here are load-bearing and were not obvious:

    `-f gif` is REQUIRED. ffmpeg picks its output format from the filename
    extension, and we deliberately write to a temporary name so a crash can
    never leave a half-written GIF in the media directory. That temporary name
    ends in `.tmp`, which ffmpeg cannot map to any format - it fails with
    "Unable to choose an output format" and exit status 234. Stating the format
    explicitly decouples the two concerns.

    The palette filtergraph is what makes the output legible. A GIF holds 256
    colours; without a palette pass ffmpeg quantises against a fixed web
    palette, which turns night footage into mud. `palettegen` derives an
    optimal palette from these actual frames and `paletteuse` maps to it, in a
    single pass via `split`. That difference matters a lot here, because the
    whole point of the GIF is to let a human look at an alert and tell what
    happened.
    """
    temporary_out_file = out_gif.with_name(out_gif.name + ".tmp")

    # lanczos is a high quality scaling algorithm; split/palettegen/paletteuse
    # builds a per-clip 256-colour palette instead of using the default one.
    video_filter = (
        f"fps={FPS},scale={SCALE_WIDTH}:-1:flags=lanczos,"
        f"split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer"
    )

    cmd = [
        "ffmpeg",
        "-y",                   # Override the output without prompting
        "-hide_banner",         # hide ffmpeg banner
        "-loglevel", "error",   # only print errors
        "-ss", "0",             # 0 seek to time 0
        "-t", str(GIF_SECONDS), # GIF_SECONDS only process that many seconds
        "-i", str(video_path),  # -i video_path input file.
        "-filter_complex", video_filter,
        "-f", "gif",            # state the format; the temp name cannot imply it
        str(temporary_out_file) # output file
    ]

    # Run ffmpeg; raise on non-zero exit
    subprocess.run(cmd, check=True)

    temporary_out_file.replace(out_gif)

def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def update_alert_status(alert_id: str, status: str, connection=None, **updates) -> None:
    """
    Update an alert's status in BOTH places it is recorded.

    The JSON file is the on-disk record the dashboard's Recent view reads; the
    `alerts` table backs the JSON API and the stats counters. Writing only the
    file (as this did before the database existed) left the table permanently
    claiming every alert was still 'gif_queued'.
    """
    alert_path = ALERTS_DIR / f"{alert_id}.json"
    if alert_path.exists():
        alert = load_json(alert_path)
        alert["status"] = status
        alert.update(updates)
        save_json(alert_path, alert)

    if connection is None:
        return

    try:
        connection.execute(
            "UPDATE alerts SET status=?, gif=COALESCE(?, gif) WHERE id=?",
            (status, updates.get("gif"), alert_id),
        )
        connection.commit()
    except Exception as exception_object:
        # A database hiccup must never lose the GIF we just successfully made.
        print(f"[🎞️ gif_worker] could not update alert row {alert_id}: {exception_object}")

def main():
    ensure_dirs()
    connection = connect()

    print(f"[🎞️ gif_worker] queue={GIF_QUEUE_DIR} media={MEDIA_DIR}")

    while True:
        # Process oldest job (global pattern matching on *.json files) first for fairness (by mtime)
        # - uses stat().st_mtime (like using ls -l) to sort by time
        jobs = sorted(
            GIF_QUEUE_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime, # Asks the OS for file metadata . make it into epoch seconds
        )

        if not jobs:
            time.sleep(0.5)
            continue

        job_file = jobs[0]

        # Atomically claim the job by renaming it (prevents double-processing if worker restarts)
        claimed = job_file.with_name(job_file.name + ".processing")
        try:
            job_file.replace(claimed)
        except FileNotFoundError:
            continue # Race Condition - Do not claim
        except Exception as exception_object:
            print(f"[🎞️ gif_worker] ERROR claiming {job_file.name}: {exception_object}")
            time.sleep(0.5)
            continue

        # Initializing before the try
        job = {}
        alert_id = ""
        try:
            job = load_json(claimed)
            alert_id = str(job.get("alert_id", "")).strip()
            video_str = str(job.get("video", "")).strip()

            if not alert_id:
                raise ValueError("job missing alert_id")
            if not video_str:
                raise ValueError("job missing video string")
            
            video_path = resolve_video_path(video_str)
            if not video_path.exists():
                raise FileNotFoundError(f"video not found: {video_str} (also tried {PROCESSED_DIR / Path(video_str).name})")
            
            out_gif = MEDIA_DIR / f"{alert_id}.gif"
            
            print(f"[🎞️ gif_worker] making gif alert={alert_id} video={video_path.name}")
            make_gif_ffmpeg(video_path, out_gif)

            update_alert_status(alert_id, status="gif_done", connection=connection, gif=out_gif.name)

            # Mark job as done
            done = job_file.with_name(job_file.name + ".done")
            claimed.replace(done)

            print(f"[🎞️ gif_worker] done alert={alert_id} gif={out_gif.name}")

        except Exception as exception_object:
            print(f"[🎞️ gif_worker] FAILED {claimed.name}: {exception_object}")
            if alert_id:
                update_alert_status(alert_id, status="gif_failed", connection=connection)            

            failed = job_file.with_name(job_file.name + ".failed")
            try:
                claimed.replace(failed)
            except Exception:
                pass
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()