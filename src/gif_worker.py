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
    GIF_QUEUE_DIR,
    MEDIA_DIR,
    PROCESSED_DIR,
    ALERTS_DIR,
)

# GIF Generation Config (Scaled for dashboard)
GIF_SECONDS = 5
FPS = 10            # Set Frames per Second (Lower == Smaller GIF)
SCALE_WIDTH = 640

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def resolve_video_path(video_str: str):
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
    """
    temporary_out_file = out_gif.with_suffix(".gif.tmp")

    video_filter = f"fps={FPS},scale={SCALE_WIDTH}:-1:flags=lanczos" # lanczos is a high scaling scaling algorithm

    cmd = [
        "ffmpeg",
        "-y",                   # Override the output without prompting
        "-hide_banner",         # hide ffmpeg banner
        "-loglevel", "error",   # only print errors
        "-ss", "0",             # 0 seek to time 0
        "-t", str(GIF_SECONDS), # GIF_SECONDS only process that many seconds
        "-i", str(video_path),  # -i video_path input file.
        "-vf", video_filter,
        str(temporary_out_file) # output file
    ]

    # Run ffmpeg; raise on non-zero exit
    subprocess.run(cmd, check=True)

    temporary_out_file.replace(out_gif)

def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def update_alert_status(alert_id, status: str, **updates) -> None:
    alert_path = ALERTS_DIR / f"{alert_id}.json"
    if not alert_path.exists():
        return

    alert = load_json(alert_path)
    alert.update(updates)
    save_json(alert_path, alert)

def main():
    ensure_dirs()

    print(f"[🎞️ gif_worker] queue={GIF_QUEUE_DIR} media={MEDIA_DIR}")

    while True:
        # Process oldest job (global pattern matching on *.json files) first for fairness (by mtime)
        # - uses stat().st_mtime (like using ls -l) to sort by time
        jobs = sorted(
            GIF_QUEUE_DIR
            .glob(
                "*.json", 
                key=lambda 
                    path_object:path_object
                    .stat()     # Asks the OS for file metadata
                        .st_mtime))  # make it in epoch seconds

        if not jobs:
            time.sleep(0.5)
            continue

        job_file = jobs[0]

        # Atomically claim the job by renaming it (prevents double-processing if worker restarts)
        claimed = job_file.with_suffix(".json.processing")
        try:
            job_file.replace(claimed)
        except FileNotFoundError:
            continue # Race Condition - Do not claim
        except Exception as expection_object:
            print(f"[🎞️ gif_worker] ERROR claiming {job_file.name}: {expection_object}")
            time.sleep(0.5)
            continue

        try:
            job = load_json(claimed)
            alert_id = str(job.get("alert_id", "").strip())
            video_str = str(job.get("video", "").strip())

            if not alert_id:
                raise ValueError("job missing alert_id")
            if not video_str:
                raise ValueError("job missing video string")
            
            video_path = resolve_video_path(video_str)
            if not video_path.exist():
                raise FileNotFoundError(f"video not found: {video_str} (also tried {PROCESSED_DIR / Path(video_str).name})")
            
            out_gif = MEDIA_DIR / f"{alert_id}.gif"
            
            print(f"[🎞️ gif_worker] making gif alert={alert_id} video={video_path.name}")
            make_gif_ffmpeg(video_path, out_gif)

            update_alert_status(alert_id, status="gif_done", gif=out_gif.name)

            # Mark job as done
            done = claimed.with_suffix(".json.done")
            claimed.replace(done)

            print(f"[🎞️ gif_worker] done alert={alert_id} gif={out_gif.name}")

        except Exception as exception_object:
            print(f"[🎞️ gif_worker] FAILED {claimed.name}: {exception_object}")
            update_alert_status(alert_id, status="gif_failed")

            failed = claimed.with_suffix(".json.done")
            try:
                claimed.replace(failed)
            except Exception:
                pass
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()