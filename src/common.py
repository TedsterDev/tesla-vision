"""
common.py

Shared configuration + utility helpers used by all services.

Design goals:
    - Keep all "where things live" decisions in one place (BASE_DIR, TESLACAM_DIR)
    - Make it easy to run inside Docker *or* directly on the host
    - Provide safe ingestion helpers (file stability detection) to avoid racing Tesla writes
"""
import os
import time
from pathlib import Path

def env_path(name : str, default:str) -> Path:
        """Read an env var as a Path, or falling back to `default`.

        Example:
            BASE_DIR=/data -> Path("/data")
        """
        return Path(os.environ.get(name, default)).expanduser()

# BASE_DIR is where our application state lives (alerts, media, jobs, logs, etc.)
# In /Docker/compose.yaml sets BASE_DIR=/data and mounts /mnt/jetsondata/tesla-alerts -> /data
BASE_DIR = env_path("BASE_DIR", str(Path.home() / "tesla-alerts"))

# TESLACAM_DIR is the *source* directory where Tesla writes Dashcam/Sentry clips.
# In Docker, compose.yaml sets TESLACAM_DIR=/teslacam/TeslaCam and mounts /mnt/teslacam -> teslacam:ro
TESLACAM_DIR = env_path("TESLACAM_DIR", str(Path("/mnt/teslacam/TeslaCam")))

# Derived directories inside BASE_DIR
INBOX_DIR = BASE_DIR / "inbox"          # where we copy stable clips for processing
PROCESSED_DIR = BASE_DIR / "processed"  # processed clips moved here
ALERTS_DIR = BASE_DIR / "alerts"        # alert metadata JSON files
MEDIA_DIR = BASE_DIR / "media"          # output of JPEGs + GIFSs
JOBS_DIR = BASE_DIR / "jobs"            # job-related files
GIF_QUEUE_DIR = JOBS_DIR / "gif_queue"  # JSON job files for GIF worker
LOGS_DIR = BASE_DIR / "logs"            # optional log files

def ensure_dirs():
    """
    Create all required directories if they don't exist.
    Safe to call repeatedly
    """
    for path in [INBOX_DIR, PROCESSED_DIR, ALERTS_DIR, MEDIA_DIR, GIF_QUEUE_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

def file_is_stable(path: Path, stable_seconds: int = 8, poll_seconds: int = 2) -> bool:
    """
    Returns True if a file's size stays unchanged for `stable_seconds`.

    Why we need this:
    - Tesla may still be writing an MP4 when we notice it.
    - If we copy mid-write, we get partial/corrupted clips.
    - This "stability window" approach is simple and effective for MVP.

    Args:
        path: life to check
        stable_seconds: how long the file must remain unchanged
        poll_seconds: how often to re-check size

    Returns:
        True if stable False otherwise.
    """
    try:
        size1 = path.stat().st_size
    except: FileNotFoundError:
        return False

    waited = 0
    while waited < stable_seconds:
        time.sleep(poll_seconds)
        waited += poll_seconds

        try:
            size2 = path.stat().st_size
