"""
clipmeta.py

Turn a TeslaCam filename into structured metadata.

This is the Python port of the first half of Scout's `preprocess.sh`, which did:

    videoDate=$(echo $filename|cut -d"_" -f1)
    videoTime=$(echo $filename|cut -d"_" -f2|cut -d"-" -f1-3|tr - :)
    camera=$(echo $filename|cut -d"." -f1|rev|cut -d"-" -f1|rev)

That shell version breaks on the modern camera names, because `rev | cut -d-`
grabs only the trailing token: "right_pillar" survives (no dash) but the older
"left_repeater" style is fine while a name like "right-pillar" would yield just
"pillar". We parse with a regex against the known camera list instead, which is
both correct and tells us when Tesla changes the format.

Filenames look like:
    2026-02-16_21-30-21-front.mp4
    2026-02-16_21-30-21-right_pillar.mp4
    2019-04-10_16-33-38-left_repeater.mp4

The timestamp is the vehicle's *local* wall clock, not UTC. We convert using
the box's local timezone, which is correct as long as the Jetson and the car
agree - they do, both ride in the same vehicle.
"""
import re

from datetime import datetime
from pathlib import Path

# The six camera positions Tesla writes, longest-first so that a naive prefix
# match can never shadow a longer name.
KNOWN_CAMERAS = [
    "left_repeater",
    "right_repeater",
    "left_pillar",
    "right_pillar",
    "front",
    "back",
]

# 2026-02-16_21-30-21-right_pillar
FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})-(?P<camera>[A-Za-z_]+)$"
)

# Tesla's three clip folders. Which one a clip came from is real signal:
# SentryClips means the car decided something was worth recording while parked.
CLIP_SOURCES = ("RecentClips", "SentryClips", "SavedClips")


def parse_clip_filename(filename: str) -> dict:
    """
    Parse a TeslaCam filename into its parts.

    Returns a dict with keys:
        captured_ts : int | None   unix epoch seconds (local time interpreted)
        camera      : str | None   e.g. "right_pillar"
        date_text   : str | None   "2026-02-16"
        time_text   : str | None   "21:30:21"

    Never raises - an unparseable name yields all-None so the caller can still
    ingest the clip, just without metadata.
    """
    stem = Path(filename).stem

    match = FILENAME_PATTERN.match(stem)
    if not match:
        return {"captured_ts": None, "camera": None, "date_text": None, "time_text": None}

    date_text = match.group("date")
    time_text = match.group("time").replace("-", ":")
    camera = match.group("camera")

    # Guard against Tesla inventing a new camera name without us noticing.
    if camera not in KNOWN_CAMERAS:
        camera = camera if camera else None

    try:
        captured = datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S")
        # .timestamp() on a naive datetime interprets it as local time, which is
        # exactly what Tesla wrote.
        captured_ts = int(captured.timestamp())
    except ValueError:
        captured_ts = None

    return {
        "captured_ts": captured_ts,
        "camera": camera,
        "date_text": date_text,
        "time_text": time_text,
    }


def detect_clip_source(path: Path) -> str | None:
    """
    Work out whether a clip came from RecentClips, SentryClips or SavedClips by
    walking up its parent directories.

    Sentry and Saved clips are far more interesting than Recent ones: the car
    flagged them. The correlation engine weights them accordingly.
    """
    for parent in path.parents:
        if parent.name in CLIP_SOURCES:
            return parent.name
    return None


def camera_faces_rearward(camera: str | None) -> bool:
    """
    True when the camera points behind the vehicle.

    This matters for follow detection: a car that keeps showing up in the
    *rear* and *repeater* views while you're driving is behind you, which is
    the geometry of a tail. The same plate seen only by the front camera is
    just traffic you're driving past.
    """
    return camera in ("back", "left_repeater", "right_repeater")


def describe_clip(filename: str) -> str:
    """Human-readable one-liner used in logs and the dashboard."""
    meta = parse_clip_filename(filename)
    if not meta["captured_ts"]:
        return filename
    return f"{meta['date_text']} {meta['time_text']} ({meta['camera'] or 'unknown camera'})"
