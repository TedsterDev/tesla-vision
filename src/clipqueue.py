"""
clipqueue.py

The order the processor works through inbox/, and the numbers that say how
far behind it is.

Why not oldest-first
--------------------
inbox/ used to be drained in filename order, which is capture order. That is
the right order for a batch and the wrong one for a lookout: if a backlog
builds (a Sentry burst, a long drive on a slow clip mix), every new clip waits
behind all the old ones, and an alert about the car behind you NOW arrives
after an alert about a parking lot from an hour ago.

So the queue is:

  1. Sentry and Saved clips first. The car itself flagged those moments.
  2. Then newest capture first. The present is what a lookout is for; older
     clips are backfilled whenever the newest are done, and the correlation
     engine still sees every clip eventually - it scores sightings, not order.
  3. Within one minute, front and back before the side cameras: a follower is
     most visible ahead of or behind you.

The processor re-reads this order after every clip, so a clip that lands
mid-backlog jumps straight to the front.

Where a clip came from
----------------------
The host ingest service copies clips into a flat inbox/, which used to throw
away the RecentClips / SentryClips / SavedClips folder they came from - and
with it both this queue's first rule and the correlation engine's
event-clip weighting. ingest now leaves a `<clip>.mp4.source` sidecar beside
each clip, written BEFORE the clip is renamed into place so the processor can
never see a clip without one.

Standard library only: the host-side ingest service imports this too, and it
runs on the host's bare Python 3.10.
"""
import os
import time
from pathlib import Path

from src.clipmeta import parse_clip_filename

SOURCE_SUFFIX = ".source"

# Lower sorts first.
SOURCE_RANK = {"SentryClips": 0, "SavedClips": 0, "RecentClips": 1}
UNKNOWN_SOURCE_RANK = 1
CAMERA_RANK = {"front": 0, "back": 1}
OTHER_CAMERA_RANK = 2

# Tesla writes one-minute clips. Used to turn "this clip's start time" into
# "the moment the last frame in it happened" for lag figures.
CLIP_SECONDS = 60


def source_sidecar(clip: Path) -> Path:
    return clip.with_name(clip.name + SOURCE_SUFFIX)


def write_source(clip: Path, source: str | None) -> None:
    """Record which TeslaCam folder `clip` came from. Atomic, like the clip."""
    if not source:
        return
    sidecar = source_sidecar(clip)
    temporary = sidecar.with_name("." + sidecar.name + ".tmp")
    temporary.write_text(source + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)


def read_source(clip: Path) -> str | None:
    try:
        value = source_sidecar(clip).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def discard_source(clip: Path) -> None:
    try:
        source_sidecar(clip).unlink()
    except OSError:
        pass


def queue_entries(inbox: Path) -> list[dict]:
    """
    Every clip waiting in `inbox`, in the order it will be processed.

    Dotfiles are skipped - ingest's in-flight copies are named `.tmp_*`.
    """
    entries = []
    try:
        candidates = list(inbox.glob("*.mp4"))
    except OSError:
        return entries
    for clip in candidates:
        if clip.name.startswith("."):
            continue
        metadata = parse_clip_filename(clip.name)
        source = read_source(clip)
        entries.append({
            "path": clip,
            "name": clip.name,
            "camera": metadata["camera"],
            "captured_ts": metadata["captured_ts"],
            "source": source,
        })
    entries.sort(key=_priority)
    return entries


def _priority(entry: dict) -> tuple:
    captured = entry["captured_ts"]
    return (
        SOURCE_RANK.get(entry["source"], UNKNOWN_SOURCE_RANK),
        # Newest first; an unparseable name has no time and goes last.
        -captured if captured is not None else float("inf"),
        CAMERA_RANK.get(entry["camera"], OTHER_CAMERA_RANK),
        entry["name"],
    )


def estimate_seconds(entries: list[dict], seconds_by_camera: dict[str, float],
                     fallback_seconds: float | None) -> float | None:
    """
    How long the queue will take to clear at recent per-camera speeds.

    Per camera because the cost is not uniform: the 5 MP front camera takes
    roughly twice as long as a side camera.
    """
    if not entries:
        return 0.0
    total = 0.0
    for entry in entries:
        seconds = seconds_by_camera.get(entry["camera"] or "", fallback_seconds)
        if seconds is None:
            return None
        total += seconds
    return total


def summarize(entries: list[dict], now: float | None = None) -> dict:
    """The queue in numbers, for the processor's heartbeat and the dashboard."""
    now = time.time() if now is None else now
    captured = [entry["captured_ts"] for entry in entries if entry["captured_ts"] is not None]
    events = sum(1 for entry in entries if SOURCE_RANK.get(entry["source"]) == 0)
    return {
        "depth": len(entries),
        "events": events,
        "oldest_captured_ts": min(captured) if captured else None,
        "newest_captured_ts": max(captured) if captured else None,
        "next": [
            {"name": entry["name"], "camera": entry["camera"],
             "captured_ts": entry["captured_ts"], "source": entry["source"]}
            for entry in entries[:10]
        ],
        "ts": int(now),
    }
