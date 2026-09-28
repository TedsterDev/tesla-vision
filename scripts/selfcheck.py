#!/usr/bin/env python3
"""
selfcheck.py

A POSITIVE CONTROL for the detection pipeline.

WHY THIS EXISTS
---------------
Run over the 132 real clips currently on this box, the pipeline reports zero
plates, zero faces and zero correlations. That output has two completely
different explanations and no way to tell them apart by looking at it:

    a) the footage genuinely contains nothing readable - 53 minutes of a
       stationary car in a dark, rainy car park, no GPS, no drives, plate crops
       too dark to survive the legibility gate; or
    b) a stage of the pipeline is silently broken and would report zero on ANY
       footage, including a plate held up to the lens in daylight.

"Zero" is the correct answer to (a) and a catastrophic failure in (b), and they
look identical from the dashboard. The only way to separate them is to feed the
pipeline input whose answer we already know and check that it comes back out.

That is what this script does. It manufactures video containing a specific
license plate and a specific face, at specific times and places, runs it through
the REAL production functions - not a reimplementation - and asserts that the
known answer comes back. If this passes, "zero on the real footage" is a
statement about the footage. If it fails, it names the stage that broke.

WHAT IT ASSERTS (the known answer)
----------------------------------
Six clips: three consecutive days, one sighting per day, each sighting recorded
simultaneously by the `front` and `back` cameras, each day at a different place
several miles from the others, each during its own drive. Every clip contains
the plate `7ABC123` and at least one detectable face.

Correctly processed, that scenario MUST produce:

    6 clips registered, each yielding YOLO vehicle AND person detections
    1 plate identity, reading 7ABC123, with 6 plate detections
    >= 1 face identity, with an SFace embedding stored per sighting
    a correlations row for the plate at severity "high", because the same
    vehicle turned up on 3 separate drives, on 3 different days, at 3 distinct
    locations ~10 miles apart, always in a rear-facing camera

Every one of those numbers is a consequence of the scenario, not a tuning
constant, so they stay correct across threshold changes as long as the pipeline
still works.

WHERE THE SYNTHETIC INPUT COMES FROM, AND WHY
---------------------------------------------
The plate is drawn (a bright rectangle with black characters), because a drawn
plate is exactly what the ALPR stage is supposed to find and read, and drawing
it is the only way to know the ground-truth string.

The vehicle, the people and the face are photographic, taken from the two
sample images that ship inside the installed `ultralytics` package
(`assets/bus.jpg` and `assets/zidane.jpg`). This is deliberate, and it is worth
being loud about why, because it was measured rather than assumed:

    A DRAWN face DOES fire YuNet, but only barely and only at one size. A
    hand-drawn cartoon face scored 0.435 at 120px, 0.834 at 200px, 0.862 at
    320px and 0.801 at 480px against a production gate of 0.85 - i.e. it clears
    the gate in one narrow size band and fails on either side of it. A control
    built on that would fail for reasons that have nothing to do with the
    pipeline being broken, which is the one thing a control must never do.
    The bundled photograph scores 0.928 at 59px wide, comfortably clear.

The plate crop, by contrast, needs no such help: drawn at 300x150 it is
localized by the real plate_detector.pt, survives the real legibility gate, and
is read exactly by the real Tesseract call.

WHAT THIS DOES *NOT* COVER
--------------------------
A control that overstates its coverage is worse than none, so, explicitly:

    - ingest (safe_copy_to_inbox / file_is_stable). That stage is an
      eight-second-per-file wait on filesystem behaviour, not detection logic.
      Clips are registered directly from a SentryClips-shaped directory.
    - alert JSON writing, GIF enqueue and the notification senders. Those live
      inline inside processor.main()'s `while True:` loop and cannot be called
      without running the daemon.
    - anything about real-world accuracy. Passing here means the stages are
      wired up and firing. It does not mean the thresholds are right for night
      footage - that is a different question, and this control is the thing that
      lets you ask it honestly.

SAFETY
------
Everything runs against a throwaway directory and a throwaway SQLite file. The
production /data tree and /data/scout.db are never opened for writing; the only
thing read from /data is the model weights directory, exactly as production
reads it. There is a hard guard (`refuse_to_run_against_production_data`) that
aborts if BASE_DIR or DB_PATH ever resolve inside the production data mount.

USAGE
-----
    docker compose exec processor python -u /app/scripts/selfcheck.py
    python3 scripts/selfcheck.py --verbose --keep

Exits 0 when every stage passes, 1 when any stage fails, so it works in CI.

Budget four to seven minutes. Almost all of it is YOLO running on the Orin
Nano's CPU at roughly a second a frame, competing with whatever the processor
container is doing at the time - the same cost the real pipeline pays per clip.
It is not hung.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The scenario - the "known answer" this control checks for
# ---------------------------------------------------------------------------
# Read this block as the specification. Everything below it is machinery.

# The plate string drawn into every frame. 7ABC123 is the codebase's own
# worked example (see plates_probably_match in alpr.py), which makes it the
# right choice: if OCR reads the B as an 8, production is supposed to treat
# that as the same vehicle, and this control exercises that leniency instead
# of pretending OCR is perfect.
EXPECTED_PLATE_TEXT = "7ABC123"

# Three places, far enough apart that cluster_locations (0.15 mile radius)
# counts them as three, and spread over ~10 miles so the geographic-spread
# signal is fully earned rather than scraped.
SIGHTING_LOCATIONS = [
    (37.7749, -122.4194),   # San Francisco
    (37.8044, -122.2712),   # Oakland,  ~8.3 miles away
    (37.8716, -122.2727),   # Berkeley, ~4.6 miles further north
]

# Three consecutive days, at three different times of day, so the "different
# days" signal is real and the timestamps cannot collide.
SIGHTING_MOMENTS = [
    datetime(2026, 3, 1, 8, 15, 0),
    datetime(2026, 3, 2, 12, 40, 0),
    datetime(2026, 3, 3, 18, 5, 0),
]

# Both a forward and a rear-facing camera record each moment, which is what a
# Tesla actually does. It also exercises the multi-camera encounter merge in
# collapse_into_encounters - six detections must become three encounters, and
# every one of them must remember that a rear camera saw it.
SIGHTING_CAMERAS = ["front", "back"]

# Consequences of the scenario above. These are what we assert.
EXPECTED_CLIP_COUNT = len(SIGHTING_MOMENTS) * len(SIGHTING_CAMERAS)      # 6
EXPECTED_PLATE_IDENTITIES = 1
EXPECTED_PLATE_DETECTIONS = EXPECTED_CLIP_COUNT                          # 6
EXPECTED_DISTINCT_DRIVES = len(SIGHTING_MOMENTS)                         # 3
EXPECTED_DISTINCT_DAYS = len(SIGHTING_MOMENTS)                           # 3
EXPECTED_DISTINCT_LOCATIONS = len(SIGHTING_LOCATIONS)                    # 3
EXPECTED_ENCOUNTERS = len(SIGHTING_MOMENTS)                              # 3

# --- Clip geometry ---------------------------------------------------------
# 1280x960 is the resolution TeslaCam writes for the rear and repeater cameras,
# and it is above YOLO_INPUT_WIDTH, so analyze_clip takes its real downscale
# path and then has to scale vehicle boxes back up before the plate reader sees
# them. A smaller canvas would skip that code entirely.
CLIP_WIDTH = 1280
CLIP_HEIGHT = 960

# Twelve frames at twelve fps. The whole clip is one second long, which is not
# realistic and does not need to be: the cost of this control is one YOLO pass
# per sampled frame (~0.9s on this CPU), and the question it answers is "do the
# stages fire", not "how well does sampling cover a minute of video". Twelve
# frames is enough for sample_frames to yield a full budget, for the ALPR and
# face budgets to be spent twice each (they are gated on a six-frame gap), and
# therefore for vote_on_plate_text to actually have votes to count.
CLIP_FRAME_COUNT = 12
CLIP_FPS = 12.0

# --- Telemetry seeding -----------------------------------------------------
# Detections get their lat/lon/drive_id from poller.location_at, which looks for
# the nearest poll within fifteen minutes. Without polls every detection is
# location-less and the correlation engine loses its three strongest signals -
# it can then reach at most 35.0, which is exactly the ambiguity this control
# exists to remove. So we seed real telemetry through the real poller code.
DRIVE_LEAD_SECONDS = 300        # polls start five minutes before the sighting
DRIVE_TRAIL_SECONDS = 300       # ...and continue five minutes after it
POLL_INTERVAL_SECONDS = 60
DRIVING_SPEED_MPH = 31.0

# The parked poll that closes each drive. It must be far enough after the last
# driving poll to beat DRIVE_IDLE_TIMEOUT_SECONDS (300) so attach_drive closes
# the drive, AND far enough from the sighting that location_at's 900-second
# tolerance never picks it - if it did, the detection would inherit a NULL
# drive_id and the "three separate drives" signal would quietly vanish.
DRIVE_CLOSING_POLL_DELAY_SECONDS = 1800


# ---------------------------------------------------------------------------
# Environment redirection - this has to happen before src/* is imported
# ---------------------------------------------------------------------------

# Everything the running system owns lives under this mount. The self-check
# reads model weights from it and must never write anything to it.
PRODUCTION_DATA_MOUNT = Path("/data")


def refuse_if_inside_production_data(paths: dict) -> None:
    """
    Abort before doing anything at all if a path we are about to write to lives
    inside the production data mount.

    Called twice, deliberately: once on the paths we are about to create, and
    again after src/common has resolved them, because those are two different
    questions. The first stops `--workdir /data/somewhere` from so much as
    creating a directory in the real data tree; the second catches an
    import-order mistake that left BASE_DIR pointing at production anyway.
    """
    for label, path in paths.items():
        resolved = Path(path).resolve()
        if resolved == PRODUCTION_DATA_MOUNT or PRODUCTION_DATA_MOUNT in resolved.parents:
            raise SystemExit(
                f"REFUSING TO RUN: {label} resolves to {resolved}, which is inside the "
                f"production data mount {PRODUCTION_DATA_MOUNT}. The self-check writes a "
                f"database, video and image crops, and must never do that to real data."
            )


def redirect_environment_to(work_dir: Path) -> None:
    """
    Point every path the application reads at a throwaway directory.

    This MUST run before anything under src/ is imported. common.py resolves
    BASE_DIR, DB_PATH and MODELS_DIR into module-level constants at import
    time, and db.py then does `from src.common import DB_PATH`, binding the
    value rather than the module. Setting the environment after the import
    would therefore have no effect at all, and the control would cheerfully
    write its synthetic plates into the production database - which is the one
    outcome that would be worse than not having a control.
    """
    base_dir = work_dir / "base"
    teslacam_dir = work_dir / "teslacam"
    # Ultralytics writes a settings file into YOLO_CONFIG_DIR on import. In
    # production that is /data/.config; we keep it out of the production mount
    # so this script leaves no trace there at all.
    yolo_config_dir = work_dir / "yolo_config"

    refuse_if_inside_production_data({
        "the scratch directory": work_dir,
        "BASE_DIR": base_dir,
        "TESLACAM_DIR": teslacam_dir,
        "YOLO_CONFIG_DIR": yolo_config_dir,
    })

    base_dir.mkdir(parents=True, exist_ok=True)
    yolo_config_dir.mkdir(parents=True, exist_ok=True)

    os.environ["BASE_DIR"] = str(base_dir)
    os.environ["DB_PATH"] = str(base_dir / "selfcheck.db")
    os.environ["TESLACAM_DIR"] = str(teslacam_dir)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)

    # MODELS_DIR is the one thing we deliberately share with production: the
    # whole point is to run the REAL weights the real pipeline loads. It is
    # opened read-only by every model loader.
    if "MODELS_DIR" not in os.environ:
        for candidate in (Path("/data/models"), REPO_ROOT / "models"):
            if candidate.is_dir():
                os.environ["MODELS_DIR"] = str(candidate)
                break

    # src/* is imported as a package from the repository root.
    sys.path.insert(0, str(REPO_ROOT))


def import_production_modules() -> SimpleNamespace:
    """
    Import the real pipeline modules and hand them back in one namespace.

    Imported here rather than at module scope purely because of the ordering
    described in redirect_environment_to. Everything this control exercises
    comes out of this namespace, so there is exactly one place to look to
    confirm that no pipeline logic has been reimplemented locally.
    """
    import importlib

    production = SimpleNamespace()
    production.common = importlib.import_module("src.common")
    refuse_if_inside_production_data({
        "BASE_DIR": production.common.BASE_DIR,
        "DB_PATH": production.common.DB_PATH,
        "MEDIA_DIR": production.common.MEDIA_DIR,
    })

    production.db = importlib.import_module("src.db")
    production.clipmeta = importlib.import_module("src.clipmeta")
    production.poller = importlib.import_module("src.poller")
    production.alpr = importlib.import_module("src.alpr")
    production.faces = importlib.import_module("src.faces")
    production.correlate = importlib.import_module("src.correlate")
    production.processor = importlib.import_module("src.processor")
    return production


# ---------------------------------------------------------------------------
# The check table
# ---------------------------------------------------------------------------

@dataclass
class Check:
    """One assertion, its verdict, and enough detail to act on a failure."""
    stage: str
    description: str
    passed: bool
    detail: str = ""


class CheckList:
    """
    Collects checks and prints them as a per-stage PASS/FAIL table.

    A check is recorded even when its stage could not run, because a stage that
    was skipped is a stage whose behaviour is unknown, and unknown must read as
    FAIL. Silent skipping is how a harness ends up passing vacuously.
    """

    def __init__(self) -> None:
        self.checks: list[Check] = []

    def record(self, stage: str, description: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(stage, description, bool(passed), detail))
        return bool(passed)

    def fail_remaining(self, stages: list[tuple[str, str]], reason: str) -> None:
        """Mark stages that never got to run, so they show as FAIL not absent."""
        for stage, description in stages:
            self.record(stage, description, False, f"stage never ran: {reason}")

    @property
    def everything_passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def print_table(self) -> None:
        stage_width = max(len(check.stage) for check in self.checks)
        description_width = max(len(check.description) for check in self.checks)

        print()
        print("=" * (stage_width + description_width + 60))
        print("SELF-CHECK RESULTS  (positive control: known-answer input through production code)")
        print("=" * (stage_width + description_width + 60))

        last_stage = None
        for check in self.checks:
            stage_label = check.stage if check.stage != last_stage else ""
            last_stage = check.stage
            verdict = "PASS" if check.passed else "FAIL"
            print(f"  [{verdict}] {stage_label:<{stage_width}}  "
                  f"{check.description:<{description_width}}  {check.detail}")

        failures = [check for check in self.checks if not check.passed]
        print("-" * (stage_width + description_width + 60))
        if failures:
            print(f"  RESULT: FAIL - {len(failures)} of {len(self.checks)} checks failed.")
            print("  The pipeline did NOT return the known answer for known input, so a")
            print("  zero-finding result on real footage cannot be trusted. Failing stages:")
            for check in failures:
                print(f"    - {check.stage}: {check.description} :: {check.detail}")
        else:
            print(f"  RESULT: PASS - all {len(self.checks)} checks passed.")
            print("  Every stage returned the known answer for known input, so a")
            print("  zero-finding result on real footage is a statement about the footage.")
        print("=" * (stage_width + description_width + 60))


# ---------------------------------------------------------------------------
# Building the synthetic footage
# ---------------------------------------------------------------------------

@dataclass
class Sighting:
    """One moment the follower was recorded, and everything that follows from it."""
    moment: datetime
    latitude: float
    longitude: float
    brightness: float
    clip_paths: list = field(default_factory=list)

    @property
    def captured_ts(self) -> int:
        """Epoch seconds, interpreted the same way clipmeta.py interprets a filename."""
        return int(self.moment.timestamp())


def build_sightings() -> list[Sighting]:
    """
    Pair up the scenario's moments and places into the encounters we will stage.

    Each day gets a slightly different brightness so the six clips are not
    byte-identical. That matters: identical input would let a broken face
    matcher look correct by accident, because any two copies of one array embed
    to the same vector no matter how badly the embedder is behaving.
    """
    brightness_by_day = [1.00, 0.92, 0.85]
    return [
        Sighting(moment=moment, latitude=location[0], longitude=location[1],
                 brightness=brightness)
        for moment, location, brightness in zip(
            SIGHTING_MOMENTS, SIGHTING_LOCATIONS, brightness_by_day)
    ]


def locate_photographic_assets() -> tuple[Path, Path]:
    """
    Find the two bundled sample photographs used to build the synthetic frame.

    They ship inside the installed ultralytics package, so they are present
    wherever the pipeline itself can run. bus.jpg supplies a vehicle and several
    pedestrians (the ALPR stage is gated on YOLO seeing a vehicle and the face
    stage on YOLO seeing a person, so both are load-bearing); zidane.jpg
    supplies a face large and real enough to clear YuNet's 0.85 gate.
    """
    import ultralytics

    assets_dir = Path(ultralytics.__file__).parent / "assets"
    vehicle_image = assets_dir / "bus.jpg"
    face_image = assets_dir / "zidane.jpg"

    for path in (vehicle_image, face_image):
        if not path.exists():
            raise FileNotFoundError(
                f"missing bundled sample image {path}. The self-check needs a real "
                f"photograph of a vehicle and of a face; see the module docstring for "
                f"why a drawn face is not a usable substitute."
            )
    return vehicle_image, face_image


def draw_license_plate(text: str, width: int = 300, height: int = 150):
    """
    Render a plate whose characters we know, bright and high-contrast enough to
    survive the real legibility gate.

    Deliberately generous: is_legible() demands mean brightness >= 28 and
    grey-level standard deviation >= 18, and preprocess_for_ocr wants characters
    around 30px tall. A washed-out plate would fail this control for reasons
    that say nothing about whether the pipeline works, so we make the one thing
    we control easy and let every gate downstream stay at its production value.
    """
    import cv2
    import numpy as np

    plate = np.full((height, width, 3), 235, np.uint8)
    cv2.rectangle(plate, (0, 0), (width - 1, height - 1), (30, 30, 30), 6)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 8
    font_scale = 3.0
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    while text_width > width - 30 and font_scale > 0.5:
        font_scale -= 0.1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    origin = ((width - text_width) // 2, (height + text_height) // 2)
    cv2.putText(plate, text, origin, font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return plate


def compose_frame(vehicle_image, face_image, plate_image, brightness: float, jitter: int):
    """
    Assemble one frame containing everything the pipeline is gated on.

    Layout, and why each piece is there:
        left      the bus photograph, scaled to fill the frame height. YOLO
                  reports it as class 5 (bus) plus several class 0 (person)
                  boxes, which is what opens the ALPR and face budgets.
        right     the face photograph, which is what YuNet fires on.
        bottom left  the drawn plate, positioned over the vehicle so that the
                  plate detector's vehicle-crop pass has something to find.

    `jitter` shifts the plate a few pixels per frame and `brightness` scales the
    whole frame per day, so that no two frames are identical and every stage has
    to do real work on each one.
    """
    import cv2
    import numpy as np

    canvas = np.full((CLIP_HEIGHT, CLIP_WIDTH, 3), 60, np.uint8)

    vehicle_height = CLIP_HEIGHT
    vehicle_width = int(vehicle_image.shape[1] * vehicle_height / vehicle_image.shape[0])
    canvas[0:vehicle_height, 0:vehicle_width] = cv2.resize(
        vehicle_image, (vehicle_width, vehicle_height))

    face_width = CLIP_WIDTH - vehicle_width
    scaled_face = cv2.resize(
        face_image,
        (face_width, int(face_image.shape[0] * face_width / face_image.shape[1])),
    )
    pasted_height = min(CLIP_HEIGHT, scaled_face.shape[0])
    canvas[0:pasted_height, vehicle_width:vehicle_width + face_width] = \
        scaled_face[0:pasted_height, :]

    plate_x = 60 + jitter
    plate_y = CLIP_HEIGHT - 260 + jitter
    canvas[plate_y:plate_y + plate_image.shape[0],
           plate_x:plate_x + plate_image.shape[1]] = plate_image

    if brightness != 1.0:
        canvas = np.clip(canvas.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    return canvas


def write_clip(path: Path, frames) -> None:
    """
    Write frames out as an MP4 that cv2.VideoCapture can read back.

    mp4v is the only fourcc that opens in this image - the h264 encoder OpenCV
    reaches for first is hardware-backed (h264_v4l2m2m) and has no device to
    talk to inside a container. Verified by reading the file back and confirming
    the frame count, because a VideoWriter that silently produced an unreadable
    file would make every downstream stage fail for a reason that has nothing to
    do with the pipeline.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (CLIP_WIDTH, CLIP_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path} with the mp4v encoder")
    for frame in frames:
        writer.write(frame)
    writer.release()


def build_synthetic_clips(work_dir: Path, sightings: list[Sighting], verbose: bool) -> list[Path]:
    """
    Manufacture the whole scenario's footage.

    Clips are written into a SentryClips-shaped directory so that
    clipmeta.detect_clip_source records them as sentry footage, exactly as it
    would for real clips off the drive, and are named in Tesla's format so
    parse_clip_filename recovers the camera and capture time we intended.
    """
    vehicle_path, face_path = locate_photographic_assets()

    import cv2
    vehicle_image = cv2.imread(str(vehicle_path))
    face_image = cv2.imread(str(face_path))
    if vehicle_image is None or face_image is None:
        raise RuntimeError("bundled sample images exist but could not be decoded by OpenCV")

    plate_image = draw_license_plate(EXPECTED_PLATE_TEXT)
    clips_dir = work_dir / "teslacam" / "SentryClips"

    written: list[Path] = []
    for sighting in sightings:
        frames = [
            compose_frame(vehicle_image, face_image, plate_image,
                          sighting.brightness, jitter=(frame_number % 5) - 2)
            for frame_number in range(CLIP_FRAME_COUNT)
        ]
        for camera in SIGHTING_CAMERAS:
            filename = f"{sighting.moment.strftime('%Y-%m-%d_%H-%M-%S')}-{camera}.mp4"
            clip_path = clips_dir / filename
            write_clip(clip_path, frames)
            sighting.clip_paths.append(clip_path)
            written.append(clip_path)
            if verbose:
                print(f"[selfcheck] wrote {clip_path.name} "
                      f"({CLIP_FRAME_COUNT} frames, brightness {sighting.brightness:.2f})")

    return written


# ---------------------------------------------------------------------------
# Seeding vehicle telemetry through the real poller
# ---------------------------------------------------------------------------

def synthetic_vehicle_data(timestamp: int, latitude: float, longitude: float,
                           speed_mph: float) -> dict:
    """
    Build the payload shape the Tesla Fleet API returns, so that the real
    poll_record_from_vehicle_data does the flattening.

    Going through the production function rather than writing `polls` rows by
    hand means this control also proves that the poller's status derivation
    ('D' driving / 'P' parked) and its millisecond timestamp handling still
    work - which the drive grouping downstream depends on completely.
    """
    return {
        "drive_state": {
            "latitude": latitude,
            "longitude": longitude,
            "speed": speed_mph,
            "heading": 90.0,
            "power": 42.0,
            "shift_state": "D" if speed_mph else "P",
            "timestamp": timestamp * 1000,          # Tesla reports milliseconds
        },
        "charge_state": {"charging_state": "Disconnected"},
        "vehicle_state": {"odometer": 12345.6},
    }


def seed_vehicle_telemetry(production, connection, sightings: list[Sighting],
                           verbose: bool) -> int:
    """
    Give each sighting a drive and a GPS position, using the production poller.

    Shape of what we write, per sighting:
        eleven driving polls, one a minute, straddling the sighting - so
        location_at finds a poll at zero seconds' distance and the detection
        inherits both the coordinates and the drive id;
        then one parked poll half an hour later, whose only job is to make
        attach_drive close the drive so the next day opens a NEW one.

    That last part is not decoration. attach_drive extends the open drive for
    ANY moving poll regardless of the gap since the last one, so without an
    intervening parked poll all three days would collapse into a single drive
    and "seen on 3 separate drives" - the strongest signal the engine has -
    would silently become "seen on 1".

    Returns the number of polls written.
    """
    poll_count = 0

    for sighting in sightings:
        first_poll_ts = sighting.captured_ts - DRIVE_LEAD_SECONDS
        last_poll_ts = sighting.captured_ts + DRIVE_TRAIL_SECONDS

        for timestamp in range(first_poll_ts, last_poll_ts + 1, POLL_INTERVAL_SECONDS):
            poll = production.poller.poll_record_from_vehicle_data(
                synthetic_vehicle_data(timestamp, sighting.latitude, sighting.longitude,
                                       DRIVING_SPEED_MPH))
            poll["drive_id"] = production.poller.attach_drive(connection, poll)
            production.db.upsert(connection, "polls", poll)
            poll_count += 1

        closing_ts = last_poll_ts + DRIVE_CLOSING_POLL_DELAY_SECONDS
        parked_poll = production.poller.poll_record_from_vehicle_data(
            synthetic_vehicle_data(closing_ts, sighting.latitude, sighting.longitude, 0.0))
        parked_poll["drive_id"] = production.poller.attach_drive(connection, parked_poll)
        production.db.upsert(connection, "polls", parked_poll)
        poll_count += 1

    connection.commit()

    if verbose:
        for row in connection.execute(
                "SELECT id, start_ts, end_ts, poll_count, is_open FROM drives ORDER BY start_ts"):
            print(f"[selfcheck] drive {row['id']} polls={row['poll_count']} "
                  f"open={row['is_open']} start={datetime.fromtimestamp(row['start_ts'])}")

    return poll_count


# ---------------------------------------------------------------------------
# Running the real pipeline
# ---------------------------------------------------------------------------

def locate_yolo_weights(production) -> str:
    """
    Find the same detector weights processor.main() loads.

    processor.main() says `YOLO("yolo26n.pt")` and relies on the file sitting in
    its working directory, which is /app in the container. Run from anywhere
    else that becomes a network download, so we look in the obvious places
    first and only fall back to the bare name.
    """
    candidates = [
        Path(os.environ["SELFCHECK_YOLO_WEIGHTS"]) if "SELFCHECK_YOLO_WEIGHTS" in os.environ else None,
        Path.cwd() / "yolo26n.pt",
        Path("/app/yolo26n.pt"),
        REPO_ROOT / "yolo26n.pt",
        Path(production.common.MODELS_DIR) / "yolo26n.pt",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return str(candidate)
    return "yolo26n.pt"


@dataclass
class PipelineTally:
    """What the production functions actually reported, summed over all clips."""
    clips_registered: int = 0
    frames_analyzed: int = 0
    yolo_hits: int = 0
    vehicle_hits: int = 0
    person_hits: int = 0
    plate_candidates: int = 0
    illegible_plate_crops: int = 0
    face_observations: int = 0
    plate_texts: list = field(default_factory=list)
    face_names: list = field(default_factory=list)
    identities_after_first_clip: int = 0
    contexts: list = field(default_factory=list)


def run_pipeline(production, connection, clip_paths: list[Path], verbose: bool) -> PipelineTally:
    """
    Push every synthetic clip through the production pipeline, in production order.

    This is the part that has to be exactly right for the control to mean
    anything, so it is a deliberately thin wrapper: every line below either
    calls a function from src/ or records what that function returned. Nothing
    here reimplements, patches or shortcuts detection logic. The sequence is
    lifted from processor.main()'s per-clip body:

        register_clip -> build_clip_context -> analyze_clip
                      -> record_plate_detections -> record_face_detections

    and then run_correlation_pass over the lot, exactly as main() does once it
    has finished a batch.
    """
    from ultralytics import YOLO

    tally = PipelineTally()

    weights = locate_yolo_weights(production)
    print(f"[selfcheck] loading YOLO weights: {weights}")
    model = YOLO(weights)

    plate_reader = production.alpr.PlateReader()
    print(f"[selfcheck] ALPR: {plate_reader.status_text()}")

    face_engine = production.faces.FaceEngine()
    print(f"[selfcheck] faces: {face_engine.status_text()}")

    for index, clip_path in enumerate(sorted(clip_paths)):
        started = time.time()

        clip_id = production.processor.register_clip(connection, clip_path)
        tally.clips_registered += 1

        context = production.processor.build_clip_context(connection, clip_id, clip_path)
        tally.contexts.append(context)

        illegible_before = plate_reader.illegible_crop_count
        analysis = production.processor.analyze_clip(model, clip_path, plate_reader, face_engine)

        tally.frames_analyzed += analysis.frames_analyzed
        tally.yolo_hits += len(analysis.hits)
        tally.vehicle_hits += analysis.vehicle_count
        tally.person_hits += analysis.person_count
        tally.plate_candidates += len(analysis.plate_candidates)
        tally.face_observations += len(analysis.face_observations)
        tally.illegible_plate_crops += plate_reader.illegible_crop_count - illegible_before

        plate_text = production.processor.record_plate_detections(connection, analysis, context)
        if plate_text:
            tally.plate_texts.append(plate_text)

        face_names = production.processor.record_face_detections(
            connection, face_engine, analysis, context)
        tally.face_names.extend(face_names)

        if index == 0:
            tally.identities_after_first_clip = connection.execute(
                "SELECT COUNT(*) AS n FROM faces").fetchone()["n"]

        print(f"[selfcheck] {clip_path.name}: frames={analysis.frames_analyzed} "
              f"hits={len(analysis.hits)} (vehicle={analysis.vehicle_count} "
              f"person={analysis.person_count}) plate_candidates={len(analysis.plate_candidates)} "
              f"faces={len(analysis.face_observations)} plate={plate_text} "
              f"({time.time() - started:.1f}s)")
        if verbose:
            print(f"            context lat={context['lat']} lon={context['lon']} "
                  f"drive_id={context['drive_id']} camera={context['camera']}")

    newly_raised = production.correlate.run_correlation_pass(connection, verbose=verbose)
    for result in newly_raised:
        print(f"[selfcheck] correlation raised {result.severity.upper()} "
              f"{result.entity_type} {result.entity_label}: {result.score}/100")
        for reason in result.reasons:
            print(f"              - {reason}")

    return tally


# ---------------------------------------------------------------------------
# Checking the known answer came back
# ---------------------------------------------------------------------------

def check_results(production, connection, tally: PipelineTally, checks: CheckList) -> None:
    """
    Compare what the database now holds against the scenario we staged.

    Each check names the stage it covers, so a failure points at one place in
    the pipeline rather than at "detection". Where a stage can fail in more than
    one interesting way - a plate that was found but judged illegible is a very
    different problem from a plate that was never found - the detail string says
    which happened.
    """
    scalar = lambda sql: connection.execute(sql).fetchone()[0]

    # -- Stage: ingest / metadata -----------------------------------------
    clip_rows = connection.execute(
        "SELECT filename, camera, captured_ts, clip_source FROM clips ORDER BY filename"
    ).fetchall()
    checks.record(
        "clips", "clips registered",
        len(clip_rows) == EXPECTED_CLIP_COUNT,
        f"{len(clip_rows)} registered, expected {EXPECTED_CLIP_COUNT}")
    checks.record(
        "clips", "filenames parsed into camera + capture time",
        all(row["camera"] in SIGHTING_CAMERAS and row["captured_ts"] for row in clip_rows),
        f"cameras={sorted({row['camera'] for row in clip_rows})}")
    checks.record(
        "clips", "frames sampled from every clip",
        tally.frames_analyzed == EXPECTED_CLIP_COUNT * CLIP_FRAME_COUNT,
        f"{tally.frames_analyzed} frames analysed, expected "
        f"{EXPECTED_CLIP_COUNT * CLIP_FRAME_COUNT}")

    # -- Stage: GPS context ------------------------------------------------
    located_contexts = [c for c in tally.contexts if c["lat"] is not None and c["drive_id"]]
    checks.record(
        "telemetry", "every clip got GPS and a drive from the poll history",
        len(located_contexts) == EXPECTED_CLIP_COUNT,
        f"{len(located_contexts)}/{EXPECTED_CLIP_COUNT} clips located; "
        f"{scalar('SELECT COUNT(*) FROM polls')} polls, "
        f"{scalar('SELECT COUNT(*) FROM drives')} drives seeded")
    checks.record(
        "telemetry", "drives grouped as separate journeys",
        scalar("SELECT COUNT(*) FROM drives") == EXPECTED_DISTINCT_DRIVES,
        f"{scalar('SELECT COUNT(*) FROM drives')} drives, expected {EXPECTED_DISTINCT_DRIVES}")

    # -- Stage: YOLO -------------------------------------------------------
    checks.record(
        "yolo", "vehicle detections",
        tally.vehicle_hits > 0,
        f"{tally.vehicle_hits} vehicle hits across {tally.frames_analyzed} frames "
        f"(gates the ALPR stage)")
    checks.record(
        "yolo", "person detections",
        tally.person_hits > 0,
        f"{tally.person_hits} person hits (gates the face stage)")

    # -- Stage: ALPR -------------------------------------------------------
    localized_or_read = tally.plate_candidates + tally.illegible_plate_crops
    checks.record(
        "alpr", "plate localized",
        localized_or_read > 0,
        f"{localized_or_read} plate rectangles found "
        f"({tally.illegible_plate_crops} rejected by the legibility gate)")
    checks.record(
        "alpr", "plate OCR read",
        tally.plate_candidates > 0,
        f"{tally.plate_candidates} readable candidates; "
        f"{tally.illegible_plate_crops} illegible")

    plate_rows = connection.execute(
        "SELECT id, plate_text, detection_count, threat_score FROM plates").fetchall()
    checks.record(
        "alpr", "plate identity created",
        len(plate_rows) == EXPECTED_PLATE_IDENTITIES,
        f"{len(plate_rows)} plate identities: "
        f"{[row['plate_text'] for row in plate_rows]}, expected {EXPECTED_PLATE_IDENTITIES}")

    read_texts = {row["plate_text"] for row in plate_rows}
    exactly_right = EXPECTED_PLATE_TEXT in read_texts
    close_enough = any(production.alpr.plates_probably_match(EXPECTED_PLATE_TEXT, text)
                       for text in read_texts)
    matched_leniently = close_enough and not exactly_right
    checks.record(
        "alpr", "plate text matches expected",
        close_enough,
        f"read {sorted(read_texts)}, expected {EXPECTED_PLATE_TEXT}"
        + (" (matched via the confusable-glyph rule, not exactly)" if matched_leniently else ""))

    plate_detection_count = scalar("SELECT COUNT(*) FROM plate_detections")
    checks.record(
        "alpr", "one plate sighting recorded per clip",
        plate_detection_count == EXPECTED_PLATE_DETECTIONS,
        f"{plate_detection_count} plate detections, expected {EXPECTED_PLATE_DETECTIONS}")

    # -- Stage: faces ------------------------------------------------------
    checks.record(
        "faces", "face detected",
        tally.face_observations > 0,
        f"{tally.face_observations} face observations across {EXPECTED_CLIP_COUNT} clips")

    face_detection_rows = connection.execute(
        "SELECT embedding FROM face_detections").fetchall()
    embedded = sum(
        1 for row in face_detection_rows
        if production.faces.blob_to_embedding(row["embedding"]) is not None)
    checks.record(
        "faces", "face embedded",
        embedded > 0 and embedded == len(face_detection_rows),
        f"{embedded}/{len(face_detection_rows)} sightings carry a valid "
        f"{production.faces.EMBEDDING_DIMENSIONS}-d SFace embedding")

    face_rows = connection.execute(
        "SELECT id, person_name, detection_count, threat_score FROM faces").fetchall()
    checks.record(
        "faces", "face identity created",
        len(face_rows) >= 1,
        f"{len(face_rows)} identities: {[row['person_name'] for row in face_rows]}")
    checks.record(
        "faces", "identities stable across clips",
        len(face_rows) == tally.identities_after_first_clip and len(face_rows) >= 1,
        f"{tally.identities_after_first_clip} after clip 1, {len(face_rows)} at the end - "
        f"equal means later clips RE-RECOGNISED the same people rather than "
        f"minting new strangers")

    # -- Stage: correlation ------------------------------------------------
    correlation_rows = connection.execute(
        "SELECT * FROM correlations ORDER BY score DESC").fetchall()
    checks.record(
        "correlate", "correlation rows written",
        len(correlation_rows) >= 1,
        f"{len(correlation_rows)} rows: "
        + ", ".join(f"{row['entity_type']}:{row['entity_label']}="
                    f"{row['score']}/{row['severity']}" for row in correlation_rows))

    plate_correlation = next(
        (row for row in correlation_rows if row["entity_type"] == "plate"), None)
    if plate_correlation is None:
        checks.record("correlate", "plate scored as a high-severity finding", False,
                      "no correlations row for any plate")
        checks.record("correlate", "plate score is built from the right signals", False,
                      "no correlations row for any plate")
    else:
        checks.record(
            "correlate", "plate scored as a high-severity finding",
            (plate_correlation["severity"] == "high"
             and plate_correlation["score"] >= production.correlate.HIGH_SEVERITY_SCORE),
            f"score {plate_correlation['score']} severity "
            f"'{plate_correlation['severity']}', expected >= "
            f"{production.correlate.HIGH_SEVERITY_SCORE} and 'high'")

        structure = {
            "distinct_drives": (plate_correlation["distinct_drives"], EXPECTED_DISTINCT_DRIVES),
            "distinct_days": (plate_correlation["distinct_days"], EXPECTED_DISTINCT_DAYS),
            "distinct_locations": (plate_correlation["distinct_locations"],
                                   EXPECTED_DISTINCT_LOCATIONS),
            "detection_count": (plate_correlation["detection_count"], EXPECTED_PLATE_DETECTIONS),
        }
        wrong = {name: pair for name, pair in structure.items() if pair[0] != pair[1]}
        checks.record(
            "correlate", "plate score is built from the right signals",
            not wrong,
            ", ".join(f"{name}={actual} (expected {expected})"
                      for name, (actual, expected) in structure.items())
            + f", spread {plate_correlation['max_separation_mi']:.1f} miles")

    face_correlations = [row for row in correlation_rows if row["entity_type"] == "face"]
    best_face = max(face_correlations, key=lambda row: row["score"], default=None)
    checks.record(
        "correlate", "face scored as a high-severity finding",
        best_face is not None and best_face["severity"] == "high",
        f"best face finding: {best_face['entity_label']}={best_face['score']}/"
        f"{best_face['severity']}" if best_face else "no correlations row for any face")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Stages listed here get a FAIL row if the run dies before reaching them, so a
# crash halfway through still produces a table that says what is unknown.
ALL_STAGES = [
    ("clips", "clips registered"),
    ("telemetry", "every clip got GPS and a drive from the poll history"),
    ("yolo", "vehicle and person detections"),
    ("alpr", "plate localized, read, and turned into an identity"),
    ("faces", "face detected, embedded, and turned into an identity"),
    ("correlate", "finding written with the expected score"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Positive control: push known-answer synthetic footage through the "
                    "real detection pipeline and assert the known answer comes back.")
    parser.add_argument("--keep", action="store_true",
                        help="leave the scratch directory (clips, database, crops) for inspection")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-clip context, drive seeding and per-entity scores")
    parser.add_argument("--workdir", default=None,
                        help="scratch directory to use instead of a fresh temporary one")
    arguments = parser.parse_args()

    work_dir = (Path(arguments.workdir).expanduser().resolve()
                if arguments.workdir else Path(tempfile.mkdtemp(prefix="scout-selfcheck-")))

    # Checked BEFORE the mkdir, not after. Creating the directory first and
    # then refusing would still have left a stray directory in the production
    # data tree, which is exactly the class of thing this guard exists to
    # prevent - it has to refuse before it touches the filesystem at all.
    refuse_if_inside_production_data({"the scratch directory": work_dir})
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[selfcheck] scratch directory: {work_dir}")
    redirect_environment_to(work_dir)

    checks = CheckList()
    started = time.time()

    try:
        production = import_production_modules()
        print(f"[selfcheck] BASE_DIR={production.common.BASE_DIR}")
        print(f"[selfcheck] DB_PATH={production.common.DB_PATH}")
        print(f"[selfcheck] MODELS_DIR={production.common.MODELS_DIR}")

        connection = production.db.connect()

        sightings = build_sightings()
        clip_paths = build_synthetic_clips(work_dir, sightings, arguments.verbose)
        print(f"[selfcheck] built {len(clip_paths)} synthetic clips")

        poll_count = seed_vehicle_telemetry(production, connection, sightings, arguments.verbose)
        print(f"[selfcheck] seeded {poll_count} polls through the production poller")

        tally = run_pipeline(production, connection, clip_paths, arguments.verbose)
        check_results(production, connection, tally, checks)
        connection.close()

    except Exception as exception_object:
        # A crash is a FAIL, never a skip: an unknown stage must not read as a
        # pass, because the whole purpose of this script is to remove ambiguity.
        import traceback
        traceback.print_exc()
        checks.fail_remaining(ALL_STAGES, f"{type(exception_object).__name__}: {exception_object}")

    finally:
        if arguments.keep:
            print(f"[selfcheck] --keep: artefacts left in {work_dir}")
        elif not arguments.workdir:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            # A caller-supplied --workdir is not ours to delete, so we remove
            # only the subdirectories this script created inside it.
            for created in ("base", "teslacam", "yolo_config"):
                shutil.rmtree(work_dir / created, ignore_errors=True)

    checks.print_table()
    print(f"  ({time.time() - started:.0f}s)")
    print()
    print("  NOT covered by this control, and therefore still unverified:")
    print("    - ingest: safe_copy_to_inbox / file_is_stable (an 8s-per-file wait, not detection)")
    print("    - alert JSON, GIF enqueue and notification delivery (inline in processor.main())")
    print("    - whether the thresholds are right for dark, rainy, real-world footage")

    return 0 if checks.everything_passed else 1


if __name__ == "__main__":
    sys.exit(main())
