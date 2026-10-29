"""
processor.py

The detection pipeline. Ingests TeslaCam clips and turns them into identities.

Per clip, in one pass over the sampled frames:

    1) INGEST     copy the clip out of the Tesla drive once it stops changing
    2) CONTEXT    parse camera + capture time from the filename, and ask the
                  poll history where the car was at that moment
    3) DETECT     YOLO for people and vehicles (the original behaviour)
    4) READ       ALPR on frames containing a vehicle  -> plate identities
    5) RECOGNISE  faces on frames containing a person  -> person identities
    6) CORRELATE  re-score every identity and push a notification for any newly
                  raised finding

Steps 2, 4, 5 and 6 are the Scout port; steps 1 and 3 are the pipeline that was
already here, kept intact.

A note on frame budgets. Inference runs on the Orin Nano's CPU inside a
python:3.10-slim container (no CUDA wheels for aarch64 in that base image), so
running three models over every sampled frame would put us well behind
real-time. Instead each expensive stage gets a budget of frames per clip and a
minimum gap between uses, which spreads the work across the whole clip rather
than burning it all in the first second. Tune with ALPR_MAX_FRAMES /
FACE_MAX_FRAMES if you move to an L4T base image with GPU inference.
"""
import json
import os
import shutil
import time
import uuid

from collections import deque
from pathlib import Path

# Open source computer vision imports
import cv2
from ultralytics import YOLO

from src.common import (
    BASE_DIR,
    JOBS_DIR,
    ensure_dirs,
    MODELS_DIR,
    env_flag,
    env_float,
    env_int,
    FACE_CROPS_DIR,
    TESLACAM_DIR,
    PROCESSED_DIR,
    INBOX_DIR,
    LOGS_DIR,
    GIF_QUEUE_DIR,
    ALERTS_DIR,
    MEDIA_DIR,
    PLATE_CROPS_DIR,
)
from src.clipmeta import detect_clip_source, describe_clip, parse_clip_filename
from src.clipqueue import CLIP_SECONDS, discard_source, estimate_seconds, queue_entries, read_source, summarize
from src.ingest import iterate_new_clips, safe_copy_to_inbox, write_heartbeat
from src.db import connect, now_ts, upsert
from src.poller import location_at

# COCO = Common Objects in Context, a big computer-vision dataset used to train and benchmark detection models. Many YOLO models (including the Ultralytics yolov8n.pt we’re using) are trained on COCO, so their outputs use COCO’s class list and class IDs.

# COCO (Common Objects in Context) class indices that we care about (Ultralytics YOLO default COCO mapping)
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 4 airplane, 5 bus, 7 truck
KEEP = {0, 1, 2, 3, 4, 5, 7}

# The subset of KEEP that is a road vehicle - these are the boxes worth
# pointing the plate reader at.
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
PERSON_CLASS_ID = 0

# Detection thresholds (tune these after you see reall data)
CONFIDENCE_THRESHOLD = 0.35     # minimum confidence per detection
MIN_HITS_PER_CLIP = 3           # require at least N detections across sampled frames

# --- Scout stage switches ---------------------------------------------------
# Each stage can be turned off independently; all default on, and each one
# self-disables anyway if its models are missing.
ENABLE_ALPR = env_flag("ENABLE_ALPR", True)
ENABLE_FACES = env_flag("ENABLE_FACES", True)
ENABLE_CORRELATION = env_flag("ENABLE_CORRELATION", True)
ENABLE_NOTIFICATIONS = env_flag("ENABLE_NOTIFICATIONS", True)

# Per-clip frame budgets for the expensive stages (see module docstring).
ALPR_MAX_FRAMES = env_int("ALPR_MAX_FRAMES", 8)
FACE_MAX_FRAMES = env_int("FACE_MAX_FRAMES", 8)
# Minimum sampled-frame gap between two uses of a budget, so the frames we
# spend are spread over the clip instead of clustered at the start.
EXPENSIVE_STAGE_FRAME_GAP = env_int("EXPENSIVE_STAGE_FRAME_GAP", 6)

# How much of each clip to analyse. This is the single biggest performance
# lever in the system: the main YOLO pass costs ~0.9s per sampled frame on the
# Orin Nano's CPU, so a 30-frame budget is ~27s per clip and accounts for
# roughly two thirds of total pipeline time.
#
# We spend that budget EVENLY ACROSS THE WHOLE CLIP rather than densely at the
# start. The original 3fps-for-10-seconds sampling looked at only the first
# sixth of each 60-second clip, and measurably lost real detections: this
# footage's clearest license plate is at frame 927 of
# 2026-02-16_21-31-48-front.mp4, about 26 seconds in, which the first-10-seconds
# window never sampled. Spreading the same 30 frames over 60 seconds costs
# exactly the same inference time and gives six times the temporal coverage.
#
# For surveillance detection that trade is strongly correct: the question is
# "was this vehicle present during this minute", not "what happened in the
# first ten seconds". A follower is present throughout; dense early sampling
# buys nothing and risks missing them entirely if they appear mid-clip.
SAMPLE_FRAMES_PER_CLIP = env_int("SAMPLE_FRAMES_PER_CLIP", 30)

# Cap the analysed window in seconds. 0 means "the whole clip", which is the
# default now that the budget is spread rather than front-loaded.
SAMPLE_MAX_SECONDS = env_int("SAMPLE_MAX_SECONDS", 0)

# A Tesla writes one-minute clips from four to six cameras simultaneously, so
# live ingest produces 4-6 clips per minute. With YOLO on the GPU and seeking
# (below), a minute of six-camera footage costs ~38 s at this budget - about
# 0.63x real time, so the queue keeps up with a recording car. The dashboard
# footer shows the live figure.

# Below this many frames between samples, seeking (which decodes forward from
# the previous keyframe each time) costs more than reading straight through.
SEEK_MIN_STEP = env_int("SEEK_MIN_STEP", 15)

# What the processor is doing and how far behind it is; read by the dashboard.
STATUS_PATH = LOGS_DIR / "processor.json"

# The dashboard's demo button: a request file it drops, the clips it replays,
# and where the demo's snapshots go (overwritten each run).
DEMO_REQUEST_PATH = JOBS_DIR / "demo.json"
DEMO_CLIPS_DIR = Path(os.environ.get("DEMO_CLIPS_DIR", str(BASE_DIR / "natix-stray")))
DEMO_MEDIA_DIR = MEDIA_DIR / "demo"

# Must match gif_worker.py: the GIF is GIF_SECONDS from t=0 at GIF_FPS, and
# the idle box pass rebuilds exactly those frames.
GIF_SECONDS = env_int("GIF_SECONDS", 5)
GIF_FPS = env_int("GIF_FPS", 8)

# After the GPU fails a clip (shared-memory exhaustion on an 8 GB Orin), run
# on the CPU for this long before trying the GPU again.
GPU_RETRY_SECONDS = env_int("GPU_RETRY_SECONDS", 300)

# The general-purpose COCO detector that finds people and vehicles. Staged in
# MODELS_DIR by scripts/fetch_models.sh so it is present offline.
COCO_MODEL_FILENAME = "yolo26n.pt"

# YOLO runs at this width; boxes are scaled back to full resolution before the
# plate reader sees them, because plate crops need every pixel they can get.
YOLO_INPUT_WIDTH = env_int("YOLO_INPUT_WIDTH", 640)


def register_clip(connection, clip_path: Path, source_path: Path | None = None) -> str:
    """
    Record an ingested clip in the `clips` table with its parsed metadata.

    This is what gives every downstream detection a camera and a capture time.
    Scout got these by string-slicing the filename in `preprocess.sh` and then
    threw them away; we keep them, because "which camera saw it" turns out to
    be one of the more useful correlation signals (see clipmeta.py).
    """
    metadata = parse_clip_filename(clip_path.name)
    clip_id = uuid.uuid4().hex[:12]

    existing = connection.execute(
        "SELECT id FROM clips WHERE filename=?", (clip_path.name,)
    ).fetchone()
    if existing:
        return existing["id"]

    upsert(connection, "clips", {
        "id": clip_id,
        "filename": clip_path.name,
        "camera": metadata["camera"],
        "captured_ts": metadata["captured_ts"],
        "clip_source": read_source(clip_path) or detect_clip_source(source_path or clip_path),
        "ingested_ts": now_ts(),
        "processed_ts": None,
        "status": "ingested",
        "lat": None,
        "lon": None,
        "drive_id": None,
    }, key="id")
    connection.commit()

    return clip_id


def sample_frames(
    video_path: Path,
    frame_budget: int = SAMPLE_FRAMES_PER_CLIP,
    max_seconds: int = SAMPLE_MAX_SECONDS,
):
    """
    Yield (frame_index, frame), spreading `frame_budget` frames across the clip.

    Why sample at all?
    - Full-frame inference on every frame is far too expensive on this hardware.

    Why spread rather than front-load?
    - A clip is a minute long and a follower is present for all of it, so even
      coverage answers "were they there" better than a dense burst at the start.
    - Front-loading demonstrably misses things: see the note on
      SAMPLE_FRAMES_PER_CLIP above.

    Args:
        frame_budget: how many frames to analyse from this clip
        max_seconds: analyse only the first N seconds (0 = the whole clip)
    """
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if max_seconds > 0:
        limit = int(max_seconds * fps)
        available = min(total_frames, limit) if total_frames else limit
    else:
        available = total_frames

    if available <= 0:
        # Frame count is unavailable on some containers; fall back to reading
        # sequentially with a fixed stride rather than giving up on the clip.
        available = 0

    step = max(1, available // max(frame_budget, 1)) if available else int(round(fps))

    if available and step >= SEEK_MIN_STEP:
        # Seek straight to each sampled frame. Decoding was the most expensive
        # stage once YOLO moved to the GPU: reading every frame of a 60 s clip
        # to keep 1 in 70 took 28.8 s for the 5 MP front camera and 7.7 s for
        # a side camera; seeking takes 5.3 s and 2.3 s. Checked against the
        # read-everything frames on this car's footage: every seeked frame is
        # a clean decoded frame, at most one frame (33 ms) from the requested
        # index - irrelevant for "was this vehicle present this minute".
        yielded = 0
        for frame_number in range(0, available, step):
            if yielded >= frame_budget:
                break
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = video_capture.read()
            if not ok:
                break
            yield frame_number, frame
            yielded += 1
        video_capture.release()
        return

    # Sequential: frames too close together for a seek (each one decodes from
    # the previous keyframe) or a container with no frame count. grab()
    # advances without the colour conversion read() pays on every frame.
    frame_number = 0
    yielded = 0
    while yielded < frame_budget:
        if available and frame_number >= available:
            break
        if frame_number % step == 0:
            ok, frame = video_capture.read()
            if not ok:
                break
            yield frame_number, frame
            yielded += 1
        elif not video_capture.grab():
            break

        frame_number += 1

    video_capture.release()


class ClipAnalysis:
    """Everything we learned from one clip, before any of it is persisted."""

    def __init__(self) -> None:
        self.hits: list[dict] = []
        self.best_frame = None
        self.best_score = 0.0
        self.plate_candidates: list = []
        self.face_observations: list = []   # (FaceDetection, embedding, frame)
        self.frames_analyzed = 0
        # Every detection on the snapshot frame, for the viewer's boxes.
        self.best_frame_number: int | None = None
        self.best_boxes: list[dict] = []
        # For the dashboard's progress display.
        self.plate_frames = 0
        self.face_frames = 0
        self.plate_budget = 0
        self.face_budget = 0

    @property
    def person_count(self) -> int:
        return sum(1 for hit in self.hits if hit["class_id"] == PERSON_CLASS_ID)

    @property
    def vehicle_count(self) -> int:
        return sum(1 for hit in self.hits if hit["class_id"] in VEHICLE_CLASS_IDS)


def analyze_clip(model: YOLO, video_path: Path, plate_reader=None, face_engine=None,
                 device: str | None = None, progress=None) -> ClipAnalysis:
    """
    Run every enabled detector over one clip in a single pass.

    Returns a ClipAnalysis. Frames are never accumulated in memory beyond the
    single best snapshot - a 10-second 1280x960 clip at 3fps would otherwise
    hold ~110MB of raw frames, which on a 7.4GB box shared with three model
    runtimes is not a trade worth making.

    `progress(stage, analysis)`, if given, is called after each step so the
    dashboard can show where a clip is: "detect" after each sampled frame,
    "plates" and "faces" after those budgeted stages run.
    """
    analysis = ClipAnalysis()

    alpr_budget = ALPR_MAX_FRAMES if (plate_reader and plate_reader.available) else 0
    face_budget = FACE_MAX_FRAMES if (face_engine and (face_engine.available or face_engine.detection_only)) else 0
    analysis.plate_budget, analysis.face_budget = alpr_budget, face_budget
    report = progress or (lambda stage, analysis: None)
    last_alpr_frame = -EXPENSIVE_STAGE_FRAME_GAP
    last_face_frame = -EXPENSIVE_STAGE_FRAME_GAP

    for frame_number, frame in sample_frames(video_path):
        analysis.frames_analyzed += 1

        # Resize for speed (YOLO generally does fine at 640 width for MVP)
        height, width = frame.shape[:2]
        if width > YOLO_INPUT_WIDTH:
            scale = YOLO_INPUT_WIDTH / width
            frame_small = cv2.resize(frame, (YOLO_INPUT_WIDTH, int(height * scale)))
        else:
            scale = 1.0
            frame_small = frame

        results = model(frame_small, verbose=False, **({"device": device} if device else {}))
        single_frame_result = results[0]
        report("detect", analysis)
        if single_frame_result.boxes is None:
            continue

        vehicle_boxes_full_res: list[tuple[int, int, int, int]] = []
        saw_person = False
        boxes_this_frame: list[dict] = []
        became_best = False

        for detection_box in single_frame_result.boxes:
            # Class Index
            class_id = int(detection_box.cls[0].item())
            # Confidence Score
            confidence_score = float(detection_box.conf[0].item())

            if class_id in KEEP and confidence_score >= CONFIDENCE_THRESHOLD:
                analysis.hits.append({
                    "frame": frame_number,
                    "class_id": class_id,
                    "confidence_score": confidence_score,
                })
                bx1, by1, bx2, by2 = detection_box.xyxy[0].tolist()
                boxes_this_frame.append(detection_box_record(
                    class_id, confidence_score,
                    (bx1 / scale, by1 / scale, (bx2 - bx1) / scale, (by2 - by1) / scale),
                    width, height, model))

                # Keep the best scoring original frame for snapshot.
                if confidence_score > analysis.best_score:
                    analysis.best_score = confidence_score
                    analysis.best_frame = frame.copy()
                    became_best = True

                if class_id in VEHICLE_CLASS_IDS:
                    # Scale the box back up to full resolution - the plate
                    # reader works on the original frame, where the plate is
                    # (however marginally) legible.
                    x1, y1, x2, y2 = detection_box.xyxy[0].tolist()
                    vehicle_boxes_full_res.append((
                        int(x1 / scale), int(y1 / scale),
                        int((x2 - x1) / scale), int((y2 - y1) / scale),
                    ))
                elif class_id == PERSON_CLASS_ID:
                    saw_person = True

        if became_best:
            analysis.best_frame_number = frame_number
            analysis.best_boxes = boxes_this_frame

        # --- ALPR, budgeted, only where there is a vehicle to read ---------
        if (vehicle_boxes_full_res and alpr_budget > 0
                and frame_number - last_alpr_frame >= EXPENSIVE_STAGE_FRAME_GAP):
            plates_here = plate_reader.read_plates(frame, vehicle_boxes_full_res, frame_number)
            analysis.plate_candidates.extend(plates_here)
            if became_best:
                for plate in plates_here:
                    analysis.best_boxes.append(normalised_box(
                        "plate", f"plate {plate.text}" if plate.text else "plate",
                        plate.ocr_confidence or plate.det_confidence, plate.bbox, width, height))
            alpr_budget -= 1
            last_alpr_frame = frame_number
            analysis.plate_frames += 1
            report("plates", analysis)

        # --- Faces, budgeted, only where there is a person -----------------
        if (saw_person and face_budget > 0
                and frame_number - last_face_frame >= EXPENSIVE_STAGE_FRAME_GAP):
            for face_detection in face_engine.detect(frame, frame_number):
                if became_best:
                    analysis.best_boxes.append(normalised_box(
                        "face", "face", face_detection.confidence, face_detection.bbox, width, height))
                embedding = face_engine.embed(frame, face_detection)
                if embedding is not None:
                    x, y, w, h = face_detection.bbox
                    crop = frame[max(0, y):y + h, max(0, x):x + w].copy()
                    analysis.face_observations.append((face_detection, embedding, crop))
            face_budget -= 1
            last_face_frame = frame_number
            analysis.face_frames += 1
            report("faces", analysis)

    return analysis


# ---------------------------------------------------------------------------
# Detection boxes for the dashboard's media viewer
# ---------------------------------------------------------------------------
# Each snapshot or GIF in media/ can have a `<file>.boxes.json` beside it:
#     {"v": 1, "source": "snapshot"|"gif", "fps": 8,
#      "frames": [[{"kind", "label", "conf", "box": [x, y, w, h]}, ...], ...]}
# Boxes are fractions of the frame (0-1), so one file serves the 1448 px
# snapshot and the 480 px GIF alike. A snapshot has one frame; a GIF has one
# entry per GIF frame.

def normalised_box(kind: str, label: str, confidence: float, box, width: int, height: int) -> dict:
    x, y, w, h = box
    return {
        "kind": kind, "label": label, "conf": round(float(confidence or 0), 3),
        "box": [round(max(0.0, x / width), 4), round(max(0.0, y / height), 4),
                round(min(1.0, w / width), 4), round(min(1.0, h / height), 4)],
    }


def detection_box_record(class_id: int, confidence: float, box, width: int, height: int, model) -> dict:
    names = getattr(model, "names", None) or {}
    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
    kind = "vehicle" if class_id in VEHICLE_CLASS_IDS else "person" if class_id == PERSON_CLASS_ID else "other"
    return normalised_box(kind, label, confidence, box, width, height)


def write_boxes(media_file: Path, source: str, frames: list[list[dict]], fps: float | None = None) -> None:
    payload = {"v": 1, "source": source, "frames": frames}
    if fps:
        payload["fps"] = fps
    target = media_file.with_name(media_file.name + ".boxes.json")
    temporary = target.with_name("." + target.name + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, target)


def frame_boxes(model, frame, device: str | None = None) -> list[dict]:
    """Every kept detection on one frame, at the same input width the pipeline uses."""
    height, width = frame.shape[:2]
    scale = YOLO_INPUT_WIDTH / width if width > YOLO_INPUT_WIDTH else 1.0
    small = cv2.resize(frame, (YOLO_INPUT_WIDTH, int(height * scale))) if scale != 1.0 else frame
    result = model(small, verbose=False, **({"device": device} if device else {}))[0]
    boxes = []
    for detection in (result.boxes if result.boxes is not None else []):
        class_id = int(detection.cls[0].item())
        confidence = float(detection.conf[0].item())
        if class_id in KEEP and confidence >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = detection.xyxy[0].tolist()
            boxes.append(detection_box_record(
                class_id, confidence, (x1 / scale, y1 / scale, (x2 - x1) / scale, (y2 - y1) / scale),
                width, height, model))
    return boxes


class BoxAnnotator:
    """
    Fills in boxes the live pipeline did not compute, only while idle.

    A GIF is the first GIF_SECONDS of its clip at GIF_FPS - 40 frames - but the
    pipeline samples 30 frames across the whole minute, so at most two or
    three of them land inside the GIF. Boxes on every GIF frame therefore mean
    running YOLO on the GIF's own frames, ~2-3 s per alert on the GPU. Done
    inline that would push a busy drive from ~0.61x toward ~0.85x real time,
    so it runs only when the queue is empty: live footage first, boxes after.
    The same pass gives snapshots from before boxes existed their boxes, by
    running YOLO on the saved JPEG itself.
    """

    RESCAN_SECONDS = 30

    def __init__(self) -> None:
        self.todo: list[tuple[str, dict]] = []
        self.failed: set[str] = set()
        self.last_scan = 0.0

    def _scan(self) -> None:
        self.last_scan = time.monotonic()
        todo = []
        alerts = sorted(ALERTS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for alert_path in alerts[:300]:
            try:
                alert = json.loads(alert_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for kind in ("jpeg", "gif"):
                name = alert.get(kind)
                if not name or f"{name}:{kind}" in self.failed:
                    continue
                media = MEDIA_DIR / name
                if media.exists() and not media.with_name(name + ".boxes.json").exists():
                    todo.append((kind, alert))
        self.todo = todo

    def pending(self) -> int:
        if time.monotonic() - self.last_scan > self.RESCAN_SECONDS:
            self._scan()
        return len(self.todo)

    def annotate_one(self, detectors: "Detectors") -> str | None:
        if not self.pending():
            return None
        kind, alert = self.todo.pop(0)
        name = alert[kind]
        try:
            if kind == "jpeg":
                image = cv2.imread(str(MEDIA_DIR / name))
                if image is None:
                    raise ValueError("unreadable JPEG")
                write_boxes(MEDIA_DIR / name, "snapshot", [frame_boxes(detectors.model, image)])
            else:
                write_boxes(MEDIA_DIR / name, "gif", self._gif_frames(detectors.model, alert), fps=GIF_FPS)
            return name
        except Exception as error:
            self.failed.add(f"{name}:{kind}")
            if is_gpu_failure(error):
                detectors.gpu_retry_after = time.time() + GPU_RETRY_SECONDS
            print(f"[⚙️ processor] could not box {name}: {error}", flush=True)
            return None

    def _gif_frames(self, model, alert: dict) -> list[list[dict]]:
        """YOLO on exactly the frames gif_worker put in the GIF (fps filter from t=0)."""
        video = PROCESSED_DIR / (alert.get("source_file") or "")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(f"source clip {video.name} is gone")
        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            wanted = [round(k / GIF_FPS * source_fps) for k in range(GIF_SECONDS * GIF_FPS)]
            frames, index, frame = [], 0, None
            for target in wanted:
                while index <= target:
                    ok, frame = capture.read() if index == target else (capture.grab(), None)
                    if not ok:
                        return frames
                    index += 1
                frames.append(frame_boxes(model, frame))
            return frames
        finally:
            capture.release()


def save_jpeg(frame, alert_id: str) -> str:
    """
    Save a JPEG snapshot to MEDIA_DIR and return filename.
    """
    out = MEDIA_DIR / f"{alert_id}.jpg"
    cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out.name


def save_crop(frame, directory: Path, detection_id: str, subdirectory: str) -> str | None:
    """
    Save a cropped piece of evidence (a plate or a face) under MEDIA_DIR.

    Returns the path *relative to MEDIA_DIR* so the web app can serve it
    straight from its /media mount without any path juggling.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return None

    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{detection_id}.jpg"
    cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return f"{subdirectory}/{out.name}"


def discard_crop(crop_name: str | None) -> None:
    """
    Delete a crop we saved but could not attach to a database row.

    The crop has to be written before the INSERT, because the row stores its
    filename - which means any failure between the two leaves a file on disk
    that nothing will ever reference or clean up. On a device recording
    continuously that is a slow, silent disk leak. Cheap insurance.
    """
    if not crop_name:
        return
    try:
        (MEDIA_DIR / crop_name).unlink(missing_ok=True)
    except OSError:
        pass


def enqueue_gif(src_video: Path, alert_id: str):
    """
    Queue a GIF by writting a small JSON file.

    Why JSON files?
    - Simple and robust for MVP (no external queue needed)
    - The worker can just scan the queue directory and process jobs
    """
    job = {"video": str(src_video), "alert_id": alert_id}
    (GIF_QUEUE_DIR / f"{alert_id}.json").write_text(json.dumps(job), encoding="utf-8")


def write_alert(alert: dict):
    """
    Persist an alert record to disk as JSON.

    Kept alongside the database row: gif_worker.py updates these files in place,
    and they are the on-disk record of record if the database is ever rebuilt.
    """
    (ALERTS_DIR / f"{alert['id']}.json").write_text(json.dumps(alert, indent=2), encoding="utf-8")


def record_plate_detections(connection, analysis: ClipAnalysis, context: dict) -> str | None:
    """
    Vote on the clip's plate reads and persist the winner.

    One clip yields at most one plate detection per distinct plate, because a
    clip is one encounter. Returns the winning plate text for logging.
    """
    from src.alpr import plates_probably_match, vote_on_plate_text

    if not analysis.plate_candidates:
        return None

    winner = vote_on_plate_text(analysis.plate_candidates)
    if winner is None:
        return None

    # Find an existing identity, tolerating the glyphs OCR always confuses so
    # that "7A8C123" and "7ABC123" don't become two separate vehicles.
    plate_id = None
    for row in connection.execute("SELECT id, plate_text FROM plates").fetchall():
        if plates_probably_match(row["plate_text"], winner.text):
            plate_id = row["id"]
            break

    detection_id = uuid.uuid4().hex[:12]
    crop_name = save_crop(winner.crop, PLATE_CROPS_DIR, detection_id, "plates")
    seen_ts = context["ts"]

    if plate_id is None:
        plate_id = uuid.uuid4().hex[:12]
        upsert(connection, "plates", {
            "id": plate_id,
            "plate_text": winner.text,
            "label": None,
            "is_known": 0,
            "first_seen_ts": seen_ts,
            "last_seen_ts": seen_ts,
            "detection_count": 1,
            "threat_score": 0.0,
            "best_image": crop_name,
        })
    else:
        connection.execute(
            "UPDATE plates SET last_seen_ts=?, detection_count=detection_count+1, "
            "best_image=COALESCE(best_image, ?) WHERE id=?",
            (seen_ts, crop_name, plate_id),
        )

    try:
        upsert(connection, "plate_detections", {
            "id": detection_id,
            "plate_id": plate_id,
            "plate_text": winner.text,
            "ts": seen_ts,
            "clip_id": context["clip_id"],
            "source_file": context["source_file"],
            "camera": context["camera"],
            "frame_number": winner.frame_number,
            "det_confidence": winner.det_confidence,
            "ocr_confidence": winner.ocr_confidence,
            "image": crop_name,
            "lat": context["lat"],
            "lon": context["lon"],
            "drive_id": context["drive_id"],
        })
        connection.commit()
    except Exception:
        # No row means nothing will ever reference this crop again.
        connection.rollback()
        discard_crop(crop_name)
        raise

    return winner.text


def record_face_detections(connection, face_engine, analysis: ClipAnalysis, context: dict) -> list[str]:
    """
    Cluster the clip's faces into identities and persist one sighting each.

    Returns the person names seen, for logging.
    """
    from src.faces import deduplicate_faces_within_clip, embedding_to_blob

    if not analysis.face_observations:
        return []

    # One clip is one encounter: collapse the same person across frames first.
    deduplicated = deduplicate_faces_within_clip(
        face_engine,
        [(observation[0], observation[1]) for observation in analysis.face_observations],
    )
    crops_by_id = {id(observation[0]): observation[2] for observation in analysis.face_observations}

    seen_names: list[str] = []
    seen_ts = context["ts"]

    for face_detection, embedding in deduplicated:
        face_id, person_name, similarity, is_new = face_engine.match_or_create_identity(
            connection, embedding, seen_ts
        )

        detection_id = uuid.uuid4().hex[:12]
        crop_name = save_crop(
            crops_by_id.get(id(face_detection)), FACE_CROPS_DIR, detection_id, "faces"
        )

        upsert(connection, "face_detections", {
            "id": detection_id,
            "face_id": face_id,
            "ts": seen_ts,
            "clip_id": context["clip_id"],
            "source_file": context["source_file"],
            "camera": context["camera"],
            "frame_number": face_detection.frame_number,
            "det_confidence": face_detection.confidence,
            "match_distance": similarity,
            "embedding": embedding_to_blob(embedding),
            "image": crop_name,
            "lat": context["lat"],
            "lon": context["lon"],
            "drive_id": context["drive_id"],
        })

        if crop_name:
            connection.execute(
                "UPDATE faces SET best_image=COALESCE(best_image, ?) WHERE id=?",
                (crop_name, face_id),
            )

        seen_names.append(f"{person_name}{' (new)' if is_new else ''}")

    connection.commit()
    return seen_names


def build_clip_context(connection, clip_id: str, clip_path: Path) -> dict:
    """
    Assemble the who/where/when a detection needs to be useful.

    The timestamp comes from the Tesla filename when we can parse it, because
    that is when the *event* happened - file mtime is when we happened to copy
    it off the drive, which can be hours later.
    """
    metadata = parse_clip_filename(clip_path.name)
    timestamp = metadata["captured_ts"] or now_ts()

    location = location_at(connection, timestamp)

    context = {
        "clip_id": clip_id,
        "source_file": clip_path.name,
        "camera": metadata["camera"],
        "ts": timestamp,
        "lat": location["lat"] if location else None,
        "lon": location["lon"] if location else None,
        "drive_id": location["drive_id"] if location else None,
    }

    connection.execute(
        "UPDATE clips SET lat=?, lon=?, drive_id=? WHERE id=?",
        (context["lat"], context["lon"], context["drive_id"], clip_id),
    )
    # Commit immediately. Python's sqlite3 opens an implicit transaction on the
    # first write and holds it until commit, and the very next thing the caller
    # does is spend 30-40 seconds running inference on the clip. Leaving this
    # uncommitted meant holding the database's single write lock for that whole
    # time, which blocked the dashboard, the gif worker and the poller from
    # writing at all - one slow clip froze every other service.
    connection.commit()
    return context


def load_coco_model() -> YOLO:
    """
    Load the general-purpose person/vehicle detector.

    Prefer a copy pre-staged in MODELS_DIR (which lives on the data volume and
    therefore survives image rebuilds). Fall back to the bare name, which makes
    Ultralytics download it into the container's writable layer.

    That fallback is a trap in this deployment and the warning matters: the
    container layer is discarded on every rebuild or `docker compose up
    --force-recreate`, and the target environment is a car with no internet. A
    recreate in the driveway would leave the processor unable to load its
    primary detector and the whole pipeline dead at startup. scripts/fetch_models.sh
    pre-stages it precisely so that cannot happen.
    """
    staged = MODELS_DIR / COCO_MODEL_FILENAME
    if staged.exists():
        print(f"[⚙️ processor] COCO detector: {staged}")
        return YOLO(str(staged))

    print(f"[⚙️ processor] ⚠️  {COCO_MODEL_FILENAME} is not staged in {MODELS_DIR}; "
          "Ultralytics will try to DOWNLOAD it into the container layer.")
    print("[⚙️ processor] ⚠️  Run ./scripts/fetch_models.sh before deploying - "
          "a car has no internet and a container recreate would kill the pipeline.")
    return YOLO(COCO_MODEL_FILENAME)


def is_gpu_failure(error: BaseException) -> bool:
    """
    True for the ways a CUDA allocation fails on the Orin.

    Its 8 GB is shared by CPU and GPU, so under memory pressure an allocation
    fails mid-clip. On Jetson that surfaces as several different exceptions -
    a CUDA OOM, CUBLAS_STATUS_ALLOC_FAILED, or an NVML assert (PyTorch tries
    to query NVML, which Jetson lacks, while reporting the OOM) - so match
    on text rather than on one type.
    """
    text = f"{type(error).__name__}: {error}"
    return any(marker in text for marker in (
        "CUDA", "CUBLAS", "CUDNN", "cuDNN", "NVML_SUCCESS", "out of memory", "OutOfMemory",
    ))


class Detectors:
    """
    The models analyze_clip needs, on the GPU while it is willing.

    When the GPU fails a clip, that clip is re-run on CPU copies of the two
    YOLO models (built on first need, then kept) and the GPU is rested for
    GPU_RETRY_SECONDS. Before this, the clip was moved aside as error_* and
    never analysed. The face engine is OpenCV on the CPU either way.
    """

    def __init__(self, model, plate_reader, face_engine) -> None:
        import torch
        self.model = model
        self.plate_reader = plate_reader
        self.face_engine = face_engine
        self.has_gpu = torch.cuda.is_available()
        self.gpu_retry_after = 0.0
        self.gpu_failures = 0
        self.cpu_fallbacks = 0
        self.last_gpu_error = None
        self._cpu_model = None
        self._cpu_plate_reader = None

    @property
    def device(self) -> str:
        return "cuda" if self.has_gpu and time.time() >= self.gpu_retry_after else "cpu"

    def analyze(self, clip: Path, progress=None) -> tuple[ClipAnalysis, str]:
        if not self.has_gpu:
            return analyze_clip(self.model, clip, self.plate_reader, self.face_engine,
                                progress=progress), "cpu"

        if self.device == "cuda":
            try:
                return analyze_clip(self.model, clip, self.plate_reader, self.face_engine,
                                    progress=progress), "cuda"
            except Exception as error:
                if not is_gpu_failure(error):
                    raise
                self.gpu_failures += 1
                self.last_gpu_error = str(error).splitlines()[0][:200]
                self.gpu_retry_after = time.time() + GPU_RETRY_SECONDS
                print(f"[⚙️ processor] GPU failed on {clip.name} ({self.last_gpu_error}); "
                      f"re-running on the CPU, GPU rested {GPU_RETRY_SECONDS}s", flush=True)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        self.cpu_fallbacks += 1
        model, plate_reader = self._cpu_detectors()
        return analyze_clip(model, clip, plate_reader, self.face_engine, device="cpu",
                            progress=progress), "cpu"

    def _cpu_detectors(self):
        if self._cpu_model is None:
            self._cpu_model = load_coco_model()
            if self.plate_reader is not None:
                from src.alpr import PlateReader
                self._cpu_plate_reader = PlateReader(device="cpu")
        return self._cpu_model, self._cpu_plate_reader


def clip_duration_seconds(clip: Path) -> float:
    capture = cv2.VideoCapture(str(clip))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    finally:
        capture.release()
    return frames / fps if fps > 0 and frames > 0 else float(CLIP_SECONDS)


class PipelineStatus:
    """
    logs/processor.json: what the processor is doing and how far behind it is.

    Written atomically (ingest's write_heartbeat) when a clip starts, when it
    ends, and every few seconds while idle, so the dashboard can tell "idle,
    nothing to do" from "dead" by the heartbeat's age - the same contract as
    gps.json and ingest.json.
    """

    IDLE_WRITE_SECONDS = 5
    PROGRESS_WRITE_SECONDS = 1
    WINDOW = 30   # recent clips the speed figures are averaged over

    def __init__(self, detectors: Detectors) -> None:
        self.detectors = detectors
        self.started_ts = int(time.time())
        self.recent: deque[dict] = deque(maxlen=self.WINDOW)
        self.current: dict | None = None
        self.processed_total = 0
        self.errors_total = 0
        self.last_error: str | None = None
        self.last_written = 0.0
        self.pending: list[dict] = []
        self.alerts_total = 0
        self.last_alert_ts: int | None = None
        self.correlating = False
        self.last_correlation_ts: int | None = None
        self.last_correlation_raised = 0
        self.demo: "DemoFeed | None" = None
        self.annotating: str | None = None
        self.boxes_pending = 0
        # The "live edge": the real-world moment the newest analysed footage
        # reaches - a clip's capture time plus its length. "Behind real time"
        # on the dashboard is now minus this. Demo clips keep their own edge,
        # on the demo's clock: a demo minute "ends" when it is released.
        self.live_edge_ts: float | None = None
        self.demo_edge_ts: float | None = None

    def begin(self, entry: dict, pending: list[dict]) -> None:
        self.current = {
            "name": entry["name"], "camera": entry["camera"],
            "captured_ts": entry["captured_ts"], "source": entry["source"],
            "demo": bool(entry.get("demo")),
            "started_ts": time.time(), "device": self.detectors.device,
            "stage": "detect", "frames_done": 0, "frames_budget": SAMPLE_FRAMES_PER_CLIP,
            "plate_frames": 0, "plate_budget": 0, "face_frames": 0, "face_budget": 0,
            "vehicles": 0, "people": 0,
        }
        self.write(pending, force=True)

    def progress(self, stage: str, analysis: "ClipAnalysis") -> None:
        """Called from inside analyze_clip; writes at most once a second."""
        if not self.current:
            return
        self.current.update(
            stage=stage, frames_done=analysis.frames_analyzed,
            plate_frames=analysis.plate_frames, plate_budget=analysis.plate_budget,
            face_frames=analysis.face_frames, face_budget=analysis.face_budget,
            vehicles=analysis.vehicle_count, people=analysis.person_count,
            device=self.detectors.device,
        )
        if time.time() - self.last_written >= self.PROGRESS_WRITE_SECONDS:
            self.write(self.pending, force=True)

    def stage(self, stage: str) -> None:
        if self.current:
            self.current["stage"] = stage
            self.write(self.pending, force=True)

    def finish(self, entry: dict, seconds: float, footage_seconds: float, device: str,
               alerted: bool, pending: list[dict], result: dict | None = None) -> None:
        self.processed_total += 1
        if alerted and not entry.get("demo"):
            self.alerts_total += 1
            self.last_alert_ts = int(time.time())
        if entry.get("demo"):
            self.demo_edge_ts = max(self.demo_edge_ts or 0, entry["release_ts"])
        elif entry["captured_ts"] is not None:
            self.live_edge_ts = max(self.live_edge_ts or 0, entry["captured_ts"] + footage_seconds)
        self.recent.append({
            "name": entry["name"], "camera": entry["camera"],
            "captured_ts": entry["captured_ts"], "source": entry["source"],
            "seconds": round(seconds, 1), "footage_seconds": round(footage_seconds, 1),
            "device": device, "alert": alerted, "finished_ts": int(time.time()),
            "demo": bool(entry.get("demo")), "result": result,
        })
        self.current = None
        self.write(pending, force=True)

    def fail(self, entry: dict, message: str, pending: list[dict]) -> None:
        self.errors_total += 1
        self.last_error = f"{entry['name']}: {message}"[:300]
        self.current = None
        self.write(pending, force=True)

    def seconds_by_camera(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for clip in self.recent:
            totals.setdefault(clip["camera"] or "", []).append(clip["seconds"])
        return {camera: sum(values) / len(values) for camera, values in totals.items()}

    def realtime_factor(self) -> float | None:
        """
        Processing time per minute of driving, over a minute.

        Below 1.0 the queue keeps up with a recording car. "A minute of
        driving" is one clip per camera, so this multiplies the mean cost of
        a clip by how many cameras are recording.
        """
        if not self.recent:
            return None
        cameras = len({clip["camera"] for clip in self.recent}) or 1
        seconds = sum(clip["seconds"] for clip in self.recent) / len(self.recent)
        footage = sum(clip["footage_seconds"] for clip in self.recent) / len(self.recent)
        return round(seconds * cameras / footage, 2) if footage > 0 else None

    def write(self, pending: list[dict], force: bool = False) -> None:
        now = time.time()
        idle_interval = self.PROGRESS_WRITE_SECONDS if self.demo and self.demo.active else self.IDLE_WRITE_SECONDS
        if not force and now - self.last_written < idle_interval:
            return
        self.last_written = now
        self.pending = pending

        by_camera = self.seconds_by_camera()
        overall = (sum(clip["seconds"] for clip in self.recent) / len(self.recent)) if self.recent else None
        last = self.recent[-1] if self.recent else None
        newest_done = max((clip["captured_ts"] for clip in self.recent
                           if clip["captured_ts"] is not None), default=None)
        current = dict(self.current) if self.current else None
        if current:
            current["elapsed_seconds"] = round(now - current["started_ts"], 1)

        payload = {
            "schema": 1,
            "pid": os.getpid(),
            "ts": int(now),
            "started_ts": self.started_ts,
            "state": ("processing" if current else "correlating" if self.correlating
                      else "queued" if pending else "annotating" if self.annotating else "idle"),
            "annotating": self.annotating,
            "boxes_pending": self.boxes_pending,
            "device": self.detectors.device,
            "has_gpu": self.detectors.has_gpu,
            "gpu_failures": self.detectors.gpu_failures,
            "cpu_fallbacks": self.detectors.cpu_fallbacks,
            "gpu_retry_after": int(self.detectors.gpu_retry_after) or None,
            "last_gpu_error": self.detectors.last_gpu_error,
            "current": current,
            "queue": summarize(pending, now),
            "eta_seconds": estimate_seconds(pending, by_camera, overall),
            "realtime_factor": self.realtime_factor(),
            "seconds_by_camera": {camera: round(value, 1) for camera, value in by_camera.items()},
            # Age of the newest footage analysed, measured from the END of
            # that clip: how stale the freshest finished answer is.
            "newest_done_age_seconds": (int(now - newest_done - CLIP_SECONDS)
                                        if newest_done is not None else None),
            "last": last,
            "recent": list(self.recent)[-10:][::-1],
            "processed_total": self.processed_total,
            "errors_total": self.errors_total,
            "last_error": self.last_error,
            "alerts_total": self.alerts_total,
            "last_alert_ts": self.last_alert_ts,
            "last_correlation_ts": self.last_correlation_ts,
            "last_correlation_raised": self.last_correlation_raised,
            "demo": self.demo.summary(now) if self.demo else None,
            "live_edge_ts": self.live_edge_ts,
            "demo_edge_ts": self.demo_edge_ts if self.demo and self.demo.active else None,
        }
        try:
            write_heartbeat(payload, STATUS_PATH)
        except OSError as error:
            print(f"[⚙️ processor] status write failed: {error}", flush=True)


class DemoFeed:
    """
    Replays the NATIX clips through the live processor, for the dashboard's
    demo button.

    The clips are read where they already sit (DEMO_CLIPS_DIR, the
    natix-stray rescue) - nothing is copied into inbox/ - and a demo clip's
    results are shown in the queue card only: no alert, plate, face, GIF or
    correlation row is written, so the demo cannot plant duplicate alerts or
    a "Stranger #3" in the real data. Real clips in inbox/ always go first.

    "realtime" releases one capture-minute's clips every 60 s, the way they
    would arrive from the car; "backlog" releases them all at once.
    """

    RELEASE_INTERVAL_SECONDS = 60

    def __init__(self, clips_dir: Path, mode: str, now: float | None = None) -> None:
        self.mode = mode if mode in ("realtime", "backlog") else "realtime"
        self.started_ts = time.time() if now is None else now
        self.finished_ts: float | None = None
        self.stopped = False
        self.done: dict[str, float] = {}          # name -> finished time
        self.entries: list[dict] = []

        minutes: dict[int, list[dict]] = {}
        for clip in sorted(clips_dir.rglob("*.mp4")) if clips_dir.exists() else []:
            metadata = parse_clip_filename(clip.name)
            if metadata["captured_ts"] is None:
                continue
            minutes.setdefault(metadata["captured_ts"], []).append({
                "path": clip, "name": clip.name, "camera": metadata["camera"],
                "captured_ts": metadata["captured_ts"],
                "source": detect_clip_source(clip), "demo": True,
            })
        for index, captured in enumerate(sorted(minutes)):
            delay = index * self.RELEASE_INTERVAL_SECONDS if self.mode == "realtime" else 0
            for entry in minutes[captured]:
                entry["release_ts"] = self.started_ts + delay
                self.entries.append(entry)

    @property
    def active(self) -> bool:
        return not self.stopped and self.finished_ts is None and bool(self.entries)

    def pending(self, now: float | None = None) -> list[dict]:
        if not self.active:
            return []
        now = time.time() if now is None else now
        waiting = [entry for entry in self.entries
                   if entry["release_ts"] <= now and entry["name"] not in self.done]
        from src.clipqueue import _priority
        return sorted(waiting, key=_priority)

    def mark_done(self, entry: dict, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.done[entry["name"]] = now
        if len(self.done) >= len(self.entries):
            self.finished_ts = now

    def summary(self, now: float) -> dict:
        released = [entry for entry in self.entries if entry["release_ts"] <= now]
        upcoming = [entry["release_ts"] for entry in self.entries if entry["release_ts"] > now]
        # "Lag" as the car would see it: from the moment a clip became
        # available to the moment its analysis finished.
        lags = [self.done[entry["name"]] - entry["release_ts"]
                for entry in self.entries if entry["name"] in self.done]
        return {
            "mode": self.mode,
            "active": self.active,
            "stopped": self.stopped,
            "started_ts": int(self.started_ts),
            "finished_ts": int(self.finished_ts) if self.finished_ts else None,
            "total": len(self.entries),
            "released": len(released),
            "done": len(self.done),
            "next_release_in": int(min(upcoming) - now) if upcoming and self.active else None,
            "max_lag_seconds": round(max(lags), 1) if lags else None,
            "last_lag_seconds": round(lags[-1], 1) if lags else None,
            "elapsed_seconds": int((self.finished_ts or now) - self.started_ts),
        }


def read_demo_request(current: "DemoFeed | None", status: "PipelineStatus | None" = None) -> "DemoFeed | None":
    """Pick up a start/stop request the dashboard left in jobs/; consume it."""
    try:
        request = json.loads(DEMO_REQUEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return current
    try:
        DEMO_REQUEST_PATH.unlink()
    except OSError:
        pass
    action = request.get("action") if isinstance(request, dict) else None
    if action == "stop":
        if current:
            current.stopped = True
            print("[⚙️ processor] demo stopped", flush=True)
        return current
    if action == "start":
        demo = DemoFeed(DEMO_CLIPS_DIR, str(request.get("mode") or "realtime"))
        if status:
            status.demo_edge_ts = None
        print(f"[⚙️ processor] demo started: {len(demo.entries)} clips from {DEMO_CLIPS_DIR} "
              f"({demo.mode})", flush=True)
        return demo
    return current


def process_demo_clip(detectors: Detectors, entry: dict, status: "PipelineStatus") -> tuple[bool, str, dict]:
    """Analyse one demo clip for real, but keep only a summary and a snapshot."""
    clip = entry["path"]
    analysis, device = detectors.analyze(clip, progress=status.progress)
    status.stage("saving")
    snapshot = None
    if analysis.best_frame is not None:
        DEMO_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        snapshot = f"demo/{clip.stem}.jpg"
        cv2.imwrite(str(MEDIA_DIR / snapshot), analysis.best_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        write_boxes(MEDIA_DIR / snapshot, "snapshot", [analysis.best_boxes])
    from src.alpr import vote_on_plate_text
    plate = vote_on_plate_text(analysis.plate_candidates) if analysis.plate_candidates else None
    result = {
        "vehicles": analysis.vehicle_count,
        "people": analysis.person_count,
        "plate": getattr(plate, "text", plate) if plate else None,
        "plate_reads": len(analysis.plate_candidates),
        "faces": len(analysis.face_observations),
        "best_score": round(analysis.best_score, 2),
        "snapshot": snapshot,
    }
    alerted = len(analysis.hits) >= MIN_HITS_PER_CLIP and analysis.best_frame is not None
    return alerted, device, result


def process_clip(connection, detectors: Detectors, mp4filefrominbox: Path,
                 status: "PipelineStatus | None" = None):
    """Analyse one inbox clip and persist everything it produced. Returns (alerted, device)."""
    clip_id = register_clip(connection, mp4filefrominbox)
    context = build_clip_context(connection, clip_id, mp4filefrominbox)

    analysis, device = detectors.analyze(mp4filefrominbox,
                                         progress=status.progress if status else None)
    if status:
        status.stage("saving")
    face_engine = detectors.face_engine

    plate_text = record_plate_detections(connection, analysis, context)
    face_names = (
        record_face_detections(connection, face_engine, analysis, context)
        if face_engine else []
    )

    alerted = False
    # Minimal alert rule: enough hits + snapshot exists
    if len(analysis.hits) >= MIN_HITS_PER_CLIP and analysis.best_frame is not None:
        alerted = True
        alert_id = uuid.uuid4().hex[:12]

        jpeg = save_jpeg(analysis.best_frame, alert_id)
        write_boxes(MEDIA_DIR / jpeg, "snapshot", [analysis.best_boxes])
        enqueue_gif(mp4filefrominbox, alert_id)

        alert = {
            "id": alert_id,
            "timestamp": context["ts"],
            "source_file": mp4filefrominbox.name,
            "camera": context["camera"],
            "score": analysis.best_score,
            "hits": analysis.hits,
            "jpeg": jpeg,
            "gif": f"{alert_id}.gif",  # generated by worker
            "status": "gif_queued",
            "plate": plate_text,
            "faces": face_names,
            "lat": context["lat"],
            "lon": context["lon"],
        }
        write_alert(alert)

        upsert(connection, "alerts", {
            "id": alert_id,
            "timestamp": context["ts"],
            "source_file": mp4filefrominbox.name,
            "clip_id": clip_id,
            "camera": context["camera"],
            "score": analysis.best_score,
            "hit_count": len(analysis.hits),
            "person_count": analysis.person_count,
            "vehicle_count": analysis.vehicle_count,
            "jpeg": jpeg,
            "gif": f"{alert_id}.gif",
            "status": "gif_queued",
            "lat": context["lat"],
            "lon": context["lon"],
        })

        extras = []
        if plate_text:
            extras.append(f"plate={plate_text}")
        if face_names:
            extras.append(f"faces={','.join(face_names)}")
        if device == "cpu" and detectors.has_gpu:
            extras.append("on=cpu")
        suffix = (" " + " ".join(extras)) if extras else ""
        print(f"[⚙️ processor] ALERT {alert_id} jpg={jpeg} score={analysis.best_score:.2f}{suffix}")

    connection.execute(
        "UPDATE clips SET status='processed', processed_ts=? WHERE id=?",
        (now_ts(), clip_id),
    )
    connection.commit()

    # Move processed clip out of the inbox to avoid reprocessing
    dest = PROCESSED_DIR / mp4filefrominbox.name
    mp4filefrominbox.rename(dest)
    discard_source(mp4filefrominbox)
    return alerted, device


def run_correlation(connection) -> int:
    newly_raised = []
    try:
        from src.correlate import run_correlation_pass
        newly_raised = run_correlation_pass(connection)
        for result in newly_raised:
            print(f"[🧭 correlate] NEW {result.severity.upper()} finding: "
                  f"{result.entity_label} ({result.score}/100)")
            if ENABLE_NOTIFICATIONS:
                from src.notify import notify_correlation
                notify_correlation(connection, result)
    except Exception as exception_object:
        print(f"[🧭 correlate] ⧱❗️ ERROR: {exception_object}")
    return len(newly_raised)


def main():
    ensure_dirs()
    connection = connect()

    print(f"[⚙️ processor] TELSACAM_DIR={TESLACAM_DIR}")
    print(f"[⚙️ processor] INBOX_DIR={INBOX_DIR} PROCESSED_DIR={PROCESSED_DIR}")

    model = load_coco_model()

    # --- Scout stages, each self-reporting whether it came up -------------
    plate_reader = None
    if ENABLE_ALPR:
        from src.alpr import PlateReader
        plate_reader = PlateReader()
        print(f"[⚙️ processor] ALPR: {plate_reader.status_text()}")

    face_engine = None
    if ENABLE_FACES:
        from src.faces import FaceEngine
        face_engine = FaceEngine()
        print(f"[⚙️ processor] faces: {face_engine.status_text()}")

    if ENABLE_NOTIFICATIONS:
        from src import notify
        print(f"[⚙️ processor] notifications: {notify.status_text()}")

    detectors = Detectors(model, plate_reader, face_engine)
    status = PipelineStatus(detectors)
    annotator = BoxAnnotator()
    print(f"[⚙️ processor] device: {detectors.device}")

    uncorrelated = 0
    last_correlation = time.monotonic()

    while True:
        # --- 1) Ingest step: copy stable Tesla clips into our inbox ---
        for clip in iterate_new_clips(TESLACAM_DIR):
            already_known = connection.execute(
                "SELECT 1 FROM clips WHERE filename=?", (clip.name,)
            ).fetchone()
            if already_known:
                continue

            inbox_clip = safe_copy_to_inbox(clip)
            if inbox_clip:
                register_clip(connection, inbox_clip, source_path=clip)
                print(f"[⚙️ processor] copied -> inbox: {inbox_clip.name}")

        # --- 2) Processing step: ONE clip, the highest-priority one ---
        # The queue is re-read after every clip so new arrivals jump ahead of
        # any backlog; see clipqueue.py for the order. Real clips always go
        # before the dashboard's demo replay.
        status.demo = read_demo_request(status.demo, status)
        demo = status.demo
        entries = queue_entries(INBOX_DIR)
        demo_entries = demo.pending() if demo else []

        def waiting() -> list[dict]:
            return queue_entries(INBOX_DIR) + (demo.pending() if demo else [])

        if not entries and demo_entries:
            entry = demo_entries[0]
            status.begin(entry, demo_entries[1:])
            started = time.monotonic()
            try:
                print(f"[⚙️ processor] demo: {describe_clip(entry['name'])} "
                      f"(queue {len(demo_entries) - 1})", flush=True)
                footage_seconds = clip_duration_seconds(entry["path"])
                alerted, device, result = process_demo_clip(detectors, entry, status)
                demo.mark_done(entry)
                status.finish(entry, time.monotonic() - started, footage_seconds, device,
                              alerted, waiting(), result=result)
            except Exception as exception_object:
                print(f"[⚙️ processor] ⧱❗️ demo ERROR {entry['name']}: {exception_object}")
                demo.mark_done(entry)
                status.fail(entry, str(exception_object), waiting())
            continue

        if entries:
            entry, pending = entries[0], entries[1:] + demo_entries
            mp4filefrominbox = entry["path"]
            status.begin(entry, pending)
            started = time.monotonic()
            try:
                print(f"[⚙️ processor] processing: {describe_clip(mp4filefrominbox.name)}"
                      f"{' [' + entry['source'] + ']' if entry['source'] else ''}"
                      f" (queue {len(pending)})")
                footage_seconds = clip_duration_seconds(mp4filefrominbox)
                alerted, device = process_clip(connection, detectors, mp4filefrominbox, status)
                status.finish(entry, time.monotonic() - started, footage_seconds, device,
                              alerted, waiting())
            except Exception as exception_object:
                print(f"[⚙️ processor] ⧱❗️ ERROR {mp4filefrominbox.name}: {exception_object}")
                # Move aside to avoid infinite loop on a bad file.
                dest = PROCESSED_DIR / f"error_{mp4filefrominbox.name}"
                try:
                    mp4filefrominbox.rename(dest)
                except Exception:
                    pass
                discard_source(mp4filefrominbox)
                status.fail(entry, str(exception_object), waiting())
            uncorrelated += 1

        # --- 3) Correlate: re-score identities, notify on new findings ---
        # Once the queue drains, or every minute through a long backlog -
        # not after every clip, which would re-score everything per clip.
        if uncorrelated and ENABLE_CORRELATION and (
            len(entries) <= 1 or time.monotonic() - last_correlation >= 60
        ):
            status.correlating = True
            status.write(waiting(), force=True)
            status.last_correlation_raised = run_correlation(connection)
            status.last_correlation_ts = int(time.time())
            status.correlating = False
            uncorrelated = 0
            last_correlation = time.monotonic()

        if not entries:
            # Idle: give one GIF or old snapshot its boxes, then look again -
            # a clip that arrives meanwhile waits at most one GIF's worth.
            status.boxes_pending = annotator.pending()
            # Not while a demo runs (it would skew the demo's lag figures), and
            # not on a resting GPU (the CPU is busy enough as the fallback).
            if (status.boxes_pending and not (demo and demo.active)
                    and (detectors.device == "cuda" or not detectors.has_gpu)):
                status.annotating = annotator.todo[0][1].get(annotator.todo[0][0])
                status.write(waiting(), force=True)
                annotator.annotate_one(detectors)
                status.annotating = None
                status.boxes_pending = annotator.pending()
                continue
            status.write(waiting())
            time.sleep(1.0)


if __name__ == "__main__":
    main()
