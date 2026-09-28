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
import shutil
import time
import uuid

from pathlib import Path

# Open source computer vision imports
import cv2
from ultralytics import YOLO

from src.common import (
    ensure_dirs,
    MODELS_DIR,
    env_flag,
    env_float,
    env_int,
    FACE_CROPS_DIR,
    TESLACAM_DIR,
    PROCESSED_DIR,
    INBOX_DIR,
    GIF_QUEUE_DIR,
    ALERTS_DIR,
    MEDIA_DIR,
    PLATE_CROPS_DIR,
)
from src.clipmeta import detect_clip_source, describe_clip, parse_clip_filename
from src.ingest import iterate_new_clips, safe_copy_to_inbox
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
# live ingest produces 4-6 clips per minute. At a 30-frame budget we process
# about 1.5-2 clips per minute, i.e. we fall behind a continuously-recording
# car. Drop SAMPLE_FRAMES_PER_CLIP to 10 for roughly real-time throughput, or
# move to an l4t-pytorch base image to run YOLO on the GPU.

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
        "clip_source": detect_clip_source(source_path or clip_path),
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

    frame_number = 0
    yielded = 0
    while yielded < frame_budget:
        ok, frame = video_capture.read()
        if not ok:
            break
        if available and frame_number >= available:
            break
        if frame_number % step == 0:
            yield frame_number, frame
            yielded += 1

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

    @property
    def person_count(self) -> int:
        return sum(1 for hit in self.hits if hit["class_id"] == PERSON_CLASS_ID)

    @property
    def vehicle_count(self) -> int:
        return sum(1 for hit in self.hits if hit["class_id"] in VEHICLE_CLASS_IDS)


def analyze_clip(model: YOLO, video_path: Path, plate_reader=None, face_engine=None) -> ClipAnalysis:
    """
    Run every enabled detector over one clip in a single pass.

    Returns a ClipAnalysis. Frames are never accumulated in memory beyond the
    single best snapshot - a 10-second 1280x960 clip at 3fps would otherwise
    hold ~110MB of raw frames, which on a 7.4GB box shared with three model
    runtimes is not a trade worth making.
    """
    analysis = ClipAnalysis()

    alpr_budget = ALPR_MAX_FRAMES if (plate_reader and plate_reader.available) else 0
    face_budget = FACE_MAX_FRAMES if (face_engine and (face_engine.available or face_engine.detection_only)) else 0
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

        results = model(frame_small, verbose=False)
        single_frame_result = results[0]
        if single_frame_result.boxes is None:
            continue

        vehicle_boxes_full_res: list[tuple[int, int, int, int]] = []
        saw_person = False

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

                # Keep the best scoring original frame for snapshot.
                if confidence_score > analysis.best_score:
                    analysis.best_score = confidence_score
                    analysis.best_frame = frame.copy()

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

        # --- ALPR, budgeted, only where there is a vehicle to read ---------
        if (vehicle_boxes_full_res and alpr_budget > 0
                and frame_number - last_alpr_frame >= EXPENSIVE_STAGE_FRAME_GAP):
            analysis.plate_candidates.extend(
                plate_reader.read_plates(frame, vehicle_boxes_full_res, frame_number)
            )
            alpr_budget -= 1
            last_alpr_frame = frame_number

        # --- Faces, budgeted, only where there is a person -----------------
        if (saw_person and face_budget > 0
                and frame_number - last_face_frame >= EXPENSIVE_STAGE_FRAME_GAP):
            for face_detection in face_engine.detect(frame, frame_number):
                embedding = face_engine.embed(frame, face_detection)
                if embedding is not None:
                    x, y, w, h = face_detection.bbox
                    crop = frame[max(0, y):y + h, max(0, x):x + w].copy()
                    analysis.face_observations.append((face_detection, embedding, crop))
            face_budget -= 1
            last_face_frame = frame_number

    return analysis


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

        # --- 2) Processing step: detect on inbox mp4 files ---
        # Note: We need to look into sorting could be done during insert to save time.
        found_work = False
        for mp4filefrominbox in sorted(INBOX_DIR.glob("*.mp4")):
            found_work = True
            try:
                print(f"[⚙️ processor] processing: {describe_clip(mp4filefrominbox.name)}")

                clip_id = register_clip(connection, mp4filefrominbox)
                context = build_clip_context(connection, clip_id, mp4filefrominbox)

                analysis = analyze_clip(model, mp4filefrominbox, plate_reader, face_engine)

                plate_text = record_plate_detections(connection, analysis, context)
                face_names = (
                    record_face_detections(connection, face_engine, analysis, context)
                    if face_engine else []
                )

                # Minimal alert rule: enough hits + snapshot exists
                if len(analysis.hits) >= MIN_HITS_PER_CLIP and analysis.best_frame is not None:
                    alert_id = uuid.uuid4().hex[:12]

                    jpeg = save_jpeg(analysis.best_frame, alert_id)
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
            except Exception as exception_object:
                print(f"[⚙️ processor] ⧱❗️ ERROR {mp4filefrominbox.name}: {exception_object}")
                # Move aside to avoid infinite loop on a bad file.
                dest = PROCESSED_DIR / f"error_{mp4filefrominbox.name}"
                try:
                    mp4filefrominbox.rename(dest)
                except Exception:
                    pass

        # --- 3) Correlate: re-score identities, notify on new findings ---
        # Only after we actually processed something, so an idle loop costs
        # nothing.
        if found_work and ENABLE_CORRELATION:
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

        time.sleep(1.0)


if __name__ == "__main__":
    main()
