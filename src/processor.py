"""
processor.py

This serice does two things in a loop :

1) Ingest :
    - Walk TESLACAM_DIR (read-only mount) looking for *.mp4 clips
    - Only copy a clip to INBOX once the file becomes "stable" (size stops changing)
    - This prvents copying while Tesla is still writing

2) Detect + alert
    - For each clip in INBOX, sample frames and run object detection You Only Look Once (YOLO)
    - If we detect people/vehicles above thresholds, create an alert
        * JSON record in ALERTS_DIR
        * JPEG snapshot in MEDIA_DIR
        * enqueue a GIF job (JOSN file) in GIF_QUEUE_DIR
    - Move clip to PROCESSED_DIR regardless (to keep inbox clean)

Notes :
- MVP detection is intentionally simple; we'll improve it with tracking + rules later.
- TESLACAM_DIR is read-only inside Docker to prevent corruption of Tesla-written media.
"""
import shutil
import uuid
import json
import time

from pathlib import Path

# Open source computer vision imports
import cv2
from ultralytics import YOLO

from src.common import (
    ensure_dirs,
    TESLACAM_DIR,
    PROCESSED_DIR,
    file_is_stable,
    INBOX_DIR,
    GIF_QUEUE_DIR,
    ALERTS_DIR,
    MEDIA_DIR
)

# COCO = Common Objects in Context, a big computer-vision dataset used to train and benchmark detection models. Many YOLO models (including the Ultralytics yolov8n.pt we’re using) are trained on COCO, so their outputs use COCO’s class list and class IDs.

# COCO (Common Objects in Context) class indices that we care about (Ultralytics YOLO default COCO mapping)
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 4 airplane, 5 bus, 7 truck
KEEP = {0, 1, 2, 3, 4, 5, 7}

# Detection thresholds (tune these after you see reall data)
CONFIDENCE_THRESHOLD = 0.35     # minimum confidence per detection
MIN_HITS_PER_CLIP = 3           # require at least N detections across sampled frames

def iterate_new_clips(root: Path):
    """
    Yield MP4 paths under the TeslaCam directory.

    Tesla typically writes into subfolders like:
        TeslaCam/RecentClips/
        TeslaCam/SentryClips/
        TeslaCam/SavedClips/
        
    We use rglob("*.mp4") so we don't have to hard-code the folder names.
    """

    # Passed directory does not exist
    if not root.exists():
        return

    for mp4file in root.rglob("*.mp4"):
        yield mp4file

def safe_copy_to_inbox(src: Path) -> Path | None:
    """
    Copy Tesla-written file to INBOX once it is stable (size stops changing).

    Implementation details:
    - If a clip is mid-write, file_is_stable returns False and we skip it for now.
    - We copy to a temporary file first, then rename to final name. Renames within the same filesystem are atomic (prevents partial files in inbox).
    """
    # Ignore dotfiles just in case
    if src.name.startswith("."):
        return None
    # Only proceed when Tesla appears to be done writing this clip.
    if not file_is_stable(src):
        return None
    
    dest = INBOX_DIR / src.name
    if dest.exists():
        # Already copied.
        return None
    
    temporary = INBOX_DIR / f".tmp_{src.name}"
    shutil.copy2(src, temporary)    # copy2 preserves timestamps/metadata where possible
    temporary.rename(dest)          # atomic rename into final path

    return dest

def sample_frames(video_path: Path, fps_sample: float = 3.0, max_seconds: int = 10):
    """
    Yield (frame_index) from the clip at a sampled rate.

    Why sample?
    - Full-frame inference on every frame is expensive
    - Sampling 3 fps for the first ~10 seconds is enough for MVP alerts.

    Arguements:
        fps_sample: sample rate in frames per second
        max_seconds: max duration to analyze from start of clip
    """
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        return
    
    fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / fps_sample )), 1)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(min(total_frames, max_seconds * fps))

    frame_number = 0
    while True:
        ok, frame = video_capture.read()
        if not ok:
            break
        if frame_number >= max_frames:
            break
        if frame_number % step == 0:
            yield frame_number, frame
        
        frame_number += 1

    video_capture.release()

def detect_hits(model: YOLO, video_path: Path):
    """
    Run YOLO on sampled frames and record detections (hits).

    Returns:
        hits: list of dicts {frame, cls, conf}
        best_frame: original full-resolution frame for snapshot (highest confidence)
        best_score: confidence score of best detection
    """
    hits = []
    best_frame = None
    best_score = 0.0

    for frame_number, frame in sample_frames(video_path):
        # Resize for speed (YOLO generally does fine at 640 width for MVP)
        height, width = frame.shape[:2]
        target_width = 640

        if width > target_width:
            scale = target_width / width
            frame_small = cv2.resize(frame, (target_width, int(height * scale)))
        else:
            frame_small = frame
        
        results = model(frame_small, verbose=False)
        single_frame_result = results[0]
        if single_frame_result.boxes is None:
            continue

        for detection_box in single_frame_result.boxes:
            # Class Index
            class_id = int(detection_box.cls[0].item())
            # Confidence Score
            confidence_score = float(detection_box.conf[0].item())

            if class_id in KEEP and confidence_score >= CONFIDENCE_THRESHOLD:
                hits.append({"frame": frame_number, "class_id": class_id, "confidence_score": confidence_score})

                # Keep the best scoring original frame for snapshot.
                if confidence_score > best_score:
                    best_score = confidence_score
                    best_frame = frame.copy()

    return hits, best_frame, best_score

def save_jpeg(frame, alert_id: str) -> str:
    """
    Save a JPEG snapshot to MEDIA_DIR and return filename.
    """
    out = MEDIA_DIR / f"{alert_id}.jpg"
    cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out.name

def enqueue_gif(src_video: Path, alert_id:str):
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
    """
    (ALERTS_DIR / f"{alert['id']}.json").write_text(json.dumps(alert, indent=2), encoding="utf-8")

def main():
    ensure_dirs()
    
    print(f"[⚙️ processor] TELSACAM_DIR={TESLACAM_DIR}")
    print(f"[⚙️ processor] INBOX_DIR={INBOX_DIR} PROCESSED_DIR={PROCESSED_DIR}")


    # Start with the small YOLO model for speed. Later we can swap for TensorRT.
    # First run will download weights into the container layer (or cache).
    model = YOLO("yolov8n.pt")

    # Track TeslaCam files we've already considered so we don't re-check endlessly.
    # Note: If we delete old clips or rotate folders, we have to revisit this logic.
    seen = set()

    while True:
        # --- 1) Ingest step: copy stable Tesla clips into our inbox ---
        for clip in iterate_new_clips(TESLACAM_DIR):
            key = str(clip)
            if key in seen:
                continue
            seen.add(key)

            inbox_clip = safe_copy_to_inbox(clip)
            if inbox_clip:
                print(f"[⚙️ processor] copied -> inbox: {inbox_clip.name}")

        # --- 2) Processing step: detect on inbox mp4 files ---
        # Note: We need to look into sorting could be done during insert to save time.
        for mp4filefrominbox in sorted(INBOX_DIR.glob("*.mp4")):
            try:
                print(f"[⚙️ processor] processing: {mp4filefrominbox.name}")
                hits, best_frame, score = detect_hits(model, mp4filefrominbox)

                # Minimal alert rule: enough hits + snapshot exists
                if len(hits) >= MIN_HITS_PER_CLIP and best_frame is not None:
                    alert_id = uuid.uuid4().hex[:12]

                    jpeg = save_jpeg(best_frame, alert_id)
                    enqueue_gif(mp4filefrominbox, alert_id)

                    alert = {
                        "id": alert_id,
                        "timestamp": int(time.time()),
                        "source_file": mp4filefrominbox.name,
                        "score": score,
                        "hits": hits,
                        "jpeg": jpeg,
                        "gif": f"{alert_id}.gif", # generated by worker
                        "status": "gif_queued"
                    }
                    write_alert(alert)
                    print(f"[⚙️ processor] ALERT {alert_id} jpg={jpeg} score={score:.2f}")

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

        time.sleep(1.0)

if __name__ == "__main__":
    main()