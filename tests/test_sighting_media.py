"""
test_sighting_media.py

Each plate or face sighting keeps the whole frame it was read from, with its
boxes, so the dashboard can show where in the picture it was.

The failure this guards against is quiet: a sighting whose frame was never
written, or was written with someone else's boxes, looks exactly like a
sighting from before frames existed. Nothing errors; the evidence is just not
there when it is wanted. So these tests follow real frames through
analyze_clip and onto disk.

Run:  python3 tests/test_sighting_media.py
"""
import json
import os
import shutil
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# BASE_DIR decides where media/ and the database live and src.common reads it
# at import time, so it has to be set before anything under src/ is imported.
SANDBOX = Path(tempfile.mkdtemp(prefix="sighting_media_test_"))
os.environ["BASE_DIR"] = str(SANDBOX)
os.environ["DB_PATH"] = str(SANDBOX / "scout.db")
os.environ.setdefault("MODELS_DIR", str(SANDBOX / "models"))

try:
    import cv2
    import numpy as np
    from src import processor
    from src.alpr import PlateCandidate
    from src.common import MEDIA_DIR, ensure_dirs
    from src.db import connect
    from src.faces import FaceDetection
except ImportError as import_error:                       # pragma: no cover
    print(f"  skip  (no processor dependencies here: {import_error})")
    sys.exit(0)

ensure_dirs()

WIDTH, HEIGHT = 320, 240


# ---------------------------------------------------------------------------
# Fakes: just enough of YOLO, the plate reader and the face engine
# ---------------------------------------------------------------------------
class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Coordinates:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = [Scalar(class_id)]
        self.conf = [Scalar(confidence)]
        self.xyxy = [Coordinates(xyxy)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Answers per frame from a script keyed by the frame's fill value."""
    names = {0: "person", 2: "car"}

    def __init__(self, script):
        self.script = script

    def __call__(self, frame, verbose=False, **kwargs):
        return [FakeResult(self.script.get(int(frame[0, 0, 0]), []))]


class FakePlateReader:
    available = True

    def __init__(self, reads):
        self.reads = reads

    def read_plates(self, frame, vehicle_boxes, frame_number):
        return [PlateCandidate(text=text, raw_text=text, det_confidence=0.8, ocr_confidence=0.9,
                               bbox=bbox, frame_number=frame_number, crop=frame[:8, :16].copy())
                for text, bbox in self.reads.get(frame_number, [])]


class FakeFaceEngine:
    available = True
    detection_only = False

    def __init__(self, faces):
        # frame_number -> [(bbox, has_embedding)]
        self.faces = faces

    def detect(self, frame, frame_number):
        return [FaceDetection(bbox=bbox, confidence=0.7 if embeds else 0.6, frame_number=frame_number)
                for bbox, embeds in self.faces.get(frame_number, [])]

    def embed(self, frame, detection):
        for bbox, embeds in self.faces.get(detection.frame_number, []):
            if bbox == detection.bbox and embeds:
                return np.ones(4, dtype=np.float32)
        return None

    def match_or_create_identity(self, connection, embedding, seen_ts):
        connection.execute(
            "INSERT OR IGNORE INTO faces (id, person_name, first_seen_ts, last_seen_ts) "
            "VALUES ('face0001', 'Stranger #1', ?, ?)", (seen_ts, seen_ts))
        return "face0001", "Stranger #1", 0.95, True


def frames(*numbers):
    """Each frame is filled with its own number, so the fake model can tell them apart."""
    return lambda video_path: ((n, np.full((HEIGHT, WIDTH, 3), n, dtype=np.uint8)) for n in numbers)


CAR = (2, 0.6, [40, 40, 200, 160])
BEST_CAR = (2, 0.9, [40, 40, 200, 160])
PERSON = (0, 0.5, [100, 20, 160, 200])


def analyse(sample, script, plate_reads, faces):
    original = processor.sample_frames
    processor.sample_frames = sample
    try:
        return processor.analyze_clip(
            FakeModel({n: [FakeBox(*box) for box in boxes] for n, boxes in script.items()}),
            Path("clip.mp4"), FakePlateReader(plate_reads), FakeFaceEngine(faces))
    finally:
        processor.sample_frames = original


def standard_analysis():
    """
    Frame 0: a car and a plate read. Frame 10: a better car and another read,
    so it becomes the snapshot. Frame 20: a person with one usable face and
    one face too poor to embed. Frame 30: a person and only the poor face.
    """
    return analyse(
        frames(0, 10, 20, 30),
        {0: [CAR], 10: [BEST_CAR], 20: [PERSON], 30: [PERSON]},
        {0: [("TEST123", (60, 120, 40, 12))], 10: [("TEST124", (70, 120, 40, 12))]},
        {20: [((110, 30, 20, 20), True), ((130, 30, 20, 20), False)],
         30: [((110, 30, 20, 20), False)]},
    )


def kinds(boxes):
    return sorted(box["kind"] for box in boxes)


def context():
    return {"clip_id": "clip0001", "source_file": "2026-01-01_10-00-00-front.mp4",
            "camera": "front", "ts": 1767261600, "lat": None, "lon": None, "drive_id": None}


def media_files(folder):
    directory = MEDIA_DIR / folder
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


def clear_media():
    for folder in ("plates", "faces"):
        shutil.rmtree(MEDIA_DIR / folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# analyze_clip keeps the frames sightings can come from, and only those
# ---------------------------------------------------------------------------
def test_only_frames_that_read_a_plate_or_an_embeddable_face_are_kept():
    analysis = standard_analysis()
    assert sorted(analysis.sighting_frames) == [0, 10, 20], sorted(analysis.sighting_frames)


def test_a_kept_frame_carries_every_box_on_it():
    analysis = standard_analysis()
    assert kinds(analysis.sighting_frames[0][1]) == ["plate", "vehicle"]
    # Both faces are boxed, though only one could be embedded.
    assert kinds(analysis.sighting_frames[20][1]) == ["face", "face", "person"]
    labels = [box["label"] for box in analysis.sighting_frames[0][1]]
    assert "plate TEST123" in labels, labels


def test_a_kept_frame_is_the_frame_itself():
    analysis = standard_analysis()
    image = cv2.imdecode(np.frombuffer(analysis.sighting_frames[20][0], np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (HEIGHT, WIDTH, 3), image.shape
    assert abs(int(image.mean()) - 20) <= 2, image.mean()


def test_the_snapshot_still_gets_plate_and_face_boxes_from_its_own_frame():
    analysis = standard_analysis()
    assert analysis.best_frame_number == 10
    assert kinds(analysis.best_boxes) == ["plate", "vehicle"], analysis.best_boxes
    assert analysis.best_boxes is analysis.sighting_frames[10][1]


def test_a_snapshot_frame_with_only_a_poor_face_still_boxes_it():
    analysis = analyse(frames(0), {0: [PERSON]}, {}, {0: [((110, 30, 20, 20), False)]})
    assert analysis.best_frame_number == 0
    assert kinds(analysis.best_boxes) == ["face", "person"], analysis.best_boxes
    assert analysis.sighting_frames == {}


def test_a_frame_that_will_not_encode_is_skipped_not_fatal():
    original = processor.cv2.imencode
    processor.cv2.imencode = lambda *args, **kwargs: (False, None)
    try:
        analysis = standard_analysis()
    finally:
        processor.cv2.imencode = original
    assert analysis.sighting_frames == {}
    assert len(analysis.plate_candidates) == 2 and len(analysis.face_observations) == 1


# ---------------------------------------------------------------------------
# Recording a sighting writes its frame beside its crop
# ---------------------------------------------------------------------------
def only_row(connection, table):
    rows = connection.execute(f"SELECT * FROM {table}").fetchall()
    assert len(rows) == 1, rows
    return rows[0]


def fresh_connection():
    connection = connect()
    for table in ("plate_detections", "plates", "face_detections", "faces"):
        connection.execute(f"DELETE FROM {table}")
    connection.commit()
    clear_media()
    return connection


def test_a_plate_sighting_saves_the_frame_it_was_read_from_with_its_boxes():
    connection = fresh_connection()
    try:
        analysis = standard_analysis()
        text = processor.record_plate_detections(connection, analysis, context())
        detection = only_row(connection, "plate_detections")
    finally:
        connection.close()
    # The vote's winner is read on frame 0 or 10; its own frame is the one saved.
    frame_name = f"plates/{detection['id']}-frame.jpg"
    assert (MEDIA_DIR / frame_name).is_file(), media_files("plates")
    boxes = json.loads((MEDIA_DIR / (frame_name + ".boxes.json")).read_text())
    assert boxes["source"] == "snapshot" and len(boxes["frames"]) == 1, boxes
    assert f"plate {text}" in [box["label"] for box in boxes["frames"][0]], boxes
    image = cv2.imread(str(MEDIA_DIR / frame_name))
    assert abs(int(image.mean()) - detection["frame_number"]) <= 2


def test_a_plate_sighting_without_a_kept_frame_saves_only_its_crop():
    connection = fresh_connection()
    try:
        analysis = standard_analysis()
        analysis.sighting_frames.clear()
        processor.record_plate_detections(connection, analysis, context())
        detection = only_row(connection, "plate_detections")
    finally:
        connection.close()
    assert media_files("plates") == [f"{detection['id']}.jpg"], media_files("plates")


def test_a_plate_row_that_fails_to_insert_leaves_no_files_behind():
    connection = fresh_connection()
    original = processor.upsert

    def failing_upsert(connection, table, record, key="id"):
        if table == "plate_detections":
            raise RuntimeError("disk I/O error")
        return original(connection, table, record, key)

    processor.upsert = failing_upsert
    try:
        try:
            processor.record_plate_detections(connection, standard_analysis(), context())
            raise AssertionError("the insert failure should propagate")
        except RuntimeError:
            pass
    finally:
        processor.upsert = original
        connection.close()
    assert media_files("plates") == [], media_files("plates")


def test_a_face_sighting_saves_the_frame_it_was_read_from_with_its_boxes():
    connection = fresh_connection()
    try:
        names = processor.record_face_detections(connection, FakeFaceEngine({}), standard_analysis(), context())
        detection = only_row(connection, "face_detections")
    finally:
        connection.close()
    assert names == ["Stranger #1 (new)"], names
    frame_name = f"faces/{detection['id']}-frame.jpg"
    assert detection["frame_number"] == 20
    assert sorted(media_files("faces")) == sorted([
        f"{detection['id']}.jpg", f"{detection['id']}-frame.jpg",
        f"{detection['id']}-frame.jpg.boxes.json"]), media_files("faces")
    boxes = json.loads((MEDIA_DIR / (frame_name + ".boxes.json")).read_text())
    assert kinds(boxes["frames"][0]) == ["face", "face", "person"], boxes


def test_discarding_nothing_is_a_no_op():
    processor.discard_crop(None)
    processor.discard_crop("")


def test_a_leftover_that_cannot_be_deleted_does_not_stop_the_cleanup():
    clear_media()
    stuck = MEDIA_DIR / "plates" / "stuck.jpg"
    stuck.mkdir(parents=True)                     # unlink() on a directory raises OSError
    (MEDIA_DIR / "plates" / "stuck.jpg.boxes.json").write_text("{}")
    processor.discard_crop("plates/stuck.jpg")
    assert media_files("plates") == ["stuck.jpg"], media_files("plates")
    clear_media()


# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {test.__name__}: {error}")
        except Exception as error:                        # noqa: BLE001
            failures += 1
            print(f"  ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    shutil.rmtree(SANDBOX, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
