"""
faces.py

Face detection, embedding and identity clustering. This is the port of Scout's
FaceNet stage ("Familiar Face Detection" in its roadmap), which produced 128-d
embeddings and grouped them into people the UI called "Stranger #N".

Scout used davidsandberg/facenet (TensorFlow 1.x) with a TensorRT variant for
the Xavier. That stack does not build on JetPack 6 / L4T R36, so we use the two
models OpenCV ships APIs for:

    YuNet  (cv2.FaceDetectorYN)   - detector, ~340KB ONNX, 5 facial landmarks
    SFace  (cv2.FaceRecognizerSF) - 128-d embedding, ~37MB ONNX

SFace produces the same 128 dimensions FaceNet did and its published accuracy
on LFW is within a point of it, so the downstream logic is unchanged. The
landmarks YuNet returns matter: SFace's alignCrop uses them to normalise pose
before embedding, which is the step that makes cross-clip matching work at all.

Identity assignment is *online* clustering, matching Scout's behaviour:
    - embed the face
    - compare against every known identity's running-mean embedding
    - above the similarity threshold: same person, fold the embedding into the
      running mean
    - below it: a new "Stranger #N"

Running means (rather than storing every embedding) keep the comparison O(n) in
number of *people*, not number of sightings, which is what lets this stay real
time on an Orin Nano after months of data.

Both models are optional. Without them this module reports `available == False`
and the pipeline skips faces entirely - see scripts/fetch_models.sh.
"""
import uuid

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.common import MODELS_DIR, env_float, env_int
from src.db import now_ts, upsert

# --- Model files ------------------------------------------------------------
YUNET_WEIGHTS = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_WEIGHTS = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# --- Tunables ---------------------------------------------------------------
# YuNet's own confidence. 0.85 is strict; dashcam frames are noisy and a loose
# threshold fills the database with reflections and headrests.
FACE_DETECTION_CONFIDENCE = env_float("FACE_DETECTION_CONFIDENCE", 0.85)

# OpenCV's published SFace operating point for cosine similarity is 0.363.
# We use 0.40 because a false merge (two people becoming one identity) is much
# worse for us than a false split: a merged identity hides a real follower
# inside a stranger's history.
FACE_MATCH_SIMILARITY = env_float("FACE_MATCH_SIMILARITY", 0.40)

# Faces smaller than this have too few pixels for a stable embedding.
#
# Was 40, which a positive control showed was silently discarding real
# detections: a genuine face composited into a real night frame from this
# vehicle's own footage scored 0.873 at 30px wide (0.858 dimmed to a third
# brightness) - comfortably past FACE_DETECTION_CONFIDENCE - and was then
# thrown away by the width filter before embedding. Someone following you on
# foot is photographed at distance, so the small end of the range is exactly
# the case that matters. 30px is about the floor at which SFace's 112x112
# align-and-crop still yields a comparable embedding rather than an upscaled
# blur; below that the embedding is noise and would fragment identities.
MIN_FACE_WIDTH_PIXELS = env_int("MIN_FACE_WIDTH_PIXELS", 30)

EMBEDDING_DIMENSIONS = 128


@dataclass
class FaceDetection:
    """One detected face in one frame, before we know who it is."""
    bbox: tuple[int, int, int, int]        # x, y, w, h
    confidence: float
    landmarks: np.ndarray | None = field(default=None, repr=False)
    raw_row: np.ndarray | None = field(default=None, repr=False)  # YuNet's 15-value row
    frame_number: int = 0


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a float32 embedding for the BLOB column."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes | None) -> np.ndarray | None:
    """
    Read an embedding back out of the database.

    Returns None for anything that isn't a full, correctly-sized embedding.
    This matters more than it looks: the box loses power every time the car
    sleeps, so a half-written BLOB is a real possibility, and np.frombuffer
    raises on a buffer that isn't a whole number of float32s. A crash here
    would take down the face matching loop for every subsequent clip because
    of one bad row.
    """
    if not blob:
        return None
    try:
        array = np.frombuffer(blob, dtype=np.float32)
    except ValueError:
        return None
    return array if array.size == EMBEDDING_DIMENSIONS else None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two embeddings, in [-1, 1].

    SFace embeddings are compared this way (not by L2 distance); 1.0 means
    identical direction in embedding space.
    """
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


class FaceEngine:
    """
    Detector + embedder + identity store.

    One instance per process. The detector needs its input size set to match
    each frame, which we do lazily in `detect` because clips from different
    cameras have different resolutions.
    """

    def __init__(self) -> None:
        self.detector = None
        self.recognizer = None
        self.haar_fallback = None
        self.reasons_unavailable: list[str] = []
        self._detector_input_size: tuple[int, int] | None = None

        self._load_models()

    def _load_models(self) -> None:
        if YUNET_WEIGHTS.exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=str(YUNET_WEIGHTS),
                    config="",
                    input_size=(320, 320),
                    score_threshold=FACE_DETECTION_CONFIDENCE,
                    nms_threshold=0.3,
                    top_k=50,
                )
            except Exception as exception_object:
                self.reasons_unavailable.append(f"YuNet failed to load: {exception_object}")
        else:
            self.reasons_unavailable.append(f"missing {YUNET_WEIGHTS.name}")

        if SFACE_WEIGHTS.exists():
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=str(SFACE_WEIGHTS), config=""
                )
            except Exception as exception_object:
                self.reasons_unavailable.append(f"SFace failed to load: {exception_object}")
        else:
            self.reasons_unavailable.append(f"missing {SFACE_WEIGHTS.name}")

        # Haar gives us *detection* without embeddings when YuNet is absent -
        # enough to say "a person's face was visible" but not who.
        if self.detector is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.haar_fallback = cascade

    @property
    def available(self) -> bool:
        """True only when we can both detect *and* identify - the useful case."""
        return self.detector is not None and self.recognizer is not None

    @property
    def detection_only(self) -> bool:
        """True when we can spot faces but not tell people apart."""
        return not self.available and (self.detector is not None or self.haar_fallback is not None)

    def status_text(self) -> str:
        if self.available:
            return "ready (detector=yunet, embedder=sface)"
        if self.detection_only:
            return "detection only - " + "; ".join(self.reasons_unavailable)
        return "disabled - " + "; ".join(self.reasons_unavailable or ["no face models"])

    # -- detection --------------------------------------------------------
    def detect(self, frame: np.ndarray, frame_number: int = 0) -> list[FaceDetection]:
        """Find faces in a BGR frame."""
        height, width = frame.shape[:2]

        if self.detector is not None:
            # setInputSize is required whenever the frame size changes, and it
            # is cheap, but skipping the redundant calls keeps the hot loop tidy.
            if self._detector_input_size != (width, height):
                self.detector.setInputSize((width, height))
                self._detector_input_size = (width, height)

            _, results = self.detector.detect(frame)
            if results is None:
                return []

            detections = []
            for row in results:
                x, y, w, h = (int(value) for value in row[:4])
                confidence = float(row[-1])
                if w < MIN_FACE_WIDTH_PIXELS:
                    continue
                detections.append(FaceDetection(
                    bbox=(max(0, x), max(0, y), w, h),
                    confidence=confidence,
                    landmarks=row[4:14].reshape(5, 2),
                    raw_row=row,
                    frame_number=frame_number,
                ))
            return detections

        if self.haar_fallback is not None:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = self.haar_fallback.detectMultiScale(
                grayscale, scaleFactor=1.1, minNeighbors=5,
                minSize=(MIN_FACE_WIDTH_PIXELS, MIN_FACE_WIDTH_PIXELS),
            )
            return [
                FaceDetection(bbox=(int(x), int(y), int(w), int(h)), confidence=0.5, frame_number=frame_number)
                for (x, y, w, h) in boxes
            ]

        return []

    # -- embedding --------------------------------------------------------
    def embed(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray | None:
        """
        Produce a 128-d embedding for one detected face.

        alignCrop uses YuNet's five landmarks to rotate and scale the face to a
        canonical 112x112 before embedding. Without landmarks (Haar fallback)
        we can't align, and an unaligned embedding is not comparable to an
        aligned one - so we return None rather than poison the identity store.
        """
        if self.recognizer is None or detection.raw_row is None:
            return None

        try:
            aligned = self.recognizer.alignCrop(frame, detection.raw_row)
            embedding = self.recognizer.feature(aligned)
        except Exception:
            return None

        return np.asarray(embedding, dtype=np.float32).flatten()

    # -- identity assignment ----------------------------------------------
    def match_or_create_identity(
        self,
        connection,
        embedding: np.ndarray,
        seen_ts: int,
    ) -> tuple[str, str, float, bool]:
        """
        Find which known person this embedding belongs to, or mint a new one.

        This is Scout's "Stranger #N" behaviour. Returns:
            (face_id, person_name, similarity, is_new_identity)

        The running-mean update is weighted by how many embeddings the identity
        already has, so an identity built from 50 good sightings isn't dragged
        away by one blurry frame.
        """
        best_face_id: str | None = None
        best_person_name = ""
        best_similarity = -1.0

        known_faces = connection.execute(
            "SELECT id, person_name, embedding, embedding_count FROM faces WHERE embedding IS NOT NULL"
        ).fetchall()

        for row in known_faces:
            known_embedding = blob_to_embedding(row["embedding"])
            if known_embedding is None:
                continue
            similarity = cosine_similarity(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_face_id = row["id"]
                best_person_name = row["person_name"]

        if best_face_id is not None and best_similarity >= FACE_MATCH_SIMILARITY:
            # Known person - fold this sighting into their running mean.
            row = connection.execute(
                "SELECT embedding, embedding_count FROM faces WHERE id=?", (best_face_id,)
            ).fetchone()
            existing = blob_to_embedding(row["embedding"])
            count = int(row["embedding_count"] or 0)

            updated = ((existing * count) + embedding) / float(count + 1)

            connection.execute(
                "UPDATE faces SET embedding=?, embedding_count=?, last_seen_ts=?, "
                "detection_count=detection_count+1 WHERE id=?",
                (embedding_to_blob(updated), count + 1, seen_ts, best_face_id),
            )
            return best_face_id, best_person_name, best_similarity, False

        # Nobody matched - this is a new person.
        stranger_number = connection.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"] + 1
        face_id = uuid.uuid4().hex[:12]
        person_name = f"Stranger #{stranger_number}"

        upsert(connection, "faces", {
            "id": face_id,
            "person_name": person_name,
            "label": None,
            "is_known": 0,
            "embedding": embedding_to_blob(embedding),
            "embedding_count": 1,
            "first_seen_ts": seen_ts,
            "last_seen_ts": seen_ts,
            "detection_count": 1,
            "threat_score": 0.0,
            "best_image": None,
        })

        return face_id, person_name, max(best_similarity, 0.0), True


# ---------------------------------------------------------------------------
# Correcting clustering mistakes
# ---------------------------------------------------------------------------
# Online clustering makes two kinds of error, and both need a human fix:
#
#   a SPLIT   one person became several identities, because a change of
#             lighting or pose pushed an embedding below the match threshold
#   a MERGE   two people became one identity, because a bad frame matched
#
# Scout exposed half of this as its `makeStranger` route. Both directions
# matter: an unfixable split hides a follower's true sighting count across
# several "strangers", and an unfixable merge hides them inside someone else's
# history. Either way the correlation engine reasons over the wrong data.
#
# Both operations recompute the affected identity's mean embedding from its
# remaining detections rather than trying to patch the running mean, which is
# exact and cheap - we store every detection's embedding.

def recompute_identity_embedding(connection, face_id: str) -> int:
    """
    Rebuild an identity's mean embedding from the detections it still owns.

    Returns the number of embeddings averaged. An identity left with none is
    given a NULL embedding, which takes it out of future matching without
    destroying its history.
    """
    rows = connection.execute(
        "SELECT embedding FROM face_detections WHERE face_id=? AND embedding IS NOT NULL",
        (face_id,),
    ).fetchall()

    embeddings = [blob_to_embedding(row["embedding"]) for row in rows]
    embeddings = [embedding for embedding in embeddings if embedding is not None]

    if not embeddings:
        connection.execute(
            "UPDATE faces SET embedding=NULL, embedding_count=0, detection_count=0 WHERE id=?",
            (face_id,),
        )
        return 0

    mean_embedding = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    timestamps = connection.execute(
        "SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n "
        "FROM face_detections WHERE face_id=?",
        (face_id,),
    ).fetchone()

    connection.execute(
        "UPDATE faces SET embedding=?, embedding_count=?, detection_count=?, "
        "first_seen_ts=?, last_seen_ts=? WHERE id=?",
        (embedding_to_blob(mean_embedding), len(embeddings), timestamps["n"],
         timestamps["first_ts"], timestamps["last_ts"], face_id),
    )
    return len(embeddings)


def merge_identities(connection, source_face_id: str, target_face_id: str) -> bool:
    """
    Fold `source` into `target`: same person, wrongly split into two identities.

    Every detection is reassigned, the target's embedding is recomputed from
    the combined set, and the source identity and its correlation row are
    removed. Returns False if either identity is missing or they are the same.
    """
    if source_face_id == target_face_id:
        return False

    both = connection.execute(
        "SELECT id FROM faces WHERE id IN (?, ?)", (source_face_id, target_face_id)
    ).fetchall()
    if len(both) != 2:
        return False

    connection.execute(
        "UPDATE face_detections SET face_id=? WHERE face_id=?",
        (target_face_id, source_face_id),
    )
    connection.execute("DELETE FROM faces WHERE id=?", (source_face_id,))
    connection.execute(
        "DELETE FROM correlations WHERE entity_type='face' AND entity_id=?",
        (source_face_id,),
    )
    recompute_identity_embedding(connection, target_face_id)
    return True


def split_detection_into_new_identity(connection, detection_id: str) -> str | None:
    """
    Pull one sighting out into a brand new person - Scout's `makeStranger`.

    Use when a detection was wrongly matched to an existing identity. The
    original identity is recomputed without it, so both sides stay consistent.

    Returns the new face id, or None if the detection doesn't exist.
    """
    detection = connection.execute(
        "SELECT id, face_id, ts, embedding, image FROM face_detections WHERE id=?",
        (detection_id,),
    ).fetchone()
    if not detection:
        return None

    original_face_id = detection["face_id"]

    stranger_number = connection.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"] + 1
    new_face_id = uuid.uuid4().hex[:12]

    upsert(connection, "faces", {
        "id": new_face_id,
        "person_name": f"Stranger #{stranger_number}",
        "label": None,
        "is_known": 0,
        "embedding": detection["embedding"],
        "embedding_count": 1,
        "first_seen_ts": detection["ts"],
        "last_seen_ts": detection["ts"],
        "detection_count": 1,
        "threat_score": 0.0,
        "best_image": detection["image"],
    })

    connection.execute(
        "UPDATE face_detections SET face_id=? WHERE id=?", (new_face_id, detection_id)
    )

    # The identity it came from now has one fewer sighting.
    recompute_identity_embedding(connection, original_face_id)
    return new_face_id


def deduplicate_faces_within_clip(
    engine: "FaceEngine",
    detections: list[tuple[FaceDetection, np.ndarray]],
    similarity_threshold: float = FACE_MATCH_SIMILARITY,
) -> list[tuple[FaceDetection, np.ndarray]]:
    """
    Collapse the same person appearing across many frames of one clip.

    Without this, a person standing in front of the car for a 60-second sentry
    clip produces 180 "detections" and single-handedly wrecks every
    frequency-based threat score. We keep the highest-confidence exemplar of
    each distinct person per clip - one clip, one sighting per person.

    The threshold MUST NOT be stricter than FACE_MATCH_SIMILARITY, which is why
    it defaults to exactly that value. If it were stricter, two frames of one
    person could score below it (staying separate here) yet above the identity
    threshold in the next step - and both would then attach to the same
    identity, recording one person as two sightings of themselves in a single
    clip. That is precisely the inflation this function exists to prevent, so
    the two thresholds are tied together deliberately.
    """
    kept: list[tuple[FaceDetection, np.ndarray]] = []

    for detection, embedding in sorted(detections, key=lambda pair: pair[0].confidence, reverse=True):
        if any(cosine_similarity(embedding, kept_embedding) >= similarity_threshold
               for _, kept_embedding in kept):
            continue
        kept.append((detection, embedding))

    return kept
