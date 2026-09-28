"""
test_faces.py

Tests for the face identity store - the "Stranger #N" clustering Scout used.

The detector and embedder are pretrained models we trust; what we wrote (and
therefore what can be wrong) is the clustering that decides whether two
embeddings are the same person. That logic is testable with synthetic
embeddings and no photographs, which is what this does.

Run:  python3 tests/test_faces.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.faces import (  # noqa: E402
    EMBEDDING_DIMENSIONS,
    FaceEngine,
    blob_to_embedding,
    cosine_similarity,
    deduplicate_faces_within_clip,
    embedding_to_blob,
)
from src.db import connect  # noqa: E402


def check(label, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def make_embedding(seed: int) -> np.ndarray:
    """Build a deterministic unit-norm 128-d embedding."""
    generator = np.random.default_rng(seed)
    base = generator.normal(size=EMBEDDING_DIMENSIONS).astype(np.float32)
    return base / np.linalg.norm(base)


def make_similar_embedding(base: np.ndarray, similarity: float, seed: int = 7) -> np.ndarray:
    """
    Build an embedding at an EXACT cosine similarity to `base`.

    Naively adding gaussian noise does not do this. At 128 dimensions a
    per-component sigma of 0.15 has norm 0.15*sqrt(128) ~= 1.7, which swamps a
    unit-norm signal and lands around 0.4 similarity - nothing like two frames
    of one person, which sit at 0.8-0.95. Getting that wrong makes the test
    assert against a scenario that never occurs.

    Instead we project out the base direction to get a true orthogonal
    component, then mix the two in the ratio that yields the requested cosine:
        v = c * base + sqrt(1 - c^2) * orthogonal
    """
    generator = np.random.default_rng(seed)
    random_direction = generator.normal(size=EMBEDDING_DIMENSIONS).astype(np.float32)

    # Remove any component along `base` so the two parts are truly independent.
    orthogonal = random_direction - np.dot(random_direction, base) * base
    orthogonal /= np.linalg.norm(orthogonal)

    mixed = similarity * base + np.sqrt(1.0 - similarity ** 2) * orthogonal
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


def test_blob_roundtrip():
    embedding = make_embedding(1)
    restored = blob_to_embedding(embedding_to_blob(embedding))
    ok = check("embedding survives a database round trip",
               restored is not None and np.allclose(embedding, restored))
    ok &= check("a NULL blob returns None", blob_to_embedding(None) is None)
    ok &= check("a truncated blob is rejected", blob_to_embedding(b"\x00\x01\x02") is None)
    return ok


def test_cosine_similarity():
    a = make_embedding(2)
    ok = check("identical embeddings score 1.0", abs(cosine_similarity(a, a) - 1.0) < 1e-5)
    target = make_similar_embedding(a, 0.75)
    ok &= check(f"constructed similarity is exact (got {cosine_similarity(a, target):.3f}, want 0.750)",
                abs(cosine_similarity(a, target) - 0.75) < 1e-3)
    ok &= check("opposite embeddings score -1.0", abs(cosine_similarity(a, -a) + 1.0) < 1e-5)
    ok &= check("a zero embedding is handled, not divided by",
                cosine_similarity(a, np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)) == 0.0)
    return ok


def test_identity_clustering():
    """
    The behaviour that matters: the same person recurs as one identity, a
    different person becomes a new one.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        import os
        os.environ["BASE_DIR"] = temporary_directory
        os.environ["DB_PATH"] = str(Path(temporary_directory) / "test.db")

        # Reload so the module-level paths pick up the temp directory.
        import importlib
        import src.common, src.db, src.faces
        importlib.reload(src.common)
        importlib.reload(src.db)
        importlib.reload(src.faces)

        connection = src.db.connect()
        engine = src.faces.FaceEngine()

        alice = make_embedding(100)
        # Same person, different clip: SFace typically lands around 0.5-0.7
        # across sessions, comfortably above the 0.40 identity threshold.
        alice_again = make_similar_embedding(alice, 0.60)
        # A different person: well below threshold.
        bob = make_similar_embedding(alice, 0.10, seed=42)

        face_id_1, name_1, _, new_1 = engine.match_or_create_identity(connection, alice, 1000)
        face_id_2, name_2, similarity, new_2 = engine.match_or_create_identity(
            connection, alice_again, 2000
        )
        face_id_3, name_3, _, new_3 = engine.match_or_create_identity(connection, bob, 3000)
        connection.commit()

        ok = check(f"first sighting creates an identity ({name_1})", new_1)
        ok &= check(f"same person matches the existing identity (sim={similarity:.2f})",
                    not new_2 and face_id_2 == face_id_1)
        ok &= check(f"a different person becomes a new identity ({name_3})",
                    new_3 and face_id_3 != face_id_1)
        ok &= check("naming follows Scout's 'Stranger #N' convention",
                    name_1 == "Stranger #1" and name_3 == "Stranger #2")

        row = connection.execute(
            "SELECT detection_count, embedding_count FROM faces WHERE id=?", (face_id_1,)
        ).fetchone()
        ok &= check("the matched identity folded in the second embedding",
                    row["embedding_count"] == 2)

        total = connection.execute("SELECT COUNT(*) n FROM faces").fetchone()["n"]
        ok &= check("two people produced exactly two identities", total == 2)
        connection.close()
        return ok


def test_within_clip_deduplication():
    """
    One person standing in front of the car for a whole sentry clip must count
    as ONE sighting, not one per frame - otherwise they dominate every score.
    """
    class FakeDetection:
        def __init__(self, confidence):
            self.confidence = confidence

    person = make_embedding(700)
    observations = [
        (FakeDetection(0.9), person),
        # Adjacent frames of one person barely differ in pose or lighting, so
        # they sit very high in embedding space.
        (FakeDetection(0.8), make_similar_embedding(person, 0.92, seed=1)),
        (FakeDetection(0.85), make_similar_embedding(person, 0.88, seed=2)),
        # Someone else entirely.
        (FakeDetection(0.95), make_similar_embedding(person, 0.05, seed=3)),
    ]

    kept = deduplicate_faces_within_clip(None, observations)
    ok = check(f"3 frames of one person + 1 other collapse to 2 (got {len(kept)})",
               len(kept) == 2)
    ok &= check("the highest-confidence exemplar is the one kept",
                max(d.confidence for d, _ in kept) == 0.95)
    return ok


def test_merge_and_split_identities():
    """
    Repairing clustering mistakes must leave BOTH sides consistent.

    A merge that reassigns detections but forgets to recompute the surviving
    identity's embedding would leave it matching against a mean that no longer
    reflects its sightings - so future frames of that person stop matching, and
    the split you just repaired silently reappears.
    """
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temporary_directory:
        os.environ["BASE_DIR"] = temporary_directory
        os.environ["DB_PATH"] = str(Path(temporary_directory) / "merge.db")

        import importlib
        import src.common, src.db, src.faces
        importlib.reload(src.common)
        importlib.reload(src.db)
        importlib.reload(src.faces)

        connection = src.db.connect()
        engine = src.faces.FaceEngine()

        # One person that clustering wrongly split into two identities.
        person = make_embedding(11)
        first_view = make_similar_embedding(person, 0.95, seed=4)
        second_view = make_similar_embedding(person, 0.93, seed=5)

        face_a, _, _, _ = engine.match_or_create_identity(connection, first_view, 1000)
        # Force a separate identity by inserting directly, simulating the split.
        face_b = "bbbbbbbbbbbb"
        src.db.upsert(connection, "faces", {
            "id": face_b, "person_name": "Stranger #2", "label": None, "is_known": 0,
            "embedding": src.faces.embedding_to_blob(second_view), "embedding_count": 1,
            "first_seen_ts": 2000, "last_seen_ts": 2000, "detection_count": 1,
            "threat_score": 0.0, "best_image": None,
        })
        for index, (face_id, embedding, ts) in enumerate(
            [(face_a, first_view, 1000), (face_b, second_view, 2000)]
        ):
            src.db.upsert(connection, "face_detections", {
                "id": f"det{index}", "face_id": face_id, "ts": ts, "clip_id": None,
                "source_file": f"clip{index}.mp4", "camera": "back", "frame_number": 0,
                "det_confidence": 0.9, "match_distance": 1.0,
                "embedding": src.faces.embedding_to_blob(embedding), "image": None,
                "lat": None, "lon": None, "drive_id": None,
            })
        connection.commit()

        # --- merge -------------------------------------------------------
        merged = src.faces.merge_identities(connection, face_b, face_a)
        connection.commit()

        ok = check("merge reports success", merged)
        ok &= check("merged-away identity is gone",
                    connection.execute("SELECT COUNT(*) n FROM faces").fetchone()["n"] == 1)
        survivor = connection.execute("SELECT * FROM faces WHERE id=?", (face_a,)).fetchone()
        ok &= check("survivor owns both sightings", survivor["detection_count"] == 2)
        ok &= check("survivor's embedding was recomputed from both",
                    survivor["embedding_count"] == 2)
        ok &= check("survivor's first/last seen span both sightings",
                    survivor["first_seen_ts"] == 1000 and survivor["last_seen_ts"] == 2000)

        recomputed = src.faces.blob_to_embedding(survivor["embedding"])
        ok &= check("recomputed embedding still matches the person",
                    src.faces.cosine_similarity(recomputed, person) > 0.9)

        # --- split -------------------------------------------------------
        new_face_id = src.faces.split_detection_into_new_identity(connection, "det1")
        connection.commit()

        ok &= check("split created a new identity", bool(new_face_id))
        ok &= check("there are two identities again",
                    connection.execute("SELECT COUNT(*) n FROM faces").fetchone()["n"] == 2)
        original = connection.execute("SELECT * FROM faces WHERE id=?", (face_a,)).fetchone()
        ok &= check("original was recomputed down to one sighting",
                    original["detection_count"] == 1 and original["embedding_count"] == 1)
        moved = connection.execute(
            "SELECT face_id FROM face_detections WHERE id='det1'"
        ).fetchone()
        ok &= check("the detection now belongs to the new identity",
                    moved["face_id"] == new_face_id)

        ok &= check("splitting a missing detection is handled",
                    src.faces.split_detection_into_new_identity(connection, "nope") is None)
        ok &= check("merging an identity into itself is refused",
                    not src.faces.merge_identities(connection, face_a, face_a))

        connection.close()
        return ok


def test_engine_reports_status():
    engine = FaceEngine()
    status = engine.status_text()
    ok = check(f"engine reports a status ('{status}')", bool(status))
    ok &= check("status distinguishes ready from disabled",
                ("ready" in status) or ("disabled" in status) or ("detection only" in status))
    return ok


def main():
    tests = [
        ("embedding serialization", test_blob_roundtrip),
        ("cosine similarity", test_cosine_similarity),
        ("within-clip deduplication", test_within_clip_deduplication),
        ("engine status reporting", test_engine_reports_status),
        ("identity clustering", test_identity_clustering),
        ("merge and split identities", test_merge_and_split_identities),
    ]
    failures = 0
    for name, test_function in tests:
        print(f"\n{name}:")
        if not test_function():
            failures += 1
    print(f"\n{'=' * 60}\n{len(tests) - failures}/{len(tests)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
