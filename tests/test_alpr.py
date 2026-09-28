"""
test_alpr.py

Tests for the ALPR stage that don't need a car.

Real TeslaCam night footage produces plates that are genuinely unreadable, so
"the pipeline read nothing" is the correct result there and tells us nothing
about whether the OCR is wired up. These tests render synthetic plates at known
text, which isolates the question: given a legible plate, do we read it?

Run:  python3 tests/test_alpr.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.alpr import (  # noqa: E402
    PlateCandidate,
    is_legible,
    looks_like_plate_text,
    normalize_plate_text,
    plates_probably_match,
    vote_on_plate_text,
)


def render_plate(text: str, width: int = 320, height: int = 160, dark: bool = False) -> np.ndarray:
    """Draw a synthetic US-style plate: dark characters on a light background."""
    background = 30 if dark else 240
    foreground = 20 if dark else 25
    image = np.full((height, width, 3), background, dtype=np.uint8)

    cv2.putText(image, text, (18, int(height * 0.65)), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (foreground, foreground, foreground), 4, cv2.LINE_AA)
    return image


def check(label, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def test_normalization():
    ok = check("strips punctuation and case", normalize_plate_text("7ab-c 123") == "7ABC123")
    ok &= check("rejects too-short reads", not looks_like_plate_text("AB"))
    ok &= check("rejects repeated-glyph hallucination", not looks_like_plate_text("IIIIIII"))
    ok &= check("accepts a normal plate", looks_like_plate_text("7ABC123"))
    return ok


def test_confusable_matching():
    ok = check("B/8 confusion still matches", plates_probably_match("7ABC123", "7A8C123"))
    ok &= check("O/0 confusion still matches", plates_probably_match("ABO123", "AB0123"))
    ok &= check("genuinely different plates do not match",
                not plates_probably_match("7ABC123", "9XYZ987"))
    ok &= check("different lengths do not match", not plates_probably_match("ABC123", "ABC1234"))
    return ok


def test_legibility_gate():
    """The gate must reject night crops and accept daylight ones."""
    ok = check("rejects a near-black crop", not is_legible(render_plate("7ABC123", dark=True)))
    ok &= check("accepts a well-lit crop", is_legible(render_plate("7ABC123")))
    ok &= check("rejects an empty crop", not is_legible(np.zeros((0, 0, 3), dtype=np.uint8)))
    ok &= check("rejects a flat grey crop",
                not is_legible(np.full((80, 240, 3), 128, dtype=np.uint8)))
    return ok


def test_ocr_reads_a_synthetic_plate():
    """
    The end-to-end proof that OCR is wired correctly.

    Skips (rather than fails) when tesseract isn't installed, so the suite
    still runs on a host that only has the Python deps.
    """
    from src.alpr import PlateReader

    reader = PlateReader()
    if reader.tesseract is None:
        print("  [SKIP] tesseract not installed on this host")
        return True

    expected = "7ABC123"
    plate_image = render_plate(expected)
    raw_text, confidence = reader.recognize(plate_image)
    read = normalize_plate_text(raw_text)

    ok = check(f"read something (got '{read}', conf={confidence:.2f})", bool(read))
    ok &= check(f"read matches '{expected}'", plates_probably_match(read, expected))
    return ok


def test_voting_prefers_frame_support():
    """
    A read backed by many frames beats a single lucky high-confidence read.

    This is the mechanism that makes plate reads survive a moving vehicle, so
    it needs to hold even when the odd frame disagrees confidently.
    """
    candidates = [
        PlateCandidate(text="7ABC123", raw_text="7ABC123", det_confidence=0.6,
                       ocr_confidence=0.6, bbox=(0, 0, 100, 50), frame_number=n)
        for n in range(6)
    ]
    # One frame read something else, very confidently.
    candidates.append(PlateCandidate(text="9XYZ987", raw_text="9XYZ987", det_confidence=0.99,
                                     ocr_confidence=0.99, bbox=(0, 0, 100, 50), frame_number=99))

    winner = vote_on_plate_text(candidates)
    ok = check("6-frame consensus beats 1 confident outlier", winner.text == "7ABC123")

    # Confusable variants should vote together rather than splitting the group.
    mixed = [
        PlateCandidate(text="7ABC123", raw_text="", det_confidence=0.5, ocr_confidence=0.5,
                       bbox=(0, 0, 1, 1), frame_number=0),
        PlateCandidate(text="7A8C123", raw_text="", det_confidence=0.5, ocr_confidence=0.5,
                       bbox=(0, 0, 1, 1), frame_number=1),
        PlateCandidate(text="7ABC1Z3", raw_text="", det_confidence=0.5, ocr_confidence=0.5,
                       bbox=(0, 0, 1, 1), frame_number=2),
    ]
    winner = vote_on_plate_text(mixed)
    ok &= check("confusable variants vote as one plate", winner is not None)

    ok &= check("empty candidate list returns None", vote_on_plate_text([]) is None)
    return ok


def main():
    tests = [
        ("text normalization", test_normalization),
        ("confusable glyph matching", test_confusable_matching),
        ("legibility gate", test_legibility_gate),
        ("OCR reads a synthetic plate", test_ocr_reads_a_synthetic_plate),
        ("cross-frame voting", test_voting_prefers_frame_support),
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
