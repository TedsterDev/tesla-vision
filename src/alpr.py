"""
alpr.py

Automatic License Plate Recognition. This is the port of Scout's ALPR stage,
which chained a custom YOLOv3 plate detector into ALPR-Unconstrained's
WPOD-NET for rectification and character recognition.

We reimplement that same three-step shape with parts that actually run on a
Jetson Orin Nano in 2026 without a TensorFlow 1.x time machine:

    1. LOCALIZE  find the plate rectangle in the frame
    2. RECTIFY   warp the (usually skewed) plate to a flat rectangle
    3. RECOGNIZE read the characters

Step 1 has a preference ladder, best first, because plate detector weights are
a licensing/redistribution mess and we can't assume they're on the box:

    a) an Ultralytics YOLO plate model at $MODELS_DIR/plate_detector.pt
       (scripts/fetch_models.sh can install one)
    b) OpenCV's haarcascade_russian_plate_number, which ships inside the
       opencv-python wheel and - despite the name - is a decent generic
       rectangular-plate finder for front/rear-on views
    c) an edge+contour search restricted to vehicle boxes YOLO already found

Step 2 is a light stand-in for WPOD-NET: we find the plate's quadrilateral in
the crop and perspective-warp it. WPOD-NET learned this; we solve it
geometrically, which is worse on extreme angles and free everywhere else.

Step 3 is Tesseract with a character whitelist. Single-frame plate OCR from a
dashcam is genuinely unreliable, so the real accuracy win is in the caller:
processor.py collects candidates across every sampled frame of a clip and votes
(see `vote_on_plate_text`). Scout read each frame independently and took what
it got; voting across ~30 frames is what turns a coin-flip into a read.
"""
import os
import re

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.common import MODELS_DIR, env_float, env_int

# --- Tunables ---------------------------------------------------------------
# A plate is far wider than it is tall. US plates are 12"x6" (2.0:1), European
# ones up to 4.7:1. We accept a generous band and reject everything else, which
# throws out most of what a Haar cascade gets wrong (headlights, badges).
# Measured against real TeslaCam night footage: the trained detector's boxes
# come out at 1.67-1.72 for US plates because it pads around the plate border,
# so a 2.0 "true" aspect arrives here nearer 1.7. A 1.7 floor was clipping real
# detections; 1.4 keeps them and costs nothing now that precision comes from the
# detector rather than from this shape filter.
MIN_PLATE_ASPECT_RATIO = env_float("MIN_PLATE_ASPECT_RATIO", 1.4)
MAX_PLATE_ASPECT_RATIO = env_float("MAX_PLATE_ASPECT_RATIO", 5.5)

# Below ~55px wide there simply aren't enough pixels per character to read.
MIN_PLATE_WIDTH_PIXELS = env_int("MIN_PLATE_WIDTH_PIXELS", 55)
MIN_PLATE_HEIGHT_PIXELS = env_int("MIN_PLATE_HEIGHT_PIXELS", 16)

# Plates are 2-8 characters in essentially every jurisdiction.
MIN_PLATE_CHARACTERS = env_int("MIN_PLATE_CHARACTERS", 5)
MAX_PLATE_CHARACTERS = env_int("MAX_PLATE_CHARACTERS", 8)

# Tesseract's per-character confidence, 0-100. Below 60 is usually noise.
MIN_OCR_CONFIDENCE = env_float("MIN_OCR_CONFIDENCE", 60.0)

PLATE_DETECTOR_WEIGHTS = MODELS_DIR / "plate_detector.pt"

# How many vehicles per frame to hand the plate detector, largest first.
# Detector cost is per-crop, and the largest box is the nearest vehicle, which
# is the only one whose plate has any chance of being legible anyway.
MAX_VEHICLE_CROPS_PER_FRAME = env_int("MAX_VEHICLE_CROPS_PER_FRAME", 3)

# Legibility gate. A plate crop from a dark car park is a real plate that
# happens to be unreadable, and OCR asked to read it will still return *some*
# characters. Those characters are noise, and noise here is expensive: it mints
# a fake plate identity that the correlation engine then reasons about as if it
# were a vehicle. Refusing to guess is strictly better than guessing.
MIN_PLATE_CONTRAST = env_float("MIN_PLATE_CONTRAST", 18.0)   # std-dev of grey levels
MIN_PLATE_BRIGHTNESS = env_float("MIN_PLATE_BRIGHTNESS", 28.0)

# Characters Tesseract is allowed to emit. Excluding punctuation and lowercase
# removes most of its creative interpretations of dirt and screw heads.
PLATE_CHARACTER_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Glyph pairs that low-resolution OCR routinely confuses. Used only when
# comparing two reads for equality, never to rewrite what we stored.
CONFUSABLE_GLYPHS = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
})


@dataclass
class PlateCandidate:
    """One plate read from one frame. Many of these get voted into one answer."""
    text: str                       # normalized, e.g. "7ABC123"
    raw_text: str                   # exactly what the OCR returned
    det_confidence: float           # confidence that this rectangle is a plate
    ocr_confidence: float           # mean per-character OCR confidence, 0-1
    bbox: tuple[int, int, int, int] # x, y, w, h in the full frame
    frame_number: int = 0
    crop: np.ndarray | None = field(default=None, repr=False)

    @property
    def combined_confidence(self) -> float:
        """Single number for ranking candidates - both stages have to agree."""
        return self.det_confidence * self.ocr_confidence


def is_legible(plate_image: np.ndarray) -> bool:
    """
    Decide whether a crop has enough signal to be worth reading.

    Two cheap checks against the grey-level histogram:
        contrast   - characters need to stand out from the plate background
        brightness - a near-black crop has no recoverable detail at all

    This is what stops a night-time car park producing a dashboard full of
    invented plates. It runs before OCR, so it also saves the Tesseract call.
    """
    if plate_image is None or plate_image.size == 0:
        return False

    grayscale = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if plate_image.ndim == 3 else plate_image

    if float(np.mean(grayscale)) < MIN_PLATE_BRIGHTNESS:
        return False
    if float(np.std(grayscale)) < MIN_PLATE_CONTRAST:
        return False
    return True


def normalize_plate_text(raw: str) -> str:
    """
    Strip a raw OCR string down to plausible plate characters.

    We uppercase, drop anything not alphanumeric, and remove the state-name and
    slogan text that OCR loves to pick up from the top and bottom of US plates.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    return cleaned


def looks_like_plate_text(text: str) -> bool:
    """
    Reject reads that can't be a plate.

    Rules kept deliberately loose because plate formats vary wildly by state
    and country; we're filtering obvious garbage, not validating a format.
    """
    if not (MIN_PLATE_CHARACTERS <= len(text) <= MAX_PLATE_CHARACTERS):
        return False
    # A plate has both structure and variety - all-identical characters
    # ("IIIIIII") is a classic Tesseract hallucination on a picket fence.
    if len(set(text)) < 3:
        return False
    # Must contain at least one digit or be a vanity plate of pure letters;
    # what we reject is a read with no letters *and* no digits, i.e. empty.
    if not any(character.isalnum() for character in text):
        return False
    return True


def plates_probably_match(text_a: str, text_b: str) -> bool:
    """
    Compare two plate reads, tolerating the glyphs OCR always confuses.

    "7ABC123" and "7A8C123" should be treated as the same vehicle - B/8 is the
    single most common dashcam OCR error. Used when voting and when looking up
    an existing plate identity.
    """
    if text_a == text_b:
        return True
    if len(text_a) != len(text_b):
        return False
    return text_a.translate(CONFUSABLE_GLYPHS) == text_b.translate(CONFUSABLE_GLYPHS)


def vote_on_plate_text(candidates: list[PlateCandidate]) -> PlateCandidate | None:
    """
    Collapse every plate read from a clip into the single most-supported answer.

    This is the accuracy trick that Scout lacked. One frame of a moving plate
    is a coin flip; thirty frames voting is a read. We group candidates by
    confusable-equivalence, score each group by summed confidence *and* how many
    frames backed it, then return the highest-confidence exemplar of the winner.
    """
    if not candidates:
        return None

    groups: list[list[PlateCandidate]] = []
    for candidate in candidates:
        for group in groups:
            if plates_probably_match(group[0].text, candidate.text):
                group.append(candidate)
                break
        else:
            groups.append([candidate])

    def group_score(group: list[PlateCandidate]) -> float:
        # Frame support is weighted heavily: a mediocre read seen ten times
        # beats one lucky high-confidence read of something else.
        support = len(group)
        confidence = sum(member.combined_confidence for member in group)
        return support * 1.5 + confidence

    winning_group = max(groups, key=group_score)
    best = max(winning_group, key=lambda member: member.combined_confidence)

    # Report the vote's breadth through ocr_confidence so downstream code can
    # see that a 20-frame consensus is stronger than a 2-frame one.
    frame_support = len(winning_group)
    consensus_bonus = min(0.25, 0.03 * frame_support)
    best.ocr_confidence = min(1.0, best.ocr_confidence + consensus_bonus)

    return best


# ---------------------------------------------------------------------------
# Step 2 - rectification (our stand-in for WPOD-NET)
# ---------------------------------------------------------------------------

def box_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """
    Intersection-over-union of two (x, y, w, h) boxes.

    Used to recognise when the whole-frame pass and a vehicle-crop pass have
    found the same physical plate, so it is reported once rather than twice.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_boxes(
    boxes: list[tuple[tuple[int, int, int, int], float]],
    overlap_threshold: float = 0.35,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """
    Collapse boxes that describe the same plate, keeping the most confident.

    Standard greedy non-maximum suppression. The threshold is loose because the
    two passes see the plate at different scales, so their boxes agree on the
    object but not to the pixel.
    """
    kept: list[tuple[tuple[int, int, int, int], float]] = []

    for box, confidence in sorted(boxes, key=lambda item: item[1], reverse=True):
        if any(box_overlap_ratio(box, other) >= overlap_threshold for other, _ in kept):
            continue
        kept.append((box, confidence))

    return kept


def rectify_plate(crop: np.ndarray) -> np.ndarray:
    """
    Perspective-correct a plate crop so the characters sit on a horizontal line.

    ALPR-Unconstrained trained WPOD-NET to regress the plate's four corners.
    We approximate: threshold the crop, take the largest 4-point contour that
    fills most of the frame, and warp it to a flat 240x80 rectangle. When no
    convincing quad is found we return the crop unchanged, which is the right
    fallback for plates that were already close to face-on.
    """
    if crop is None or crop.size == 0:
        return crop

    height, width = crop.shape[:2]
    if width < MIN_PLATE_WIDTH_PIXELS or height < MIN_PLATE_HEIGHT_PIXELS:
        return crop

    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blurred = cv2.bilateralFilter(grayscale, 9, 75, 75)
    edges = cv2.Canny(blurred, 40, 140)

    # Close small gaps so the plate border reads as one contour.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop

    crop_area = float(width * height)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        # The plate border should dominate its own crop.
        if cv2.contourArea(contour) < 0.35 * crop_area:
            break

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approximation) != 4:
            continue

        source_points = _order_quadrilateral(approximation.reshape(4, 2).astype(np.float32))
        target_points = np.array(
            [[0, 0], [239, 0], [239, 79], [0, 79]], dtype=np.float32
        )
        transform = cv2.getPerspectiveTransform(source_points, target_points)
        return cv2.warpPerspective(crop, transform, (240, 80))

    return crop


def _order_quadrilateral(points: np.ndarray) -> np.ndarray:
    """
    Sort four corner points into top-left, top-right, bottom-right, bottom-left.

    getPerspectiveTransform needs a consistent winding or the warp comes out
    mirrored or rotated.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)

    coordinate_sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(coordinate_sums)]   # top-left has smallest x+y
    ordered[2] = points[np.argmax(coordinate_sums)]   # bottom-right, largest x+y

    coordinate_diffs = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(coordinate_diffs)]  # top-right, smallest y-x
    ordered[3] = points[np.argmax(coordinate_diffs)]  # bottom-left, largest y-x

    return ordered


# ---------------------------------------------------------------------------
# Step 3 - character recognition
# ---------------------------------------------------------------------------

def preprocess_for_ocr(plate_image: np.ndarray) -> np.ndarray:
    """
    Clean a rectified plate crop up for Tesseract.

    The pipeline (upscale -> CLAHE -> Otsu -> open) matters more than the OCR
    engine choice at these resolutions. Upscaling first is the single biggest
    win: Tesseract's models expect roughly 30px-tall characters and a dashcam
    plate is often half that.
    """
    grayscale = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if plate_image.ndim == 3 else plate_image

    height, width = grayscale.shape[:2]
    if height < 60:
        scale = 60.0 / max(height, 1)
        grayscale = cv2.resize(
            grayscale, (int(width * scale), 60), interpolation=cv2.INTER_CUBIC
        )

    # Local contrast equalisation copes with the headlight-glare / deep-shadow
    # split that ruins a global threshold at night.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(grayscale)

    denoised = cv2.bilateralFilter(equalized, 7, 60, 60)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Plates are dark-on-light; if we came out mostly black, invert.
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


class PlateReader:
    """
    The full localize -> rectify -> recognize pipeline.

    Constructed once per process and reused; loading Tesseract and any YOLO
    weights on every frame would dominate runtime.
    """

    def __init__(self) -> None:
        self.yolo_plate_model = None
        self.haar_cascade = None
        self.tesseract = None
        self.reasons_unavailable: list[str] = []
        # Plates we found but could not read. Surfaced in logs and the Settings
        # page so "no plates" is distinguishable from "plates too dark to read",
        # which are very different problems with very different fixes.
        self.illegible_crop_count = 0

        self._load_localizers()
        self._load_ocr()

    # -- loading ----------------------------------------------------------
    def _load_localizers(self) -> None:
        """Set up whichever plate localizers are actually available."""
        if PLATE_DETECTOR_WEIGHTS.exists():
            try:
                from ultralytics import YOLO
                self.yolo_plate_model = YOLO(str(PLATE_DETECTOR_WEIGHTS))
                print(f"[🔎 alpr] YOLO plate detector loaded: {PLATE_DETECTOR_WEIGHTS.name}")
            except Exception as exception_object:
                self.reasons_unavailable.append(f"YOLO plate weights failed to load: {exception_object}")

        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.haar_cascade = cascade
                if self.yolo_plate_model is None:
                    print("[🔎 alpr] using Haar cascade plate localizer (no YOLO plate weights found)")

    def _load_ocr(self) -> None:
        """Import pytesseract lazily so a missing binary degrades, not crashes."""
        try:
            import pytesseract
            # Fail fast and loudly here rather than once per frame later.
            pytesseract.get_tesseract_version()
            self.tesseract = pytesseract
        except Exception as exception_object:
            self.reasons_unavailable.append(f"tesseract unavailable: {exception_object}")

    @property
    def available(self) -> bool:
        """True when we can both find and read a plate."""
        has_localizer = self.yolo_plate_model is not None or self.haar_cascade is not None
        return has_localizer and self.tesseract is not None

    def status_text(self) -> str:
        """One-line summary for startup logs and the dashboard's Settings view."""
        if self.available:
            localizer = "yolo" if self.yolo_plate_model else "haar"
            return f"ready (localizer={localizer}, ocr=tesseract)"
        return "disabled - " + "; ".join(self.reasons_unavailable or ["no plate localizer available"])

    # -- step 1: localize -------------------------------------------------
    def _run_plate_model(self, image: np.ndarray, offset_x: int = 0, offset_y: int = 0):
        """Run the plate detector on one image, returning full-frame coordinates."""
        boxes: list[tuple[tuple[int, int, int, int], float]] = []
        for result in self.yolo_plate_model(image, verbose=False):
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                confidence = float(box.conf[0].item())
                boxes.append((
                    (x1 + offset_x, y1 + offset_y, x2 - x1, y2 - y1), confidence
                ))
        return boxes

    def _localize_with_yolo(
        self,
        frame: np.ndarray,
        vehicle_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Find plates, giving the detector as many real pixels as possible.

        Ultralytics letterboxes whatever you hand it down to imgsz (640 by
        default). A TeslaCam frame is 1448x938, and the front camera 2896x1876,
        so passing the whole frame silently scales plates down 2.3x-4.5x before
        the model ever sees them - which defeats the point of doing detection
        at full resolution.

        Raising imgsz is the wrong fix: this checkpoint was TRAINED at 640
        (train_args imgsz=640), so inferring at 1920 is off-distribution, and
        measured on this footage it mostly produced confident false positives -
        the most convincing "plate" it recovered turned out to be a Chipotle
        storefront sign.

        The right fix is to crop to the vehicles YOLO already found and run the
        detector on each crop at its native resolution. A car occupying 400px
        of a 2896px frame goes from 88px wide (after letterboxing) to 400px, so
        its plate lands in the size range the model was trained on. We fall
        back to the whole frame when no vehicle boxes were supplied.
        """
        # Always do the whole-frame pass as well. Measured on real footage,
        # the two passes are complementary rather than redundant: cropping
        # found a plate at 102x69/conf 0.79 that the frame pass saw as
        # 75x47/conf 0.42, but the frame pass caught plates on vehicles that
        # fell outside the per-frame crop budget and would otherwise be lost.
        # Union then de-duplicate, keeping the more confident box of any pair.
        candidates = self._run_plate_model(frame)

        if not vehicle_boxes:
            return candidates

        frame_height, frame_width = frame.shape[:2]
        # Nearest vehicles first - a distant car's plate is unreadable regardless.
        largest_first = sorted(
            vehicle_boxes, key=lambda box: box[2] * box[3], reverse=True
        )[:MAX_VEHICLE_CROPS_PER_FRAME]

        for x, y, width, height in largest_first:
            # Pad: plates sit at the very edge of a vehicle box, and Tesla's
            # detector boxes crop tight.
            pad = int(0.08 * max(width, height))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(frame_width, x + width + pad)
            y1 = min(frame_height, y + height + pad)

            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            candidates.extend(self._run_plate_model(crop, offset_x=x0, offset_y=y0))

        return deduplicate_boxes(candidates)

    def _localize_with_haar(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)

        detections = self.haar_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.06,
            minNeighbors=6,
            minSize=(MIN_PLATE_WIDTH_PIXELS, MIN_PLATE_HEIGHT_PIXELS),
        )
        # A cascade gives no confidence score. 0.5 is an honest "maybe" that
        # keeps the combined-confidence maths meaningful.
        return [((int(x), int(y), int(w), int(h)), 0.5) for (x, y, w, h) in detections]

    def _localize_with_contours(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Last-resort localizer: hunt for plate-shaped bright rectangles inside a
        region YOLO already told us contains a vehicle.

        Restricting to the vehicle box is what makes this tractable - run over a
        whole frame it would return every window and road sign in view.
        """
        region_x, region_y, region_width, region_height = region
        patch = frame[region_y:region_y + region_height, region_x:region_x + region_width]
        if patch.size == 0:
            return []

        grayscale = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # A morphological blackhat pulls out dark characters on a light plate.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

        gradient = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
        gradient = np.absolute(gradient)
        minimum, maximum = float(gradient.min()), float(gradient.max())
        if maximum - minimum < 1e-6:
            return []
        gradient = (255 * (gradient - minimum) / (maximum - minimum)).astype("uint8")

        gradient = cv2.GaussianBlur(gradient, (5, 5), 0)
        gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
        _, threshold = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[tuple[int, int, int, int], float]] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            x, y, w, h = cv2.boundingRect(contour)
            if w < MIN_PLATE_WIDTH_PIXELS or h < MIN_PLATE_HEIGHT_PIXELS:
                continue
            aspect_ratio = w / float(h)
            if not (MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO):
                continue
            # Translate back into full-frame coordinates.
            boxes.append(((region_x + x, region_y + y, w, h), 0.35))

        return boxes

    def localize(self, frame: np.ndarray, vehicle_boxes: list[tuple[int, int, int, int]] | None = None) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Find candidate plate rectangles, walking down the preference ladder.

        Args:
            frame: full BGR frame
            vehicle_boxes: optional (x, y, w, h) boxes YOLO flagged as vehicles,
                used to focus the contour fallback and to filter false positives

        Returns list of ((x, y, w, h), det_confidence).
        """
        candidates: list[tuple[tuple[int, int, int, int], float]] = []

        # Use the BEST available localizer only - never cascade down to a weaker
        # one just because the better one found nothing in this frame.
        #
        # Measured on real footage: cascading that way produced 62 boxes from
        # the contour fallback across two clips, every one of them a wheel arch,
        # against 26 genuine plates from the trained detector. "The good model
        # says there is no plate here" is an answer, not a failure to answer.
        if self.yolo_plate_model is not None:
            candidates.extend(self._localize_with_yolo(frame, vehicle_boxes))
        elif self.haar_cascade is not None:
            candidates.extend(self._localize_with_haar(frame))
        elif vehicle_boxes:
            for vehicle_box in vehicle_boxes:
                candidates.extend(self._localize_with_contours(frame, vehicle_box))

        # Shape sanity check applies to every localizer's output.
        accepted = []
        for (x, y, w, h), confidence in candidates:
            if w < MIN_PLATE_WIDTH_PIXELS or h < MIN_PLATE_HEIGHT_PIXELS:
                continue
            aspect_ratio = w / float(max(h, 1))
            if not (MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO):
                continue
            accepted.append(((x, y, w, h), confidence))

        return accepted

    # -- step 3: recognize ------------------------------------------------
    def recognize(self, plate_image: np.ndarray) -> tuple[str, float]:
        """
        Read characters off a rectified plate crop.

        Returns (raw_text, mean_confidence_0_to_1). ("", 0.0) when unreadable.
        """
        if self.tesseract is None:
            return "", 0.0

        prepared = preprocess_for_ocr(plate_image)

        # psm 7 = "treat the image as a single text line", which is what a
        # rectified plate is. oem 3 lets Tesseract pick its best engine.
        config = (
            f"--psm 7 --oem 3 -c tessedit_char_whitelist={PLATE_CHARACTER_WHITELIST}"
        )

        try:
            data = self.tesseract.image_to_data(
                prepared, config=config, output_type=self.tesseract.Output.DICT
            )
        except Exception:
            return "", 0.0

        words: list[str] = []
        confidences: list[float] = []
        for index, text in enumerate(data.get("text", [])):
            text = (text or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except (ValueError, KeyError, IndexError):
                continue
            if confidence < MIN_OCR_CONFIDENCE:
                continue
            words.append(text)
            confidences.append(confidence)

        if not words:
            return "", 0.0

        raw_text = "".join(words)
        mean_confidence = sum(confidences) / len(confidences) / 100.0
        return raw_text, mean_confidence

    # -- the whole pipeline -----------------------------------------------
    def read_plates(
        self,
        frame: np.ndarray,
        vehicle_boxes: list[tuple[int, int, int, int]] | None = None,
        frame_number: int = 0,
    ) -> list[PlateCandidate]:
        """
        Run localize -> rectify -> recognize over one frame.

        Returns every plausible read; the caller votes across frames.
        """
        if not self.available:
            return []

        found: list[PlateCandidate] = []

        for (x, y, w, h), det_confidence in self.localize(frame, vehicle_boxes):
            # Pad slightly - localizers tend to clip the outer characters.
            pad_x, pad_y = int(w * 0.04), int(h * 0.10)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(frame.shape[1], x + w + pad_x)
            y1 = min(frame.shape[0], y + h + pad_y)

            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            rectified = rectify_plate(crop)

            # Refuse to guess at a crop with no recoverable detail.
            if not is_legible(rectified):
                self.illegible_crop_count += 1
                continue

            raw_text, ocr_confidence = self.recognize(rectified)

            normalized = normalize_plate_text(raw_text)
            if not looks_like_plate_text(normalized):
                continue

            found.append(PlateCandidate(
                text=normalized,
                raw_text=raw_text,
                det_confidence=det_confidence,
                ocr_confidence=ocr_confidence,
                bbox=(x, y, w, h),
                frame_number=frame_number,
                crop=crop,
            ))

        return found
