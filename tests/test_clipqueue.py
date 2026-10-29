"""
test_clipqueue.py

The processing queue: what goes first, that a clip never arrives without the
folder it came from, and that frame sampling still lands on the frames it
asks for now that it seeks instead of decoding everything.

Run:  python3 tests/test_clipqueue.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BASE_DIR", tempfile.mkdtemp(prefix="clipqueue-test-"))

from src import clipqueue  # noqa: E402
from src import ingest     # noqa: E402


def make_inbox(names_and_sources: list[tuple[str, str | None]]) -> Path:
    inbox = Path(tempfile.mkdtemp(prefix="inbox-"))
    for name, source in names_and_sources:
        (inbox / name).write_bytes(b"")
        clipqueue.write_source(inbox / name, source)
    return inbox


def order(inbox: Path) -> list[str]:
    return [entry["name"] for entry in clipqueue.queue_entries(inbox)]


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------
def test_newest_minute_goes_first():
    inbox = make_inbox([
        ("2026-02-16_21-31-48-back.mp4", "RecentClips"),
        ("2026-02-16_21-33-22-back.mp4", "RecentClips"),
        ("2026-02-16_21-32-22-back.mp4", "RecentClips"),
    ])
    assert order(inbox) == [
        "2026-02-16_21-33-22-back.mp4",
        "2026-02-16_21-32-22-back.mp4",
        "2026-02-16_21-31-48-back.mp4",
    ], order(inbox)


def test_events_jump_ahead_of_newer_recent_clips():
    inbox = make_inbox([
        ("2026-02-16_22-00-00-back.mp4", "RecentClips"),
        ("2026-02-16_21-00-00-back.mp4", "SentryClips"),
        ("2026-02-16_20-00-00-back.mp4", "SavedClips"),
    ])
    assert order(inbox)[:2] == [
        "2026-02-16_21-00-00-back.mp4", "2026-02-16_20-00-00-back.mp4",
    ], order(inbox)


def test_front_and_back_lead_their_minute():
    names = [f"2026-02-16_21-31-48-{camera}.mp4" for camera in
             ("left_pillar", "right_repeater", "back", "front", "left_repeater")]
    inbox = make_inbox([(name, "RecentClips") for name in names])
    got = order(inbox)
    assert got[0].endswith("-front.mp4") and got[1].endswith("-back.mp4"), got


def test_a_clip_with_no_sidecar_is_treated_as_recent_not_dropped():
    inbox = make_inbox([("2026-02-16_21-31-48-back.mp4", None)])
    entries = clipqueue.queue_entries(inbox)
    assert len(entries) == 1 and entries[0]["source"] is None, entries


def test_in_flight_copies_and_sidecars_are_not_queued():
    inbox = make_inbox([("2026-02-16_21-31-48-back.mp4", "RecentClips")])
    (inbox / ".tmp_2026-02-16_21-32-22-back.mp4").write_bytes(b"")
    assert order(inbox) == ["2026-02-16_21-31-48-back.mp4"], order(inbox)


def test_an_unparseable_name_waits_behind_everything():
    inbox = make_inbox([
        ("mystery.mp4", "RecentClips"),
        ("2026-02-16_21-31-48-back.mp4", "RecentClips"),
    ])
    assert order(inbox)[-1] == "mystery.mp4", order(inbox)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def test_the_estimate_is_per_camera():
    entries = [{"camera": "front"}, {"camera": "back"}, {"camera": "back"}]
    assert clipqueue.estimate_seconds(entries, {"front": 12.0, "back": 5.0}, None) == 22.0
    # An unseen camera falls back to the overall mean...
    assert clipqueue.estimate_seconds([{"camera": "left_pillar"}], {"front": 12.0}, 8.0) == 8.0
    # ...and with nothing to go on the answer is "unknown", not zero.
    assert clipqueue.estimate_seconds([{"camera": "front"}], {}, None) is None
    assert clipqueue.estimate_seconds([], {}, None) == 0.0


def test_the_summary_counts_events_and_spans():
    inbox = make_inbox([
        ("2026-02-16_21-31-48-back.mp4", "SentryClips"),
        ("2026-02-16_21-33-22-back.mp4", "RecentClips"),
    ])
    summary = clipqueue.summarize(clipqueue.queue_entries(inbox))
    assert summary["depth"] == 2 and summary["events"] == 1, summary
    assert summary["oldest_captured_ts"] < summary["newest_captured_ts"], summary
    assert "path" not in summary["next"][0], "Path objects do not serialise to JSON"


# ---------------------------------------------------------------------------
# Ingest keeps the source
# ---------------------------------------------------------------------------
def test_ingest_records_the_folder_before_the_clip_appears():
    with tempfile.TemporaryDirectory() as tmp:
        event = Path(tmp) / "TeslaCam" / "SentryClips" / "2026-02-16_21-31-48"
        event.mkdir(parents=True)
        src = event / "2026-02-16_21-31-48-front.mp4"
        src.write_bytes(b"x" * 64)
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()

        seen_at_rename = {}
        real_rename = Path.rename

        def spying_rename(self, target):
            if Path(target).suffix == ".mp4":
                seen_at_rename["source"] = clipqueue.read_source(Path(target))
            return real_rename(self, target)

        original_stable = ingest.file_is_stable
        ingest.file_is_stable = lambda path, **_: True
        Path.rename = spying_rename
        try:
            landed = ingest.safe_copy_to_inbox(src, inbox)
        finally:
            Path.rename = real_rename
            ingest.file_is_stable = original_stable

        assert landed and landed.exists(), landed
        assert seen_at_rename.get("source") == "SentryClips", seen_at_rename
        assert clipqueue.queue_entries(inbox)[0]["source"] == "SentryClips"


def test_discard_source_removes_the_sidecar():
    inbox = make_inbox([("2026-02-16_21-31-48-back.mp4", "SentryClips")])
    clip = inbox / "2026-02-16_21-31-48-back.mp4"
    clipqueue.discard_source(clip)
    assert clipqueue.read_source(clip) is None
    clipqueue.discard_source(clip)   # twice is fine


# ---------------------------------------------------------------------------
# Frame sampling (needs OpenCV - runs in the processor image)
# ---------------------------------------------------------------------------
def synthetic_clip(path: Path, frames: int) -> None:
    """Every frame's pixels encode its own index, so a sample can be checked."""
    import cv2
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index % 256, dtype=np.uint8))
    writer.release()


def test_seeking_lands_on_the_frames_it_asks_for():
    try:
        import cv2  # noqa: F401
        from src.processor import sample_frames
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        synthetic_clip(clip, 240)
        samples = list(sample_frames(clip, frame_budget=8))
        assert [index for index, _ in samples] == [0, 30, 60, 90, 120, 150, 180, 210], samples
        for index, frame in samples:
            # Compression blurs the flat value a little; a wrong frame would
            # be off by ~30.
            assert abs(int(frame.mean()) - index % 256) <= 3, (index, frame.mean())


def test_dense_sampling_reads_sequentially_and_still_lands():
    try:
        from src.processor import sample_frames
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        synthetic_clip(clip, 40)
        samples = list(sample_frames(clip, frame_budget=10))   # step 4: no seeks
        assert [index for index, _ in samples] == list(range(0, 40, 4)), samples
        for index, frame in samples:
            assert abs(int(frame.mean()) - index) <= 3, (index, frame.mean())


# ---------------------------------------------------------------------------
# GPU fallback (needs the processor image)
# ---------------------------------------------------------------------------
def test_a_gpu_failure_reruns_the_clip_on_the_cpu_and_rests_the_gpu():
    try:
        from src import processor
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    calls = []

    def fake_analyze(model, clip, plate_reader=None, face_engine=None, device=None, progress=None):
        calls.append(device)
        if device is None:
            raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`")
        return "analysis"

    original = processor.analyze_clip
    processor.analyze_clip = fake_analyze
    try:
        detectors = processor.Detectors("gpu-model", None, None)
        detectors.has_gpu = True
        detectors._cpu_detectors = lambda: ("cpu-model", None)
        analysis, device = detectors.analyze(Path("clip.mp4"))
        assert (analysis, device) == ("analysis", "cpu"), (analysis, device)
        assert calls == [None, "cpu"], calls
        assert detectors.device == "cpu", "the GPU should be resting"
        assert detectors.gpu_failures == 1 and detectors.cpu_fallbacks == 1
        # While it rests, the next clip goes straight to the CPU.
        detectors.analyze(Path("clip2.mp4"))
        assert calls == [None, "cpu", "cpu"], calls
    finally:
        processor.analyze_clip = original


def test_an_ordinary_error_is_not_mistaken_for_a_gpu_failure():
    try:
        from src import processor
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    assert processor.is_gpu_failure(RuntimeError("CUDA error: out of memory"))
    assert processor.is_gpu_failure(RuntimeError(
        'NVML_SUCCESS == r INTERNAL ASSERT FAILED at "CUDACachingAllocator.cpp"'))
    assert not processor.is_gpu_failure(ValueError("moov atom not found"))


# ---------------------------------------------------------------------------
# Demo feed (needs the processor image)
# ---------------------------------------------------------------------------
def demo_clips() -> Path:
    root = Path(tempfile.mkdtemp(prefix="natix-"))
    for minute in ("2026-02-16_21-31-48", "2026-02-16_21-32-22", "2026-02-16_21-33-22"):
        event = root / "TeslaCam" / "SentryClips" / minute
        event.mkdir(parents=True)
        for camera in ("front", "back"):
            (event / f"{minute}-{camera}.mp4").write_bytes(b"")
    return root


def test_the_realtime_demo_releases_one_minute_per_minute():
    try:
        from src.processor import DemoFeed
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    feed = DemoFeed(demo_clips(), "realtime", now=1000.0)
    assert len(feed.entries) == 6
    first = feed.pending(now=1000.0)
    assert [entry["name"] for entry in first] == [
        "2026-02-16_21-31-48-front.mp4", "2026-02-16_21-31-48-back.mp4"], first
    assert len(feed.pending(now=1059.0)) == 2          # minute two not yet
    assert len(feed.pending(now=1061.0)) == 4
    # Within what has arrived, the newest minute goes first - as in the car.
    assert feed.pending(now=1061.0)[0]["name"] == "2026-02-16_21-32-22-front.mp4"
    assert all(entry["demo"] for entry in feed.entries)


def test_the_demo_reports_its_worst_wait_and_finishes():
    try:
        from src.processor import DemoFeed
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    feed = DemoFeed(demo_clips(), "backlog", now=1000.0)
    assert len(feed.pending(now=1000.0)) == 6           # everything at once
    for offset, entry in enumerate(feed.pending(now=1000.0), start=1):
        feed.mark_done(entry, now=1000.0 + 10 * offset)
    summary = feed.summary(1100.0)
    assert summary["done"] == 6 and not summary["active"], summary
    assert summary["max_lag_seconds"] == 60.0, summary
    assert feed.pending(now=1100.0) == []


def test_a_stopped_demo_releases_nothing_more():
    try:
        from src.processor import DemoFeed
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    feed = DemoFeed(demo_clips(), "realtime", now=1000.0)
    feed.stopped = True
    assert feed.pending(now=5000.0) == [] and not feed.active


# ---------------------------------------------------------------------------
# Detection boxes (needs the processor image)
# ---------------------------------------------------------------------------
class FakeModel:
    """Stands in for YOLO: records the frames it is shown, detects nothing."""
    names = {0: "person", 2: "car"}

    def __init__(self):
        self.seen = []

    def __call__(self, image, verbose=False, **kwargs):
        self.seen.append(int(image.mean()))
        return [type("Result", (), {"boxes": None})()]


def test_boxes_are_fractions_of_the_frame():
    try:
        from src.processor import normalised_box
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    box = normalised_box("vehicle", "car", 0.8612, (144.8, 93.8, 724.0, 469.0), 1448, 938)
    assert box == {"kind": "vehicle", "label": "car", "conf": 0.861, "box": [0.1, 0.1, 0.5, 0.5]}, box


def test_the_idle_pass_boxes_exactly_the_frames_the_gif_shows():
    try:
        from src import processor
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        synthetic_clip(clip, 200)          # 30 fps, ~6.7 s: frame value = index % 256
        original = processor.PROCESSED_DIR
        processor.PROCESSED_DIR = Path(tmp)
        try:
            model = FakeModel()
            frames = processor.BoxAnnotator()._gif_frames(model, {"source_file": "clip.mp4"})
        finally:
            processor.PROCESSED_DIR = original
        expected = [round(k / processor.GIF_FPS * 30.0) for k in range(processor.GIF_SECONDS * processor.GIF_FPS)]
        assert len(frames) == len(expected) == 40, (len(frames), len(expected))
        # ffmpeg's fps filter from t=0: GIF frame k is video frame round(k/8*30).
        assert all(abs(seen - want) <= 3 for seen, want in zip(model.seen, expected)), list(zip(model.seen, expected))[:8]


def test_the_idle_pass_boxes_old_snapshots_from_the_jpeg_itself():
    try:
        import cv2
        import numpy as np
        from src import processor
    except ImportError as error:
        print(f"  skip  (no processor dependencies here: {error})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        media, alerts = Path(tmp) / "media", Path(tmp) / "alerts"
        media.mkdir(); alerts.mkdir()
        cv2.imwrite(str(media / "abc.jpg"), np.full((48, 64, 3), 90, dtype=np.uint8))
        (alerts / "abc.json").write_text('{"id": "abc", "jpeg": "abc.jpg", "gif": "abc.gif"}')
        saved = processor.MEDIA_DIR, processor.ALERTS_DIR
        processor.MEDIA_DIR, processor.ALERTS_DIR = media, alerts
        try:
            annotator = processor.BoxAnnotator()
            assert annotator.pending() == 1, annotator.todo     # the GIF does not exist: skipped
            detectors = type("D", (), {"model": FakeModel(), "gpu_retry_after": 0})()
            assert annotator.annotate_one(detectors) == "abc.jpg"
            sidecar = json.loads((media / "abc.jpg.boxes.json").read_text())
            assert sidecar["source"] == "snapshot" and sidecar["frames"] == [[]], sidecar
            annotator.last_scan = 0
            assert annotator.pending() == 0
        finally:
            processor.MEDIA_DIR, processor.ALERTS_DIR = saved


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
    shutil.rmtree(os.environ["BASE_DIR"], ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
