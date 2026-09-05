"""
test_correlate.py

Behavioural tests for the surveillance detection engine.

These aren't unit tests of arithmetic - they're scenario tests. Each one builds
the detection history of a plausible real-world situation and asserts that the
engine reaches the conclusion a careful human would. If a change to the weights
makes the neighbour look like a stalker, this file fails.

Run:  python3 -m pytest tests/ -v      (or: python3 tests/test_correlate.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.correlate import (  # noqa: E402
    collapse_into_encounters,
    detect_active_tail,
    score_encounters,
)

HOUR = 3600
DAY = 86400
BASE_TS = 1771300000

# A few coordinates far enough apart to be distinct locations.
HOME = (37.7749, -122.4194)
OFFICE = (37.8044, -122.2712)          # ~10 miles from home
GROCERY = (37.7849, -122.4094)         # ~1 mile from home
MIDPOINT = (37.7900, -122.3400)        # between home and office


def detection(ts, location=None, drive_id=None, camera="front"):
    """Build a detection row shaped like the sqlite3.Row the engine expects."""
    lat, lon = location if location else (None, None)
    return {"ts": ts, "lat": lat, "lon": lon, "drive_id": drive_id, "camera": camera}


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def test_dedupe_collapses_burst():
    """Thirty frames of the same car at one red light is ONE encounter."""
    rows = [detection(BASE_TS + n, HOME, "drive-1") for n in range(0, 30)]
    encounters = collapse_into_encounters(rows)

    ok = check("30 rapid detections collapse to 1 encounter", len(encounters) == 1)
    ok &= check("collapsed encounter remembers all 30 detections",
                encounters[0].detection_count == 30)
    return ok


def test_dedupe_separates_across_gap():
    """Sightings more than 60s apart are separate encounters."""
    rows = [
        detection(BASE_TS, HOME, "drive-1"),
        detection(BASE_TS + 30, HOME, "drive-1"),      # same encounter
        detection(BASE_TS + 600, OFFICE, "drive-1"),   # new encounter
    ]
    encounters = collapse_into_encounters(rows)
    return check("60s gap splits encounters", len(encounters) == 2)


def test_encounter_merge_is_order_independent():
    """
    A Tesla records all six camera views of one moment with the SAME timestamp,
    so an event arrives as six rows zero seconds apart. Which of them sorts
    first must not change the verdict.

    This is a regression test for a real bug: collapse_into_encounters used to
    keep only the first row's camera and discard the other five, so "was this
    encounter rear-facing?" was decided by SQL row order. Identical sightings
    scored 35.0 or 20.0 depending purely on whether the 'back' row happened to
    come first - which it did, only because 'back' sorts alphabetically ahead
    of 'front'. Rename that camera to 'rear' and the score would have flipped.
    """
    ALL_SIX = ["back", "front", "left_pillar", "left_repeater", "right_pillar", "right_repeater"]

    def six_camera_event(ts, order):
        return [detection(ts, HOME, "drive-1", camera=c) for c in order]

    rows_back_first = six_camera_event(BASE_TS, ALL_SIX)
    rows_front_first = six_camera_event(BASE_TS, ["front", "left_pillar", "back",
                                                  "right_pillar", "left_repeater", "right_repeater"])
    rows_no_rear = six_camera_event(BASE_TS, ["front", "left_pillar", "right_pillar"])

    back_first = collapse_into_encounters(rows_back_first)
    front_first = collapse_into_encounters(rows_front_first)
    no_rear = collapse_into_encounters(rows_no_rear)

    ok = check("six simultaneous views collapse to one encounter",
               len(back_first) == 1 and len(front_first) == 1)
    ok &= check("all six detections are counted", back_first[0].detection_count == 6)
    ok &= check("rear-facing is detected regardless of row order",
                back_first[0].saw_rear_camera and front_first[0].saw_rear_camera)
    ok &= check("an event with no rear view is not marked rear-facing",
                not no_rear[0].saw_rear_camera)

    # The score itself must be identical across orderings.
    def score_of(rows_at_times):
        rows = []
        for index, ts in enumerate(rows_at_times):
            rows.extend(six_camera_event(ts, ALL_SIX if index % 2 == 0
                                         else ["front", "back", "left_pillar",
                                               "right_pillar", "left_repeater", "right_repeater"]))
        return score_encounters("plate", "p9", "ORDER1", collapse_into_encounters(rows)).score

    times = [BASE_TS + n * 300 for n in range(8)]
    shuffled_score = score_of(times)

    all_back_first = []
    for ts in times:
        all_back_first.extend(six_camera_event(ts, ALL_SIX))
    stable_score = score_encounters(
        "plate", "p9", "ORDER1", collapse_into_encounters(all_back_first)
    ).score

    ok &= check(f"score is stable across camera orderings ({shuffled_score} == {stable_score})",
                shuffled_score == stable_score)
    return ok


def test_neighbour_scores_low():
    """
    The neighbour: parked in the same spot, seen constantly, never anywhere
    else. High detection count, zero threat.
    """
    rows = []
    for day in range(10):
        for n in range(20):
            rows.append(detection(BASE_TS + day * DAY + n * 5, HOME, f"drive-{day}"))

    encounters = collapse_into_encounters(rows)
    result = score_encounters("plate", "p1", "NEIGHBOR1", encounters)

    ok = check(f"neighbour scores low (got {result.score})", result.severity == "low")
    ok &= check("neighbour resolves to a single location", result.distinct_locations == 1)
    ok &= check("engine explains the single-location finding",
                any("one location" in reason for reason in result.reasons))
    return ok


def test_random_traffic_scores_low():
    """A car seen once on one drive is traffic."""
    rows = [detection(BASE_TS + n, MIDPOINT, "drive-1") for n in range(3)]
    encounters = collapse_into_encounters(rows)
    result = score_encounters("plate", "p2", "RANDOM1", encounters)

    ok = check(f"one-off traffic scores low (got {result.score})", result.severity == "low")
    ok &= check("engine says it was seen only once",
                any("only once" in reason for reason in result.reasons))
    return ok


def test_multi_drive_follower_scores_high():
    """
    The real case: same plate on four separate drives, across three days, at
    four different places, mostly behind us.
    """
    rows = [
        detection(BASE_TS, HOME, "drive-1", "back"),
        detection(BASE_TS + HOUR, OFFICE, "drive-1", "left_repeater"),
        detection(BASE_TS + DAY, GROCERY, "drive-2", "back"),
        detection(BASE_TS + DAY + HOUR, OFFICE, "drive-2", "back"),
        detection(BASE_TS + 2 * DAY, MIDPOINT, "drive-3", "right_repeater"),
        detection(BASE_TS + 2 * DAY + HOUR, HOME, "drive-4", "back"),
    ]
    encounters = collapse_into_encounters(rows)
    result = score_encounters("plate", "p3", "FOLLOW1", encounters)

    ok = check(f"multi-drive follower scores high (got {result.score})",
               result.severity == "high")
    ok &= check("counted 4 distinct drives", result.distinct_drives == 4)
    ok &= check("counted 3 distinct days", result.distinct_days == 3)
    ok &= check("noticed the rear-camera bias",
                any("rear-facing" in reason for reason in result.reasons))
    return ok


def test_active_tail_within_single_drive():
    """
    Being followed right now: one drive, four encounters, 10 miles, 90 minutes.
    This must trigger regardless of how the other signals land.
    """
    rows = [
        detection(BASE_TS, HOME, "drive-9", "back"),
        detection(BASE_TS + 1800, MIDPOINT, "drive-9", "back"),
        detection(BASE_TS + 3600, OFFICE, "drive-9", "left_repeater"),
        detection(BASE_TS + 5400, OFFICE, "drive-9", "back"),
    ]
    encounters = collapse_into_encounters(rows)

    ok = check("active tail detected inside the drive",
               detect_active_tail(encounters) == "drive-9")

    result = score_encounters("plate", "p4", "TAIL1", encounters)
    ok &= check(f"active tail scores high (got {result.score})", result.severity == "high")
    ok &= check("FOLLOWED is the first reason given",
                result.reasons and result.reasons[0].startswith("FOLLOWED"))
    return ok


def test_short_hop_is_not_a_tail():
    """Three encounters in one drive but only 4 minutes apart is a traffic light."""
    rows = [
        detection(BASE_TS, HOME, "drive-8"),
        detection(BASE_TS + 120, HOME, "drive-8"),
        detection(BASE_TS + 240, HOME, "drive-8"),
    ]
    encounters = collapse_into_encounters(rows)
    return check("brief co-travel is not flagged as a tail",
                 detect_active_tail(encounters) is None)


def test_score_is_independent_of_row_order():
    """
    A Tesla writes all six cameras with the SAME timestamp, so one encounter
    absorbs up to six detections. Scoring must not depend on which of those
    rows the database happened to return first.

    This was a real bug: the encounter took its `camera` from the first row and
    discarded the other five, so an identical set of sightings scored 35.0 or
    20.0 purely according to SQL row order — and it only looked correct on real
    data because "back" sorts alphabetically ahead of the other five names.
    """
    ALL_SIX = ["back", "front", "left_pillar", "left_repeater", "right_pillar", "right_repeater"]

    def sightings(camera_order):
        rows = []
        for group in range(6):
            ts = BASE_TS + group * 600      # 10 min apart: separate encounters
            for camera in camera_order:
                rows.append(detection(ts, HOME, f"drive-{group}", camera))
        return collapse_into_encounters(rows)

    back_first = score_encounters("plate", "p8", "ORDER1", sightings(ALL_SIX))
    front_first = score_encounters("plate", "p8", "ORDER1", sightings(list(reversed(ALL_SIX))))
    middle_first = score_encounters("plate", "p8", "ORDER1", sightings(ALL_SIX[2:] + ALL_SIX[:2]))

    ok = check(f"score is identical regardless of camera row order "
               f"({back_first.score} / {front_first.score} / {middle_first.score})",
               back_first.score == front_first.score == middle_first.score)
    ok &= check("six cameras at one instant collapse to one encounter per group",
                back_first.encounter_count == 6)
    ok &= check("the rear-facing signal survives whichever row came first",
                any("rear-facing" in r for r in front_first.reasons))
    return ok


def test_rear_signal_requires_an_actual_rear_camera():
    """The aggregate must not fire when no rear camera saw anything."""
    rows = []
    for group in range(4):
        ts = BASE_TS + group * 600
        for camera in ("front", "left_pillar", "right_pillar"):
            rows.append(detection(ts, HOME, f"drive-{group}", camera))

    result = score_encounters("plate", "p9", "FRONTONLY", collapse_into_encounters(rows))
    return check("front/pillar-only sightings earn no rear-facing reason",
                 not any("rear-facing" in r for r in result.reasons))


def test_known_entity_is_suppressed():
    """A whitelisted plate scores zero no matter how damning the history."""
    rows = [
        detection(BASE_TS, HOME, "drive-1", "back"),
        detection(BASE_TS + DAY, OFFICE, "drive-2", "back"),
        detection(BASE_TS + 2 * DAY, GROCERY, "drive-3", "back"),
        detection(BASE_TS + 3 * DAY, MIDPOINT, "drive-4", "back"),
    ]
    encounters = collapse_into_encounters(rows)
    result = score_encounters("plate", "p5", "MYSPOUSE", encounters, is_known=True)

    ok = check("known entity scores zero", result.score == 0.0)
    ok &= check("known entity explains itself",
                any("known" in reason.lower() for reason in result.reasons))
    return ok


def test_dwell_location_suppression():
    """
    A vehicle seen at two spots that are both places YOU habitually park (home
    and office) is background, not a follower - even though it technically has
    multiple locations across multiple drives and days.
    """
    rows = [
        detection(BASE_TS, HOME, "drive-1", "back"),
        detection(BASE_TS + DAY, OFFICE, "drive-2", "back"),
        detection(BASE_TS + 2 * DAY, HOME, "drive-3", "back"),
        detection(BASE_TS + 3 * DAY, OFFICE, "drive-4", "back"),
    ]
    encounters = collapse_into_encounters(rows)

    without_dwell = score_encounters("plate", "p7", "COWORKER", encounters)
    with_dwell = score_encounters(
        "plate", "p7", "COWORKER", encounters, dwell_locations=[HOME, OFFICE]
    )

    ok = check(f"scores high without dwell knowledge (got {without_dwell.score})",
               without_dwell.severity == "high")
    ok &= check(f"dwell knowledge suppresses it (got {with_dwell.score})",
                with_dwell.score < without_dwell.score)
    ok &= check("engine explains the dwell suppression",
                any("regularly parks" in reason for reason in with_dwell.reasons))
    return ok


def test_works_without_gps():
    """
    No Tesla API credentials means no coordinates. The engine must still reach
    a sane conclusion from drives, days and cameras alone.
    """
    rows = [
        detection(BASE_TS, None, "drive-1", "back"),
        detection(BASE_TS + DAY, None, "drive-2", "back"),
        detection(BASE_TS + 2 * DAY, None, "drive-3", "back"),
    ]
    encounters = collapse_into_encounters(rows)
    result = score_encounters("plate", "p6", "NOGPS1", encounters)

    ok = check(f"GPS-less multi-drive still scores (got {result.score})", result.score > 0)
    ok &= check("no phantom locations invented", result.distinct_locations == 0)
    ok &= check("no phantom distance invented", result.max_separation_miles == 0.0)
    return ok


def main():
    tests = [
        ("dedupe: burst collapses", test_dedupe_collapses_burst),
        ("dedupe: gap separates", test_dedupe_separates_across_gap),
        ("dedupe: merge is order-independent", test_encounter_merge_is_order_independent),
        ("neighbour scores low", test_neighbour_scores_low),
        ("random traffic scores low", test_random_traffic_scores_low),
        ("multi-drive follower scores high", test_multi_drive_follower_scores_high),
        ("active tail detected", test_active_tail_within_single_drive),
        ("short hop is not a tail", test_short_hop_is_not_a_tail),
        ("score independent of row order", test_score_is_independent_of_row_order),
        ("rear signal needs a rear camera", test_rear_signal_requires_an_actual_rear_camera),
        ("known entity suppressed", test_known_entity_is_suppressed),
        ("dwell locations suppress background", test_dwell_location_suppression),
        ("works without GPS", test_works_without_gps),
    ]

    failures = 0
    for name, test_function in tests:
        print(f"\n{name}:")
        if not test_function():
            failures += 1

    print(f"\n{'=' * 60}")
    print(f"{len(tests) - failures}/{len(tests)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
