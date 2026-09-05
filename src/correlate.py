"""
correlate.py

The surveillance detection engine - the part that actually answers Scout's
question: "is this vehicle or person following me?"

Scout's shipped implementation of this was thin. Its `/plates/alldetectionsdd`
route deduplicated detections (same plate text, more than 60 seconds apart) and
handed the list to a Vue timeline for a human to eyeball. The judgement was
left to you.

We keep the 60-second dedupe - it is the right primitive - and then actually
score the result, because the signal that separates a tail from traffic is not
"how many times did I see this plate" but *how those sightings are distributed
over space, time and separate journeys*:

    - A plate seen 40 times in your driveway is your neighbour.       (0 points)
    - A plate seen 3 times across 3 different drives on 2 days is a
      vehicle that keeps appearing wherever you go.                   (alarming)
    - A plate seen 4 times within one drive, 6 miles apart, always in
      the rear-facing cameras, is behind you right now.               (alarming)

So we compute five independent signals per entity and combine them. Each one
contributes a bounded number of points and a human-readable reason, so the
dashboard can explain *why* something scored, rather than showing a number
nobody trusts.

Everything degrades without GPS. If you haven't connected the Tesla API (see
poller.py) there are no coordinates on detections, so the geographic signals
return zero and scoring leans on the drive/day/camera signals instead. That is
weaker but still useful, and it means this runs on day one with no credentials.
"""
import json
import uuid

from dataclasses import dataclass, field
from datetime import datetime

from src.common import env_float, env_int
from src.clipmeta import camera_faces_rearward
from src.geo import cluster_locations, haversine_miles, spatial_spread_miles
from src.db import now_ts, upsert

# --- Dedupe -----------------------------------------------------------------
# Scout's rule, kept verbatim: two detections of the same entity less than this
# far apart in time are the same encounter, not two independent sightings.
ENCOUNTER_GAP_SECONDS = env_int("ENCOUNTER_GAP_SECONDS", 60)

# --- Score weights ----------------------------------------------------------
# Each signal is capped so no single one can max out the score alone; a real
# alert needs corroboration from more than one axis.
POINTS_PER_EXTRA_DRIVE = env_float("POINTS_PER_EXTRA_DRIVE", 18.0)
MAX_DRIVE_POINTS = 45.0

POINTS_PER_EXTRA_DAY = env_float("POINTS_PER_EXTRA_DAY", 12.0)
MAX_DAY_POINTS = 30.0

POINTS_PER_EXTRA_LOCATION = env_float("POINTS_PER_EXTRA_LOCATION", 10.0)
MAX_LOCATION_POINTS = 30.0

MAX_SPREAD_POINTS = 25.0
POINTS_PER_ENCOUNTER = env_float("POINTS_PER_ENCOUNTER", 4.0)
MAX_ENCOUNTER_POINTS = 20.0
MAX_REAR_CAMERA_POINTS = 15.0

# Distance beyond which two sightings of one entity can't be coincidence in
# ordinary traffic. Below it we scale points linearly.
SIGNIFICANT_SPREAD_MILES = env_float("SIGNIFICANT_SPREAD_MILES", 5.0)

# --- Suppression --------------------------------------------------------
# The single most important noise control after the whitelist. An entity whose
# every sighting falls in one place is geographically ANCHORED - a neighbour's
# parked car, a shop sign, a doorman. It cannot be following you, because it
# never went anywhere. We multiply its score down rather than zeroing it, so a
# genuinely odd anchored entity still surfaces below the real findings.
#
# Only applied when we actually have GPS. With no coordinates we cannot tell
# "always in one place" from "we don't know where", and guessing would silence
# real findings for every user who hasn't connected the Tesla API.
ANCHORED_SCORE_MULTIPLIER = env_float("ANCHORED_SCORE_MULTIPLIER", 0.15)

# Weaker penalty for an entity seen only where *you* habitually park. Those are
# your home, office and gym - the places your own background is densest.
DWELL_SCORE_MULTIPLIER = env_float("DWELL_SCORE_MULTIPLIER", 0.35)
DWELL_RADIUS_MILES = env_float("DWELL_RADIUS_MILES", 0.20)
MIN_POLLS_FOR_DWELL = env_int("MIN_POLLS_FOR_DWELL", 20)

# An entity needs at least this many *encounters* before we score it at all.
# One sighting is not a pattern, and scoring singletons floods the dashboard.
MIN_ENCOUNTERS_TO_SCORE = env_int("MIN_ENCOUNTERS_TO_SCORE", 2)

# Severity bands for the 0-100 score.
HIGH_SEVERITY_SCORE = env_float("HIGH_SEVERITY_SCORE", 65.0)
MEDIUM_SEVERITY_SCORE = env_float("MEDIUM_SEVERITY_SCORE", 35.0)

# --- Active-tail detection (the real-time case) -----------------------------
# Within a single drive: this many separated encounters, spanning at least this
# far, means something stayed with you across the journey.
TAIL_MIN_ENCOUNTERS_IN_DRIVE = env_int("TAIL_MIN_ENCOUNTERS_IN_DRIVE", 3)
TAIL_MIN_SPREAD_MILES = env_float("TAIL_MIN_SPREAD_MILES", 2.0)
TAIL_MIN_DURATION_SECONDS = env_int("TAIL_MIN_DURATION_SECONDS", 300)


@dataclass
class Encounter:
    """A deduplicated sighting: one entity, one place, one moment."""
    ts: int
    lat: float | None
    lon: float | None
    drive_id: str | None
    camera: str | None
    detection_count: int = 1
    # True if ANY detection folded into this encounter came from a rear-facing
    # camera. This must be an aggregate, not a property of `camera`: a Tesla
    # writes all six cameras with the SAME timestamp, so a single encounter
    # routinely absorbs six detections and `camera` is merely whichever row the
    # database happened to return first. Scoring off `camera` made the result
    # depend on SQL row order - measured, the identical sighting scored 35.0 or
    # 20.0 purely according to whether the 'back' clip's row sorted first,
    # which it does only because "back" happens to precede "front" and
    # "left_*"/"right_*" alphabetically. Aggregating removes that accident.
    saw_rear_camera: bool = False
    cameras: set = field(default_factory=set)
    # True if ANY detection folded into this encounter came from a rear-facing
    # camera. Kept separately from `camera` because `camera` can only hold one
    # value while a Tesla records all six views of the same moment - see



@dataclass
class CorrelationResult:
    """The scored verdict for one entity."""
    entity_type: str
    entity_id: str
    entity_label: str
    score: float
    severity: str
    reasons: list[str] = field(default_factory=list)
    distinct_drives: int = 0
    distinct_days: int = 0
    distinct_locations: int = 0
    span_seconds: int = 0
    max_separation_miles: float = 0.0
    detection_count: int = 0
    encounter_count: int = 0
    active_tail_drive_id: str | None = None


def collapse_into_encounters(detection_rows: list) -> list[Encounter]:
    """
    Apply Scout's 60-second dedupe rule to a time-ordered list of detections.

    Detections arrive one per analysed video frame, so a car sitting behind you
    at a red light can generate dozens. Collapsing them into encounters is what
    makes every count downstream mean "separate occasion" instead of "number of
    frames the camera happened to catch it in".

    Merging must be ORDER-INDEPENDENT. A Tesla records all six camera views of
    the same moment and stamps them with the same clip timestamp, so one event
    arrives here as six rows that are zero seconds apart. An earlier version of
    this function kept only the first row's `camera` and discarded the other
    five, which made "was this encounter rear-facing?" a function of SQL row
    order rather than of where the vehicle actually was: identical sightings
    scored 35.0 or 20.0 depending purely on which camera's row sorted first,
    and 'back' only won because it sorts alphabetically ahead of 'front'.

    So we accumulate across every merged detection instead of overwriting:
    `saw_rear_camera` is true if ANY view of the moment was rear-facing, which
    is the question the threat score actually wants answered.
    """
    encounters: list[Encounter] = []

    for row in sorted(detection_rows, key=lambda r: r["ts"] or 0):
        timestamp = int(row["ts"] or 0)
        drive_id = row["drive_id"] if "drive_id" in row.keys() else None
        camera = row["camera"] if "camera" in row.keys() else None

        if encounters:
            previous = encounters[-1]
            same_drive = previous.drive_id == drive_id
            within_gap = (timestamp - previous.ts) <= ENCOUNTER_GAP_SECONDS
            if within_gap and same_drive:
                previous.detection_count += 1
                if camera:
                    previous.cameras.add(camera)
                # Rear-facing is sticky: one rear view of the moment is enough.
                previous.saw_rear_camera = (
                    previous.saw_rear_camera or camera_faces_rearward(camera)
                )
                # Prefer a rear-facing camera as the encounter's display camera,
                # since that is the one that matters when reviewing a tail.
                if camera_faces_rearward(camera) and not camera_faces_rearward(previous.camera):
                    previous.camera = camera
                # Keep the last known position for the encounter; a GPS fix
                # later in the group is usually better than the first one.
                if row["lat"] is not None:
                    previous.lat, previous.lon = row["lat"], row["lon"]
                continue

        encounters.append(Encounter(
            ts=timestamp,
            lat=row["lat"],
            lon=row["lon"],
            drive_id=drive_id,
            camera=camera,
            saw_rear_camera=camera_faces_rearward(camera),
            cameras={camera} if camera else set(),
        ))

    return encounters


def detect_active_tail(encounters: list[Encounter]) -> str | None:
    """
    Look for a single drive during which this entity stayed with us.

    This is the alert Scout's README promised ("tell you if you're being
    followed in real-time"). The test is deliberately strict, because a false
    positive here is the kind that makes someone stop trusting the tool:

        >= 3 separate encounters within one drive
        AND those encounters spread over >= 2 miles
        AND >= 5 minutes between first and last

    Something that satisfies all three did not merely share a road with you.

    Returns the drive_id of the worst offending drive, or None.
    """
    by_drive: dict[str, list[Encounter]] = {}
    for encounter in encounters:
        if not encounter.drive_id:
            continue
        by_drive.setdefault(encounter.drive_id, []).append(encounter)

    for drive_id, drive_encounters in by_drive.items():
        if len(drive_encounters) < TAIL_MIN_ENCOUNTERS_IN_DRIVE:
            continue

        duration = max(e.ts for e in drive_encounters) - min(e.ts for e in drive_encounters)
        if duration < TAIL_MIN_DURATION_SECONDS:
            continue

        points = [(e.lat, e.lon) for e in drive_encounters if e.lat is not None and e.lon is not None]
        if len(points) >= 2 and spatial_spread_miles(points) >= TAIL_MIN_SPREAD_MILES:
            return drive_id

        # Without GPS we can't measure spread, but a long multi-encounter
        # presence inside one drive is still worth flagging on its own.
        if not points and duration >= TAIL_MIN_DURATION_SECONDS * 2:
            return drive_id

    return None


def score_encounters(
    entity_type: str,
    entity_id: str,
    entity_label: str,
    encounters: list[Encounter],
    is_known: bool = False,
    dwell_locations: list[tuple[float, float]] | None = None,
) -> CorrelationResult:
    """
    Turn a list of encounters into a 0-100 threat score plus an explanation.

    The score is the sum of five bounded signals. Read the reasons, not the
    number - the number exists to sort the dashboard.
    """
    result = CorrelationResult(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_label=entity_label,
        score=0.0,
        severity="low",
        detection_count=sum(e.detection_count for e in encounters),
        encounter_count=len(encounters),
    )

    if not encounters:
        return result

    # A whitelisted entity (your own car, family, coworkers) never scores. This
    # is the single most important noise control in the system - without it the
    # top of your dashboard is permanently occupied by your own household.
    if is_known:
        result.reasons.append("Marked as known - excluded from scoring")
        return result

    timestamps = [e.ts for e in encounters]
    result.span_seconds = max(timestamps) - min(timestamps)

    drive_ids = {e.drive_id for e in encounters if e.drive_id}
    result.distinct_drives = len(drive_ids)

    days = {datetime.fromtimestamp(ts).date() for ts in timestamps if ts}
    result.distinct_days = len(days)

    points = [(e.lat, e.lon) for e in encounters if e.lat is not None and e.lon is not None]
    result.distinct_locations = cluster_locations(points) if points else 0
    result.max_separation_miles = spatial_spread_miles(points) if len(points) >= 2 else 0.0

    if len(encounters) < MIN_ENCOUNTERS_TO_SCORE:
        result.reasons.append("Seen only once - not a pattern")
        return result

    score = 0.0

    # -- Signal 1: separate journeys -------------------------------------
    # The strongest signal we have. Two independent trips is a coincidence you
    # can explain; three is not.
    if result.distinct_drives >= 2:
        drive_points = min(MAX_DRIVE_POINTS, (result.distinct_drives - 1) * POINTS_PER_EXTRA_DRIVE)
        score += drive_points
        result.reasons.append(
            f"Seen on {result.distinct_drives} separate drives"
        )

    # -- Signal 2: separate days -----------------------------------------
    if result.distinct_days >= 2:
        day_points = min(MAX_DAY_POINTS, (result.distinct_days - 1) * POINTS_PER_EXTRA_DAY)
        score += day_points
        result.reasons.append(f"Appeared on {result.distinct_days} different days")

    # -- Signal 3: distinct places ---------------------------------------
    # This is what tells a follower apart from a neighbour: the neighbour is
    # always in the same place.
    if result.distinct_locations >= 2:
        location_points = min(
            MAX_LOCATION_POINTS, (result.distinct_locations - 1) * POINTS_PER_EXTRA_LOCATION
        )
        score += location_points
        result.reasons.append(f"Seen at {result.distinct_locations} distinct locations")
    elif result.distinct_locations == 1 and len(encounters) > 3:
        result.reasons.append(
            "All sightings at one location - likely a neighbour or regular parking"
        )

    # -- Signal 4: geographic spread --------------------------------------
    if result.max_separation_miles > 0.25:
        spread_points = min(
            MAX_SPREAD_POINTS,
            MAX_SPREAD_POINTS * (result.max_separation_miles / SIGNIFICANT_SPREAD_MILES),
        )
        score += spread_points
        result.reasons.append(
            f"Sightings span {result.max_separation_miles:.1f} miles"
        )

    # -- Signal 5: how often, and from which cameras ----------------------
    encounter_points = min(
        MAX_ENCOUNTER_POINTS, (len(encounters) - 1) * POINTS_PER_ENCOUNTER
    )
    score += encounter_points

    # Use the accumulated flag, not e.camera: one encounter can span all six
    # camera views and only the flag survives that merge intact.
    rear_encounters = sum(1 for e in encounters if e.saw_rear_camera)
    if rear_encounters >= 2:
        rear_ratio = rear_encounters / len(encounters)
        rear_points = MAX_REAR_CAMERA_POINTS * rear_ratio
        score += rear_points
        result.reasons.append(
            f"{rear_encounters} of {len(encounters)} sightings were in rear-facing cameras"
        )

    # -- Suppression: is this thing actually going anywhere? --------------
    # Applied as a multiplier on everything above, because an anchored entity
    # is not a weaker follower - it is a different kind of thing entirely.
    if points:
        if result.distinct_locations <= 1:
            score *= ANCHORED_SCORE_MULTIPLIER
            result.reasons.append(
                "Anchored to one location - cannot be following the vehicle"
            )
        elif dwell_locations and _all_within_dwell_locations(points, dwell_locations):
            score *= DWELL_SCORE_MULTIPLIER
            result.reasons.append(
                "Only ever seen where this vehicle regularly parks - likely background"
            )

    # -- Active tail overrides everything ---------------------------------
    tail_drive_id = detect_active_tail(encounters)
    if tail_drive_id:
        result.active_tail_drive_id = tail_drive_id
        score = max(score, HIGH_SEVERITY_SCORE + 10.0)
        result.reasons.insert(
            0, "FOLLOWED: stayed with the vehicle across a single journey"
        )

    result.score = round(min(100.0, score), 1)
    result.severity = (
        "high" if result.score >= HIGH_SEVERITY_SCORE
        else "medium" if result.score >= MEDIUM_SEVERITY_SCORE
        else "low"
    )

    if not result.reasons:
        result.reasons.append("Repeated sightings with no distinguishing pattern")

    return result


def _all_within_dwell_locations(
    points: list[tuple[float, float]],
    dwell_locations: list[tuple[float, float]],
) -> bool:
    """True when every sighting sits near a place the vehicle habitually parks."""
    return all(
        any(haversine_miles(lat, lon, d_lat, d_lon) <= DWELL_RADIUS_MILES
            for d_lat, d_lon in dwell_locations)
        for lat, lon in points
    )


def compute_dwell_locations(connection) -> list[tuple[float, float]]:
    """
    Work out where this vehicle habitually parks, from its own telemetry.

    We take every parked poll, cluster them, and keep the clusters with enough
    polls to represent real dwell time. These come out as your home, your
    office, your gym - exactly the places where innocent vehicles and faces
    will accumulate sightings simply because you are there a lot.

    Returns [] when there is no poll data, which disables the dwell penalty
    rather than guessing.
    """
    parked_polls = connection.execute(
        "SELECT lat, lon FROM polls WHERE status='P' AND lat IS NOT NULL AND lon IS NOT NULL"
    ).fetchall()

    if len(parked_polls) < MIN_POLLS_FOR_DWELL:
        return []

    # Greedy clustering with a poll count per centre, so we can drop the
    # one-off stops (a red light, a drive-through) and keep real destinations.
    centres: list[list] = []   # [lat, lon, count]
    for row in parked_polls:
        lat, lon = row["lat"], row["lon"]
        for centre in centres:
            if haversine_miles(lat, lon, centre[0], centre[1]) <= DWELL_RADIUS_MILES:
                centre[2] += 1
                break
        else:
            centres.append([lat, lon, 1])

    return [(lat, lon) for lat, lon, count in centres if count >= MIN_POLLS_FOR_DWELL]


# ---------------------------------------------------------------------------
# Database-facing entry points
# ---------------------------------------------------------------------------

def score_plate(connection, plate_row, dwell_locations=None) -> CorrelationResult:
    """Score one plate identity from its detection history."""
    detections = connection.execute(
        "SELECT ts, lat, lon, drive_id, camera FROM plate_detections "
        "WHERE plate_id=? ORDER BY ts",
        (plate_row["id"],),
    ).fetchall()

    encounters = collapse_into_encounters(detections)
    return score_encounters(
        entity_type="plate",
        entity_id=plate_row["id"],
        entity_label=plate_row["label"] or plate_row["plate_text"],
        encounters=encounters,
        is_known=bool(plate_row["is_known"]),
        dwell_locations=dwell_locations,
    )


def score_face(connection, face_row, dwell_locations=None) -> CorrelationResult:
    """Score one face identity from its detection history."""
    detections = connection.execute(
        "SELECT ts, lat, lon, drive_id, camera FROM face_detections "
        "WHERE face_id=? ORDER BY ts",
        (face_row["id"],),
    ).fetchall()

    encounters = collapse_into_encounters(detections)
    return score_encounters(
        entity_type="face",
        entity_id=face_row["id"],
        entity_label=face_row["label"] or face_row["person_name"],
        encounters=encounters,
        is_known=bool(face_row["is_known"]),
        dwell_locations=dwell_locations,
    )


def persist_result(connection, result: CorrelationResult) -> bool:
    """
    Write a score into the `correlations` table.

    Returns True when this is a *newly* medium-or-worse finding, which is the
    signal notify.py uses to decide whether to wake you up. We deliberately
    return False for a finding that merely got worse - you were already told.
    """
    # One read covers both questions: is there already a row (and what is its
    # id), and have we already told the user about it.
    existing = connection.execute(
        "SELECT id, score, severity, notified FROM correlations "
        "WHERE entity_type=? AND entity_id=?",
        (result.entity_type, result.entity_id),
    ).fetchone()

    # "New" means newly worth telling someone about. A finding that was already
    # notified and merely got worse does not page you again.
    is_new_finding = (
        result.severity in ("medium", "high")
        and (existing is None or not existing["notified"])
    )

    row_id = existing["id"] if existing is not None else uuid.uuid4().hex[:12]

    connection.execute(
        """
        INSERT INTO correlations (
            id, entity_type, entity_id, entity_label, score, severity, reasons,
            distinct_drives, distinct_days, distinct_locations, span_seconds,
            max_separation_mi, detection_count, created_ts, notified, acknowledged
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                  COALESCE((SELECT notified FROM correlations WHERE entity_type=? AND entity_id=?), 0),
                  COALESCE((SELECT acknowledged FROM correlations WHERE entity_type=? AND entity_id=?), 0))
        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
            score=excluded.score,
            severity=excluded.severity,
            reasons=excluded.reasons,
            entity_label=excluded.entity_label,
            distinct_drives=excluded.distinct_drives,
            distinct_days=excluded.distinct_days,
            distinct_locations=excluded.distinct_locations,
            span_seconds=excluded.span_seconds,
            max_separation_mi=excluded.max_separation_mi,
            detection_count=excluded.detection_count,
            created_ts=excluded.created_ts
        """,
        (
            row_id, result.entity_type, result.entity_id, result.entity_label,
            result.score, result.severity, json.dumps(result.reasons),
            result.distinct_drives, result.distinct_days, result.distinct_locations,
            result.span_seconds, result.max_separation_miles, result.detection_count,
            now_ts(),
            result.entity_type, result.entity_id,
            result.entity_type, result.entity_id,
        ),
    )

    # Keep the entity's own denormalised score in sync so list views can sort
    # without joining.
    table = "plates" if result.entity_type == "plate" else "faces"
    connection.execute(
        f"UPDATE {table} SET threat_score=? WHERE id=?", (result.score, result.entity_id)
    )

    return is_new_finding


def run_correlation_pass(connection, verbose: bool = False) -> list[CorrelationResult]:
    """
    Re-score every entity and persist the results.

    Cheap enough to run after each clip: the work is proportional to total
    detections, and the 60-second dedupe keeps that number small even after
    months of driving.

    Returns the list of newly-raised medium/high findings, for notification.
    """
    newly_raised: list[CorrelationResult] = []

    # Computed once per pass, not per entity - it scans the whole poll table.
    dwell_locations = compute_dwell_locations(connection)

    for plate_row in connection.execute("SELECT * FROM plates").fetchall():
        result = score_plate(connection, plate_row, dwell_locations)
        if persist_result(connection, result):
            newly_raised.append(result)
        if verbose and result.score > 0:
            print(f"[🧭 correlate] plate {result.entity_label}: {result.score} ({result.severity})")

    for face_row in connection.execute("SELECT * FROM faces").fetchall():
        result = score_face(connection, face_row, dwell_locations)
        if persist_result(connection, result):
            newly_raised.append(result)
        if verbose and result.score > 0:
            print(f"[🧭 correlate] face {result.entity_label}: {result.score} ({result.severity})")

    connection.commit()
    return newly_raised
