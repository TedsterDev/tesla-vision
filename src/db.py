"""
db.py

SQLite persistence layer. This is the port of Scout's MongoDB data model.

Why SQLite instead of MongoDB?
    - Scout ran Mongo on a Jetson Xavier (32GB). We're on an Orin Nano (7.4GB),
      and a Mongo daemon would eat RAM we'd rather give to inference.
    - Every query Scout actually performs is a filter + sort + join. SQL does
      that better, and SQLite gives us real indexes for free.
    - One file on disk means backups are `cp`, and there's no service to babysit
      in the car when power drops out mid-write (WAL handles that).

The collections from Scout map to tables as follows:

    Scout (MongoDB)      ->  here (SQLite)
    ------------------------------------------
    polls                ->  polls
    drives               ->  drives
    geocodes             ->  geocodes
    plates               ->  plates
    plateDetections      ->  plate_detections
    faces                ->  faces
    faceDetections       ->  face_detections
    (none)               ->  clips           - one row per ingested Tesla video
    (none)               ->  correlations    - output of the threat scorer

The "entity vs detection" split is the important idea we keep from Scout:
a *plate* / *face* is a persistent identity, and a *detection* is one sighting
of that identity at a place and time. Surveillance detection is entirely a
question of how a single entity's detections are distributed over space+time.
"""
import json
import sqlite3
import time

from pathlib import Path
from typing import Any, Iterable

from src.common import DB_PATH, ensure_dirs

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Notes on conventions used throughout:
#   *_ts columns are INTEGER unix epoch seconds (UTC), matching Scout's unixTS.
#   lat/lon are REAL degrees, NULL when we had no GPS fix for that moment.
#   embeddings are stored as raw float32 bytes (BLOB) - see faces.py.

SCHEMA_STATEMENTS = [
    # -- Video clips pulled off the TeslaCam drive -------------------------
    """
    CREATE TABLE IF NOT EXISTS clips (
        id              TEXT PRIMARY KEY,
        filename        TEXT NOT NULL UNIQUE,
        camera          TEXT,           -- front | back | left_repeater | ...
        captured_ts     INTEGER,        -- parsed from the Tesla filename
        clip_source     TEXT,           -- RecentClips | SentryClips | SavedClips
        ingested_ts     INTEGER NOT NULL,
        processed_ts    INTEGER,
        status          TEXT NOT NULL DEFAULT 'ingested',
        lat             REAL,
        lon             REAL,
        drive_id        TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clips_captured ON clips(captured_ts)",
    "CREATE INDEX IF NOT EXISTS idx_clips_status   ON clips(status)",

    # -- Alerts (mirrors the JSON files the dashboard already reads) -------
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id              TEXT PRIMARY KEY,
        timestamp       INTEGER NOT NULL,
        source_file     TEXT,
        clip_id         TEXT,
        camera          TEXT,
        score           REAL,
        hit_count       INTEGER DEFAULT 0,
        person_count    INTEGER DEFAULT 0,
        vehicle_count   INTEGER DEFAULT 0,
        jpeg            TEXT,
        gif             TEXT,
        status          TEXT,
        lat             REAL,
        lon             REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp DESC)",

    # -- Plate identities --------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS plates (
        id              TEXT PRIMARY KEY,
        plate_text      TEXT NOT NULL UNIQUE,
        label           TEXT,           -- user-supplied name ("mom's car")
        is_known        INTEGER NOT NULL DEFAULT 0,   -- 1 = whitelisted, never alert
        first_seen_ts   INTEGER NOT NULL,
        last_seen_ts    INTEGER NOT NULL,
        detection_count INTEGER NOT NULL DEFAULT 0,
        threat_score    REAL NOT NULL DEFAULT 0.0,
        best_image      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plates_score ON plates(threat_score DESC)",
    "CREATE INDEX IF NOT EXISTS idx_plates_seen  ON plates(last_seen_ts DESC)",

    # -- Individual plate sightings ---------------------------------------
    """
    CREATE TABLE IF NOT EXISTS plate_detections (
        id              TEXT PRIMARY KEY,
        plate_id        TEXT NOT NULL,
        plate_text      TEXT NOT NULL,
        ts              INTEGER NOT NULL,
        clip_id         TEXT,
        source_file     TEXT,
        camera          TEXT,
        frame_number    INTEGER,
        det_confidence  REAL,           -- how sure we were it *is* a plate
        ocr_confidence  REAL,           -- how sure we were of the characters
        image           TEXT,           -- cropped plate jpeg in MEDIA_DIR
        lat             REAL,
        lon             REAL,
        drive_id        TEXT,
        FOREIGN KEY (plate_id) REFERENCES plates(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pdet_plate ON plate_detections(plate_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_pdet_ts    ON plate_detections(ts DESC)",

    # -- Face identities ---------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS faces (
        id              TEXT PRIMARY KEY,
        person_name     TEXT NOT NULL,  -- "Stranger #7" until the user renames it
        label           TEXT,
        is_known        INTEGER NOT NULL DEFAULT 0,
        embedding       BLOB,           -- running mean embedding (float32)
        embedding_count INTEGER NOT NULL DEFAULT 0,
        first_seen_ts   INTEGER NOT NULL,
        last_seen_ts    INTEGER NOT NULL,
        detection_count INTEGER NOT NULL DEFAULT 0,
        threat_score    REAL NOT NULL DEFAULT 0.0,
        best_image      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_faces_score ON faces(threat_score DESC)",
    "CREATE INDEX IF NOT EXISTS idx_faces_seen  ON faces(last_seen_ts DESC)",

    # -- Individual face sightings ----------------------------------------
    """
    CREATE TABLE IF NOT EXISTS face_detections (
        id              TEXT PRIMARY KEY,
        face_id         TEXT NOT NULL,
        ts              INTEGER NOT NULL,
        clip_id         TEXT,
        source_file     TEXT,
        camera          TEXT,
        frame_number    INTEGER,
        det_confidence  REAL,
        match_distance  REAL,           -- cosine similarity to the identity
        embedding       BLOB,
        image           TEXT,
        lat             REAL,
        lon             REAL,
        drive_id        TEXT,
        FOREIGN KEY (face_id) REFERENCES faces(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_fdet_face ON face_detections(face_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_fdet_ts   ON face_detections(ts DESC)",

    # -- Vehicle telemetry (Scout's `polls`) -------------------------------
    """
    CREATE TABLE IF NOT EXISTS polls (
        id              TEXT PRIMARY KEY,
        ts              INTEGER NOT NULL,
        lat             REAL,
        lon             REAL,
        heading         REAL,
        speed           REAL,           -- mph, as Tesla reports it
        power           REAL,
        shift_state     TEXT,
        status          TEXT,           -- 'D' driving | 'P' parked | 'C' charging
        loc_available   INTEGER NOT NULL DEFAULT 0,
        odometer        REAL,
        street          TEXT,
        city            TEXT,
        drive_id        TEXT,
        geocode_id      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_polls_ts    ON polls(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_polls_drive ON polls(drive_id, ts)",
    # Find My asks one question - "the newest row that actually has a position"
    # - and idx_polls_ts cannot answer it, because the Tesla path writes
    # lat-NULL rows with ts=now_ts() whenever drive_state is missing, so the
    # newest row is often not the newest *located* row. This partial index
    # holds only the located rows, which turns that lookup from a scan that
    # walks past the NULLs into a single seek (10.81ms -> 0.56ms at 200k rows).
    # SQLite only uses a partial index when the query's WHERE clause matches
    # the index's predicate TEXTUALLY: writing the semantically identical
    # `WHERE loc_available=1` instead of `WHERE lat IS NOT NULL` silently falls
    # back to the full index walk. No error, no warning, just the 19x back.
    "CREATE INDEX IF NOT EXISTS idx_polls_fix ON polls(ts DESC) WHERE lat IS NOT NULL",

    # -- Drives (a contiguous run of driving polls) ------------------------
    """
    CREATE TABLE IF NOT EXISTS drives (
        id              TEXT PRIMARY KEY,
        start_ts        INTEGER NOT NULL,
        start_lat       REAL,
        start_lon       REAL,
        start_heading   REAL,
        end_ts          INTEGER,
        end_lat         REAL,
        end_lon         REAL,
        end_heading     REAL,
        distance_miles  REAL DEFAULT 0.0,
        poll_count      INTEGER DEFAULT 0,
        is_open         INTEGER NOT NULL DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_drives_start ON drives(start_ts DESC)",

    # -- Reverse geocode cache (Scout's `geocodes`) ------------------------
    # Keyed by rounded lat/lon so we hit Nominatim once per ~10m of road.
    """
    CREATE TABLE IF NOT EXISTS geocodes (
        id              TEXT PRIMARY KEY,
        cache_key       TEXT NOT NULL UNIQUE,
        lat             REAL,
        lon             REAL,
        display_name    TEXT,
        house_number    TEXT,
        road            TEXT,
        suburb          TEXT,
        city            TEXT,
        county          TEXT,
        state           TEXT,
        postcode        TEXT,
        country         TEXT,
        country_code    TEXT,
        fetched_ts      INTEGER NOT NULL
    )
    """,

    # -- Surveillance detection output ------------------------------------
    """
    CREATE TABLE IF NOT EXISTS correlations (
        id                  TEXT PRIMARY KEY,
        entity_type         TEXT NOT NULL,      -- 'plate' | 'face'
        entity_id           TEXT NOT NULL,
        entity_label        TEXT,
        score               REAL NOT NULL,
        severity            TEXT NOT NULL,      -- 'low' | 'medium' | 'high'
        reasons             TEXT,               -- JSON array of human strings
        distinct_drives     INTEGER DEFAULT 0,
        distinct_days       INTEGER DEFAULT 0,
        distinct_locations  INTEGER DEFAULT 0,
        span_seconds        INTEGER DEFAULT 0,
        max_separation_mi   REAL DEFAULT 0.0,
        detection_count     INTEGER DEFAULT 0,
        created_ts          INTEGER NOT NULL,
        notified            INTEGER NOT NULL DEFAULT 0,
        acknowledged        INTEGER NOT NULL DEFAULT 0,
        UNIQUE(entity_type, entity_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corr_score ON correlations(score DESC)",

    # -- NATIX VX360 mirror bookkeeping -----------------------------------
    # The VX360 is a 128-256GB USB stick with its own ARM SoC that uploads
    # TeslaCam footage to NATIX's cloud over WiFi. In our topology the car no
    # longer writes to it directly - the car writes to the Jetson - so the
    # Jetson has to put the footage on it instead. These two tables are the
    # bookkeeping for that copy: which sticks we have seen, and which clip
    # landed on which stick.
    """
    CREATE TABLE IF NOT EXISTS natix_devices (
        id              TEXT PRIMARY KEY,   -- stable key: serial, else uuid, else label
        serial          TEXT,
        volume_uuid     TEXT,
        label           TEXT,
        vendor          TEXT,
        model           TEXT,
        usb_vendor_id   TEXT,
        usb_product_id  TEXT,
        size_bytes      INTEGER,
        fstype          TEXT,
        confidence      TEXT,               -- pinned | strong | likely
        first_seen_ts   INTEGER NOT NULL,
        last_seen_ts    INTEGER NOT NULL,
        last_mount      TEXT,
        free_bytes      INTEGER,
        mirrored_count  INTEGER NOT NULL DEFAULT 0,
        mirrored_bytes  INTEGER NOT NULL DEFAULT 0,
        is_approved     INTEGER NOT NULL DEFAULT 0,  -- 1 = user confirmed this stick
        note            TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_natixdev_seen ON natix_devices(last_seen_ts DESC)",

    """
    CREATE TABLE IF NOT EXISTS natix_mirror (
        id              TEXT PRIMARY KEY,   -- device_id + ':' + clip_id
        device_id       TEXT NOT NULL,
        clip_id         TEXT NOT NULL,
        filename        TEXT NOT NULL,
        event_folder    TEXT,               -- TeslaCam/SentryClips/2026-02-16_20-49-20
        bucket          TEXT,               -- SentryClips | SavedClips | RecentClips
        dest_path       TEXT,               -- path relative to the stick's root
        size_bytes      INTEGER,
        source_mtime    INTEGER,
        state           TEXT NOT NULL,      -- done | missing | failed | pruned
        error           TEXT,
        copied_ts       INTEGER,
        UNIQUE(device_id, clip_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_natixmir_dev   ON natix_mirror(device_id, state)",
    "CREATE INDEX IF NOT EXISTS idx_natixmir_event ON natix_mirror(device_id, event_folder)",

    # -- Small key/value store for runtime settings ------------------------
    """
    CREATE TABLE IF NOT EXISTS settings (
        key             TEXT PRIMARY KEY,
        value           TEXT
    )
    """,
]


# Schema creation is idempotent but NOT free: `CREATE TABLE IF NOT EXISTS`
# still takes a brief write lock, and running 25 of them on every connection
# means every dashboard page load contends with the processor for the write
# lock. We do it once per process instead, tracked here.
_schema_initialized_for: set[str] = set()


def connect(ensure_schema: bool = True, timeout_seconds: float = 30.0) -> sqlite3.Connection:
    """
    Open a connection to the Scout database, creating the schema if needed.

    Safe to call per request as well as per process: the schema DDL only runs
    the first time this process touches a given database file.

    SQLite is happy with several processes writing to one file as long as WAL
    is on and we tolerate short lock waits - which is exactly our topology
    (processor, gif_worker, poller and web all share one file).
    """
    ensure_dirs()

    # `timeout` becomes SQLite's busy timeout: how long a statement waits for
    # another process's write lock before giving up. Background writers want a
    # generous value; a web request wants a short one, because a request that
    # blocks for 30 seconds is indistinguishable from a hung dashboard. Callers
    # that serve users should pass a few seconds and let write_with_retry
    # handle the rest.
    connection = sqlite3.connect(str(DB_PATH), timeout=timeout_seconds)
    connection.row_factory = sqlite3.Row

    # WAL lets readers (the dashboard) keep working while a writer (the
    # processor) commits. NORMAL sync is the right trade for a device that can
    # lose power: we may lose the last transaction, never the database.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")

    database_key = str(DB_PATH)
    if ensure_schema and database_key not in _schema_initialized_for:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.commit()
        _schema_initialized_for.add(database_key)

    return connection


def write_with_retry(connection: sqlite3.Connection, operation, attempts: int = 5):
    """
    Run a write, retrying while another process holds the write lock.

    SQLite's busy timeout covers waiting for a lock, but not every busy case:
    a connection that already holds a read snapshot and then tries to write
    gets SQLITE_BUSY back immediately, without waiting, to avoid deadlocking.
    That is exactly the shape of a dashboard request - read the entity, then
    update it - while the processor is committing a clip.

    `operation` is a callable taking the connection and doing the writes; it
    may be invoked more than once, so it must be safe to repeat.
    """
    delay = 0.15
    last_error: sqlite3.OperationalError | None = None

    for attempt in range(attempts):
        try:
            result = operation(connection)
            connection.commit()
            return result
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) and "busy" not in str(error).lower():
                raise
            last_error = error
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            time.sleep(delay)
            delay *= 2   # 0.15, 0.3, 0.6, 1.2, 2.4 - about 4.5s of patience

    raise last_error if last_error else sqlite3.OperationalError("write failed")


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    """sqlite3.Row is tuple-like; the API layer wants plain JSON-able dicts."""
    return [dict(row) for row in rows]


def upsert(connection: sqlite3.Connection, table: str, record: dict[str, Any], key: str = "id") -> None:
    """
    Insert `record`, or update the existing row when `key` collides.

    Kept generic because every writer in this codebase has the same shape of
    need and hand-writing eight nearly-identical INSERT ... ON CONFLICT
    statements is how typos get shipped.
    """
    columns = list(record.keys())
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column != key)

    sql = f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"
    if updates:
        sql += f" ON CONFLICT({key}) DO UPDATE SET {updates}"
    else:
        sql += f" ON CONFLICT({key}) DO NOTHING"

    connection.execute(sql, [record[column] for column in columns])


def get_setting(connection: sqlite3.Connection, key: str, default: str = "") -> str:
    row = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    connection.commit()


def import_legacy_alert_json(connection: sqlite3.Connection, alerts_dir: Path) -> int:
    """
    Backfill the `alerts` table from the JSON files written by the original
    processor, so alerts that predate the database still show up in the UI.

    Two properties this needs and did not originally have:

    1. It only writes rows that are actually missing or stale. Re-importing all
       ~100 files on every web start meant a burst of ~100 writes competing
       with the processor for the write lock, for no benefit.
    2. It writes in ONE transaction with retry. It runs at web startup, and a
       lock timeout here used to abort startup and put the container into a
       restart loop - taking the dashboard down precisely when the processor
       was busiest, which is when you would most want to look at it.
    """
    # What we already have, so we can skip the unchanged majority.
    existing_status = {
        row["id"]: row["status"]
        for row in connection.execute("SELECT id, status FROM alerts").fetchall()
    }

    pending: list[dict[str, Any]] = []
    for alert_path in sorted(alerts_dir.glob("*.json")):
        try:
            alert = json.loads(alert_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        alert_id = alert.get("id", alert_path.stem)
        status = alert.get("status", "")
        if existing_status.get(alert_id) == status:
            continue   # already imported and unchanged

        hits = alert.get("hits", []) or []
        pending.append({
            # NOTE: clip_id is deliberately ABSENT from this dict. upsert() builds
            # its ON CONFLICT DO UPDATE clause from the record's keys, so including
            # clip_id here would set every re-imported alert's clip_id to NULL -
            # severing the alert -> clip link (camera, capture time, GPS, drive)
            # for the whole history, silently, on every web restart. Omitting the
            # key leaves the column untouched on update and NULL on a fresh insert.
            "id": alert_id,
            "timestamp": int(alert.get("timestamp", 0)),
            "source_file": alert.get("source_file", ""),
            "camera": alert.get("camera"),
            "score": float(alert.get("score", 0.0) or 0.0),
            "hit_count": len(hits),
            "person_count": sum(1 for hit in hits if hit.get("class_id") == 0),
            "vehicle_count": sum(1 for hit in hits if hit.get("class_id") in (2, 3, 5, 7)),
            "jpeg": alert.get("jpeg", ""),
            "gif": alert.get("gif", ""),
            "status": status,
            "lat": alert.get("lat"),
            "lon": alert.get("lon"),
        })

    if not pending:
        return 0

    def write_all(active_connection):
        for record in pending:
            upsert(active_connection, "alerts", record)

    write_with_retry(connection, write_all)
    return len(pending)


def now_ts() -> int:
    """Single definition of 'now' so every table agrees on units (epoch seconds)."""
    return int(time.time())
