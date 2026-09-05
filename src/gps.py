"""
gps.py

Location from a USB GNSS receiver, as a replacement for poller.py's half of
the job that actually matters.

poller.py talks to Tesla's Fleet API. That works, but it needs a registered
developer application, it needs OAuth tokens that expire, it keeps the car
awake if you are careless with it, and Tesla has already retired one location
API out from under this project (Scout's owner-api.teslamotors.com). A $35
u-blox receiver on the Jetson's own USB has none of those properties. It needs
no credentials, cannot be deprecated, draws no battery from the car, and works
with no connectivity at all - which matters, because the pipeline it feeds is
running in a car.

What it gives up: odometer, charge state and shift position. Nothing in
correlate.py scores on any of those - it reasons over lat/lon, drive_id and
haversine spread - so the loss is cosmetic. `status` here is only ever 'D' or
'P'; a charging car reads as parked, which for "was the car stationary here"
is the right answer anyway.

The two tables are shared. This writes the same `polls` rows poller.py writes
and calls poller.attach_drive for segmentation, so the two can run together and
interleave without either knowing about the other.

Why geocoding is decoupled from sampling
----------------------------------------
geo.py caches by coordinates rounded to 4 decimal places - about 11 metres. At
highway speed every single sample falls in a fresh cell, so geocoding once per
sample would be a fresh Nominatim request every time, and reverse_geocode
sleeps 1.1s to honour OSM's rate limit. A 30-minute drive would be several
hundred requests and half an hour of sleeping. That is the exact abuse geo.py's
docstring says it exists to prevent.

So we geocode on *distance travelled*, not on sample count: once the car has
moved GPS_GEOCODE_MIN_MOVE_MILES since the last lookup, or when it settles into
being parked. Detections still get lat/lon from every sample, because
location_at only reads lat/lon and drive_id; street and city are for the
dashboard.

Configuration (all via environment, normally .env):
    GPS_DEVICE                  serial device (default: the u-blox by-id path)
    GPS_MOVING_SAMPLE_SECONDS   how often to record a row while driving
    GPS_PARKED_SAMPLE_SECONDS   how often to record a row while stopped
    GPS_GEOCODE_MIN_MOVE_MILES  distance before we spend a Nominatim request
    GPS_HEARTBEAT_SECONDS       how often to rewrite logs/gps.json
    GPS_LOG_MAX_BYTES           rotate logs/gps.log past this size

Diagnostics:
    python3 src/gps.py --status     print live fix state, write nothing
    python3 src/gps.py --raw        echo raw NMEA sentences

Why this service writes two files nobody asked for
--------------------------------------------------
An invalid fix is skipped before the insert, so a healthy receiver in an
underground garage writes nothing to `polls` - which is byte-identical to a
crashed reader, an unplugged receiver, and a service that was never installed.
No reader downstream can tell those four apart from the database alone, and
they are exactly the four someone needs told apart at the moment they go
looking. So the service also writes logs/gps.json every GPS_HEARTBEAT_SECONDS
(liveness, satellites, the last error, the last row written) and mirrors its
stdout to logs/gps.log. Both live inside BASE_DIR, which the web container
already mounts, so the dashboard reads them with no new privilege and no
journald access.

Neither file carries coordinates - see redact_position(). LOGS_DIR is mounted
by four containers and the dashboard serves the log verbatim, so an unredacted
stream would be a continuous location history of the owner's car sitting in a
debug panel.
"""
import calendar
import json
import os
import re
import sys
import time
import uuid

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from src.common import LOGS_DIR, ensure_dirs, env_float, env_int
from src.db import connect, now_ts, upsert
from src.geo import haversine_miles, reverse_geocode
from src.nmea_sky import SkyView
from src.poller import MOVING_SPEED_THRESHOLD_MPH, attach_drive

# The by-id path is derived from the device's own USB descriptor and survives
# replugging and reboots. /dev/ttyACM0 is assigned in enumeration order, so it
# silently becomes ttyACM1 the moment anything else that speaks CDC-ACM is
# plugged in - and in a car that reboots every drive, that is a real event.
GPS_DEVICE = os.environ.get(
    "GPS_DEVICE",
    "/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00",
)

GPS_MOVING_SAMPLE_SECONDS = env_int("GPS_MOVING_SAMPLE_SECONDS", 5)
GPS_PARKED_SAMPLE_SECONDS = env_int("GPS_PARKED_SAMPLE_SECONDS", 300)
GPS_GEOCODE_MIN_MOVE_MILES = env_float("GPS_GEOCODE_MIN_MOVE_MILES", 0.25)

# How often the heartbeat file is rewritten. 5s is a deliberate compromise: the
# dashboard polls at 2s, so a slower beat makes a live service look stale on
# every other request, while 1Hz would be ~86k fsync+rename pairs a day onto
# the SD card this whole appliance boots from.
GPS_HEARTBEAT_SECONDS = env_int("GPS_HEARTBEAT_SECONDS", 5)

# Rotated one generation deep, for natix_worker.py's reason: the storage budget
# is a 119GB card shared with the clip archive, and nobody is ever going to
# read gps.log.7.
GPS_LOG_MAX_BYTES = env_int("GPS_LOG_MAX_BYTES", 2_000_000)

# Sentences kept in the heartbeat for the debug page. Twenty is roughly two
# seconds of a multi-constellation receiver - enough to see the shape of an
# epoch, short enough that gps.json stays a small fixed-size file that
# os.replace can swap in one step.
RECENT_SENTENCE_LIMIT = 20

# NMEA's saturation value for every dilution-of-precision field. Rendered as a
# number it looks like catastrophically bad geometry and sends the reader
# hunting a sky-view problem; it actually means "no value".
DOP_SATURATION = 99.99

# NMEA reports speed over ground in knots. Everything downstream - the polls
# table, the dashboard, MOVING_SPEED_THRESHOLD_MPH - is in mph.
MPH_PER_KNOT = 1.150779

RECONNECT_DELAY_SECONDS = 5


# ---------------------------------------------------------------------------
# NMEA parsing
# ---------------------------------------------------------------------------
# All of this is pure text handling with no hardware in it, which is the point:
# the parser is the part that can be wrong in ways that quietly corrupt a
# year of location history, so it is testable off recorded sentences.

def checksum_ok(sentence: str) -> bool:
    """
    Verify an NMEA checksum: XOR of every byte between '$' and '*'.

    Worth doing rather than trusting the line. A receiver on a 5m USB extension
    in a car is on the wrong end of a noisy cable, and a corrupted digit in a
    latitude field is not detectable any other way - it just puts you in the
    next county.
    """
    sentence = sentence.strip()
    if not sentence.startswith("$") or "*" not in sentence:
        return False

    body, _, given = sentence[1:].partition("*")
    if len(given) < 2:
        return False

    computed = 0
    for character in body:
        computed ^= ord(character)

    try:
        return computed == int(given[:2], 16)
    except ValueError:
        return False


def sentence_kind(sentence: str) -> str:
    """
    The 3-letter sentence type, ignoring the talker prefix.

    This matters more than it looks. Everyone writes `$GPRMC` in examples, but
    a multi-constellation u-blox 8 emits `$GNRMC` - GN for the combined GPS +
    GLONASS + Galileo solution. Matching on '$GP' is the single most common way
    to build a parser that works on a bench receiver and returns nothing at all
    on the device you actually bought.
    """
    body = sentence.strip().lstrip("$")
    return body[2:5].upper() if len(body) >= 5 else ""


def parse_degrees(value: str, hemisphere: str) -> float | None:
    """
    Convert NMEA's ddmm.mmmm / dddmm.mmmm into signed decimal degrees.

    NMEA does not report decimal degrees. It reports degrees and *minutes*
    glued together, so 4807.038 is 48 degrees 7.038 minutes, which is 48.1173
    degrees - not 48.07. Reading it as a float directly is wrong by up to
    ~0.17 degrees, roughly 11 miles, and it is wrong in a way that still looks
    like a plausible coordinate.
    """
    if not value or not hemisphere:
        return None

    try:
        # Longitude has three degree digits, latitude two. Rather than switch
        # on which field we were handed, take everything before the last two
        # digits of the integer part as degrees.
        point = value.find(".")
        split_at = (point if point >= 0 else len(value)) - 2
        if split_at <= 0:
            return None
        degrees = float(value[:split_at])
        minutes = float(value[split_at:])
    except ValueError:
        return None

    result = degrees + minutes / 60.0
    if hemisphere.upper() in ("S", "W"):
        result = -result
    return result


def parse_timestamp(time_field: str, date_field: str) -> int | None:
    """
    Combine NMEA's hhmmss.ss and ddmmyy into a Unix timestamp.

    We prefer this over the system clock deliberately. The Jetson is a headless
    box in a car that may boot with no network, so its clock can be wrong by
    hours - and a detection stamped with the wrong time gets matched against
    the wrong location, which is worse than having no location. The receiver's
    time comes from the satellites.
    """
    if not time_field or not date_field or len(date_field) < 6:
        return None

    try:
        hours, minutes = int(time_field[0:2]), int(time_field[2:4])
        seconds = int(float(time_field[4:]))
        day, month = int(date_field[0:2]), int(date_field[2:4])
        year = int(date_field[4:6])
    except ValueError:
        return None

    year += 2000 if year < 80 else 1900
    try:
        return calendar.timegm((year, month, day, hours, minutes, seconds, 0, 0, 0))
    except (ValueError, OverflowError):
        return None


@dataclass
class Fix:
    """One position solution, assembled from the sentences that describe it."""
    valid: bool = False
    lat: float | None = None
    lon: float | None = None
    speed_mph: float = 0.0
    heading: float | None = None
    satellites: int = 0
    hdop: float | None = None
    ts: int | None = None
    # GGA's quality indicator: 0 no fix, 1 GPS, 2 DGPS, 4/5 RTK, 6 dead
    # reckoning. It distinguishes "searching" from "solving badly", which
    # neither the satellite count nor HDOP can. Defaulted like every field
    # here, because the tests build a Fix by keyword and a field without a
    # default turns that into a TypeError rather than a readable failure.
    quality: int | None = None

    @property
    def status(self) -> str:
        """'D' or 'P'. Never 'C' - a GPS cannot see a charge port."""
        return "D" if self.speed_mph > MOVING_SPEED_THRESHOLD_MPH else "P"

    def describe(self) -> str:
        if not self.valid:
            # Not "visible". This count is GGA's numSV - the satellites USED in
            # the solution - and indoors it reads 0 while twelve are in view at
            # good SNR. Printing that as "0 satellites visible" answers "is the
            # antenna dead?" with a confident no when the truth is the exact
            # opposite. In-view and tracked come from GSV and live on SkyView;
            # NmeaAssembler.describe() puts all three in one line.
            return f"no fix ({self.satellites} satellites used in the solution)"
        return (
            f"{self.status} {self.speed_mph:.0f}mph "
            f"@ {self.lat:.5f},{self.lon:.5f} "
            f"({self.satellites} sats, hdop {self.hdop or 0:.1f})"
        )


class NmeaAssembler:
    """
    Accumulates sentences into a Fix.

    RMC carries position, speed, course and the date; GGA carries fix quality,
    satellite count and HDOP. Neither alone is enough, so we keep the latest of
    each and hand out a fix when RMC says the solution is valid.
    """

    def __init__(self) -> None:
        self.fix = Fix()
        # GSA and GSV describe the sky, not a position, and they have the
        # opposite lifetime to a Fix: a fix is one instant, the sky is a
        # rolling window that survives across epochs and expires on its own.
        # Keeping them in a separate object is what stops satellite bookkeeping
        # leaking into the polls row.
        self.sky = SkyView()

    def describe(self) -> str:
        """
        The fix line plus the sky counts.

        In-view, tracked and used are three different numbers and the whole
        diagnosis lives in how they differ: 11/11/0 is a working antenna that
        needs sky, 3/0/0 is an antenna or LNA fault, and 0/0/0 is a receiver
        that is not being read. Printing any one of them alone gives an answer
        that sounds definite and is wrong two times in three.
        """
        sky = self.sky.snapshot(time.monotonic())
        return (
            f"{self.fix.describe()} "
            f"[{sky['satellites_in_view']} in view, "
            f"{sky['satellites_tracked']} tracked, "
            f"{self.fix.satellites} used]"
        )

    def feed(self, sentence: str) -> Fix | None:
        """Consume one sentence. Returns a Fix when RMC completes a cycle."""
        if not checksum_ok(sentence):
            return None

        fields = sentence.strip().split("*")[0].split(",")[1:]
        kind = sentence_kind(sentence)

        if kind in ("GSA", "GSV"):
            # GSV is the one sentence family that does NOT collapse to the GN
            # talker - a u-blox emits $GPGSV, $GLGSV and $GAGSV as separate
            # interleaved cycles - so the two-letter prefix has to travel with
            # the fields or GLONASS satellites merge into the GPS list.
            talker = sentence.strip().lstrip("$")[0:2].upper()
            self.sky.feed(kind, talker, fields, time.monotonic())
            # None, unconditionally. Only RMC completes a cycle; returning a
            # Fix here would push satellite sentences into main()'s database
            # path carrying the PREVIOUS epoch's timestamp and position under a
            # fresh row id - a silent data-quality regression in `polls` that
            # the sample-interval gate would rate-limit but not stop.
            return None

        if kind == "GGA" and len(fields) >= 8:
            try:
                self.fix.satellites = int(fields[6] or 0)
            except ValueError:
                self.fix.satellites = 0
            try:
                self.fix.hdop = float(fields[7]) if fields[7] else None
            except ValueError:
                self.fix.hdop = None
            try:
                self.fix.quality = int(fields[5] or 0)
            except ValueError:
                # A receiver mid-warm-start can emit a blank or garbage
                # quality field. None means "not reported", which is a
                # different statement from 0 ("reported, and there is no fix").
                self.fix.quality = None
            return None

        if kind != "RMC" or len(fields) < 9:
            return None

        # RMC is the epoch boundary for the GSA burst. A receiver emits all of
        # its per-constellation GSA sentences and then the RMC, so RMC is the
        # only marker guaranteed to fall between two bursts rather than through
        # the middle of one - a wall-clock timer would split them and report
        # "no fix" for whichever constellations landed on the far side.
        self.sky.on_rmc()

        # 'A' is active, 'V' is void. On a void fix the coordinate fields are
        # usually empty, but not always - some receivers keep echoing the last
        # known position. Trusting them writes a stale location as if it were
        # current, so status is the only thing we believe here.
        self.fix.valid = fields[1].upper() == "A"

        if not self.fix.valid:
            self.fix.lat = self.fix.lon = None
            self.fix.speed_mph = 0.0
            self.fix.heading = None
            self.fix.ts = None
            return self.fix

        self.fix.lat = parse_degrees(fields[2], fields[3])
        self.fix.lon = parse_degrees(fields[4], fields[5])

        try:
            self.fix.speed_mph = float(fields[6]) * MPH_PER_KNOT if fields[6] else 0.0
        except ValueError:
            self.fix.speed_mph = 0.0

        try:
            self.fix.heading = float(fields[7]) if fields[7] else None
        except ValueError:
            self.fix.heading = None

        self.fix.ts = parse_timestamp(fields[0], fields[8]) or now_ts()

        # A valid status with no parseable coordinates is not a fix.
        if self.fix.lat is None or self.fix.lon is None:
            self.fix.valid = False

        return self.fix


def poll_record_from_fix(fix: Fix) -> dict:
    """
    Build a `polls` row from a fix, matching poller.poll_record_from_vehicle_data.

    The fields a GPS cannot know - power, shift_state, odometer - are None
    rather than 0.0, so a reader can tell "no source for this" apart from "the
    car reported zero".
    """
    return {
        "id": uuid.uuid4().hex[:12],
        "ts": fix.ts or now_ts(),
        "lat": fix.lat,
        "lon": fix.lon,
        "heading": fix.heading,
        "speed": fix.speed_mph,
        "power": None,
        "shift_state": None,
        "status": fix.status,
        "loc_available": 1 if (fix.lat is not None and fix.lon is not None) else 0,
        "odometer": None,
        "street": None,
        "city": None,
        "drive_id": None,
        "geocode_id": None,
    }


# ---------------------------------------------------------------------------
# Heartbeat and log mirror
# ---------------------------------------------------------------------------
# A file, not a row in `settings`. Part of this heartbeat's job is to report
# "fixes are good but the database write is failing" - main()'s except clause
# swallows a persistent `database is locked` into a printed line nobody reads -
# and a heartbeat that travelled through the same sqlite connection would be
# blind to exactly the failure it exists to name. The file is also readable by
# --raw and --status, which hold no database connection at all.

HEARTBEAT_SCHEMA = 1

# Stamped once at import. Restart=always with RestartSec=10 means a service
# crash-looping every ten seconds writes a fresh heartbeat every ten seconds,
# so a freshness check calls it healthy forever; a started_ts that changes
# under a reader's feet is the only thing that gives it away.
_STARTED_TS = int(time.time())

# Off until main() turns it on, and main() turns it on only after the --raw and
# --status early returns. `python3 src/gps.py --status` is documented as safe
# to run against a live service; ungated, two processes would race to replace
# the same file and the diagnostic tool would corrupt the diagnostic page's
# view of the service it was run to inspect.
_HEARTBEAT_ENABLED = False

# Shared between read_sentences() and main() because they cannot see each
# other: when the receiver is unplugged read_sentences yields nothing at all,
# so main()'s loop body never executes and anything it owned would freeze.
_STATE: dict = {
    "assembler": None,
    "port_open": False,
    "last_error": None,
    "last_error_ts": None,
    "sentences_total": 0,
    "checksum_failures": 0,
    "last_sentence_ts": None,
    "last_fix_ts": None,
    "last_fix_lat": None,
    "last_fix_lon": None,
    "last_fix_speed_mph": None,
    "last_fix_at": None,
    "last_row_written_ts": None,
    "last_db_error": None,
    "last_db_error_ts": None,
    "recent_sentences": deque(maxlen=RECENT_SENTENCE_LIMIT),
    "last_beat_at": 0.0,
}

# Which comma-separated fields of a whole sentence carry a position, by kind.
# These are indices into the sentence INCLUDING its "$GPRMC" token, so each is
# one higher than the sliced index the parser above uses.
_POSITION_FIELDS = {"RMC": (3, 5), "GGA": (2, 4), "GLL": (1, 3)}

# A decimal lat/lon pair as this module prints one, e.g. "33.20000,-117.24000".
_COORDINATE_PAIR = re.compile(r"-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}")


def redact_position(sentence: str) -> str:
    """
    Blank the latitude and longitude out of an NMEA sentence, keep the rest.

    Watching "$GNRMC,123519,V,,N,,W,,,,,,N*xx" scroll past proves the cable,
    the tty, the framing and the parser all work, which is the entire reason
    the debug page shows raw sentences. The coordinates prove none of that and
    turn the page into a precise, continuously updating record of where the
    owner's car has been. Hemisphere, time, status, satellite counts, HDOP and
    checksum all survive, because every one of them is diagnostic.

    Empty coordinate fields are left empty rather than filled with a marker: a
    receiver emitting blank lat/lon is the single most common indoor case, and
    disguising it would hide the one thing the line is being read for.
    """
    text = sentence.strip()
    indices = _POSITION_FIELDS.get(sentence_kind(text))
    if indices is None:
        return text

    parts = text.split(",")
    for index in indices:
        if index < len(parts) and parts[index]:
            parts[index] = "..."
    return ",".join(parts)


_PLACE_AFTER_AT = re.compile(r"@ .*$")


def redact_log_line(line: str) -> str:
    """
    The same rule applied to a free-form log line.

    gps.log is served to the dashboard verbatim and sits on a volume four
    containers mount. A single un-redacted "parked at 33.20000,-117.24000" line
    is a home address, and it would be written once every five minutes forever.
    """
    if line.lstrip().startswith("$"):
        line = redact_position(line)
    line = _COORDINATE_PAIR.sub("...,...", line)
    # Place names, not just numbers. A geocoded street IS the home address once
    # the car is parked, so "@ 1425 Camino De Los Coches" leaks exactly what
    # scrubbing the coordinates was meant to prevent. Callers are not supposed
    # to pass one - the poll line logs status fields only - but a redactor that
    # handles just the numeric forms invites the next caller to assume anything
    # it returns is safe. Anything after an "@ " marker goes.
    return _PLACE_AFTER_AT.sub("@ ...", line)


def _gps_log_path() -> Path:
    """
    Resolved on every call rather than frozen at import.

    LOGS_DIR is derived from BASE_DIR, which differs between the host unit
    (/mnt/jetsondata/tesla-alerts) and the container (/data). Reading it late
    keeps the tests able to point this somewhere disposable, and keeps one
    module-level constant from silently outliving an environment change.
    """
    return LOGS_DIR / "gps.log"


def _append_to_log_file(line: str) -> None:
    """
    Append one line, rotating one generation deep at GPS_LOG_MAX_BYTES.

    Every failure is swallowed. A GPS service that died because it could not
    write its own debug log would be a poor trade, and the journal has the same
    line already.
    """
    try:
        ensure_dirs()
        path = _gps_log_path()
        if path.exists() and path.stat().st_size > GPS_LOG_MAX_BYTES:
            path.replace(path.with_suffix(".log.1"))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def log(message: str) -> None:
    """
    Print one line and mirror it into logs/gps.log.

    The journal already has all of this, but the dashboard runs in a container
    with no access to journald and no reasonable way to get it - the same
    problem natix_worker.py solves the same way. The file carries its own
    timestamp because the dashboard reads it with no other source of time.

    The mirror is gated on the same flag as the heartbeat, so --raw and
    --status stay strictly read-only: a diagnostic run must not interleave its
    lines into the running service's log.
    """
    # Redacted ONCE, before either sink, so the two cannot disagree.
    #
    # An earlier version redacted only the file and printed the raw message.
    # That made the docstring above false in the direction that matters: the
    # file was a redacted derivative, not a mirror, and anyone who later added
    # `StandardOutput=append:...` to gps.service - a natural thing to do - would
    # silently start writing raw coordinates into the served file with nothing
    # anywhere to warn them. journald being root-only today is a property of
    # this deployment, not of this function.
    safe = redact_log_line(message)
    print(f"[🛰️ gps] {safe}", flush=True)
    if _HEARTBEAT_ENABLED:
        stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [🛰️ gps] {safe}"
        _append_to_log_file(stamped)


def _dop_or_none(value: float | None) -> float | None:
    """99.99 is NMEA for "no value", not for terrible geometry."""
    if value is None:
        return None
    return None if value >= DOP_SATURATION else value


def _record_port_error(message: str, alarm: bool = False) -> None:
    """Remember why the device is unreadable, and say so once per attempt."""
    _STATE["last_error"] = message
    _STATE["last_error_ts"] = int(time.time())
    log(f"⧱❗️ {message}" if alarm else message)


def _heartbeat_payload() -> dict:
    """
    Build the payload the dashboard reads.

    EVERY key is always present; `null` means "no value yet" and an absent key
    is a bug in this function. The page is written against this exact shape, so
    a missing key surfaces as a blank field on the one page someone opens when
    nothing else is working, with no error raised anywhere to notice it by.
    """
    assembler = _STATE["assembler"]
    fix = assembler.fix if assembler is not None else Fix()
    # An empty SkyView rather than a hand-written dict of nulls, so this path
    # cannot drift out of step with the eight keys snapshot() actually returns.
    sky = (assembler.sky if assembler is not None else SkyView()).snapshot(
        time.monotonic()
    )

    return {
        "schema": HEARTBEAT_SCHEMA,
        "written_ts": int(time.time()),
        "heartbeat_interval": GPS_HEARTBEAT_SECONDS,
        "pid": os.getpid(),
        "started_ts": _STARTED_TS,
        "device": GPS_DEVICE,
        "device_present": os.path.exists(GPS_DEVICE),
        "port_open": bool(_STATE["port_open"]),
        "last_error": _STATE["last_error"],
        "last_error_ts": _STATE["last_error_ts"],
        "sentences_total": _STATE["sentences_total"],
        "checksum_failures": _STATE["checksum_failures"],
        "last_sentence_ts": _STATE["last_sentence_ts"],
        "fix_valid": bool(fix.valid),
        "fix_quality": fix.quality,
        "satellites_used": fix.satellites,
        "satellites_in_view": sky["satellites_in_view"],
        "satellites_tracked": sky["satellites_tracked"],
        "hdop": _dop_or_none(fix.hdop),
        "pdop": sky["pdop"],
        "vdop": sky["vdop"],
        "gsa_fix_type": sky["gsa_fix_type"],
        "gsa_used_prns": sky["gsa_used_prns"],
        "constellations": sky["constellations"],
        "sky": sky["sky"],
        "last_fix_ts": _STATE["last_fix_ts"],
        "last_fix_lat": _STATE["last_fix_lat"],
        "last_fix_lon": _STATE["last_fix_lon"],
        "last_fix_speed_mph": _STATE["last_fix_speed_mph"],
        "last_fix_at": _STATE["last_fix_at"],
        "last_row_written_ts": _STATE["last_row_written_ts"],
        "last_db_error": _STATE["last_db_error"],
        "last_db_error_ts": _STATE["last_db_error_ts"],
        "recent_sentences": list(_STATE["recent_sentences"]),
    }


def _write_heartbeat(payload: dict) -> None:
    """
    Replace logs/gps.json in one step, or leave the old one alone.

    The temp file is created in the SAME directory, so os.replace is a
    same-filesystem rename and therefore atomic: a reader polling at 2s sees
    either the whole previous payload or the whole new one, never a torn read.
    The leading dot keeps a partial file from being mistaken for the real one
    if the process dies between the write and the rename.

    The mode is set explicitly rather than inherited. gps.service runs as root
    and the web container also runs as uid 0, so today a 0644 file is readable
    by luck of the default umask; adding UMask=0077 to the unit would make it
    0600 and the page would fail with a bare PermissionError.

    Never opened for append or truncate: that would defeat both the atomicity
    and the mtime semantics the reader's staleness check depends on. The file
    is fixed-size and must not rotate, because rotation makes the reader's
    existence check race the rename.
    """
    try:
        if not _HEARTBEAT_ENABLED:
            return
        ensure_dirs()
        target = LOGS_DIR / "gps.json"
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
        try:
            with open(temporary, "wb") as handle:
                handle.write(json.dumps(payload).encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, target)
        except BaseException:
            # Including KeyboardInterrupt and SystemExit on purpose: a dot-file
            # left behind on every restart accumulates in the directory the
            # dashboard lists, and the next reader has to guess which one is
            # real.
            temporary.unlink(missing_ok=True)
            raise
    except OSError:
        pass                        # a GPS service must not die over a debug file


def _beat(force: bool = False) -> None:
    """
    Write the heartbeat, at most once per GPS_HEARTBEAT_SECONDS.

    Throttled on time.monotonic() rather than the wall clock because this box
    boots in a car with no network: NTP can step time.time() by hours the
    moment it finds one, which would either stop the beat entirely or spin it
    in a tight loop for the length of the step.
    """
    if not _HEARTBEAT_ENABLED:
        return
    now = time.monotonic()
    if not force and now - _STATE["last_beat_at"] < GPS_HEARTBEAT_SECONDS:
        return
    _STATE["last_beat_at"] = now
    _write_heartbeat(_heartbeat_payload())


# ---------------------------------------------------------------------------
# Reading the device
# ---------------------------------------------------------------------------

def configure_tty(handle) -> None:
    """
    Put the tty in raw mode.

    CDC-ACM ignores baud rate, but the tty line discipline above it does not
    ignore canonical mode - left alone it can buffer, echo, and translate CR.
    Failing is fine: this is a no-op on a regular file, which is what the tests
    hand us.
    """
    try:
        import termios
        import tty
        tty.setraw(handle.fileno())
        attributes = termios.tcgetattr(handle.fileno())
        termios.tcsetattr(handle.fileno(), termios.TCSANOW, attributes)
    except Exception:                                      # noqa: BLE001
        pass


def read_sentences(path: str):
    """
    Yield NMEA sentences from the device, reopening it if it goes away.

    USB in a car is not reliable - connectors vibrate loose, the hub browns out
    on cranking. A reader that raises on the first disconnect is a reader that
    logs one drive and then nothing for a month, which is exactly the sort of
    silent failure this project keeps running into.
    """
    while True:
        try:
            with open(path, "rb", buffering=0) as handle:
                configure_tty(handle)
                _STATE["port_open"] = True
                # A successful open answers the question the last failure
                # asked, so the stale error has to go. Leaving it set makes a
                # receiver that was unplugged an hour ago read as unplugged
                # now, on a page whose whole job is telling those apart.
                _STATE["last_error"] = None
                _STATE["last_error_ts"] = None
                log(f"reading {path}")
                _beat(force=True)
                buffer = b""
                while True:
                    chunk = handle.read(256)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, _, buffer = buffer.partition(b"\n")
                        sentence = line.decode("ascii", errors="replace").strip()
                        # Counted here, before any validation, so the total
                        # includes lines that fail checksum: "receiving nothing"
                        # and "receiving noise" are different faults with
                        # different fixes, and comparing the two counters is
                        # the only way to see a bad cable.
                        _STATE["sentences_total"] += 1
                        _STATE["last_sentence_ts"] = int(time.time())
                        _STATE["recent_sentences"].append(redact_position(sentence))
                        yield sentence
                    # A receiver that stops framing shouldn't grow us without
                    # bound; 4KB is far more than one sentence cycle.
                    if len(buffer) > 4096:
                        buffer = b""
        except FileNotFoundError:
            _record_port_error(f"{path} not present - is the receiver plugged in?")
        except PermissionError:
            _record_port_error(f"cannot open {path} - need root or the dialout group")
        except OSError as error:
            _record_port_error(f"read failed: {error}", alarm=True)

        # Every path out of the try lands here - the three failures and a clean
        # EOF - and the beat has to happen before the sleep. With the receiver
        # unplugged this generator yields NOTHING: it records, sleeps and
        # retries forever, so main()'s loop body never runs. A heartbeat placed
        # only in main() would freeze at the last good write and the page would
        # report "GPS service dead" for a service that is alive and waiting,
        # which is precisely the wrong diagnosis and the wrong remedy.
        _STATE["port_open"] = False
        _beat(force=True)
        time.sleep(RECONNECT_DELAY_SECONDS)


# ---------------------------------------------------------------------------
def main() -> int:
    global _HEARTBEAT_ENABLED

    arguments = sys.argv[1:]
    raw_mode = "--raw" in arguments
    status_mode = "--status" in arguments

    if raw_mode:
        for sentence in read_sentences(GPS_DEVICE):
            print(sentence)
        return 0

    assembler = NmeaAssembler()
    _STATE["assembler"] = assembler

    if status_mode:
        # Diagnostics only - touches no database, and _HEARTBEAT_ENABLED is
        # still False here, so it also writes no heartbeat and no log file.
        # That is what makes it safe to run against a live deployment.
        for sentence in read_sentences(GPS_DEVICE):
            fix = assembler.feed(sentence)
            if fix:
                log(assembler.describe())
        return 0

    # Past both early returns: this process is the service, and from here it is
    # the single writer of logs/gps.json and logs/gps.log.
    _HEARTBEAT_ENABLED = True

    connection = connect()
    log(f"device {GPS_DEVICE}")
    # Beat once before the first sentence. A receiver that never enumerates
    # otherwise leaves an empty directory, which the page cannot tell apart
    # from "the service was never installed" - a different sentence and a
    # different command for the user.
    _beat(force=True)

    last_written_at = 0.0
    last_geocode_point: tuple[float, float] | None = None
    last_unfixed_report_at = 0.0
    previous_fix_status: str | None = None
    geocode_on_park = False

    for sentence in read_sentences(GPS_DEVICE):
        try:
            if not checksum_ok(sentence):
                # Checked again inside feed(). The duplicate XOR costs
                # microseconds and keeps the parser's own guard where it
                # belongs - moving the check out of feed() would leave
                # NmeaAssembler trusting whatever it is handed.
                _STATE["checksum_failures"] += 1

            fix = assembler.feed(sentence)

            if fix is not None and fix.valid:
                # Snapshotted here rather than read off assembler.fix at write
                # time, because the next void RMC blanks lat/lon/ts in place.
                # last_fix_ts is SATELLITE time and last_fix_at is host time;
                # they are kept apart deliberately, since the two clocks
                # disagree by hours on a box that boots without a network.
                _STATE["last_fix_ts"] = fix.ts
                _STATE["last_fix_lat"] = fix.lat
                _STATE["last_fix_lon"] = fix.lon
                _STATE["last_fix_speed_mph"] = fix.speed_mph
                _STATE["last_fix_at"] = int(time.time())

            # Above every `continue` below, on purpose. A heartbeat written
            # only on the success path is indistinguishable from a dead
            # service, and "no valid fix" is the ordinary state of a car in a
            # garage - which is exactly when someone opens the page.
            _beat()

            if fix is None:
                continue

            now = time.monotonic()

            if not fix.valid:
                # Say so occasionally, so a receiver that never gets a fix is
                # visibly different from one that is not being read at all.
                if now - last_unfixed_report_at > 60:
                    log(assembler.describe())
                    last_unfixed_report_at = now
                continue

            if previous_fix_status == "D" and fix.status == "P":
                # The car has just stopped. The module docstring promises a
                # geocode "when it settles into being parked" and the distance
                # gate alone never delivers one: the last lookup sits roughly
                # GPS_GEOCODE_MIN_MOVE_MILES back down the road while geo.py's
                # cache cell is about 11 metres, so the parked row - the only
                # row a "where is my car" page ever shows - lands with street
                # and city NULL and the page renders bare coordinates.
                #
                # Deferred to the next written row rather than done here,
                # because the sample-interval gate below is about to skip this
                # iteration and a lookup here would be thrown away.
                geocode_on_park = True
            previous_fix_status = fix.status

            interval = (
                GPS_MOVING_SAMPLE_SECONDS if fix.status == "D"
                else GPS_PARKED_SAMPLE_SECONDS
            )
            if now - last_written_at < interval:
                continue
            last_written_at = now

            poll = poll_record_from_fix(fix)
            poll["drive_id"] = attach_drive(connection, poll)

            # Distance-gated, not sample-gated. See the module docstring.
            moved_enough = last_geocode_point is None or haversine_miles(
                last_geocode_point[0], last_geocode_point[1], fix.lat, fix.lon
            ) >= GPS_GEOCODE_MIN_MOVE_MILES

            if moved_enough or geocode_on_park:
                geocode = reverse_geocode(connection, poll["lat"], poll["lon"])
                if geocode:
                    poll["geocode_id"] = geocode["id"]
                    poll["street"] = geocode.get("road")
                    poll["city"] = geocode.get("city")
                last_geocode_point = (fix.lat, fix.lon)
                geocode_on_park = False

            upsert(connection, "polls", poll)
            connection.commit()
            # Stamped only after the commit returns. Comparing this against
            # last_fix_at is the only way to see the failure below: a
            # persistent `database is locked` from SD contention means good
            # fixes are acquired and discarded indefinitely while the receiver
            # looks perfectly healthy from every other angle.
            _STATE["last_row_written_ts"] = int(time.time())

            # Deliberately says WHERE NOTHING. Not the coordinates, and not
            # the geocoded street either - the street name IS the home address
            # once the car is parked, and this line would write it once every
            # GPS_PARKED_SAMPLE_SECONDS, forever, into a file that four
            # containers mount and that /api/gps/logs serves verbatim to any
            # dashboard viewer.
            #
            # An earlier version logged `poll["street"] or "<lat>,<lon>"` and
            # leaned on redact_log_line to scrub it. That scrubbed the numbers
            # and passed "1425 Camino De Los Coches" through untouched, which
            # is worse than not redacting at all: the redaction that IS there
            # makes the file look safe. Position lives in the database, which
            # is where something asking for it has to authenticate.
            log(
                f"{poll['status']} {poll['speed']:.0f}mph "
                f"sats={fix.satellites}used "
                f"hdop={fix.hdop if fix.hdop is not None else '-'} "
                f"quality={fix.quality if fix.quality is not None else '-'}"
            )

        except Exception as exception_object:              # noqa: BLE001
            # Deliberately not cleared on a later success. "This has failed at
            # some point since start" is the useful statement; a field that
            # clears itself hides an error that recurs every few minutes, which
            # is what a lock-contention failure actually looks like.
            _STATE["last_db_error"] = str(exception_object)
            _STATE["last_db_error_ts"] = int(time.time())
            log(f"⧱❗️ ERROR: {exception_object}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
