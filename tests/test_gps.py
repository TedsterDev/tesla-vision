"""
test_gps.py

Behavioural tests for the USB GNSS receiver reader.

The dangerous failure here is not "no fix". It is a fix that parses into a
plausible-looking coordinate that is wrong - because nothing downstream can
tell. A latitude off by 11 miles still geocodes, still plots, still produces
"seen at 4 distinct locations", and quietly poisons every correlation the
engine draws for as long as it runs. So most of these tests are about the
conversions, and about refusing a solution the receiver has not vouched for.

Run:  python3 tests/test_gps.py
"""
import contextlib
import json
import shutil
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import gps      # noqa: E402
from src import poller   # noqa: E402


def with_checksum(body: str) -> str:
    """Append a correct NMEA checksum, so fixtures stay readable."""
    computed = 0
    for character in body:
        computed ^= ord(character)
    return f"${body}*{computed:02X}"


# A real sentence with a hand-published checksum, used as the golden case: if
# our XOR is wrong, this fails and every other fixture (which computes its own)
# would still have passed.
GOLDEN_RMC = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------
def test_a_known_good_sentence_validates():
    assert gps.checksum_ok(GOLDEN_RMC), "XOR checksum disagrees with a published sentence"


def test_a_corrupted_sentence_is_rejected():
    # One digit of latitude flipped - the exact damage a noisy 5m USB
    # extension does, and undetectable except by the checksum.
    corrupted = GOLDEN_RMC.replace("4807.038", "4907.038")
    assert not gps.checksum_ok(corrupted)


def test_malformed_sentences_do_not_raise():
    for rubbish in ["", "$", "hello", "$GPRMC,no,star", "$GPRMC,x*ZZ", "*6A"]:
        assert gps.checksum_ok(rubbish) is False, rubbish


# ---------------------------------------------------------------------------
# The talker-prefix trap
# ---------------------------------------------------------------------------
def test_multi_constellation_gn_sentences_are_recognised():
    # A u-blox 8 solving across GPS + GLONASS + Galileo emits GN, not GP.
    # Matching on '$GP' is the classic way to write a parser that works on
    # every example on the internet and returns nothing on real hardware.
    assert gps.sentence_kind("$GNRMC,,V,,,,,,,,,,N,V*37") == "RMC"
    assert gps.sentence_kind("$GPRMC,,V,,,*00") == "RMC"
    assert gps.sentence_kind("$GLGSV,,,*00") == "GSV"
    assert gps.sentence_kind("$GNGGA,,,,,,0,00,99.99,,,,,,*56") == "GGA"


def test_a_gn_fix_actually_parses_end_to_end():
    assembler = gps.NmeaAssembler()
    fix = assembler.feed(with_checksum(
        "GNRMC,123519.00,A,4807.038,N,01131.000,E,022.4,084.4,230394,,,A"
    ))
    assert fix is not None and fix.valid, "a GN-talker fix was not parsed"


# ---------------------------------------------------------------------------
# Degrees and minutes
# ---------------------------------------------------------------------------
def test_latitude_is_degrees_and_minutes_not_decimal_degrees():
    # 4807.038 is 48 degrees 7.038 minutes = 48.1173 degrees.
    # Read as a plain float it looks like 48.07 - a plausible coordinate that
    # is 7 miles from the truth.
    value = gps.parse_degrees("4807.038", "N")
    assert abs(value - 48.1173) < 0.0001, value
    assert abs(value - 48.07) > 0.04, "parsed as decimal degrees"


def test_longitude_takes_three_degree_digits():
    # 01131.000 is 11 degrees 31 minutes, not 113 degrees 1 minute.
    value = gps.parse_degrees("01131.000", "E")
    assert abs(value - 11.51667) < 0.0001, value


def test_southern_and_western_hemispheres_are_negative():
    assert gps.parse_degrees("3352.000", "S") < 0
    assert gps.parse_degrees("11722.000", "W") < 0
    # San Diego, roughly - the sign convention that puts us in California
    # rather than in China.
    assert abs(gps.parse_degrees("3312.000", "N") - 33.2) < 0.001
    assert abs(gps.parse_degrees("11714.400", "W") + 117.24) < 0.001


def test_empty_or_junk_coordinates_return_none():
    assert gps.parse_degrees("", "N") is None
    assert gps.parse_degrees("4807.038", "") is None
    assert gps.parse_degrees("junk", "N") is None
    assert gps.parse_degrees("12", "N") is None


# ---------------------------------------------------------------------------
# Speed, time and status
# ---------------------------------------------------------------------------
def test_speed_is_converted_from_knots_to_mph():
    assembler = gps.NmeaAssembler()
    fix = assembler.feed(with_checksum(
        "GNRMC,123519.00,A,4807.038,N,01131.000,E,60.0,084.4,230394,,,A"
    ))
    # 60 knots is 69 mph. Storing 60 would under-report every speed by 13%
    # and mislabel highway driving.
    assert abs(fix.speed_mph - 69.05) < 0.1, fix.speed_mph


def test_timestamp_comes_from_the_satellites_not_the_system_clock():
    # 23 March 1994, 12:35:19 UTC - the golden sentence's own date.
    assert gps.parse_timestamp("123519.00", "230394") == 764426119


def test_timestamp_rejects_incomplete_fields():
    assert gps.parse_timestamp("", "230394") is None
    assert gps.parse_timestamp("123519", "") is None
    assert gps.parse_timestamp("123519", "2303") is None
    assert gps.parse_timestamp("xxxxxx", "230394") is None


def test_status_is_driving_only_above_the_threshold():
    assert gps.Fix(valid=True, speed_mph=30.0).status == "D"
    assert gps.Fix(valid=True, speed_mph=0.0).status == "P"
    # A GPS cannot see a charge port, so 'C' must never appear - the dashboard
    # and correlate.py both read this field.
    assert gps.Fix(valid=True, speed_mph=0.0).status != "C"


# ---------------------------------------------------------------------------
# Refusing a solution the receiver has not vouched for
# ---------------------------------------------------------------------------
def test_a_void_fix_yields_no_position():
    # What the receiver actually emits sitting indoors.
    assembler = gps.NmeaAssembler()
    fix = assembler.feed(with_checksum("GNRMC,,V,,,,,,,,,,N,V"))
    assert fix is not None
    assert not fix.valid
    assert fix.lat is None and fix.lon is None


def test_a_void_fix_discards_a_previously_good_position():
    # Some receivers keep echoing the last known position after losing the
    # solution. Keeping it would write a stale location as if it were current -
    # a car reported still sitting where it was an hour ago.
    assembler = gps.NmeaAssembler()
    assembler.feed(with_checksum(
        "GNRMC,123519.00,A,4807.038,N,01131.000,E,022.4,084.4,230394,,,A"
    ))
    assert assembler.fix.lat is not None

    fix = assembler.feed(with_checksum(
        "GNRMC,123529.00,V,4807.038,N,01131.000,E,022.4,084.4,230394,,,N"
    ))
    assert not fix.valid
    assert fix.lat is None, "stale position survived loss of fix"


def test_an_active_status_with_unparseable_coordinates_is_not_a_fix():
    assembler = gps.NmeaAssembler()
    fix = assembler.feed(with_checksum(
        "GNRMC,123519.00,A,,,,,022.4,084.4,230394,,,A"
    ))
    assert not fix.valid


def test_a_sentence_failing_its_checksum_is_ignored_entirely():
    assembler = gps.NmeaAssembler()
    good = with_checksum(
        "GNRMC,123519.00,A,4807.038,N,01131.000,E,022.4,084.4,230394,,,A"
    )
    assert assembler.feed(good).valid

    # Corrupt the payload but leave the old checksum: must not update state.
    assert assembler.feed(good.replace("4807.038", "5807.038")) is None
    assert abs(assembler.fix.lat - 48.1173) < 0.0001, "a corrupt sentence moved us"


# ---------------------------------------------------------------------------
# Satellite metadata
# ---------------------------------------------------------------------------
def test_gga_supplies_satellite_count_and_hdop():
    assembler = gps.NmeaAssembler()
    assert assembler.feed(with_checksum(
        "GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
    )) is None                                  # GGA alone completes nothing
    assert assembler.fix.satellites == 8
    assert abs(assembler.fix.hdop - 0.9) < 0.001


def test_gga_with_no_fix_does_not_invent_satellites():
    assembler = gps.NmeaAssembler()
    assembler.feed(with_checksum("GNGGA,,,,,,0,00,99.99,,,,,,"))
    assert assembler.fix.satellites == 0


# ---------------------------------------------------------------------------
# Schema parity with the Tesla poller
# ---------------------------------------------------------------------------
def test_a_gps_poll_row_matches_the_tesla_poll_row_exactly():
    # Both writers share the `polls` table and poller.attach_drive. If the
    # shapes drift, upsert silently writes a row missing a column and the
    # divergence only shows up as absent data months later.
    tesla = poller.poll_record_from_vehicle_data({
        "drive_state": {
            "latitude": 33.2, "longitude": -117.24, "speed": 42,
            "heading": 90, "shift_state": "D", "timestamp": 1770000000000,
        },
        "charge_state": {"charging_state": "Disconnected"},
        "vehicle_state": {"odometer": 12345.6},
    })
    from_gps = gps.poll_record_from_fix(gps.Fix(
        valid=True, lat=33.2, lon=-117.24, speed_mph=42.0,
        heading=90.0, satellites=9, ts=1770000000,
    ))
    assert set(from_gps) == set(tesla), (
        f"only in gps: {set(from_gps) - set(tesla)}; "
        f"only in tesla: {set(tesla) - set(from_gps)}"
    )


def test_fields_a_gps_cannot_know_are_none_not_zero():
    # None means "no source for this". 0.0 means "the car reported zero", and
    # an odometer of 0.0 would look like a brand new vehicle.
    row = gps.poll_record_from_fix(gps.Fix(valid=True, lat=1.0, lon=2.0))
    for field in ("power", "shift_state", "odometer"):
        assert row[field] is None, field


def test_loc_available_tracks_whether_there_is_a_position():
    assert gps.poll_record_from_fix(
        gps.Fix(valid=True, lat=1.0, lon=2.0))["loc_available"] == 1
    assert gps.poll_record_from_fix(gps.Fix())["loc_available"] == 0


# ---------------------------------------------------------------------------
# The heartbeat
# ---------------------------------------------------------------------------
# Every key the /gps page reads, transcribed from the contract rather than from
# the writer. Written out in full on purpose: generating this list from
# _heartbeat_payload() would make the test agree with whatever the writer does,
# including dropping a key, which is the one thing it exists to catch.
HEARTBEAT_KEYS = (
    "schema", "written_ts", "heartbeat_interval", "pid", "started_ts",
    "device", "device_present", "port_open",
    "last_error", "last_error_ts",
    "sentences_total", "checksum_failures", "last_sentence_ts",
    "motion",
    "fix_valid", "fix_quality", "satellites_used",
    "satellites_in_view", "satellites_tracked",
    "hdop", "pdop", "vdop", "gsa_fix_type", "gsa_used_prns",
    "constellations", "sky",
    "last_fix_ts", "last_fix_lat", "last_fix_lon", "last_fix_speed_mph",
    "last_fix_at", "last_row_written_ts",
    "last_db_error", "last_db_error_ts", "recent_sentences",
)


class OsThatCannotRename:
    """
    Stands in for `os` inside gps.py so that os.replace fails the way a full or
    read-only filesystem fails: after the temporary file exists.
    """

    def __init__(self, real_module):
        self._real_module = real_module

    def __getattr__(self, name):
        return getattr(self._real_module, name)

    def replace(self, *arguments, **keywords):
        raise OSError("simulated: no space left on device")


@contextlib.contextmanager
def heartbeat_sandbox(enabled: bool = True, assembler=None):
    """
    Point gps.py's heartbeat at a throwaway directory and restore everything.

    ensure_dirs is replaced too, because the real one creates ten directories
    under BASE_DIR - running the test suite must not scatter an empty
    deployment across the developer's home directory.
    """
    directory = Path(tempfile.mkdtemp(prefix="gps-heartbeat-"))
    saved_logs_dir = gps.LOGS_DIR
    saved_ensure_dirs = gps.ensure_dirs
    saved_enabled = gps._HEARTBEAT_ENABLED
    saved_state = dict(gps._STATE)

    gps.LOGS_DIR = directory
    gps.ensure_dirs = lambda: directory.mkdir(parents=True, exist_ok=True)
    gps._HEARTBEAT_ENABLED = enabled
    gps._STATE["assembler"] = assembler
    try:
        yield directory
    finally:
        gps.LOGS_DIR = saved_logs_dir
        gps.ensure_dirs = saved_ensure_dirs
        gps._HEARTBEAT_ENABLED = saved_enabled
        gps._STATE.clear()
        gps._STATE.update(saved_state)
        shutil.rmtree(directory, ignore_errors=True)


def test_the_heartbeat_carries_every_key_the_dashboard_reads():
    # The /gps page is written against this exact shape and reads each key
    # unconditionally. A key this writer forgets is not a signal, it is a blank
    # field on the one page someone opens when nothing else is working - and
    # nothing raises, so it is never noticed until someone is standing in a
    # garage trying to find out why there is no position.
    assembler = gps.NmeaAssembler()
    assembler.feed(with_checksum(
        "GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
    ))
    with heartbeat_sandbox(assembler=assembler) as directory:
        gps._write_heartbeat(gps._heartbeat_payload())
        payload = json.loads((directory / "gps.json").read_text())

    missing = [key for key in HEARTBEAT_KEYS if key not in payload]
    assert not missing, f"heartbeat is missing {missing}"
    unexpected = sorted(set(payload) - set(HEARTBEAT_KEYS))
    assert not unexpected, f"heartbeat grew keys the page does not know: {unexpected}"
    assert payload["schema"] == 1, "the reader refuses anything but schema 1"


def test_an_unknown_value_is_null_rather_than_zero_in_the_heartbeat():
    # 0 satellites and "we have never heard from the receiver" render as very
    # different sentences, and the page picks between them on null-ness. A
    # writer that helpfully defaults everything to 0 makes a dead receiver look
    # like a receiver reporting a confident zero.
    with heartbeat_sandbox(assembler=gps.NmeaAssembler()):
        payload = gps._heartbeat_payload()
    for field in ("fix_quality", "hdop", "last_fix_ts", "last_fix_lat",
                  "last_row_written_ts", "last_error", "last_sentence_ts"):
        assert payload[field] is None, field


def test_a_disabled_heartbeat_writes_absolutely_nothing():
    # `python3 src/gps.py --status` is documented as safe to run against a live
    # service. Ungated, the diagnostic tool and the service would race to
    # replace the same file, so the thing you ran to inspect the service would
    # start corrupting the page's view of it.
    with heartbeat_sandbox(enabled=False) as directory:
        gps._write_heartbeat({"schema": 1})
        gps._beat(force=True)
        assert list(directory.iterdir()) == [], "a disabled heartbeat wrote a file"


def test_a_failed_heartbeat_write_leaves_no_partial_file_behind():
    # The temp file is a dot-file in the directory the dashboard lists. One
    # orphan per failed write accumulates until the reader has to guess which
    # file is the real heartbeat, and the failure that produces them - a full
    # SD card - is exactly when nobody wants a second problem.
    with heartbeat_sandbox() as directory:
        gps._write_heartbeat({"schema": 1})
        assert (directory / "gps.json").exists()

        real_os = gps.os
        gps.os = OsThatCannotRename(real_os)
        try:
            gps._write_heartbeat({"schema": 1, "written_ts": 2})
        finally:
            gps.os = real_os

        leftovers = sorted(
            entry.name for entry in directory.iterdir() if entry.name != "gps.json"
        )
        assert leftovers == [], f"partial files survived: {leftovers}"


def test_a_heartbeat_never_carries_a_coordinate():
    # logs/gps.json and logs/gps.log sit on a volume four containers mount, and
    # the dashboard serves both. Unredacted, the debug panel becomes a precise,
    # continuously updating record of where the owner's car has been - while
    # the cable, tty, framing and parser are all provable without one.
    redacted = gps.redact_position(GOLDEN_RMC)
    assert "4807.038" not in redacted and "01131.000" not in redacted
    assert ",N," in redacted and ",E," in redacted, "hemisphere was redacted too"
    assert "123519" in redacted and redacted.endswith("*6A"), "time or checksum lost"
    assert "..." not in gps.redact_position(with_checksum("GNRMC,,V,,,,,,,,,,N,V")), (
        "a blank coordinate was disguised as a redacted one"
    )
    assert "33.2" not in gps.redact_log_line("P 0mph @ 33.20000,-117.24000")


# ---------------------------------------------------------------------------
# Satellite sentences
# ---------------------------------------------------------------------------
def test_satellite_sentences_never_complete_a_cycle():
    # Only RMC completes a fix. If GSA or GSV returned one, main() would write
    # a `polls` row under a fresh id carrying the PREVIOUS epoch's timestamp
    # and position - a silent data-quality regression that the sample-interval
    # gate would rate-limit but never stop.
    assembler = gps.NmeaAssembler()
    assert assembler.feed("$GPGSV,1,1,01,07,72,310,41*4D") is None
    assert assembler.feed("$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,1.3,2.1*39") is None
    assert assembler.fix.valid is False


def test_gga_captures_the_fix_quality_indicator():
    # Satellite count and HDOP cannot tell "searching" from "solving off dead
    # reckoning": both report plausible numbers in either case. The quality
    # indicator is the only field that says which one is happening.
    assembler = gps.NmeaAssembler()
    assembler.feed(with_checksum(
        "GNGGA,123519,4807.038,N,01131.000,E,2,08,0.9,545.4,M,46.9,M,,"
    ))
    assert assembler.fix.quality == 2, "DGPS quality was not captured"

    assembler.feed(with_checksum("GNGGA,,,,,,0,00,99.99,,,,,,"))
    assert assembler.fix.quality == 0, "a reported no-fix must be 0, not None"


def test_a_void_rmc_keeps_the_satellite_diagnostics_from_the_last_gga():
    # A void RMC blanks position, speed and time - and must leave satellites
    # and HDOP alone. Those two are the only evidence the page has that the
    # receiver is working, and indoors EVERY RMC is void, so a branch tidied
    # up to "reset everything" would wipe the diagnostics on every single
    # sentence at exactly the moment someone is reading them.
    assembler = gps.NmeaAssembler()
    assembler.feed(with_checksum(
        "GNGGA,123519,4807.038,N,01131.000,E,1,09,1.4,545.4,M,46.9,M,,"
    ))
    fix = assembler.feed(with_checksum("GNRMC,,V,,,,,,,,,,N,V"))

    assert not fix.valid and fix.lat is None, "the void RMC did not clear position"
    assert fix.satellites == 9, "satellite count was wiped by a void RMC"
    assert abs(fix.hdop - 1.4) < 0.001, "hdop was wiped by a void RMC"
    assert fix.quality == 1, "fix quality was wiped by a void RMC"


def test_the_no_fix_line_does_not_call_used_satellites_visible():
    # "0 satellites visible" indoors, with twelve in view at good SNR, answers
    # "is the antenna dead?" with a confident yes when the truth is the exact
    # opposite. This count is GGA numSV - satellites USED - and the line has to
    # say so.
    assembler = gps.NmeaAssembler()
    assembler.feed("$GPGSV,1,1,01,07,72,310,41*4D")
    assembler.feed(with_checksum("GNRMC,,V,,,,,,,,,,N,V"))

    assert "visible" not in assembler.fix.describe()
    combined = assembler.describe()
    assert "1 in view" in combined and "1 tracked" in combined and "0 used" in combined


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------
# These exist because 31 passing tests did not stop the shipped code marking a
# receiver on a desk as driving on 201 of 202 consecutive samples. Every test
# above asked "did we parse this sentence correctly", and every one was right.
# None asked "does a sequence of correct fixes mean the car is moving", which
# is a question no single sentence can answer.
#
# The numbers below are measured, not chosen: 33 minutes of this receiver
# sitting still produced a peak displacement of 13.9m and speeds up to 3.46mph.

METERS_PER_DEGREE_LAT = 111320.0


def stationary_track(seconds: int, scatter_m: float = 7.0):
    """
    A receiver that is not moving, reported the way one actually reports.

    The walk is deterministic rather than random so a failure is reproducible;
    what matters is the amplitude, which is set from the measured p95 scatter.
    """
    step = scatter_m / METERS_PER_DEGREE_LAT
    offsets = [(0.0, 0.0), (step, -step * 0.6), (-step * 0.7, step * 0.8),
               (step * 0.5, step), (-step, -step * 0.4), (step * 0.3, -step * 0.9)]
    for tick in range(seconds):
        north, east = offsets[tick % len(offsets)]
        yield float(tick), 33.1696 + north, -117.2259 + east


def driving_track(speeds_mph, scatter_m: float = 7.0):
    """The same receiver, on a car travelling north at the given speeds."""
    step = scatter_m / METERS_PER_DEGREE_LAT
    offsets = [(0.0, 0.0), (step, -step * 0.6), (-step * 0.7, step * 0.8),
               (step * 0.5, step), (-step, -step * 0.4)]
    latitude = 33.1696
    for tick, mph in enumerate(speeds_mph):
        latitude += (mph * 0.44704) / METERS_PER_DEGREE_LAT
        north, east = offsets[tick % len(offsets)]
        yield float(tick), latitude + north, -117.2259 + east


def classify(track) -> list[str]:
    gate = gps.MotionGate()
    return [gate.update(lat, lon, now) for now, lat, lon in track]


def test_a_stationary_receiver_is_never_driving():
    # The whole bug, in one assertion. The shipped speed threshold called this
    # 'D' 96% of the time and fabricated 23.6 miles a day from a desk.
    verdicts = classify(stationary_track(600))
    assert "D" not in verdicts, f"{verdicts.count('D')} of {len(verdicts)} samples read as driving"


def test_a_parking_lot_crawl_still_registers_as_driving():
    # The failure mode a naive threshold-bump would introduce. 5mph is slower
    # than a brisk cyclist and it must still count, because stop-and-go is
    # exactly when you want to know who is behind you.
    verdicts = classify(driving_track([5.0] * 200))
    assert "D" in verdicts, "a 5mph drive never registered"
    assert verdicts.index("D") < 40, f"took {verdicts.index('D')}s to notice a 5mph drive"


def test_detection_gets_faster_as_the_car_goes_faster():
    # Latency is the time taken to cover GPS_MOTION_ENTER_METERS, so this
    # ordering is a property of the design, not a coincidence to be tuned.
    crawl = classify(driving_track([5.0] * 120)).index("D")
    town = classify(driving_track([25.0] * 120)).index("D")
    highway = classify(driving_track([65.0] * 120)).index("D")
    assert highway < town < crawl, (highway, town, crawl)
    assert highway <= 6, f"highway driving took {highway}s to detect"


def test_a_long_red_light_does_not_flip_the_car_to_parked():
    # Without the exit hysteresis every light drops the write cadence from
    # GPS_MOVING_SAMPLE_SECONDS to GPS_PARKED_SAMPLE_SECONDS, and the track
    # goes sparse in traffic - where it is worth the most.
    verdicts = classify(driving_track(([20.0] * 30 + [0.0] * 70) * 3))
    assert "D" in verdicts
    after_first = verdicts[verdicts.index("D"):]
    assert "P" not in after_first, "flipped to parked while stopped mid-drive"


def test_a_car_parked_for_good_does_eventually_read_as_parked():
    # The other side of that hysteresis: it must not latch on forever, or a
    # parked car keeps writing at the moving cadence all night.
    verdicts = classify(driving_track([20.0] * 30 + [0.0] * 400))
    assert verdicts[-1] == "P", "never returned to parked after the drive ended"


def test_a_replug_does_not_invent_a_drive():
    # A receiver that vanishes and comes back somewhere else would otherwise
    # show one enormous displacement and open a journey that never happened.
    gate = gps.MotionGate()
    for now, lat, lon in stationary_track(120):
        gate.update(lat, lon, now)
    gate.reset()
    # Comes back 3km away, as if the car were driven while the port was down.
    verdicts = [gate.update(33.1696 + 0.027, -117.2259, 200.0 + tick)
                for tick in range(10)]
    assert "D" not in verdicts, "a replug opened a drive"


def test_the_gate_says_parked_until_it_has_enough_samples():
    # A false 'P' loses a few seconds off the front of a drive. A false 'D'
    # invents a journey, and distinct_drives is what correlate.py leans on.
    gate = gps.MotionGate()
    assert gate.update(33.1696, -117.2259, 0.0) == "P"
    assert gate.update(33.2000, -117.2259, 1.0) == "P", "opened a drive on two samples"


def test_the_verdict_beats_the_fixs_own_speed_reading():
    # poll_record_from_fix must write what the gate concluded, not what one
    # noisy sample happened to report.
    fix = gps.Fix(valid=True, lat=33.1696, lon=-117.2259, speed_mph=2.5)
    assert fix.status == "D", "fixture no longer exercises the disagreement"
    assert gps.poll_record_from_fix(fix, status="P")["status"] == "P"
    assert gps.poll_record_from_fix(fix)["status"] == "D"


def test_path_ratio_is_reported_but_never_decides():
    # Kept for the /gps page. It is not a decision variable: over a window
    # short enough to be useful it reads as low as 1.4 while stationary, which
    # is inside the range real driving occupies.
    gate = gps.MotionGate()
    assert gate.path_ratio is None, "claimed a ratio with no displacement"
    for now, lat, lon in stationary_track(120):
        gate.update(lat, lon, now)
    snapshot = gate.snapshot()
    assert snapshot["driving"] is False
    assert snapshot["displacement_m"] < gate.enter_meters


def test_the_heartbeat_carries_the_motion_block():
    # /gps renders this. A missing key is a blank field on the page someone
    # opens precisely when nothing else is working.
    payload = gps._heartbeat_payload()
    assert "motion" in payload, sorted(payload)
    for key in ("driving", "samples", "displacement_m", "path_ratio", "enter_m"):
        assert key in payload["motion"], sorted(payload["motion"])


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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
