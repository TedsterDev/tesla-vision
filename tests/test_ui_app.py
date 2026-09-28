"""
test_ui_app.py

Behavioural tests for the dashboard's two location pages.

The dangerous failure here is not a 500. It is a sentence: a page that is up,
fast, well laid out, and confidently wrong about where the car is or about
whether anything is still watching it. "Car is asleep" printed over a row that
says 61 mph in drive is worse than an error page, because an error page sends
the reader to look for themselves.

So almost everything below asserts about words - which sentence a state
produces, and which sentences a state may never produce - rather than about
status codes. The two exceptions are the fail-closed gate on the location
routes, which is about who may read a position at all, and the cache and
referrer headers, which are about who else ends up holding one.

Run:  python3 tests/test_ui_app.py
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# BASE_DIR decides where gps.json and the database live and src.common reads it
# at import time, so it has to be set before anything under src/ is imported.
SANDBOX = Path(tempfile.mkdtemp(prefix="ui_app_test_"))
os.environ["BASE_DIR"] = str(SANDBOX)
os.environ["DB_PATH"] = str(SANDBOX / "scout.db")
os.environ.setdefault("MODELS_DIR", str(SANDBOX / "models"))

# ensure_dirs() before ui_app is imported: the app mounts MEDIA_DIR at import
# time and StaticFiles refuses to start on a directory that does not exist.
from src.common import LOGS_DIR, ensure_dirs              # noqa: E402

ensure_dirs()

from src import cfaccess                                  # noqa: E402
from src import ui_app                                    # noqa: E402

NOW = int(time.time())
HEARTBEAT_PATH = LOGS_DIR / "gps.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def write_heartbeat(text: str | None = None, **fields) -> None:
    """Put a gps.json in place. `text` writes raw bytes, for the corrupt cases."""
    if text is not None:
        HEARTBEAT_PATH.write_text(text, encoding="utf-8")
        return
    payload = {
        "schema": 1,
        "written_ts": NOW,
        "heartbeat_interval": 5,
        "pid": 1234,
        "started_ts": NOW - 3600,
        "device": "/dev/ttyACM0",
        "device_present": True,
        "port_open": True,
        "fix_valid": True,
        "sentences_total": 90000,
        "checksum_failures": 0,
        "last_sentence_ts": NOW,
        "last_fix_at": NOW,
        "last_row_written_ts": NOW,
        "satellites_in_view": 11,
        "satellites_tracked": 9,
        "satellites_used": 8,
        "hdop": 0.9,
    }
    payload.update(fields)
    HEARTBEAT_PATH.write_text(json.dumps(payload), encoding="utf-8")


def remove_heartbeat() -> None:
    if HEARTBEAT_PATH.exists():
        HEARTBEAT_PATH.unlink()


def set_only_poll(**fields) -> dict:
    """Replace the polls table with exactly one row, and return it."""
    row = {
        "id": "testpoll0001", "ts": NOW - 20, "lat": 37.7749, "lon": -122.4194,
        "heading": 90.0, "speed": 0.0, "power": None, "shift_state": None,
        "status": "P", "loc_available": 1, "odometer": None, "street": None,
        "city": None, "drive_id": None, "geocode_id": None,
    }
    row.update(fields)
    connection = ui_app.get_connection()
    try:
        connection.execute("DELETE FROM polls")
        connection.execute(
            "INSERT INTO polls (id, ts, lat, lon, heading, speed, power, shift_state, "
            "status, loc_available, odometer, street, city, drive_id, geocode_id) "
            "VALUES (:id, :ts, :lat, :lon, :heading, :speed, :power, :shift_state, "
            ":status, :loc_available, :odometer, :street, :city, :drive_id, :geocode_id)",
            row,
        )
        connection.commit()
    finally:
        connection.close()
    return row


def verdict_for(*, position, band, motion, source, service_state,
                fix_valid=True, age_seconds=20, gps_state=None, heartbeat=None,
                tesla_configured=True) -> dict:
    """Call the verdict builder the way _findmy_state calls it."""
    payload = None
    if position is not None:
        payload = {
            "lat": position.get("lat"), "lon": position.get("lon"),
            "ts": position.get("ts"), "ts_display": "2026-09-04 09:13:00",
            "ts_clock": "Fri 9:13 AM", "age_seconds": age_seconds,
            "heading": position.get("heading"), "speed_mph": position.get("speed"),
            "status": position.get("status"), "shift_state": position.get("shift_state"),
            "source": source,
        }
    return ui_app._findmy_verdict(
        position=position, position_payload=payload, band=band,
        age_seconds=age_seconds, motion=motion, source=source,
        gps_state=gps_state if gps_state is not None else {"service_state": service_state},
        heartbeat=heartbeat or {}, service_state=service_state, fix_valid=fix_valid,
        tesla_configured=tesla_configured, server_ts=NOW,
    )


def all_prose(verdict: dict) -> str:
    return " ".join(str(value) for value in verdict.values()).lower()


def render(path: str, headers: dict | None = None) -> tuple[int, dict, str]:
    """Drive the real ASGI stack, so the middleware is exercised, not bypassed."""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "server": ("testserver", 80),
        "client": ("10.0.0.5", 51234),
        "headers": [(name.lower().encode(), value.encode())
                    for name, value in (headers or {}).items()],
    }
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(ui_app.app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    response_headers = {name.decode().lower(): value.decode() for name, value in start["headers"]}
    return start["status"], response_headers, body.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# The car is never described as asleep on evidence nobody collected
# ---------------------------------------------------------------------------
def test_a_car_being_driven_away_is_never_described_as_asleep():
    # A Tesla-only deployment: no gps.json has ever been written. One poll,
    # twenty seconds old, status D at 61 mph. This rendered "Car is asleep.
    # Position as of Fri 9:13 AM." above a motion line reading "Moving".
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 20, "status": "D",
                "shift_state": "D", "speed": 61.0, "heading": 270.0}
    verdict = verdict_for(position=position, band="live", motion="moving",
                          source="tesla", service_state="never_installed")
    assert verdict["position_state"] == "healthy_moving", verdict["position_state"]
    assert "asleep" not in all_prose(verdict), verdict["headline"]
    assert "61 mph" in verdict["headline"], verdict["headline"]


def test_a_charging_car_is_never_described_as_asleep():
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 20, "status": "C",
                "shift_state": None, "speed": 0.0, "heading": None}
    verdict = verdict_for(position=position, band="live", motion="charging",
                          source="tesla", service_state="never_installed")
    assert "asleep" not in all_prose(verdict), verdict["headline"]
    assert "charging" in all_prose(verdict), verdict["note"]


def test_no_state_this_page_can_reach_asserts_the_car_is_asleep():
    # Nothing in this module reads a vehicle_state or a poller heartbeat, so
    # "asleep" is an assertion no input can support. Sweep the matrix.
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 900, "status": "P",
                "shift_state": None, "speed": 0.0, "heading": None}
    for service_state in ("never_installed", "live", "uncertain", "stopped",
                          "unknown", "unknown_version"):
        for band in ("clock_skew", "live", "recent", "stale", "unreliable"):
            for motion in ("moving", "charging", "parked", "stationary", "unknown"):
                for source in ("tesla", "gps", "unknown"):
                    verdict = verdict_for(
                        position=position, band=band, motion=motion, source=source,
                        service_state=service_state,
                        age_seconds=-3600 if band == "clock_skew" else 900,
                    )
                    assert "asleep" not in all_prose(verdict), (
                        service_state, band, motion, source, verdict["headline"])


def test_a_stale_tesla_position_with_no_heartbeat_says_what_is_actually_known():
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 7200, "status": "P",
                "shift_state": "P", "speed": 0.0, "heading": None}
    verdict = verdict_for(position=position, band="stale", motion="parked",
                          source="tesla", service_state="never_installed",
                          age_seconds=7200)
    assert verdict["position_state"] == "gps_never_ran"
    assert "GPS logging is not running" in verdict["headline"], verdict["headline"]
    assert "Tesla API" in verdict["headline"], verdict["headline"]
    assert "Fri 9:13 AM" in verdict["headline"], verdict["headline"]


def test_a_moving_car_outranks_a_writer_that_is_down():
    # motion "moving" needs a live band, so the position is the freshest
    # evidence on the page. Printing "last known position, not a current one"
    # over the word "Moving" is a contradiction in one card.
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 20, "status": "D",
                "shift_state": "D", "speed": 44.0, "heading": 10.0}
    verdict = verdict_for(position=position, band="live", motion="moving",
                          source="tesla", service_state="stopped",
                          heartbeat={"written_ts": NOW - 900})
    assert verdict["position_state"] == "healthy_moving"
    assert "not a current one" not in all_prose(verdict)
    assert "gps logging is not running" in all_prose(verdict), verdict["note"]


# ---------------------------------------------------------------------------
# A restart is not an outage
# ---------------------------------------------------------------------------
def test_a_restart_sized_gap_is_amber_and_says_restart_on_the_gps_page():
    write_heartbeat(written_ts=NOW - 30)
    verdict = ui_app._gps_verdict(ui_app._read_gps_heartbeat())
    assert verdict["level"] == "amber", verdict
    assert "not running" not in verdict["headline"].lower(), verdict["headline"]
    assert "restart" in verdict["headline"].lower(), verdict["headline"]


def test_a_real_outage_is_still_red_on_the_gps_page():
    write_heartbeat(written_ts=NOW - 90)
    verdict = ui_app._gps_verdict(ui_app._read_gps_heartbeat())
    assert verdict["level"] == "red", verdict
    assert "GPS service is not running" in verdict["headline"], verdict["headline"]


def test_a_restart_sized_gap_does_not_call_the_position_a_last_known_one():
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 900, "status": "P",
                "shift_state": None, "speed": 0.0, "heading": None}
    verdict = verdict_for(position=position, band="recent", motion="stationary",
                          source="gps", service_state="uncertain", age_seconds=900,
                          heartbeat={"written_ts": NOW - 30})
    assert verdict["position_state"] == "writer_restarting", verdict
    assert verdict["level"] == "amber", verdict
    assert "not a current one" not in all_prose(verdict), verdict["headline"]


def test_a_stopped_writer_is_still_red_on_find_my():
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 900, "status": "P",
                "shift_state": None, "speed": 0.0, "heading": None}
    verdict = verdict_for(position=position, band="recent", motion="stationary",
                          source="gps", service_state="stopped", age_seconds=900,
                          heartbeat={"written_ts": NOW - 900})
    assert verdict["position_state"] == "writer_down", verdict
    assert verdict["level"] == "red", verdict


# ---------------------------------------------------------------------------
# A heartbeat that cannot be read is not a healthy parked car
# ---------------------------------------------------------------------------
def test_an_unreadable_heartbeat_is_not_reported_as_a_parked_car():
    write_heartbeat(text="{not json")
    gps_state = ui_app._read_gps_heartbeat()
    assert gps_state["service_state"] == "unknown"
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 10800, "status": "P",
                "shift_state": None, "speed": 0.0, "heading": None}
    verdict = verdict_for(position=position, band="stale", motion="stationary",
                          source="unknown", service_state="unknown", age_seconds=10800,
                          gps_state=gps_state)
    assert verdict["position_state"] == "writer_unreadable", verdict
    assert "This is where your car was." != verdict["headline"], verdict


def test_a_wrong_schema_heartbeat_is_not_reported_as_a_parked_car():
    write_heartbeat(schema=2)
    gps_state = ui_app._read_gps_heartbeat()
    assert gps_state["service_state"] == "unknown_version"
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 10800, "status": "P",
                "shift_state": None, "speed": 0.0, "heading": None}
    verdict = verdict_for(position=position, band="stale", motion="stationary",
                          source="unknown", service_state="unknown_version",
                          age_seconds=10800, gps_state=gps_state,
                          heartbeat={"schema": 2})
    assert verdict["position_state"] == "writer_unknown_version", verdict
    assert "This is where your car was." != verdict["headline"], verdict


# ---------------------------------------------------------------------------
# Which process wrote the row
# ---------------------------------------------------------------------------
def test_a_source_is_never_claimed_from_a_heartbeat_that_did_not_parse():
    for broken in ({"exists": True, "service_state": "unknown", "error": "boom"},
                   {"exists": True, "service_state": "unknown_version"}):
        writer = ui_app._position_writer(
            {"status": "P", "shift_state": None, "odometer": None, "power": None}, broken
        )
        assert writer == "unknown", (broken, writer)


def test_a_parked_tesla_row_is_not_mistaken_for_a_gps_row():
    # Tesla reports shift_state null while parked, and a response with no
    # vehicle_state carries no odometer either - but drive_state.power is there.
    writer = ui_app._position_writer(
        {"status": "P", "shift_state": None, "odometer": None, "power": 0.0},
        {"exists": True, "service_state": "live"},
    )
    assert writer == "tesla", writer


def test_a_charging_row_can_only_have_come_from_tesla():
    writer = ui_app._position_writer(
        {"status": "C", "shift_state": None, "odometer": None, "power": None},
        {"exists": True, "service_state": "live"},
    )
    assert writer == "tesla", writer


def test_a_gps_row_is_named_gps_only_when_a_heartbeat_vouches_for_it():
    gps_row = {"status": "P", "shift_state": None, "odometer": None, "power": None}
    assert ui_app._position_writer(gps_row, {"service_state": "live"}) == "gps"
    assert ui_app._position_writer(gps_row, {"service_state": "never_installed"}) == "unknown"


# ---------------------------------------------------------------------------
# Staleness bands
# ---------------------------------------------------------------------------
def test_a_future_timestamp_gets_a_verdict_of_its_own():
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW + 3600, "status": "P",
                "shift_state": None, "speed": 0.0, "heading": None}
    verdict = verdict_for(position=position, band="clock_skew", motion="stationary",
                          source="gps", service_state="live", age_seconds=-3599)
    assert verdict["position_state"] == "clock_skew", verdict
    assert "future" in verdict["headline"].lower(), verdict["headline"]
    assert "where your car was" not in all_prose(verdict), verdict["headline"]


def test_the_clock_skew_sentence_reads_forwards_not_backwards():
    template = ui_app._findmy_band_template("clock_skew", NOW + 3600, True)
    assert "{age} in the future" in template, template
    sentence = ui_app._findmy_band_sentence("clock_skew", -3599, NOW + 3600, True)
    assert "59 minutes in the future" in sentence, sentence
    assert "ago" not in sentence, sentence


def test_an_empty_database_does_not_render_a_sentence_with_a_hole_in_it():
    sentence = ui_app._findmy_band_sentence("unreliable", None, None, False)
    assert sentence == "No position has ever been recorded.", sentence
    assert "  " not in sentence, sentence
    assert "an unknown time" not in sentence, sentence


def test_every_band_template_leaves_exactly_one_hole_for_the_age():
    for band in ("clock_skew", "live", "recent", "stale", "unreliable"):
        template = ui_app._findmy_band_template(band, NOW - 600, True)
        assert template.count("{age}") == 1, (band, template)
        assert "{age}" not in ui_app._findmy_band_sentence(band, 600, NOW - 600, True)


def test_the_band_colour_follows_the_age_and_not_the_verdict():
    assert ui_app._findmy_band_level("live") == "green"
    assert ui_app._findmy_band_level("stale") == "amber"
    assert ui_app._findmy_band_level("unreliable") == "grey"
    # A twelve-hour-old position under a green verdict: the band stays amber.
    position = {"lat": 37.7, "lon": -122.4, "ts": NOW - 40000, "status": "P",
                "shift_state": None, "speed": 0.0, "heading": None}
    verdict = verdict_for(position=position, band="stale", motion="parked",
                          source="tesla", service_state="live", age_seconds=40000)
    assert verdict["level"] == ui_app._findmy_band_level("stale")


# ---------------------------------------------------------------------------
# The ticking age
# ---------------------------------------------------------------------------
def test_the_gps_page_hands_the_browser_its_age_as_a_number():
    write_heartbeat(written_ts=NOW - 20)
    page = ui_app.gps_view().body.decode("utf-8")
    assert 'id="gps-age" data-server-age="' in page, "no seed for the ticker"
    seeded = page.split('data-server-age="')[1].split('"')[0]
    assert seeded.isdigit() and int(seeded) >= 20, seeded


def test_the_find_my_page_hands_the_browser_everything_its_ticker_needs():
    remove_heartbeat()
    set_only_poll(ts=NOW - 20, status="P", shift_state="P", power=0.0)
    page = ui_app.findmy_view().body.decode("utf-8")
    seed = json.loads(page.split('id="findmy-seed">')[1].split("</script>")[0])
    assert isinstance(seed["age_seconds"], int), seed
    assert seed["band_ticker"]["template"].count("{age}") == 1, seed
    assert "expired_template" in seed["band_ticker"], seed
    assert seed["band_ticker"]["valid_until_age"] == ui_app.FINDMY_LIVE_SECONDS, seed


def test_the_ticking_age_keeps_its_sign_on_both_pages():
    # Math.abs() over the whole age is what let a clock_skew age of -3599 count
    # DOWN to "0 seconds ago" and arrive at "just now".
    for page in (ui_app.gps_view().body.decode("utf-8"),
                 ui_app.findmy_view().body.decode("utf-8")):
        assert "Math.abs(Math.floor(seconds))" not in page
        assert "agedPhrase" in page
        assert "in the future" in page


def test_the_find_my_band_is_re_rendered_by_the_ticker_not_written_once():
    page = ui_app.findmy_view().body.decode("utf-8")
    assert "function paintBand(" in page
    assert "paintBand(age)" in page
    assert "valid_until_age" in page


# ---------------------------------------------------------------------------
# The silent-port verdict
# ---------------------------------------------------------------------------
def test_a_service_that_has_just_started_is_not_told_to_replug_the_cable():
    write_heartbeat(written_ts=NOW, started_ts=NOW - 2, port_open=True,
                    last_sentence_ts=None, fix_valid=False, sentences_total=0,
                    satellites_in_view=0, satellites_tracked=0, satellites_used=0)
    verdict = ui_app._gps_verdict(ui_app._read_gps_heartbeat())
    assert "Replug" not in verdict["action"], verdict


def test_a_receiver_silent_for_minutes_is_still_told_to_replug_the_cable():
    write_heartbeat(written_ts=NOW, started_ts=NOW - 600, port_open=True,
                    last_sentence_ts=None, fix_valid=False, sentences_total=0)
    verdict = ui_app._gps_verdict(ui_app._read_gps_heartbeat())
    assert "sent nothing" in verdict["headline"], verdict["headline"]
    assert "Replug" in verdict["action"], verdict


# ---------------------------------------------------------------------------
# The location routes fail closed
# ---------------------------------------------------------------------------
def with_auth_settings(dashboard_pass, access_configured, access_claims):
    """Swap the module's auth configuration for one test, then put it back."""
    saved = (ui_app.DASHBOARD_PASS, cfaccess.is_configured, cfaccess.verify)
    ui_app.DASHBOARD_PASS = dashboard_pass
    cfaccess.is_configured = lambda: access_configured
    cfaccess.verify = lambda token: access_claims
    return saved


def restore_auth_settings(saved):
    ui_app.DASHBOARD_PASS, cfaccess.is_configured, cfaccess.verify = saved


def test_a_location_route_refuses_when_access_is_configured_but_nobody_proved_it():
    # DASHBOARD_PASS='' with CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD both set
    # used to serve /gps, /findmy, /api/gps and /api/findmy with 200 and no
    # credential at all: the anonymous early return fired before the JWT check.
    saved = with_auth_settings("", True, None)
    try:
        for path in ("/gps", "/findmy", "/api/gps", "/api/findmy"):
            status, _, body = render(path)
            assert status == 503, (path, status)
            assert "no authentication" in body, (path, body[:120])
    finally:
        restore_auth_settings(saved)


def test_a_location_route_serves_a_verified_access_identity_with_no_password():
    saved = with_auth_settings("", True, {"email": "owner@example.com"})
    try:
        status, _, _ = render("/api/gps", {"cf-access-jwt-assertion": "pretend-token"})
        assert status == 200, status
    finally:
        restore_auth_settings(saved)


def test_the_rest_of_the_dashboard_stays_open_when_no_password_is_set():
    # The narrow blast radius is deliberate - see the comment on the gate.
    saved = with_auth_settings("", True, None)
    try:
        status, _, _ = render("/healthz")
        assert status == 200, status
        status, _, _ = render("/settings")
        assert status == 200, status
    finally:
        restore_auth_settings(saved)


def test_a_location_route_serves_when_a_password_is_configured():
    saved = with_auth_settings("hunter2", False, None)
    ui_app.DASHBOARD_USER = "owner"
    try:
        import base64
        token = base64.b64encode(b"owner:hunter2").decode()
        status, _, _ = render("/api/gps", {"authorization": f"Basic {token}"})
        assert status == 200, status
        status, _, _ = render("/api/gps")
        assert status == 401, status
    finally:
        restore_auth_settings(saved)


# ---------------------------------------------------------------------------
# What leaves the building
# ---------------------------------------------------------------------------
def test_the_timeline_page_is_uncacheable_and_referrer_free():
    # It draws the same vehicle positions /findmy pins, as a track.
    status, headers, _ = render("/timeline")
    assert status == 200, status
    assert headers.get("cache-control") == "no-store, private", headers
    assert headers.get("referrer-policy") == "no-referrer", headers


def test_the_location_pages_are_uncacheable_and_referrer_free():
    # Authenticated, because unauthenticated they do not serve a position at
    # all - which is the subject of the fail-closed tests above.
    import base64
    saved = with_auth_settings("hunter2", False, None)
    ui_app.DASHBOARD_USER = "owner"
    token = base64.b64encode(b"owner:hunter2").decode()
    try:
        for path in ("/gps", "/findmy", "/api/gps", "/api/findmy"):
            status, headers, _ = render(path, {"authorization": f"Basic {token}"})
            assert status == 200, (path, status)
            assert headers.get("cache-control") == "no-store, private", (path, headers)
            assert headers.get("referrer-policy") == "no-referrer", (path, headers)
    finally:
        restore_auth_settings(saved)


# ---------------------------------------------------------------------------
# The map, and going stale
# ---------------------------------------------------------------------------
def test_the_map_is_loaded_from_somewhere_that_exists():
    page = ui_app.findmy_view().body.decode("utf-8")
    head = ui_app._findmy_map_head()
    assert "/media/leaflet/leaflet.js" not in head or (
        (ui_app.MEDIA_DIR / "leaflet" / "leaflet.js").exists()), head
    assert "leaflet.js" in head, head
    assert "__leafletSource" in head, head
    assert "__leafletSource" in page, "the failure message cannot name its source"


def test_going_stale_demotes_the_map_links_and_the_pin():
    page = ui_app.findmy_view().body.decode("utf-8")
    assert ".stale .maplinks a" in page, "stale links keep full-strength styling"
    assert ".stale #map" in page, "a stale pin keeps full-opacity styling"
    assert 'links.className = "maplinks demoted"' in page
    assert "band !== \"unreliable\" && !isStale" in page, "no confirmation on a stale link"


def test_the_crash_loop_banner_needs_more_than_one_restart_and_can_clear():
    page = ui_app.gps_view().body.decode("utf-8")
    assert "CRASH_MIN_RESTARTS = 2" in page.replace("CRASH_WINDOW_MS = 300000, ", "")
    assert "identities.shift()" in page, "the window never slides"
    assert "crash.hidden = true" in page, "the banner can never clear"


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
    shutil.rmtree(SANDBOX, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
