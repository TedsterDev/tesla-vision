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
    head = ui_app.leaflet_head()
    assert "/media/leaflet/leaflet.js" not in head or (
        (ui_app.MEDIA_DIR / "leaflet" / "leaflet.js").exists()), head
    assert "leaflet.js" in head, head
    assert "__leafletSource" in head, head
    assert "__leafletSource" in page, "the failure message cannot name its source"


def test_the_timeline_prefers_the_local_leaflet_and_loads_it_synchronously():
    leaflet = ui_app.MEDIA_DIR / "leaflet"
    leaflet.mkdir(parents=True, exist_ok=True)
    (leaflet / "leaflet.js").write_text("")
    (leaflet / "leaflet.css").write_text("")
    try:
        _, _, page = render("/timeline")
        assert "/media/leaflet/leaflet.js" in page and "unpkg.com" not in page, page[:400]
        # Its inline map script needs L at once: no defer.
        assert "defer src=" not in page.split("leaflet.js")[0].rsplit("<script", 1)[-1] + "leaflet.js"
    finally:
        shutil.rmtree(leaflet)


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
# Served under a path prefix (azula.tedebyte.dev/tesla/)
# ---------------------------------------------------------------------------
def with_url_root(root):
    saved = ui_app.URL_ROOT
    ui_app.URL_ROOT = root
    return saved


def test_prefixed_and_bare_paths_reach_the_same_route():
    saved = with_url_root("/tesla")
    try:
        for path in ("/tesla/healthz", "/healthz", "/tesla/settings", "/settings"):
            status, _, _ = render(path)
            assert status == 200, (path, status)
    finally:
        ui_app.URL_ROOT = saved


def test_the_bare_prefix_redirects_to_its_trailing_slash():
    saved = with_url_root("/tesla")
    try:
        status, headers, _ = render("/tesla")
        assert status == 308, status
        assert headers.get("location") == "/tesla/", headers
        # A segment boundary, not a string prefix: /teslafoo is not ours.
        status, _, _ = render("/teslafoo")
        assert status == 404, status
    finally:
        ui_app.URL_ROOT = saved


def test_the_location_gate_still_sees_a_prefixed_location_route():
    # The prefix is stripped OUTSIDE the auth middleware. Were it the other way
    # round, /tesla/findmy would not match LOCATION_ROUTES and would serve.
    saved_root = with_url_root("/tesla")
    saved_auth = with_auth_settings("", True, None)
    try:
        for path in ("/tesla/gps", "/tesla/findmy", "/tesla/api/gps", "/tesla/api/findmy"):
            status, _, _ = render(path)
            assert status == 503, (path, status)
    finally:
        restore_auth_settings(saved_auth)
        ui_app.URL_ROOT = saved_root


def test_every_page_link_carries_the_prefix():
    import re
    saved = with_url_root("/tesla")
    try:
        for path in ("/", "/findings", "/plates", "/faces", "/timeline", "/natix", "/settings"):
            status, _, body = render("/tesla" + path)
            assert status == 200, (path, status)
            assert 'var URL_ROOT = "/tesla";' in body, path
            # Any root-relative href/src that skipped the prefix. "//" is a
            # scheme-relative URL to another host, which is not ours to fix.
            stray = re.findall(r'(?:href|src)="(/(?!/)(?!tesla(?:/|"))[^"]*)"', body)
            assert not stray, (path, stray[:5])
    finally:
        ui_app.URL_ROOT = saved


# ---------------------------------------------------------------------------
# Orin box footer
# ---------------------------------------------------------------------------
def test_every_page_carries_the_orin_box_footer():
    for path in ("/", "/findings", "/plates", "/faces", "/timeline", "/natix", "/settings"):
        status, _, body = render(path)
        assert status == 200, (path, status)
        assert 'id="sysbar"' in body, path
        assert '/api/system' in body, path


def test_the_system_endpoint_reports_the_box_and_not_nonsense():
    if not Path("/proc/meminfo").is_file():
        print("  skip  (no /proc here: system stats are read from the Linux kernel)")
        return
    status, _, body = render("/api/system")
    assert status == 200, status
    stats = json.loads(body)
    for key in ("mem_used", "mem_total", "disk_used", "disk_total", "cpu_percent"):
        assert stats[key] is not None, (key, stats)
    assert 0 < stats["mem_used"] <= stats["mem_total"], stats
    assert 0 < stats["disk_used"] <= stats["disk_total"], stats
    assert 0 <= stats["cpu_percent"] <= 100, stats


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------
def test_every_page_carries_the_pipeline_row():
    status, _, body = render("/plates")
    assert status == 200, status
    assert 'id="pipebar"' in body and "/api/pipeline" in body


def test_the_recent_page_has_the_queue_card():
    status, _, body = render("/")
    assert status == 200, status
    assert 'id="pipeline-card"' in body and 'id="pipeline-next"' in body


def test_no_processor_heartbeat_reads_as_unknown_not_idle():
    (LOGS_DIR / "processor.json").unlink(missing_ok=True)
    status, _, body = render("/api/pipeline")
    assert status == 200, status
    assert json.loads(body)["state"] == "unknown", body


def test_a_silent_idle_processor_reads_as_down():
    (LOGS_DIR / "processor.json").write_text(json.dumps({
        "ts": NOW - 600, "current": None, "device": "cuda", "seconds_by_camera": {},
    }))
    try:
        state = json.loads(render("/api/pipeline")[2])["state"]
        assert state == "down", state
    finally:
        (LOGS_DIR / "processor.json").unlink()


def test_a_long_clip_reads_as_processing_until_it_is_stuck():
    from src.common import INBOX_DIR
    for name in ("2026-02-16_21-31-48-front.mp4", "2026-02-16_21-31-48-back.mp4"):
        (INBOX_DIR / name).write_bytes(b"")
    current = {"name": "2026-02-16_21-31-48-front.mp4", "camera": "front",
               "captured_ts": NOW - 3600}
    try:
        # A clip 40 s in with no heartbeat since it began is normal.
        (LOGS_DIR / "processor.json").write_text(json.dumps({
            "ts": NOW - 40, "current": dict(current, started_ts=NOW - 40),
            "seconds_by_camera": {"back": 5.0, "front": 12.0},
        }))
        result = json.loads(render("/api/pipeline")[2])
        assert result["state"] == "processing", result
        # The clip being processed is not also counted as waiting.
        assert result["queue"]["depth"] == 1, result["queue"]
        assert result["eta_seconds"] == 5.0, result["eta_seconds"]

        (LOGS_DIR / "processor.json").write_text(json.dumps({
            "ts": NOW - 1000, "current": dict(current, started_ts=NOW - 1000),
            "seconds_by_camera": {},
        }))
        assert json.loads(render("/api/pipeline")[2])["state"] == "stuck"
    finally:
        (LOGS_DIR / "processor.json").unlink(missing_ok=True)
        for clip in INBOX_DIR.glob("*.mp4"):
            clip.unlink()


def post_json(path: str, payload, content_type: str = "application/json",
              raw: bytes | None = None) -> tuple[int, dict]:
    body = raw if raw is not None else json.dumps(payload).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "server": ("testserver", 80),
        "client": ("10.0.0.5", 51234),
        "headers": [(b"content-type", content_type.encode()),
                    (b"content-length", str(len(body)).encode())],
    }
    messages: list[dict] = []
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(ui_app.app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], json.loads(data or b"{}")


def test_the_demo_endpoint_refuses_anything_but_json():
    status, _ = post_json("/api/demo", {"action": "start", "mode": "realtime"}, "text/plain")
    assert status == 415, status
    status, _ = post_json("/api/demo", {"action": "explode"})
    assert status == 400, status
    status, _ = post_json("/api/demo", {"action": "start", "mode": "sideways"})
    assert status == 400, status


def test_the_demo_endpoint_leaves_the_processor_a_request():
    from src.common import BASE_DIR, JOBS_DIR
    stray = BASE_DIR / "natix-stray" / "x" / "TeslaCam" / "SentryClips" / "2026-02-16_21-31-48"
    request = JOBS_DIR / "demo.json"
    try:
        status, answer = post_json("/api/demo", {"action": "start", "mode": "backlog"})
        assert status == 409, (status, answer)          # nothing to replay yet
        stray.mkdir(parents=True)
        (stray / "2026-02-16_21-31-48-front.mp4").write_bytes(b"")
        status, answer = post_json("/api/demo", {"action": "start", "mode": "backlog"})
        assert status == 200 and answer["ok"], (status, answer)
        assert json.loads(request.read_text())["mode"] == "backlog"
        status, _ = post_json("/api/demo", {"action": "stop"})
        assert status == 200 and json.loads(request.read_text())["action"] == "stop"
    finally:
        request.unlink(missing_ok=True)
        shutil.rmtree(BASE_DIR / "natix-stray", ignore_errors=True)


def test_the_recent_page_has_the_flow_diagram_and_demo_buttons():
    status, _, body = render("/")
    assert status == 200, status
    for marker in ('id="n-car"', 'id="n-proc"', 'id="proc-steps"', 'id="n-alerts"',
                   'data-demo="realtime"', 'data-demo="backlog"'):
        assert marker in body, marker


def test_behind_real_time_is_hidden_when_no_footage_is_live():
    (LOGS_DIR / "processor.json").write_text(json.dumps({
        "ts": NOW, "current": None, "live_edge_ts": NOW - 86400 * 200,
    }))
    (LOGS_DIR / "ingest.json").write_text(json.dumps({"state": "absent", "last_pass_ts": NOW}))
    try:
        assert json.loads(render("/api/pipeline")[2])["lag"] == {"mode": "offline", "edge_ts": None}
        # Recent footage counts as live even before ingest says the car is there.
        (LOGS_DIR / "processor.json").write_text(json.dumps({
            "ts": NOW, "current": None, "live_edge_ts": NOW - 70,
        }))
        lag = json.loads(render("/api/pipeline")[2])["lag"]
        assert lag == {"mode": "live", "edge_ts": NOW - 70}, lag
    finally:
        (LOGS_DIR / "processor.json").unlink(missing_ok=True)
        (LOGS_DIR / "ingest.json").unlink(missing_ok=True)


def test_a_running_demo_counts_on_its_own_clock():
    (LOGS_DIR / "processor.json").write_text(json.dumps({
        "ts": NOW, "current": None, "live_edge_ts": None, "demo_edge_ts": NOW - 30,
        "demo": {"active": True}, "queue": {"depth": 0, "events": 0, "demo": 0, "next": []},
    }))
    try:
        lag = json.loads(render("/api/pipeline")[2])["lag"]
        assert lag == {"mode": "demo", "edge_ts": NOW - 30}, lag
    finally:
        (LOGS_DIR / "processor.json").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Activity layer
# ---------------------------------------------------------------------------
def test_every_page_carries_the_loading_screen_and_activity_layer():
    for path in ("/", "/plates", "/timeline", "/settings"):
        status, _, body = render(path)
        assert status == 200, (path, status)
        for marker in ('id="tv-boot"', 'id="tv-bar"', 'id="tv-pill"', "window.fetch = function",
                       'class="metric tv-loadbadge"', "<noscript><style>#tv-boot"):
            assert marker in body, (path, marker)
        # The fetch wrapper must be installed before any page script runs.
        assert body.index("window.fetch = function") < body.index("<main>"), path


def test_every_response_says_how_long_the_orin_took():
    for path in ("/healthz", "/plates", "/api/system"):
        _, headers, _ = render(path)
        timing = headers.get("server-timing", "")
        assert timing.startswith("app;dur="), (path, headers)
        assert float(timing.split("dur=")[1].split(";")[0]) >= 0


def test_the_settings_page_does_not_reload_the_plate_model_every_visit():
    calls = []
    original = ui_app._load_stage_status
    ui_app._load_stage_status = lambda stage: calls.append(stage) or f"{stage}: ready"
    ui_app._stage_status_cache.clear()
    try:
        for _ in range(3):
            assert render("/settings")[0] == 200
        assert sorted(calls) == ["alpr", "faces"], calls
    finally:
        ui_app._load_stage_status = original
        ui_app._stage_status_cache.clear()


def test_every_page_carries_the_media_viewer():
    for path in ("/", "/plates", "/faces", "/timeline"):
        status, _, body = render(path)
        assert status == 200, (path, status)
        for marker in ('id="vw"', 'id="vw-scrub"', 'data-vw="play"', "function decodeGif",
                       'id="vw-overlay"', 'data-vw="boxes"', 'id="vw-minconf"'):
            assert marker in body, (path, marker)


# ---------------------------------------------------------------------------
# Plates and Faces link each sighting to its frame, snapshot and GIF
# ---------------------------------------------------------------------------
def seed_sightings():
    """
    One plate seen three times and one person seen once:

    - pdet-new (newest): a clip with no alert and no saved frame - nothing to show
    - pdet-mid: its own frame, and an alert matched on clip_id whose GIF is
      still queued, so no GIF file yet
    - pdet-old: no frame; an alert from the legacy JSON with no clip_id,
      matched on the clip's filename, snapshot and GIF both on disk
    - fdet-1: a face with its frame and nothing else
    - alert-bare: an unrelated alert with no filename or media
    """
    from src.common import MEDIA_DIR
    connection = ui_app.get_connection()
    try:
        for table in ("plate_detections", "plates", "face_detections", "faces", "alerts"):
            connection.execute(f"DELETE FROM {table}")
        connection.execute(
            "INSERT INTO plates (id, plate_text, first_seen_ts, last_seen_ts, detection_count) "
            "VALUES ('plate01', 'TEST123', ?, ?, 3)", (NOW - 300, NOW - 100))
        for detection_id, ts, clip_id, source_file in (
            ("pdet-new", NOW - 100, "clip-new", "new-front.mp4"),
            ("pdet-mid", NOW - 200, "clip-mid", "mid-front.mp4"),
            ("pdet-old", NOW - 300, "clip-old", "old-front.mp4"),
        ):
            connection.execute(
                "INSERT INTO plate_detections (id, plate_id, plate_text, ts, clip_id, source_file, camera) "
                "VALUES (?, 'plate01', 'TEST123', ?, ?, ?, 'front')", (detection_id, ts, clip_id, source_file))
        connection.execute(
            "INSERT INTO faces (id, person_name, first_seen_ts, last_seen_ts, detection_count) "
            "VALUES ('face01', 'Stranger #1', ?, ?, 1)", (NOW - 50, NOW - 50))
        connection.execute(
            "INSERT INTO face_detections (id, face_id, ts, clip_id, source_file, camera) "
            "VALUES ('fdet-1', 'face01', ?, 'clip-face', 'face-front.mp4', 'front')", (NOW - 50,))
        connection.execute(
            "INSERT INTO alerts (id, timestamp, source_file, clip_id, jpeg, gif, status) "
            "VALUES ('alert-mid', ?, 'mid-front.mp4', 'clip-mid', 'alert-mid.jpg', 'alert-mid.gif', 'gif_queued')",
            (NOW - 200,))
        connection.execute(
            "INSERT INTO alerts (id, timestamp, source_file, clip_id, jpeg, gif, status) "
            "VALUES ('alert-old', ?, 'old-front.mp4', NULL, 'alert-old.jpg', 'alert-old.gif', 'gif_done')",
            (NOW - 300,))
        # An alert row with no filename at all must not trip the lookup.
        connection.execute(
            "INSERT INTO alerts (id, timestamp, source_file, clip_id, jpeg, gif, status) "
            "VALUES ('alert-bare', ?, NULL, 'clip-unrelated', NULL, NULL, 'gif_queued')", (NOW - 400,))
        connection.commit()
    finally:
        connection.close()
    for name in ("plates/pdet-mid-frame.jpg", "faces/fdet-1-frame.jpg",
                 "alert-mid.jpg", "alert-old.jpg", "alert-old.gif"):
        (MEDIA_DIR / name).parent.mkdir(parents=True, exist_ok=True)
        (MEDIA_DIR / name).write_bytes(b"x")


def clear_sightings():
    connection = ui_app.get_connection()
    try:
        for table in ("plate_detections", "plates", "face_detections", "faces", "alerts"):
            connection.execute(f"DELETE FROM {table}")
        connection.commit()
    finally:
        connection.close()


def sighting_rows(body: str) -> dict[str, str]:
    """Each sighting row of an entity page, keyed by its clip filename."""
    rows = {}
    for chunk in body.split("<tr>")[2:]:
        for name in ("new-front.mp4", "mid-front.mp4", "old-front.mp4", "face-front.mp4"):
            if name in chunk:
                rows[name] = chunk.split("</tr>")[0]
    return rows


def test_each_plate_sighting_links_to_exactly_the_media_that_exists():
    seed_sightings()
    try:
        status, _, body = render("/entity/plate/plate01")
    finally:
        clear_sightings()
    assert status == 200, status
    rows = sighting_rows(body)
    assert set(rows) == {"new-front.mp4", "mid-front.mp4", "old-front.mp4"}, rows.keys()
    # Nothing saved: said in words, not left as an empty cell.
    assert "none saved" in rows["new-front.mp4"] and "/media/" not in rows["new-front.mp4"]
    # Matched on clip_id; the queued GIF is not offered until it exists.
    assert "/media/plates/pdet-mid-frame.jpg" in rows["mid-front.mp4"]
    assert "/media/alert-mid.jpg" in rows["mid-front.mp4"]
    assert "alert-mid.gif" not in rows["mid-front.mp4"]
    # Matched on the filename when the alert has no clip_id; no frame saved.
    assert "/media/alert-old.jpg" in rows["old-front.mp4"] and "/media/alert-old.gif" in rows["old-front.mp4"]
    assert "-frame.jpg" not in rows["old-front.mp4"]


def test_sighting_links_open_in_the_viewer_and_say_what_they_are():
    seed_sightings()
    try:
        _, _, body = render("/entity/plate/plate01")
    finally:
        clear_sightings()
    row = sighting_rows(body)["old-front.mp4"]
    assert row.count('data-viewer="1"') == 2, row
    for label in ("📷 snapshot", "🎞️ GIF"):
        assert label in row, (label, row)
    assert "🎯 frame" in sighting_rows(body)["mid-front.mp4"]
    assert "<th>media</th>" in body


def test_the_plates_page_links_each_plate_to_its_latest_sighting_with_media():
    seed_sightings()
    try:
        status, _, body = render("/plates")
    finally:
        clear_sightings()
    assert status == 200, status
    # The newest sighting has nothing, so the card falls back to the one before.
    assert "latest sighting" in body
    assert "/media/plates/pdet-mid-frame.jpg" in body and "/media/alert-mid.jpg" in body
    assert "alert-old" not in body


def test_the_faces_pages_link_sightings_to_their_frames():
    seed_sightings()
    try:
        _, _, faces_body = render("/faces")
        _, _, entity_body = render("/entity/face/face01")
    finally:
        clear_sightings()
    assert "/media/faces/fdet-1-frame.jpg" in faces_body
    row = sighting_rows(entity_body)["face-front.mp4"]
    assert "/media/faces/fdet-1-frame.jpg" in row and "snapshot" not in row, row
    assert "not this person" in row


def test_an_identity_with_no_sighting_media_gets_no_media_block():
    seed_sightings()
    try:
        from src.common import MEDIA_DIR
        for name in ("plates/pdet-mid-frame.jpg", "alert-mid.jpg", "alert-old.jpg", "alert-old.gif"):
            (MEDIA_DIR / name).unlink()
        _, _, body = render("/plates")
    finally:
        clear_sightings()
    assert "TEST123" in body and "latest sighting" not in body


# ---------------------------------------------------------------------------
# Storage: the drive the car records to
# ---------------------------------------------------------------------------
GB = 1000 ** 3
CARSTORE_JSON = LOGS_DIR / "carstore.json"


def write_carstore(**fields) -> dict:
    status = {"schema": 1, "written_ts": int(time.time()), "state": "no_storage", "nonce": "abc123",
              "drives": [], "backing": None, "export": None, "job": None,
              "snapshot": {"last_ts": None}, "settings": {"tesla_min_image_gb": 64}}
    status.update(fields)
    CARSTORE_JSON.write_text(json.dumps(status), encoding="utf-8")
    return status


def drive(path, size_gb, partitions, role="available", blocked=None, drive_id="0123456789ab"):
    return {"id": drive_id, "path": path, "model": "SanDisk Extreme", "size_bytes": size_gb * GB,
            "role": role, "blocked": blocked,
            "partitions": [{"path": p[0], "label": p[1], "fstype": p[2], "size_bytes": p[3] * GB,
                            "mountpoint": p[4] if len(p) > 4 else None} for p in partitions]}


def storage_page() -> str:
    status, _, body = render("/storage")
    assert status == 200, status
    return body


def test_the_storage_page_says_when_the_service_is_not_installed():
    CARSTORE_JSON.unlink(missing_ok=True)
    body = storage_page()
    assert "not running on the Jetson" in body and "install_carstore.sh" in body
    assert "Drives are listed by the car storage service" in body
    assert 'href="/storage"' in body, "Storage is in the nav"


def test_each_format_button_says_exactly_what_it_erases():
    write_carstore(drives=[
        drive("/dev/sdb", 250, [("/dev/sdb1", "TESLACAM", "vfat", 120), ("/dev/sdb2", "jetsondata", "ext4", 130)],
              role="old TESLACAM drive"),
        drive("/dev/sda", 35, [("/dev/sda1", "VX360", "exfat", 35)], role="NATIX stick",
              blocked="this is the NATIX VX360 stick", drive_id="1111111111aa"),
        drive("/dev/sdc", 32, [("/dev/sdc1", "TESLACAM", "exfat", 32)], drive_id="2222222222bb"),
        drive("/dev/sdd", 128, [("/dev/sdd1", "teslacam-backing", "btrfs", 128, "/backingfiles")],
              role="car storage", drive_id="3333333333cc"),
        drive("/dev/sde", 64, [], drive_id="4444444444dd"),
    ])
    try:
        body = storage_page()
    finally:
        CARSTORE_JSON.unlink(missing_ok=True)
    assert "No car storage drive yet" in body
    assert "Format whole drive as car storage" in body and "Format /dev/sdb1 only" in body
    assert "Clips on it that were never processed are copied to the Jetson first." in body
    assert "Other partitions on the drive are not touched." in body
    assert "exfat, which the Jetson cannot read, so its clips cannot be copied off first" in body
    assert "smaller than Tesla&#x27;s 64 GB minimum" in body or "smaller than Tesla's 64 GB minimum" in body
    assert "Not offered: this is the NATIX VX360 stick" in body
    assert 'data-format-drive="1111111111aa"' not in body
    assert "its recordings are erased too" in body
    assert "Format /dev/sdd1 only" not in body, "the backing partition is reformatted only as a whole drive"
    assert "mounted at /backingfiles" in body and "no partitions" in body
    assert 'data-nonce="abc123"' in body and 'id="fmt-confirm"' in body


def test_an_exporting_store_says_whether_the_car_has_it_and_how_far_behind_it_is():
    now = int(time.time())
    write_carstore(state="exporting",
                   backing={"device": "/dev/sdd1", "image_bytes": 60 * GB, "free_bytes": 50 * GB,
                            "below_tesla_minimum": True},
                   export={"car_connected": False},
                   snapshot={"last_ts": now - 30, "seconds": 2.5, "copied_last": 2, "copied_total": 9,
                             "newest_clip_ts": now - 80, "paused": "less than 5 GB free on the Jetson"})
    try:
        body = storage_page()
        write_carstore(state="exporting", backing={"device": "/dev/sdd1", "image_bytes": 120 * GB,
                                                   "free_bytes": 5 * GB},
                       export={"car_connected": True},
                       snapshot={"last_ts": now - 10, "error": "cp --reflink failed (1): No space left"})
        connected = storage_page()
    finally:
        CARSTORE_JSON.unlink(missing_ok=True)
    assert "Recording to a 60 GB image on /dev/sdd1 (50 GB free)" in body
    assert "not connected right now" in body
    assert "smaller than Tesla" in body
    assert "picked up 2 new clip(s); 9 since the service started" in body
    assert "Newest clip received was recorded 80 seconds ago" in body
    assert "Copying paused: less than 5 GB free on the Jetson" in body
    assert "The car is connected and has the drive." in connected
    assert "failed: cp --reflink failed (1): No space left" in connected
    write_carstore(state="exporting", snapshot={"last_ts": int(time.time()) - 5, "seconds": 1.0,
                                                "copied_last": 0, "copied_total": 0, "copying": 7,
                                                "deferred": True})
    try:
        first = storage_page()
    finally:
        CARSTORE_JSON.unlink(missing_ok=True)
    assert "picked up 0 new clip(s)" in first and "Newest clip received" not in first
    assert "Copying now: 7 clip(s) so far" in first
    assert "More finished clips are waiting; they follow over the next snapshots, newest first." in first


def test_a_stopped_or_failing_service_is_never_shown_as_fine():
    write_carstore(state="error", reason="no USB device controller (UDC)")
    data = json.loads(CARSTORE_JSON.read_text())
    data["written_ts"] = int(time.time()) - 600
    CARSTORE_JSON.write_text(json.dumps(data))
    try:
        body = storage_page()
        api = ui_app.api_storage()
    finally:
        CARSTORE_JSON.unlink(missing_ok=True)
    assert "last reported 10 minutes ago - it may have stopped" in body
    assert "Car storage error: no USB device controller (UDC)" in body
    assert api["service"] == "stale" and api["age_seconds"] >= 600


def test_the_last_format_request_is_reported_in_words():
    sentences = {}
    for state, extra in (("running", {"step": "erasing /dev/sdb1"}),
                         ("done", {"message": "formatted /dev/sdb1 as car storage; old clips: 4 copied"}),
                         ("refused", {"message": "the drive list changed since the page loaded"})):
        write_carstore(job={"state": state, "partition": "/dev/sdb1", **extra})
        sentences[state] = storage_page()
    CARSTORE_JSON.unlink(missing_ok=True)
    assert "Formatting /dev/sdb1: erasing /dev/sdb1" in sentences["running"]
    assert "Done: formatted /dev/sdb1 as car storage; old clips: 4 copied" in sentences["done"]
    assert "Not formatted: the drive list changed" in sentences["refused"]


def test_no_drives_says_where_to_plug_one_in_and_the_page_refreshes_in_place():
    write_carstore()
    try:
        body = storage_page()
        status, _, live = render("/storage/live")
    finally:
        CARSTORE_JSON.unlink(missing_ok=True)
    assert "No USB drives attached to the Jetson. Plug one into a USB-A port" in body
    assert status == 200 and live.strip().startswith('<div id="storage-live" data-nonce="abc123"')
    assert "<html" not in live


def test_the_format_endpoint_refuses_everything_but_a_confirmed_json_request():
    from src.common import JOBS_DIR
    request_path = JOBS_DIR / "carstore-request.json"
    request_path.unlink(missing_ok=True)
    good = {"drive_id": "0123456789ab", "partition": "/dev/sdb1", "confirm": "ERASE", "nonce": "abc123"}
    CARSTORE_JSON.unlink(missing_ok=True)
    cases = [
        (post_json("/api/storage/format", good, content_type="text/plain"), 415, "application/json"),
        (post_json("/api/storage/format", None, raw=b"{nope"), 400, "JSON object"),
        (post_json("/api/storage/format", ["x"]), 400, "JSON object"),
        (post_json("/api/storage/format", {**good, "drive_id": "../../etc"}), 400, "unknown drive"),
        (post_json("/api/storage/format", {**good, "partition": "sdb1"}), 400, "unknown partition"),
        (post_json("/api/storage/format", {**good, "confirm": "yes"}), 400, "ERASE"),
        (post_json("/api/storage/format", good), 409, "not running"),
    ]
    write_carstore(job={"state": "running"})
    cases.append((post_json("/api/storage/format", good), 409, "already running"))
    for (status, answer), expected_status, expected_text in cases:
        assert status == expected_status and expected_text in answer["error"], (status, answer)
    assert not request_path.exists(), "nothing refused may reach the host service"

    write_carstore()
    try:
        status, answer = post_json("/api/storage/format", {**good, "partition": None})
        written = json.loads(request_path.read_text())
    finally:
        CARSTORE_JSON.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
    assert status == 200 and answer == {"ok": True}
    assert written["action"] == "format" and written["partition"] is None and written["nonce"] == "abc123"
    assert written["confirm"] == "ERASE" and abs(written["ts"] - time.time()) < 5


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
