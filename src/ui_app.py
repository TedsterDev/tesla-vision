"""
ui_app.py

Web dashboard and JSON API.

This is the port of Scout's `app/` - an Express server plus a Vue client whose
views were Recent, AllPlates, AllFaces, Timeline and Settings. We serve the
same five views (plus a Findings view Scout didn't have) as server-rendered
HTML, because:

    - the whole client was a read-mostly list-and-map UI, which server rendering
      does with no build step, no node_modules and no second container
    - it has to work from a phone over a hotspot in a parked car, where a Vue
      bundle is the slowest part of the page

Routes
    /               Recent - newest alerts with snapshot and GIF
    /findings       the surveillance detection verdicts (this is the point)
    /plates         every plate identity we've built
    /faces          every person identity we've built
    /entity/...     one identity's full sighting history
    /timeline       Leaflet map of drives and geo-located detections
    /settings       pipeline status, model availability, thresholds

    /api/...        JSON, mirroring Scout's /api/plates, /api/faces, /api/polls
    /media/...      JPEG and GIF evidence
    /healthz        {"ok": true}

Run (in container):
  uvicorn src.ui_app:app --host 0.0.0.0 --port 8080
"""
import asyncio
import html
import json
import os
import time

import base64
import hmac

from pathlib import Path
from urllib.parse import quote

from typing import Any

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from fastapi import Request
from fastapi import Response

from src import cfaccess
from fastapi.responses import PlainTextResponse

from src.common import (
    MEDIA_DIR,
    ALERTS_DIR,
    ensure_dirs
)
from src.db import connect, import_legacy_alert_json, rows_to_dicts, write_with_retry


def get_connection():
    """
    One SQLite connection per request path.

    FastAPI serves requests from a thread pool and SQLite connections are not
    safe to share across threads, so we open per call. WAL mode makes this
    cheap, and the dashboard's query volume is a handful per page load.

    The short busy timeout is deliberate: paired with write_with_retry's
    backoff it bounds the worst case at a few seconds. A user-facing request
    that blocks for 30 seconds waiting on a lock reads as a hung dashboard, and
    failing fast with a retry is a better experience than succeeding slowly.
    """
    return connect(timeout_seconds=3.0)


@asynccontextmanager
async def tesla_dashboard_app_lifespan(app: FastAPI):
    # On Start-Up
    ensure_dirs()
    if AUTH_ENABLED and not DASHBOARD_USER:
        raise RuntimeError("DASHBOARD_PASS is set but DASHBOARD_USER is empty. Set DASHBOARD_USER to enable auth.")
    if AUTH_ENABLED:
        print(f"[🔐 ui] Basic Auth ENABLED user={DASHBOARD_USER}")
    else:
        print("[🔐 ui] Basic Auth DISABLED (set DASHBOARD_PASS to enable)")

    # Pull any alert JSON written before the database existed into the alerts
    # table, so upgrading doesn't appear to lose history.
    #
    # Wrapped defensively: this is a convenience, not a prerequisite. An
    # exception escaping the lifespan aborts uvicorn startup, and with
    # `restart: unless-stopped` that becomes a restart loop - the dashboard
    # would go down exactly when the processor is busy, which is when you most
    # want to look at it. Serving a slightly stale alert list beats serving 502.
    try:
        connection = connect()
        imported = import_legacy_alert_json(connection, ALERTS_DIR)
        connection.close()
        print(f"[🗄️ ui] alert JSON synced to database ({imported} new/changed)")
    except Exception as exception_object:
        print(f"[🗄️ ui] alert JSON sync skipped ({exception_object}) - dashboard still starting")

    yield
    # Shutdown (nothing to do yet)

app = FastAPI(title="Tesla Vision - Surveillance Detection", lifespan=tesla_dashboard_app_lifespan)

# --- Simple password protection (HTTP Basic) ---
# Configure via environment variables (recommended via docker-compose + .env):
#   DASHBOARD_USER (default: "")
#   DASHBOARD_PASS (no default; if unset/empty, auth is DISABLED)
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "").strip()
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "").strip()
AUTH_ENABLED = bool(DASHBOARD_PASS)

def _constant_time_equal(time_a: str, time_b: str) -> bool:
    return hmac.compare_digest(time_a.encode("utf-8"), time_b.encode("utf-8"))


# --- Brute-force brake -------------------------------------------------------
# The security audit measured 322 wrong Basic Auth attempts per second against
# this origin: enough to walk a four-digit PIN's entire keyspace in 31 seconds.
# Cloudflare Access is not in the path for LAN or tailnet requests, so on that
# side the PIN is the whole boundary and it needs a brake.
#
# This is deliberately NOT a per-client counter, and the reason is specific to
# how this thing is actually deployed. Two measurements killed that design:
#
#   1. uvicorn runs with proxy_headers=True, so scope["client"] is overwritten
#      from X-Forwarded-For whenever the peer is a trusted proxy. A spray of 16
#      wrong passwords carrying a different forged XFF each time never tripped
#      a limit of 10 - the attacker simply picks their own bucket.
#   2. Every request the live service has ever served arrives from one address
#      (the compose bridge gateway), because cloudflared and docker-proxy both
#      NAT. So in practice there is exactly ONE bucket, and any attacker who
#      trips it locks out the owner. The brake became the denial of service.
#
# So the brake is a GLOBAL SERIALISED DELAY on failures instead. Every failed
# attempt takes the same lock and waits, so attempts cannot be run in parallel
# and cannot be spread across forged source addresses - neither of the escapes
# above applies. A caller presenting the right password never touches this path
# at all, so unlike a lockout it CANNOT deny service to the legitimate user.
#
# 10,000 codes x 0.5s serialised is ~83 minutes to exhaust, ~42 to expect a
# hit, against 31 seconds unbraked. That is a brake, not a wall. Stated plainly:
# the real fix is a secret longer than four digits, and this only buys time.
AUTH_FAILURE_DELAY_SECONDS = 0.5

# Past this many already queued, refuse instantly instead of joining the queue.
# Without it a flood of bad credentials grows an unbounded set of sleeping
# tasks. Legitimate users never queue here, so this cannot lock anybody out.
AUTH_FAILURE_MAX_WAITING = 64

_auth_failure_lock = asyncio.Lock()
_auth_failures_waiting = 0


async def _penalise_auth_failure() -> bool:
    """
    Serialise and slow one failed authentication attempt.

    Returns True if the caller should be told to back off entirely (the queue
    is already long), False once the delay has been served. The delay is held
    for the duration of the lock rather than slept before taking it, because
    the point is that two wrong guesses can never be in flight at once.
    """
    global _auth_failures_waiting

    if _auth_failures_waiting >= AUTH_FAILURE_MAX_WAITING:
        return True

    _auth_failures_waiting += 1
    try:
        async with _auth_failure_lock:
            await asyncio.sleep(AUTH_FAILURE_DELAY_SECONDS)
    finally:
        _auth_failures_waiting -= 1
    return False


# --- The location routes fail CLOSED -----------------------------------------
LOCATION_ROUTES = ("/gps", "/findmy", "/api/gps", "/api/findmy")


def _is_location_route(path: str) -> bool:
    # Exact match or a complete path segment, never a bare prefix. A plain
    # startswith would hand /gpsanything and /findmyphone these rules too -
    # the same trap the cloudflared ingress regex has to dodge with (/|$).
    return any(path == route or path.startswith(route + "/") for route in LOCATION_ROUTES)


def _parse_basic_auth(auth_header: str) -> tuple[str, str] | None:
    # Expect: "Basic base64(user:pass)"
    try:
        scheme, b64 = auth_header.split(" ", 1)
        if scheme.lower() != "basic":
            return None
        raw = base64.b64decode(b64.strip()).decode("utf-8")
        user, pw = raw.split(":", 1)
        return user, pw
    except Exception:
        return None


@app.middleware("http")
async def require_basic_auth(request: Request, call_next):
    # Always allow health checks
    if request.url.path in ("/healthz",):
        return await call_next(request)

    if request.url.path in (
        "/favicon.ico",
        "/apple-touch-icon.png",
        "/apple-touch-icon-precomposed.png",
    ):
        return await call_next(request)


    # Cloudflare Access first, because when a request arrives through it there
    # will BE no Authorization header - Access consumes it. Basic Auth simply
    # cannot work on that path, so the browser would prompt, have the header
    # stripped, get a 401, and prompt again forever.
    #
    # Verified HERE, above the "no password configured" early return, because
    # the location gate below has to be able to see the result. While this ran
    # second, DASHBOARD_PASS='' with Cloudflare Access fully configured served
    # /gps, /findmy, /api/gps and /api/findmy with 200 and no credential at
    # all: the early return fired before the JWT was ever looked at. That is
    # the configuration an operator lands on by concluding, reasonably, that
    # the PIN is redundant once Access is in front of the tunnel.
    access_token = request.headers.get("cf-access-jwt-assertion")
    access_identity = None
    if access_token:
        claims = cfaccess.verify(access_token)
        if claims:
            access_identity = claims.get("email") or claims.get("sub") or "access"
        else:
            # A present-but-invalid token is worth one line: it is either a
            # misconfigured CF_ACCESS_AUD or something genuinely wrong.
            print(f"[🔐 ui] Access token present but not accepted for "
                  f"{request.url.path}", flush=True)

    if access_identity:
        request.state.identity = access_identity
        return await call_next(request)

    # The location routes fail CLOSED. Everywhere else this dashboard stays
    # open when no password is configured, which is a deliberate
    # backwards-compatibility choice. For plate crops that is embarrassing. For
    # continuous real-time vehicle location - where the owner is right now, not
    # where some car was once - it is a different category of harm, so these
    # routes refuse to serve rather than serve to nobody in particular.
    #
    # The test is "no password AND no verified Access identity", never "no
    # password and Access is not configured". Configuration is not a
    # credential: requests arriving straight at 0.0.0.0:8080 over the LAN or
    # the tailnet carry no assertion no matter how completely Access is set up
    # in front of the public hostname, and only the branch above can tell the
    # difference.
    #
    # Scoped to the four routes in LOCATION_ROUTES (and what sits under them)
    # ON PURPOSE. Flipping the whole dashboard to
    # fail-closed is the right end state, but it is a far wider blast radius
    # than this feature warrants: it would black out every page the moment one
    # env line is wrong, including the page you would use to diagnose that.
    # That change belongs in its own commit with its own rollout.
    if not DASHBOARD_PASS and _is_location_route(request.url.path):
        print(
            f"[🔐 ui] 503 location route with no verified identity "
            f"path={request.url.path} "
            f"cf_access_configured={cfaccess.is_configured()} "
            f"access_jwt={bool(access_token)}",
            flush=True,
        )
        return PlainTextResponse(
            "Location endpoints are disabled because this request carried no "
            "authentication. Set DASHBOARD_USER and DASHBOARD_PASS, or reach this page "
            "through Cloudflare Access so a verified Cf-Access-Jwt-Assertion arrives "
            "with the request. Configuring CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD does "
            "not by itself authenticate anything that connects directly to this port.",
            status_code=503,
        )

    # If no password is configured, leave the rest of the dashboard open
    # (backwards compatible). Set DASHBOARD_PASS to enable protection.
    if not DASHBOARD_PASS:
        return await call_next(request)

    auth = request.headers.get("authorization")
    if not auth:
        # Say *why* this was rejected. "401" alone cannot distinguish the
        # causes that look identical from outside: no prompt, a cancelled
        # prompt, or an upstream that removed the header.
        access_email = request.headers.get("cf-access-authenticated-user-email")
        via_cloudflare = bool(request.headers.get("cf-ray"))
        print(
            f"[🔐 ui] 401 no-authorization path={request.url.path} "
            f"via_cloudflare={via_cloudflare} "
            f"access_email={access_email or '-'} "
            f"access_jwt={bool(access_token)} "
            f"cf_access_configured={cfaccess.is_configured()}",
            flush=True,
        )
        return PlainTextResponse(
            "Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
        )

    # Brake goes here, immediately before the credential check and not after
    # it: a constant-time compare is still an oracle you can run hundreds of
    # times a second, and this is the last point at which refusing costs
    # nothing. Counting only *failures* means a client using the right password
    # is never rate limited no matter how many requests the page makes.
    parsed = _parse_basic_auth(auth)
    if not parsed:
        if await _penalise_auth_failure():
            return PlainTextResponse(
                "Too many failed authentication attempts in flight.",
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return PlainTextResponse(
            "Invalid authentication",
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
        )

    user, pw = parsed
    if not (
        _constant_time_equal(user, DASHBOARD_USER)
        and _constant_time_equal(pw, DASHBOARD_PASS)
    ):
        if await _penalise_auth_failure():
            return PlainTextResponse(
                "Too many failed authentication attempts in flight.",
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return PlainTextResponse(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
        )

    return await call_next(request)


# Serve files in /data/media at /media/<filename>
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
# Everything user- or OCR-derived goes through `esc`. Plate text comes from
# Tesseract and person labels come from a text box in this very UI, so both are
# untrusted input that ends up in HTML.

def esc(value: Any) -> str:
    """HTML-escape any value for safe interpolation into a template."""
    return html.escape(str(value if value is not None else ""))


PAGE_STYLES = """
  :root {
    --bg:#0b0b0f; --panel:#15151d; --panel-2:#1c1c26; --line:#2a2a38;
    --text:#ececf2; --muted:#9a9ab0; --accent:#7aa2ff;
    --high:#ff5c5c; --medium:#ffb020; --low:#4ad07a;
  }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); margin:0;
         font-family:ui-sans-serif, system-ui, -apple-system, sans-serif; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  header { position:sticky; top:0; z-index:10; background:rgba(11,11,15,0.95);
           backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }
  nav { display:flex; gap:4px; padding:12px 20px; align-items:center; flex-wrap:wrap; }
  nav .brand { font-weight:800; font-size:17px; margin-right:14px; letter-spacing:-0.02em; }
  nav a.tab { padding:7px 13px; border-radius:9px; color:var(--muted); font-size:14px; font-weight:500; }
  nav a.tab:hover { background:var(--panel-2); color:var(--text); text-decoration:none; }
  nav a.tab.active { background:var(--accent); color:#0b0b0f; font-weight:600; }
  main { padding:22px; max-width:1400px; margin:0 auto; }
  h1 { margin:0 0 4px 0; font-size:24px; letter-spacing:-0.02em; }
  .sub { color:var(--muted); margin-bottom:22px; font-size:14px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:14px;
          padding:16px; margin-bottom:14px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
  .muted { color:var(--muted); font-size:13px; }
  .mono { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
  .pill { display:inline-block; padding:3px 9px; border-radius:999px; font-size:11px;
          font-weight:700; text-transform:uppercase; letter-spacing:0.04em; }
  .pill.high { background:rgba(255,92,92,0.18); color:var(--high); }
  .pill.medium { background:rgba(255,176,32,0.18); color:var(--medium); }
  .pill.low { background:rgba(74,208,122,0.15); color:var(--low); }
  .plate { font-family:ui-monospace,monospace; font-size:21px; font-weight:800;
           letter-spacing:0.09em; background:#f2f2f0; color:#111; padding:5px 12px;
           border-radius:6px; display:inline-block; border:2px solid #999; }
  img.evidence { max-width:100%; border-radius:10px; display:block; background:var(--panel-2); }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
       letter-spacing:0.05em; }
  .stat { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:14px 16px; min-width:112px; }
  .stat .n { font-size:26px; font-weight:800; letter-spacing:-0.02em; }
  .stat .l { color:var(--muted); font-size:12px; margin-top:2px; }
  .empty { color:var(--muted); padding:36px; text-align:center;
           border:1px dashed var(--line); border-radius:14px; }
  button, .btn { background:var(--panel-2); color:var(--text); border:1px solid var(--line);
                 border-radius:9px; padding:7px 13px; font-size:13px; cursor:pointer;
                 font-family:inherit; }
  button:hover, .btn:hover { border-color:var(--accent); text-decoration:none; }
  input[type=text] { background:var(--panel-2); border:1px solid var(--line); color:var(--text);
                     border-radius:9px; padding:7px 11px; font-family:inherit; font-size:13px; }
  #map { height:520px; border-radius:14px; border:1px solid var(--line); }
  .reason { padding:3px 0; font-size:13px; }
  .reason::before { content:"▸ "; color:var(--accent); }
  .reason.alarm { color:var(--high); font-weight:700; }
  .reason.alarm::before { content:"⚠ "; }
"""

NAV_TABS = [
    ("/", "Recent"),
    ("/findings", "Findings"),
    ("/plates", "Plates"),
    ("/faces", "Faces"),
    ("/timeline", "Timeline"),
    ("/natix", "NATIX"),
    ("/findmy", "Find My"),
    ("/gps", "GPS"),
    ("/settings", "Settings"),
]


def render_page(title: str, body: str, active_path: str = "", head_extra: str = "") -> HTMLResponse:
    """Wrap page content in the shared shell (nav, styles, viewport)."""
    tabs = "".join(
        f'<a class="tab{" active" if path == active_path else ""}" href="{path}">{esc(label)}</a>'
        for path, label in NAV_TABS
    )

    return HTMLResponse(f"""<!doctype html>
<html><head>
  <meta charset="utf-8" />
  <title>{esc(title)} - Tesla Vision</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>{PAGE_STYLES}</style>
  {head_extra}
</head>
<body>
  <header><nav>
    <span class="brand">🛰️ Tesla Vision</span>
    {tabs}
  </nav></header>
  <main>{body}</main>
</body></html>""")


def format_timestamp(timestamp: Any) -> str:
    """Render an epoch timestamp in local time, tolerating junk."""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(timestamp or "")


def severity_pill(severity: str) -> str:
    severity = (severity or "low").lower()
    return f'<span class="pill {esc(severity)}">{esc(severity)}</span>'


# ---------------------------------------------------------------------------
# Legacy JSON alert access (kept: gif_worker updates these files in place)
# ---------------------------------------------------------------------------

def _load_alert(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _list_alerts() -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for alert_path in ALERTS_DIR.glob("*.json"):
        try:
            alerts.append(_load_alert(alert_path))
        except Exception:
            continue
    alerts.sort(key=lambda alert: alert.get("timestamp", 0), reverse=True)
    return alerts


# ---------------------------------------------------------------------------
# JSON API - mirrors Scout's app/server/routes/api/*
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/alerts")
def alerts_json():
    return _list_alerts()


@app.get("/api/stats")
def api_stats():
    """Counts for the dashboard header and the Settings page."""
    connection = get_connection()
    try:
        def count(table: str, where: str = "") -> int:
            return connection.execute(f"SELECT COUNT(*) AS n FROM {table} {where}").fetchone()["n"]

        return {
            "alerts": count("alerts"),
            "clips": count("clips"),
            "plates": count("plates"),
            "plate_detections": count("plate_detections"),
            "faces": count("faces"),
            "face_detections": count("face_detections"),
            "polls": count("polls"),
            "drives": count("drives"),
            "findings_high": count("correlations", "WHERE severity='high'"),
            "findings_medium": count("correlations", "WHERE severity='medium'"),
        }
    finally:
        connection.close()


@app.get("/api/plates")
def api_plates():
    """Every plate identity, worst first. Scout's GET /api/plates."""
    connection = get_connection()
    try:
        return rows_to_dicts(connection.execute(
            "SELECT * FROM plates ORDER BY threat_score DESC, last_seen_ts DESC"
        ).fetchall())
    finally:
        connection.close()


@app.get("/api/plates/detections")
def api_plate_detections(limit: int = 50):
    """Recent plate sightings. Scout's GET /api/plates/detections."""
    connection = get_connection()
    try:
        return rows_to_dicts(connection.execute(
            "SELECT * FROM plate_detections ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall())
    finally:
        connection.close()


@app.get("/api/plates/{plate_id}")
def api_plate_detail(plate_id: str):
    connection = get_connection()
    try:
        plate = connection.execute("SELECT * FROM plates WHERE id=?", (plate_id,)).fetchone()
        if not plate:
            return {"error": "not found"}
        detections = connection.execute(
            "SELECT * FROM plate_detections WHERE plate_id=? ORDER BY ts DESC", (plate_id,)
        ).fetchall()
        return {"plate": dict(plate), "detections": rows_to_dicts(detections)}
    finally:
        connection.close()


@app.get("/api/faces")
def api_faces():
    """Every person identity. Scout's GET /api/faces."""
    connection = get_connection()
    try:
        # The embedding BLOB is large and meaningless over JSON - omit it.
        return rows_to_dicts(connection.execute(
            "SELECT id, person_name, label, is_known, embedding_count, first_seen_ts, "
            "last_seen_ts, detection_count, threat_score, best_image FROM faces "
            "ORDER BY threat_score DESC, last_seen_ts DESC"
        ).fetchall())
    finally:
        connection.close()


@app.get("/api/faces/{face_id}")
def api_face_detail(face_id: str):
    connection = get_connection()
    try:
        face = connection.execute(
            "SELECT id, person_name, label, is_known, embedding_count, first_seen_ts, "
            "last_seen_ts, detection_count, threat_score, best_image FROM faces WHERE id=?",
            (face_id,),
        ).fetchone()
        if not face:
            return {"error": "not found"}
        detections = connection.execute(
            "SELECT id, face_id, ts, source_file, camera, frame_number, det_confidence, "
            "match_distance, image, lat, lon, drive_id FROM face_detections "
            "WHERE face_id=? ORDER BY ts DESC",
            (face_id,),
        ).fetchall()
        return {"face": dict(face), "detections": rows_to_dicts(detections)}
    finally:
        connection.close()


@app.post("/api/{entity_type}/{entity_id}/label")
async def api_set_label(entity_type: str, entity_id: str, request: Request):
    """
    Name an identity, or mark it known.

    This is Scout's "Name Face" route generalised to plates as well. Marking
    something known is the highest-leverage action in the whole system: it is
    how you tell the correlation engine that your partner's car is not a
    threat, and it takes that entity out of scoring permanently.
    """
    if entity_type not in ("plates", "faces"):
        return {"error": "unknown entity type"}

    payload = await request.json()
    table = entity_type

    def apply_changes(connection):
        if "label" in payload:
            label = (payload.get("label") or "").strip()[:120]
            connection.execute(
                f"UPDATE {table} SET label=? WHERE id=?", (label or None, entity_id)
            )
        if "is_known" in payload:
            connection.execute(
                f"UPDATE {table} SET is_known=? WHERE id=?",
                (1 if payload["is_known"] else 0, entity_id),
            )
            # Marking known must take effect immediately, not at the next clip.
            connection.execute(
                "UPDATE correlations SET score=0, severity='low' "
                "WHERE entity_type=? AND entity_id=?",
                ("plate" if entity_type == "plates" else "face", entity_id),
            )
            connection.execute(f"UPDATE {table} SET threat_score=0 WHERE id=?", (entity_id,))

    connection = get_connection()
    try:
        # Retry rather than 500: the processor holds the write lock in bursts
        # while committing a clip, and the person clicking "mark as known" in
        # the dashboard should not have to care.
        write_with_retry(connection, apply_changes)
        return {"ok": True}
    except Exception as exception_object:
        return {"ok": False, "error": str(exception_object)}
    finally:
        connection.close()


@app.post("/api/faces/{face_id}/merge")
async def api_merge_faces(face_id: str, request: Request):
    """
    Fold this identity into another - "these two strangers are the same person".

    Online clustering splits one person into several identities whenever
    lighting or pose changes enough, and an unfixable split hides a follower's
    real sighting count across several strangers. This is the repair.
    """
    from src.faces import merge_identities

    payload = await request.json()
    target_face_id = (payload.get("into") or "").strip()
    if not target_face_id:
        return {"ok": False, "error": "missing target identity"}

    connection = get_connection()
    try:
        merged = write_with_retry(
            connection, lambda cn: merge_identities(cn, face_id, target_face_id)
        )
        if not merged:
            # merge_identities refuses a self-merge or an unknown identity;
            # say which rather than returning a bare false.
            return {"ok": False, "error": "identity not found, or merging into itself"}
        return {"ok": True, "merged_into": target_face_id}
    except Exception as exception_object:
        return {"ok": False, "error": str(exception_object)}
    finally:
        connection.close()


@app.post("/api/faces/detections/{detection_id}/split")
def api_split_face_detection(detection_id: str):
    """
    Pull one sighting out into a new person - Scout's `makeStranger`.

    The repair for the opposite error: a bad frame matched two different people
    onto one identity, which hides one of them inside the other's history.
    """
    from src.faces import split_detection_into_new_identity

    connection = get_connection()
    try:
        new_face_id = write_with_retry(
            connection, lambda cn: split_detection_into_new_identity(cn, detection_id)
        )
        if not new_face_id:
            return {"ok": False, "error": "no such detection"}
        return {"ok": True, "new_face_id": new_face_id}
    except Exception as exception_object:
        return {"ok": False, "error": str(exception_object)}
    finally:
        connection.close()


@app.get("/api/findings")
def api_findings():
    """The surveillance detection verdicts, worst first."""
    connection = get_connection()
    try:
        rows = connection.execute(
            "SELECT * FROM correlations ORDER BY score DESC, created_ts DESC"
        ).fetchall()
        findings = []
        for row in rows:
            finding = dict(row)
            try:
                finding["reasons"] = json.loads(finding.get("reasons") or "[]")
            except json.JSONDecodeError:
                finding["reasons"] = []
            findings.append(finding)
        return findings
    finally:
        connection.close()


@app.get("/api/polls")
def api_polls(limit: int = 500):
    """Vehicle telemetry. Scout's GET /api/polls."""
    connection = get_connection()
    try:
        return rows_to_dicts(connection.execute(
            "SELECT * FROM polls WHERE lat IS NOT NULL ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall())
    finally:
        connection.close()


@app.get("/api/drives")
def api_drives():
    """Scout's GET /api/drives."""
    connection = get_connection()
    try:
        return rows_to_dicts(connection.execute(
            "SELECT * FROM drives ORDER BY start_ts DESC LIMIT 200"
        ).fetchall())
    finally:
        connection.close()


@app.get("/api/geocodes")
def api_geocodes():
    """Scout's GET /api/geocodes."""
    connection = get_connection()
    try:
        return rows_to_dicts(connection.execute(
            "SELECT * FROM geocodes ORDER BY fetched_ts DESC LIMIT 500"
        ).fetchall())
    finally:
        connection.close()


@app.get("/api/map")
def api_map():
    """
    Everything the Timeline map needs, in one request.

    Scout's client made four separate calls (polls, drives, plates, faces) and
    stitched them together in Vue. One payload is a better fit for a phone on a
    hotspot.
    """
    connection = get_connection()
    try:
        polls = connection.execute(
            "SELECT ts, lat, lon, speed, status, drive_id FROM polls "
            "WHERE lat IS NOT NULL ORDER BY ts ASC LIMIT 5000"
        ).fetchall()
        plate_points = connection.execute(
            "SELECT plate_text AS label, ts, lat, lon, image, plate_id AS entity_id "
            "FROM plate_detections WHERE lat IS NOT NULL ORDER BY ts DESC LIMIT 1000"
        ).fetchall()
        face_points = connection.execute(
            "SELECT f.person_name AS label, d.ts, d.lat, d.lon, d.image, d.face_id AS entity_id "
            "FROM face_detections d JOIN faces f ON f.id = d.face_id "
            "WHERE d.lat IS NOT NULL ORDER BY d.ts DESC LIMIT 1000"
        ).fetchall()
        return {
            "polls": rows_to_dicts(polls),
            "plates": rows_to_dicts(plate_points),
            "faces": rows_to_dicts(face_points),
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# HTML views
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(limit: int = 25):
    """Recent - the alert feed. Scout's Recent.vue."""
    alerts = _list_alerts()
    total_alerts = len(alerts)
    limit = max(1, min(limit, 500))

    connection = get_connection()
    try:
        stats = api_stats()
        top_findings = connection.execute(
            "SELECT * FROM correlations WHERE severity IN ('high','medium') "
            "ORDER BY score DESC LIMIT 3"
        ).fetchall()
    finally:
        connection.close()

    banner = ""
    if top_findings:
        items = "".join(
            f'<div style="padding:6px 0;">{severity_pill(row["severity"])} '
            f'<a href="/entity/{esc(row["entity_type"])}/{esc(row["entity_id"])}">'
            f'<b>{esc(row["entity_label"])}</b></a> '
            f'<span class="muted">scored {esc(row["score"])}/100</span></div>'
            for row in top_findings
        )
        banner = (
            '<div class="card" style="border-color:var(--high);">'
            '<div style="font-weight:800; margin-bottom:6px;">⚠️ Active surveillance findings</div>'
            f'{items}'
            '<div style="margin-top:8px;"><a href="/findings">See all findings →</a></div>'
            '</div>'
        )

    stat_tiles = "".join(
        f'<div class="stat"><div class="n">{stats[key]}</div><div class="l">{esc(label)}</div></div>'
        for key, label in [
            ("alerts", "alerts"), ("plates", "plates"), ("faces", "people"),
            ("clips", "clips"), ("drives", "drives"),
        ]
    )

    more_link = (
        f'<div style="text-align:center; padding:16px;">'
        f'<a class="btn" href="/?limit={limit + 50}">Show more '
        f'({limit} of {total_alerts})</a></div>'
        if total_alerts > limit else ""
    )

    rows = []
    for alert in alerts[:limit]:
        jpeg = alert.get("jpeg", "")
        gif = alert.get("gif", "")
        # loading="lazy" matters more than it looks: GIF previews are megabytes
        # each and this view lists many alerts, so eager loading would pull the
        # whole media directory on one page load over a phone hotspot.
        jpeg_tag = f'<img class="evidence" loading="lazy" src="/media/{esc(jpeg)}" style="max-width:420px;" />' if jpeg else ""
        gif_tag = f'<img class="evidence" loading="lazy" src="/media/{esc(gif)}" style="max-width:420px;" />' if gif else ""

        extra = ""
        if alert.get("plate"):
            extra += f'<div style="margin-top:8px;"><span class="plate">{esc(alert["plate"])}</span></div>'
        if alert.get("faces"):
            extra += f'<div class="muted" style="margin-top:6px;">people: {esc(", ".join(alert["faces"]))}</div>'

        rows.append(f"""
          <div class="card"><div class="row">
            <div style="min-width:320px; flex:1;">
              <div style="font-size:17px; font-weight:700;">Alert {esc(alert.get("id",""))}</div>
              <div class="muted">{esc(format_timestamp(alert.get("timestamp")))}</div>
              <div class="muted mono" style="margin-top:6px;">{esc(alert.get("source_file",""))}</div>
              <div class="muted">camera: {esc(alert.get("camera") or "unknown")}</div>
              <div class="muted">score: {esc(round(float(alert.get("score") or 0), 3))}</div>
              <div class="muted">status: {esc(alert.get("status",""))}</div>
              {extra}
            </div>
            <div class="row" style="flex:2;">
              <div>{jpeg_tag}</div>
              <div>{gif_tag}</div>
            </div>
          </div></div>
        """)

    body = f"""
      <h1>Recent activity</h1>
      <div class="sub">Newest detections from the TeslaCam feed.</div>
      {banner}
      <div class="row" style="margin-bottom:20px;">{stat_tiles}</div>
      {"".join(rows) if rows else '<div class="empty">No alerts yet.</div>'}
      {more_link}
    """
    return render_page("Recent", body, "/")


@app.get("/findings", response_class=HTMLResponse)
def findings_view():
    """
    The surveillance detection verdicts.

    Scout left this judgement to the operator staring at a timeline. This page
    is the answer to "am I being followed", ordered worst first, with the
    reasoning shown rather than just a score.
    """
    findings = api_findings()
    scored = [finding for finding in findings if finding["score"] > 0]

    cards = []
    for finding in scored:
        reasons = "".join(
            f'<div class="reason{" alarm" if reason.startswith("FOLLOWED") else ""}">{esc(reason)}</div>'
            for reason in finding["reasons"]
        )
        entity_kind = "Vehicle" if finding["entity_type"] == "plate" else "Person"
        cards.append(f"""
          <div class="card">
            <div class="row" style="justify-content:space-between; align-items:center;">
              <div>
                {severity_pill(finding["severity"])}
                <span style="font-size:18px; font-weight:800; margin-left:8px;">
                  <a href="/entity/{esc(finding["entity_type"])}/{esc(finding["entity_id"])}">{esc(finding["entity_label"])}</a>
                </span>
                <span class="muted"> · {esc(entity_kind)}</span>
              </div>
              <div style="font-size:26px; font-weight:800;">{esc(finding["score"])}<span class="muted" style="font-size:14px;">/100</span></div>
            </div>
            <div style="margin-top:10px;">{reasons}</div>
            <div class="muted" style="margin-top:10px;">
              {esc(finding["detection_count"])} sightings ·
              {esc(finding["distinct_drives"])} drives ·
              {esc(finding["distinct_days"])} days ·
              {esc(finding["distinct_locations"])} locations ·
              {esc(round(finding["max_separation_mi"] or 0, 1))} mi apart
            </div>
          </div>
        """)

    explainer = """
      <div class="card" style="background:var(--panel-2);">
        <div class="muted">
          Scores combine five signals: how many <b>separate drives</b> an entity appeared on,
          how many <b>different days</b>, how many <b>distinct locations</b>, how far those
          sightings were <b>spread apart</b>, and whether it was seen in the
          <b>rear-facing cameras</b>. Anything anchored to a single place, or seen only where
          you regularly park, is suppressed - it cannot be following you.
          Mark your own vehicles and household as <b>known</b> to remove them entirely.
        </div>
      </div>
    """

    body = f"""
      <h1>Surveillance findings</h1>
      <div class="sub">Entities scored on how their sightings are distributed across space, time and journeys.</div>
      {explainer}
      {"".join(cards) if cards else '<div class="empty">Nothing has scored yet. Findings appear once an entity is seen on more than one occasion.</div>'}
    """
    return render_page("Findings", body, "/findings")


@app.get("/plates", response_class=HTMLResponse)
def plates_view():
    """Scout's AllPlates.vue."""
    plates = api_plates()

    cards = []
    for plate in plates:
        image = (
            f'<img class="evidence" loading="lazy" src="/media/{esc(plate["best_image"])}" style="max-height:96px;" />'
            if plate.get("best_image") else '<div class="muted">no crop saved</div>'
        )
        known_badge = '<span class="pill low">known</span>' if plate["is_known"] else ""
        cards.append(f"""
          <div class="card">
            <div style="margin-bottom:10px;">
              <a href="/entity/plate/{esc(plate["id"])}"><span class="plate">{esc(plate["plate_text"])}</span></a>
              {known_badge}
            </div>
            {image}
            <div class="muted" style="margin-top:10px;">
              {esc(plate["detection_count"])} sightings<br/>
              first {esc(format_timestamp(plate["first_seen_ts"]))}<br/>
              last {esc(format_timestamp(plate["last_seen_ts"]))}
            </div>
            <div style="margin-top:8px;">threat {esc(plate["threat_score"])}/100</div>
          </div>
        """)

    body = f"""
      <h1>License plates</h1>
      <div class="sub">Every plate identity built from the camera feed, highest threat first.</div>
      <div class="grid">{"".join(cards)}</div>
      {'<div class="empty">No plates read yet. Check Settings to confirm the ALPR stage is running.</div>' if not cards else ''}
    """
    return render_page("Plates", body, "/plates")


@app.get("/faces", response_class=HTMLResponse)
def faces_view():
    """Scout's AllFaces.vue, including its 'Stranger #N' naming."""
    faces = api_faces()

    cards = []
    for face in faces:
        image = (
            f'<img class="evidence" loading="lazy" src="/media/{esc(face["best_image"])}" style="max-height:150px;" />'
            if face.get("best_image") else '<div class="muted">no crop saved</div>'
        )
        known_badge = '<span class="pill low">known</span>' if face["is_known"] else ""
        display_name = face["label"] or face["person_name"]
        cards.append(f"""
          <div class="card">
            <div style="font-weight:700; font-size:16px; margin-bottom:8px;">
              <a href="/entity/face/{esc(face["id"])}">{esc(display_name)}</a> {known_badge}
            </div>
            {image}
            <div class="muted" style="margin-top:10px;">
              {esc(face["detection_count"])} sightings · {esc(face["embedding_count"])} embeddings<br/>
              first {esc(format_timestamp(face["first_seen_ts"]))}<br/>
              last {esc(format_timestamp(face["last_seen_ts"]))}
            </div>
            <div style="margin-top:8px;">threat {esc(face["threat_score"])}/100</div>
          </div>
        """)

    body = f"""
      <h1>People</h1>
      <div class="sub">Faces clustered into identities. Unnamed people are numbered as strangers until you label them.</div>
      <div class="grid">{"".join(cards)}</div>
      {'<div class="empty">No faces recognised yet. Check Settings to confirm the face models are installed.</div>' if not cards else ''}
    """
    return render_page("Faces", body, "/faces")


@app.get("/entity/{entity_type}/{entity_id}", response_class=HTMLResponse)
def entity_view(entity_type: str, entity_id: str):
    """
    One identity's full history: every sighting, with the finding that explains
    its score, plus the controls to label it or mark it known.
    """
    if entity_type not in ("plate", "face"):
        return render_page("Not found", '<div class="empty">Unknown entity type.</div>')

    connection = get_connection()
    try:
        if entity_type == "plate":
            entity = connection.execute("SELECT * FROM plates WHERE id=?", (entity_id,)).fetchone()
            detections = connection.execute(
                "SELECT * FROM plate_detections WHERE plate_id=? ORDER BY ts DESC", (entity_id,)
            ).fetchall()
            display_name = entity["label"] or entity["plate_text"] if entity else ""
        else:
            entity = connection.execute(
                "SELECT id, person_name, label, is_known, embedding_count, first_seen_ts, "
                "last_seen_ts, detection_count, threat_score, best_image FROM faces WHERE id=?",
                (entity_id,),
            ).fetchone()
            detections = connection.execute(
                "SELECT id, ts, source_file, camera, frame_number, det_confidence, "
                "match_distance, image, lat, lon, drive_id FROM face_detections "
                "WHERE face_id=? ORDER BY ts DESC",
                (entity_id,),
            ).fetchall()
            display_name = entity["label"] or entity["person_name"] if entity else ""

        if not entity:
            return render_page("Not found", '<div class="empty">No such entity.</div>')

        finding = connection.execute(
            "SELECT * FROM correlations WHERE entity_type=? AND entity_id=?",
            (entity_type, entity_id),
        ).fetchone()
    finally:
        connection.close()

    finding_block = '<div class="muted">Not yet scored.</div>'
    if finding:
        try:
            reasons = json.loads(finding["reasons"] or "[]")
        except json.JSONDecodeError:
            reasons = []
        reason_html = "".join(
            f'<div class="reason{" alarm" if reason.startswith("FOLLOWED") else ""}">{esc(reason)}</div>'
            for reason in reasons
        )
        finding_block = f"""
          {severity_pill(finding["severity"])}
          <span style="font-size:22px; font-weight:800; margin-left:8px;">{esc(finding["score"])}/100</span>
          <div style="margin-top:10px;">{reason_html}</div>
        """

    # Only faces can be split: a plate detection's identity is its text, so
    # there is nothing to disambiguate.
    def split_cell(row) -> str:
        if entity_type != "face":
            return ""
        return (f'<td><button onclick="splitDetection(\'{esc(row["id"])}\')" '
                f'title="This sighting is a different person">not this person</button></td>')

    detection_rows = "".join(f"""
        <tr>
          <td>{esc(format_timestamp(row["ts"]))}</td>
          <td>{esc(row["camera"] or "-")}</td>
          <td class="mono" style="font-size:12px;">{esc(row["source_file"] or "-")}</td>
          <td>{esc(f'{row["lat"]:.5f}, {row["lon"]:.5f}' if row["lat"] is not None else "no GPS")}</td>
          <td>{'<img class="evidence" loading="lazy" src="/media/' + esc(row["image"]) + '" style="max-height:52px;" />' if row["image"] else ""}</td>
          {split_cell(row)}
        </tr>
    """ for row in detections)

    split_header = "<th>correct</th>" if entity_type == "face" else ""

    # Merge control, faces only, listing the other identities to merge into.
    merge_block = ""
    if entity_type == "face":
        connection = get_connection()
        try:
            others = connection.execute(
                "SELECT id, person_name, label, detection_count FROM faces "
                "WHERE id != ? ORDER BY last_seen_ts DESC LIMIT 200",
                (entity_id,),
            ).fetchall()
        finally:
            connection.close()

        if others:
            options = "".join(
                f'<option value="{esc(row["id"])}">'
                f'{esc(row["label"] or row["person_name"])} ({esc(row["detection_count"])} sightings)'
                f'</option>'
                for row in others
            )
            merge_block = f"""
              <div class="card">
                <div style="font-weight:700; margin-bottom:10px;">Same person as someone else?</div>
                <div class="row" style="align-items:center;">
                  <select id="mergeTarget" style="background:var(--panel-2); color:var(--text);
                          border:1px solid var(--line); border-radius:9px; padding:7px 11px;
                          font-family:inherit; font-size:13px;">{options}</select>
                  <button onclick="mergeInto()">Merge this into the selected person</button>
                </div>
                <div class="muted" style="margin-top:10px;">
                  Clustering splits one person into several identities when lighting or pose
                  changes a lot. Merging combines their sighting histories, which is what the
                  threat score is computed from - so a follower split across three strangers
                  only scores correctly once they are one person again.
                </div>
              </div>
            """

    header = (
        f'<span class="plate">{esc(display_name)}</span>' if entity_type == "plate"
        else f'<span style="font-size:24px; font-weight:800;">{esc(display_name)}</span>'
    )

    body = f"""
      <h1>{"Vehicle" if entity_type == "plate" else "Person"}</h1>
      <div style="margin-bottom:18px;">{header}</div>

      <div class="card">
        <div style="font-weight:700; margin-bottom:8px;">Surveillance assessment</div>
        {finding_block}
      </div>

      <div class="card">
        <div style="font-weight:700; margin-bottom:10px;">Classification</div>
        <div class="row" style="align-items:center;">
          <input type="text" id="label" placeholder="Give this a name" value="{esc(entity["label"] or "")}" />
          <button onclick="saveLabel()">Save name</button>
          <button onclick="toggleKnown()" id="knownBtn">
            {"Un-mark as known" if entity["is_known"] else "Mark as known (stop scoring)"}
          </button>
        </div>
        <div class="muted" style="margin-top:10px;">
          Marking an entity known removes it from threat scoring permanently. Use it for your
          own vehicles, household and coworkers - it is the most effective way to quiet the
          Findings page.
        </div>
      </div>

      {merge_block}

      <div class="card">
        <div style="font-weight:700; margin-bottom:10px;">Sighting history ({len(detections)})</div>
        <table>
          <tr><th>when</th><th>camera</th><th>clip</th><th>location</th><th>evidence</th>{split_header}</tr>
          {detection_rows}
        </table>
      </div>

      <script>
        const ENTITY_TABLE = "{esc(entity_type)}s";
        const ENTITY_ID = "{esc(entity_id)}";
        let isKnown = {"true" if entity["is_known"] else "false"};

        async function post(payload) {{
          const response = await fetch(`/api/${{ENTITY_TABLE}}/${{ENTITY_ID}}/label`, {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(payload),
          }});
          return response.ok;
        }}
        async function saveLabel() {{
          const value = document.getElementById("label").value;
          if (await post({{ label: value }})) location.reload();
        }}
        async function toggleKnown() {{
          if (await post({{ is_known: !isKnown }})) location.reload();
        }}
        async function mergeInto() {{
          const target = document.getElementById("mergeTarget").value;
          const response = await fetch(`/api/faces/${{ENTITY_ID}}/merge`, {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ into: target }}),
          }});
          const result = await response.json();
          if (result.ok) location.href = `/entity/face/${{target}}`;
          else alert("Merge failed: " + (result.error || "unknown error"));
        }}
        async function splitDetection(detectionId) {{
          const response = await fetch(`/api/faces/detections/${{detectionId}}/split`, {{
            method: "POST",
          }});
          const result = await response.json();
          if (result.ok) location.href = `/entity/face/${{result.new_face_id}}`;
          else alert("Split failed: " + (result.error || "unknown error"));
        }}
      </script>
    """
    return render_page(display_name or "Entity", body, "")


@app.get("/timeline", response_class=HTMLResponse)
def timeline_view():
    """
    Scout's Timeline.vue - drives drawn on a Leaflet map with detections pinned
    where they happened.

    Leaflet loads from a CDN. In a parked car with no signal the map tiles
    won't render; the rest of the dashboard is unaffected, which is why the map
    lives on its own page rather than on the front one.
    """
    head_extra = """
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    """

    body = """
      <h1>Timeline</h1>
      <div class="sub">Drive tracks with plate and face sightings pinned where they were recorded.</div>
      <div id="map"></div>
      <div class="card" style="margin-top:14px;" id="summary"><span class="muted">Loading…</span></div>

      <script>
        (async function () {
          const summary = document.getElementById("summary");
          let data;
          try {
            data = await (await fetch("/api/map")).json();
          } catch (error) {
            summary.innerHTML = '<span class="muted">Could not load map data.</span>';
            return;
          }

          if (typeof L === "undefined") {
            summary.innerHTML = '<span class="muted">Leaflet could not load (no internet). ' +
              'Map data is still available at <a href="/api/map">/api/map</a>.</span>';
            return;
          }

          const map = L.map("map");
          L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "&copy; OpenStreetMap", maxZoom: 19,
          }).addTo(map);

          const bounds = [];

          // Draw each drive as its own polyline so separate journeys stay visually separate.
          const byDrive = {};
          for (const poll of data.polls) {
            const key = poll.drive_id || "unassigned";
            (byDrive[key] = byDrive[key] || []).push([poll.lat, poll.lon]);
            bounds.push([poll.lat, poll.lon]);
          }
          for (const key of Object.keys(byDrive)) {
            if (byDrive[key].length > 1) {
              L.polyline(byDrive[key], { color: "#7aa2ff", weight: 3, opacity: 0.75 }).addTo(map);
            }
          }

          function addMarkers(points, color, kind) {
            for (const point of points) {
              const marker = L.circleMarker([point.lat, point.lon], {
                radius: 7, color: color, fillColor: color, fillOpacity: 0.85, weight: 2,
              }).addTo(map);
              const when = new Date(point.ts * 1000).toLocaleString();

              // Built as DOM nodes, never as an HTML string. point.label is
              // Tesseract OCR output or a person name typed into this very UI,
              // point.image lands in a src= and point.entity_id in an href=,
              // and bindPopup parses a string argument as HTML - so a plate
              // reading `<img onerror=...>` used to become script execution in
              // the operator's browser with the dashboard's session, with no
              // server-side error to notice. textContent cannot do that.
              const popup = document.createElement("div");
              const title = document.createElement("b");
              title.textContent = point.label == null ? "(unlabelled)" : String(point.label);
              popup.appendChild(title);
              popup.appendChild(document.createElement("br"));
              popup.appendChild(document.createTextNode(when));
              popup.appendChild(document.createElement("br"));
              const history = document.createElement("a");
              // encodeURIComponent, so an id carrying a slash or a quote
              // cannot climb out of the path it is supposed to be part of.
              history.href = "/entity/" + encodeURIComponent(kind) + "/" +
                encodeURIComponent(point.entity_id == null ? "" : point.entity_id);
              history.textContent = "history";
              popup.appendChild(history);
              if (point.image) {
                popup.appendChild(document.createElement("br"));
                const evidence = document.createElement("img");
                evidence.src = "/media/" + encodeURIComponent(point.image);
                evidence.style.maxWidth = "180px";
                evidence.style.borderRadius = "6px";
                evidence.style.marginTop = "6px";
                popup.appendChild(evidence);
              }
              marker.bindPopup(popup);
              bounds.push([point.lat, point.lon]);
            }
          }
          addMarkers(data.plates, "#ffb020", "plate");
          addMarkers(data.faces, "#ff5c5c", "face");

          if (bounds.length) {
            map.fitBounds(bounds, { padding: [40, 40] });
          } else {
            map.setView([37.7749, -122.4194], 4);
          }

          summary.innerHTML =
            `<b>${Object.keys(byDrive).length}</b> drives · ` +
            `<b>${data.polls.length}</b> telemetry points · ` +
            `<b>${data.plates.length}</b> plate sightings · ` +
            `<b>${data.faces.length}</b> face sightings` +
            (data.polls.length === 0
              ? '<div class="muted" style="margin-top:8px;">No GPS data yet. Connect the Tesla ' +
                'Fleet API (see Settings) to place detections on the map.</div>'
              : "");
        })();
      </script>
    """
    page = render_page("Timeline", body, "/timeline", head_extra)
    # /timeline renders the same vehicle positions /findmy does, drawn as a
    # track rather than a pin, so it gets the same no-store and no-referrer
    # treatment: a cached drive history and a lat/lon leaked in a Referer are
    # not less of a disclosure for being historic.
    _apply_location_headers(page)
    return page


@app.get("/api/natix")
def api_natix():
    """
    State of the VX360 mirror.

    Everything here is read out of the database, which is deliberate: the
    mirror runs as a host systemd service (it mounts block devices, which a
    container cannot usefully do), so the dashboard learns about the stick the
    same way it learns about anything else - from a table.
    """
    from src import natix

    connection = get_connection()
    try:
        return natix.status(connection)
    finally:
        connection.close()


@app.get("/api/natix/logs")
def api_natix_logs(lines: int = 200):
    """
    The tail of the host mirror service's log.

    The worker is a systemd unit on the host and this is a container, so
    journald is out of reach. The worker mirrors its output into
    BASE_DIR/logs/natix.log, which we already have mounted - see the comment on
    NATIX_LOG_PATH in natix_worker.py for why that beats plumbing journald in.
    """
    from src.common import LOGS_DIR

    lines = max(1, min(lines, 2000))
    log_path = LOGS_DIR / "natix.log"
    if not log_path.exists():
        return {
            "lines": [],
            "exists": False,
            "note": "the mirror service has not written a log yet - it may not "
                    "be installed (sudo ./scripts/install_natix.sh)",
        }

    # Read only the tail. This file rotates at 2MB, so reading the last 256KB
    # covers far more than the 2000-line cap without pulling the whole thing
    # into memory on every poll.
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as handle:
            if size > 262_144:
                handle.seek(-262_144, os.SEEK_END)
            blob = handle.read().decode("utf-8", errors="replace")
    except OSError as error:
        return {"lines": [], "exists": True, "error": str(error)}

    tail = blob.splitlines()[-lines:]
    return {
        "lines": tail,
        "exists": True,
        "size": size,
        "modified": int(log_path.stat().st_mtime),
    }


@app.get("/natix", response_class=HTMLResponse)
def natix_view():
    """
    The NATIX VX360 page.

    Context for anyone reading this cold: the VX360 is a USB stick with a
    computer inside it that uploads TeslaCam footage to NATIX's cloud over
    WiFi. Normally the car writes to it directly. Here the car writes to the
    Jetson instead, so the Jetson has to hand the footage on - and this page is
    where you see whether that is actually happening.
    """
    state = api_natix()
    last = state.get("last_pass") or {}

    if not state["discovery_available"]:
        scan_note = (
            '<div class="muted">This dashboard runs in a container and cannot '
            'enumerate USB devices itself. Everything below is what the host '
            'mirror service last recorded.</div>'
        )
    elif not state["attached"]:
        scan_note = (
            '<div class="muted">No candidate device is attached right now.</div>'
        )
    else:
        rows = []
        for device in state["attached"]:
            verdict = (
                '<span class="pill high">writing</span>' if device["usable"]
                else '<span class="pill low">not used</span>'
            )
            detail = "".join(
                f'<div class="reason">{esc(reason)}</div>' for reason in device["reasons"]
            ) + "".join(
                f'<div class="reason alarm">{esc(problem)}</div>'
                for problem in device["disqualifiers"]
            )
            rows.append(f"""
              <div class="card">
                <div class="row" style="justify-content:space-between; align-items:center;">
                  <div>
                    {verdict}
                    <span style="font-size:17px; font-weight:800; margin-left:8px;">
                      {esc(device["label"] or device["path"])}
                    </span>
                    <span class="muted"> · {esc(device["size_gb"])} GB {esc(device["fstype"] or "?")}</span>
                  </div>
                  <div class="muted">{esc(device["confidence"])}</div>
                </div>
                <div class="muted" style="margin-top:6px;">
                  {esc(device["path"])} · usb {esc(device["usb_id"] or "?")} ·
                  key <code>{esc(device["device_id"])}</code> ·
                  {esc(device["mountpoint"] or "not mounted")}
                </div>
                <div style="margin-top:8px;">{detail}</div>
              </div>
            """)
        scan_note = "".join(rows)

    known_rows = []
    for device in state["known"]:
        mirrored = device.get("mirrored_count") or 0
        pending = device.get("pending") or 0
        total = mirrored + pending
        percent = int(100 * mirrored / total) if total else 0
        free_gb = (device.get("free_bytes") or 0) / (1000 ** 3)
        known_rows.append(f"""
          <tr>
            <td><code>{esc(device["id"])}</code><div class="muted">{esc(device["label"] or "")}</div></td>
            <td>{esc(mirrored)} / {esc(total)}
                <div class="muted">{percent}% · {esc(round((device.get("mirrored_bytes") or 0) / (1024 ** 3), 2))} GB</div></td>
            <td>{esc(round(free_gb, 1))} GB free</td>
            <td>{esc(format_timestamp(device["last_seen_ts"]))}</td>
          </tr>
        """)

    known_table = f"""
      <table>
        <thead><tr><th>Device</th><th>Mirrored</th><th>Space</th><th>Last seen</th></tr></thead>
        <tbody>{"".join(known_rows)}</tbody>
      </table>
    """ if known_rows else '<div class="empty">No VX360 has been mirrored to yet.</div>'

    if last:
        state_label = last.get("state", "unknown")
        if state_label == "ok":
            summary = (
                f'Copied {last.get("copied", 0)} clips '
                f'({round((last.get("copied_bytes") or 0) / (1024 ** 2))} MB), '
                f'skipped {last.get("skipped", 0)}, pruned {last.get("pruned", 0)}, '
                f'failed {last.get("failed", 0)}.'
            )
        else:
            summary = f'{state_label}: {last.get("detail", "")}'
        stopped = last.get("stopped_reason")
        pass_card = f"""
          <div class="card">
            <b>Last mirror pass</b>
            <div class="muted" style="margin-top:4px;">{esc(format_timestamp(last.get("updated_ts")))}</div>
            <div style="margin-top:8px;">{esc(summary)}</div>
            {f'<div class="reason alarm" style="margin-top:8px;">{esc(stopped)}</div>' if stopped else ""}
          </div>
        """
    else:
        pass_card = (
            '<div class="empty">The mirror service has not reported a pass yet. '
            'Install it with <code>sudo ./scripts/install_natix.sh</code>.</div>'
        )

    explainer = f"""
      <div class="card" style="background:var(--panel-2);">
        <div class="muted">
          The car's USB line goes to <b>this Jetson</b>, not to the VX360 - that is what
          lets the Scout pipeline read every clip. The consequence is that the stick sees
          nothing unless we copy it across, so a host service mirrors each clip into
          <code>TeslaCam/{esc(state["default_bucket"])}/&lt;event&gt;/</code> exactly as the car
          would have written it, and the stick's own firmware uploads from there.
          Files are copied under a temporary name and renamed into place, so a power cut
          can never leave a truncated clip. When the stick fills up, our oldest mirrored
          events are pruned first - never anything we did not write.
          Mount point <code>{esc(state["mountpoint"])}</code>, reserve {esc(state["reserve_mb"])} MB,
          confidence floor <code>{esc(state["min_confidence"])}</code>.
        </div>
      </div>
    """

    body = f"""
      <h1>NATIX VX360</h1>
      <div class="sub">{esc(state["total_clips"])} clips in the archive · mirror status and device identity.</div>

      <div class="card" id="live">
        <div class="row" style="justify-content:space-between; align-items:center;">
          <div>
            <span class="pill" id="live-dot">live</span>
            <span style="margin-left:8px; font-weight:700;" id="live-summary">connecting…</span>
          </div>
          <div class="muted" id="live-age">—</div>
        </div>
        <div class="muted" style="margin-top:6px;" id="live-detail"></div>
      </div>

      {explainer}
      {pass_card}
      <h2>Attached now</h2>
      {scan_note}
      <h2>Known devices</h2>
      {known_table}

      <h2>Service log</h2>
      <div class="muted" style="margin-bottom:6px;">
        Written by the host <code>natix-mirror</code> service. Newest at the bottom;
        the view follows new lines unless you scroll up.
      </div>
      <pre id="natixlog" style="background:var(--panel-2); padding:12px; border-radius:8px;
           max-height:460px; overflow:auto; font-size:12px; line-height:1.45;
           white-space:pre-wrap; word-break:break-word;">loading…</pre>
    """
    # Polling rather than SSE: this is served through a Cloudflare tunnel, where
    # a long-lived streaming response is the thing most likely to be buffered or
    # cut. A 5s poll of two small JSON endpoints is boring and it works.
    script = """
<script>
(function () {
  const REFRESH_MS = 5000;
  const logBox = document.getElementById('natixlog');
  const summary = document.getElementById('live-summary');
  const detail = document.getElementById('live-detail');
  const age = document.getElementById('live-age');
  const dot = document.getElementById('live-dot');

  function atBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }

  async function refresh() {
    try {
      const [statusRes, logRes] = await Promise.all([
        fetch('/api/natix', {cache: 'no-store'}),
        fetch('/api/natix/logs?lines=400', {cache: 'no-store'})
      ]);
      const status = await statusRes.json();
      const logs = await logRes.json();

      const known = (status.known || [])[0];
      if (known) {
        const freeGb = (known.free_bytes || 0) / (1024 ** 3);
        summary.textContent = known.mirrored_count + ' / ' + status.total_clips
          + ' clips mirrored';
        detail.textContent = (known.label || known.id)
          + ' · ' + freeGb.toFixed(2) + ' GB free'
          + ' · ' + (known.pending || 0) + ' not on the stick';
      } else {
        summary.textContent = 'no device recorded yet';
        detail.textContent = status.discovery_available
          ? 'nothing attached'
          : 'this dashboard cannot enumerate USB from inside its container';
      }

      const last = status.last_pass || {};
      if (last.updated_ts) {
        const secs = Math.max(0, Math.floor(Date.now() / 1000 - last.updated_ts));
        age.textContent = 'last pass ' + (secs < 90 ? secs + 's' : Math.floor(secs / 60) + 'm') + ' ago';
        dot.className = 'pill ' + (secs < 180 ? 'high' : 'low');
      } else {
        age.textContent = 'no pass recorded';
        dot.className = 'pill low';
      }

      const stick = atBottom(logBox);
      if (logs.exists === false) {
        logBox.textContent = logs.note || 'no log yet';
      } else if (logs.error) {
        logBox.textContent = 'could not read the log: ' + logs.error;
      } else {
        logBox.textContent = (logs.lines || []).join('\\n') || '(log is empty)';
      }
      if (stick) { logBox.scrollTop = logBox.scrollHeight; }
    } catch (err) {
      dot.className = 'pill low';
      age.textContent = 'dashboard unreachable';
    }
  }

  refresh();
  setInterval(refresh, REFRESH_MS);
})();
</script>
"""
    return render_page("NATIX", body + script, "/natix")


# ---------------------------------------------------------------------------
# GPS receiver debug (/gps) and Find My Car (/findmy)
# ---------------------------------------------------------------------------
# Two pages answering two different questions, and the whole design problem is
# stopping either from answering the other's.
#
#   /gps    "is the receiver working?"
#   /findmy "where is the car?"
#
# What connects them is a trap. src/gps.py `continue`s past the database write
# whenever the fix is invalid, so a perfectly healthy receiver in an
# underground garage writes NOTHING to `polls` - byte-identical to a crashed
# gps.py, an unplugged antenna, and a service that was never installed. The
# Tesla path has the same hole from the other side: poller.py never writes in
# the not-online branch, so a correctly parked car at an airport for a week
# produces zero rows for a week. Neither gap is visible in the database.
#
# So "dead service" versus "parked car" cannot be answered from `polls` at all.
# It is answered by joining the polls row against the GPS heartbeat file, and
# that join happens HERE, server-side, once, for both the API and the page.

GPS_HEARTBEAT_FILENAME = "gps.json"

# One definition of every band boundary, because the page and the JSON must
# never disagree about what "stale" means - a page that says "Live" over an API
# that says "stale" is worse than either answer alone.
FINDMY_CLOCK_SKEW_SECONDS = -120

# Same idea for the heartbeat, but a much tighter tolerance: the writer and the
# reader share a kernel clock, so anything more than a couple of seconds into
# the future is a stepped clock rather than jitter.
GPS_CLOCK_SKEW_SECONDS = -5
FINDMY_LIVE_SECONDS = 120
FINDMY_RECENT_SECONDS = 30 * 60
FINDMY_STALE_SECONDS = 12 * 3600

# The GPS heartbeat writes every GPS_HEARTBEAT_SECONDS (default 5). We only use
# this when the file itself fails to declare its interval.
GPS_HEARTBEAT_FALLBACK_INTERVAL = 5

# How long a receiver may be silent before silence means something. Generous on
# purpose: a u-blox that has just been powered up sends nothing for a moment,
# and "replug the cable" is expensive advice to give a healthy cable.
GPS_SILENCE_GRACE_SECONDS = 30


def _apply_location_headers(response: Response) -> None:
    """
    Mark a location response uncacheable and referrer-free.

    no-store keeps a position out of any shared proxy and out of the browser's
    back/forward cache, where the pin would be re-presented later with no age
    attached at all - the exact failure these pages are built to prevent.
    no-referrer matters because both pages link out to Google and Apple Maps:
    without it the outbound navigation carries this dashboard's URL, and any
    lat/lon that ever lands in a query string travels with it to a third party.
    """
    response.headers["Cache-Control"] = "no-store, private"
    response.headers["Referrer-Policy"] = "no-referrer"


def _as_int(value: Any) -> int | None:
    """
    Coerce a heartbeat field to int, or None.

    The heartbeat contract says every key is always present and that absence is
    a writer bug rather than a signal. This exists anyway: the reader and the
    writer are separate processes that get deployed separately, and a debug
    page that raises on a malformed debug file is a page that is unavailable
    exactly when something is malformed.
    """
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_age(seconds: Any) -> str:
    """
    A bare quantity: "12 seconds", "14 minutes", "4 hours", "3 days".

    Deliberately no "ago" and no leading word. The staleness bands own the
    sentence ("Last seen X ago", "No position since ..."), and letting this
    helper generate prose is how a page ends up rendering "Live - last known
    position 4 hours ago" out of two half-correct fragments.
    """
    seconds = _as_int(seconds)
    if seconds is None:
        return "an unknown time"
    seconds = abs(seconds)
    if seconds < 90:
        return f"{seconds} second{'' if seconds == 1 else 's'}"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    hours = seconds // 3600
    if hours < 48:
        return f"{hours} hour{'' if hours == 1 else 's'}"
    days = seconds // 86400
    return f"{days} day{'' if days == 1 else 's'}"


def _format_clock(timestamp: Any) -> str:
    """
    "Tue 3:12 PM" - the short absolute time that sits beside every relative one.

    A relative age alone is the single most common way a Find My page misleads:
    "4 hours ago" recomputed from a frozen page after a phone wakes from sleep
    is indistinguishable from "4 hours ago" that was true when it was written.
    An absolute wall-clock time cannot drift like that, so one is always shown.
    """
    from datetime import datetime
    try:
        moment = datetime.fromtimestamp(int(timestamp))
    except (TypeError, ValueError, OSError):
        return ""
    twelve_hour = moment.hour % 12 or 12
    return f"{moment:%a} {twelve_hour}:{moment:%M} {moment:%p}"


def _hdop_badge(hdop: Any) -> str:
    """Qualitative HDOP, because the bare number means nothing to most readers."""
    value = _as_float(hdop)
    if value is None:
        return "unknown"
    if value <= 2:
        return "good"
    if value <= 5:
        return "fair"
    return "poor"


def _read_gps_heartbeat() -> dict[str, Any]:
    """
    Read ${LOGS_DIR}/gps.json and add the server-computed age and state.

    Opens NO sqlite connection: the whole point of putting the heartbeat in a
    file is that it can still report "fixes are good but the database write is
    failing", and a heartbeat that travelled through the same sqlite connection
    would be blind to exactly the failure it exists to name.

    LOGS_DIR is imported here rather than at module scope, following
    api_natix_logs: the host writes to /mnt/jetsondata/tesla-alerts/logs and
    this container sees the same bytes at /data/logs, so the path must resolve
    through src.common's BASE_DIR and never be written literally.
    """
    from src.common import LOGS_DIR

    server_ts = int(time.time())
    heartbeat_path = LOGS_DIR / GPS_HEARTBEAT_FILENAME

    if not heartbeat_path.exists():
        return {
            "exists": False,
            "note": "the GPS service has not written a heartbeat yet - it may "
                    "not be installed (sudo ./scripts/install_gps.sh)",
            "service_state": "never_installed",
        }

    try:
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exception_object:
        # ValueError covers json.JSONDecodeError. The writer renames the file
        # into place so a torn read is impossible, but a file left behind by an
        # older writer, or truncated by a full disk, is not - and raising here
        # would take down the one page that could explain why.
        return {"exists": True, "error": str(exception_object), "service_state": "unknown"}

    if not isinstance(payload, dict):
        return {"exists": True, "error": "heartbeat is not a JSON object", "service_state": "unknown"}

    written_ts = _as_int(payload.get("written_ts"))
    heartbeat_interval = _as_int(payload.get("heartbeat_interval")) or GPS_HEARTBEAT_FALLBACK_INTERVAL

    if payload.get("schema") != 1:
        # A heartbeat we cannot interpret is not a heartbeat. Rendering an
        # unknown schema's fields would invent a diagnosis out of key names
        # that happen to match.
        service_state = "unknown_version"
        age_seconds = 0 if written_ts is None else max(0, server_ts - written_ts)
    elif written_ts is None:
        service_state = "unknown"
        age_seconds = 0
    else:
        # The host and this container share one kernel clock, so this
        # subtraction is meaningful in a way that the phone's clock is not.
        #
        # NOT clamped at zero. An earlier version did `max(0, ...)`, which made
        # a heartbeat written in the future indistinguishable from one written
        # this instant - so a host whose clock had stepped was reported as
        # "live", maximally healthy, by the one page whose whole job is saying
        # whether the writer is alive. A negative age is a real, diagnosable
        # condition and it gets its own state rather than being flattened into
        # the healthiest one. The tolerance absorbs sub-second clock jitter
        # between the write and the read.
        age_seconds = server_ts - written_ts
        if age_seconds < GPS_CLOCK_SKEW_SECONDS:
            service_state = "clock_skew"
        elif age_seconds < 0:
            age_seconds = 0
            service_state = "live"
        elif age_seconds <= 3 * heartbeat_interval:
            service_state = "live"
        elif age_seconds <= 60:
            # RestartSec=10 plus process startup legitimately produces a gap of
            # roughly this size. Calling it "stopped" would cry wolf on every
            # single restart.
            service_state = "uncertain"
        else:
            service_state = "stopped"

    return {
        "exists": True,
        "service_state": service_state,
        "age_seconds": age_seconds,
        "server_ts": server_ts,
        "heartbeat": payload,
    }


def _gps_verdict(gps_state: dict[str, Any]) -> dict[str, str]:
    """
    The one sentence and the one action at the top of /gps.

    Ordered most-specific-cause first, first match wins, because the states
    overlap heavily: a stopped service also has a stale fix, an unplugged
    receiver also has zero satellites, and a dead antenna also has no fix.
    Reporting the shallowest symptom instead of the deepest cause is how a
    debug page sends somebody out to buy an antenna for a service that is not
    running.
    """
    heartbeat = gps_state.get("heartbeat") or {}
    service_state = gps_state.get("service_state") or "unknown"
    server_ts = _as_int(gps_state.get("server_ts")) or int(time.time())
    heartbeat_age = _as_int(gps_state.get("age_seconds"))

    # Strict `is False`: a missing or null discriminator must not fall through
    # into the branch that assumes a heartbeat exists.
    if gps_state.get("exists") is False:
        return {
            "level": "amber",
            "headline": "The GPS service has never run on this machine.",
            "action": "sudo ./scripts/install_gps.sh",
            "detail": "No heartbeat file has ever been written, so every layer below is unknown "
                      "- this is not the same as a service that ran and stopped.",
        }

    if gps_state.get("error"):
        return {
            "level": "amber",
            "headline": "The GPS heartbeat exists but could not be read.",
            "action": "Check ownership and permissions on the logs directory.",
            "detail": str(gps_state.get("error")),
        }

    if service_state == "unknown_version":
        return {
            "level": "amber",
            "headline": "This heartbeat was written by a different version of gps.py.",
            "action": "Redeploy the GPS service so the writer and this dashboard agree.",
            "detail": f"expected schema 1, found {heartbeat.get('schema')!r}. Nothing below is "
                      f"trustworthy: the field names may mean something else entirely.",
        }

    if service_state == "clock_skew":
        # Ahead of the reader's own clock, which is a stronger statement here
        # than on /findmy: the writer and this container share one kernel
        # clock, so unlike a satellite timestamp there is no legitimate way for
        # this to be in the future. Everything below is timestamped by that
        # same clock, so no freshness claim on this page can be trusted until
        # it is fixed - which is why this outranks the states that merely
        # describe the receiver.
        return {
            "level": "amber",
            "headline": "The heartbeat is timestamped in the future - this host's clock is wrong.",
            "action": "Check the Jetson's clock (timedatectl). Nothing below can be aged until it agrees.",
            "detail": f"written_ts is {abs(heartbeat_age or 0)}s ahead of this container's clock. "
                      f"A GPS receiver can discipline the system clock once it has a fix, which is "
                      f"the usual fix for a headless box that boots with no network.",
        }

    if service_state == "uncertain":
        # Amber and calm, and never the red "not running" sentence. The
        # uncertain band exists SOLELY so that a routine restart does not read
        # as an outage: RestartSec=10 plus process startup legitimately
        # produces a gap of this size, and rendering a 30-second-old heartbeat
        # identically to a 90-second-old one put a service-down banner on this
        # page for 15 to 60 seconds after every `systemctl restart gps`. A
        # banner that cries wolf on every restart is a banner nobody reads on
        # the day it is right.
        return {
            "level": "amber",
            "headline": f"Heartbeat is {_relative_age(heartbeat_age)} late - this is what a "
                        f"restart looks like.",
            "action": "Reload in a minute. If it is still late: systemctl status gps",
            "detail": "Under a minute of silence is inside the window a restart takes "
                      "(RestartSec=10 plus startup), so this is not yet evidence of anything. "
                      "The values below are from that heartbeat, not from this moment.",
        }

    if service_state == "stopped":
        last_fix_at = _as_int(heartbeat.get("last_fix_at"))
        sentences_total = _as_int(heartbeat.get("sentences_total")) or 0
        fix_note = (
            f"It last had a fix at {_format_clock(last_fix_at)}."
            if last_fix_at else "It never reported a fix."
        )
        return {
            "level": "red",
            "headline": f"GPS service is not running. Last heartbeat "
                        f"{_relative_age(heartbeat_age)} ago. {fix_note}",
            "action": "systemctl status gps   /   journalctl -u gps -n 50",
            "detail": ("Every value below is frozen at that heartbeat, not current."
                       if sentences_total > 0 else
                       "It wrote a heartbeat but never read a single sentence, so it died "
                       "before or during the first open of the serial port."),
        }

    device = str(heartbeat.get("device") or "the configured device")
    last_error = str(heartbeat.get("last_error") or "")

    # Permission before presence: "cannot open it" and "it is not there" print
    # different errors and need different fixes, and the permission case still
    # has device_present true, so it must be tested first.
    if "dialout" in last_error or "need root" in last_error:
        return {
            "level": "red",
            "headline": f"Found {device} but cannot open it - it needs root or the dialout group.",
            "action": "Check the unit runs as User=root (deploy/gps.service).",
            "detail": last_error,
        }

    if heartbeat.get("device_present") is False:
        return {
            "level": "red",
            "headline": f"The u-blox receiver is not attached at {device}.",
            "action": f"Plug it in, then compare {device} against `ls /dev/serial/by-id`.",
            "detail": "The service reopens the port by itself within 5 seconds "
                      "(RECONNECT_DELAY_SECONDS) - do not restart it.",
        }

    last_sentence_ts = _as_int(heartbeat.get("last_sentence_ts"))
    silence_seconds = None if last_sentence_ts is None else max(0, server_ts - last_sentence_ts)
    service_started_ts = _as_int(heartbeat.get("started_ts"))
    service_uptime = None if service_started_ts is None else max(0, server_ts - service_started_ts)
    # A freshly started service has an open port and no sentence yet for up to
    # one heartbeat interval, which is not a fault - it is the first second of
    # a healthy start. Without this grace the page told the user to replug a
    # working cable on every single restart, and the one thing this verdict
    # must not do is spend its credibility on the normal case.
    just_started = service_uptime is not None and service_uptime < GPS_SILENCE_GRACE_SECONDS
    if (heartbeat.get("port_open")
            and (silence_seconds is None or silence_seconds > GPS_SILENCE_GRACE_SECONDS)
            and not just_started):
        silent_for = ("since the port opened" if silence_seconds is None
                      else f"for {_relative_age(silence_seconds)}")
        return {
            "level": "amber",
            "headline": f"Serial port is open but the receiver has sent nothing {silent_for}.",
            "action": "Replug or swap the cable, then run `python3 src/gps.py --raw` on the host.",
            "detail": "A charge-only USB cable and a dead receiver module are identical to every "
                      "fix-level check there is; only the absence of bytes distinguishes them.",
        }

    sentences_total = _as_int(heartbeat.get("sentences_total")) or 0
    checksum_failures = _as_int(heartbeat.get("checksum_failures")) or 0
    failure_ratio = checksum_failures / max(1, sentences_total)
    if sentences_total > 0 and failure_ratio > 0.05:
        return {
            "level": "amber",
            "headline": f"Receiving NMEA but {round(failure_ratio * 100)}% of sentences fail "
                        f"checksum - bad cable or interference.",
            "action": "Swap the USB cable; route it away from the inverter and the camera loom.",
            "detail": f"{checksum_failures} bad of {sentences_total} sentences. This is exactly "
                      f"what the checksum exists to catch and is otherwise completely silent.",
        }

    satellites_in_view = _as_int(heartbeat.get("satellites_in_view")) or 0
    satellites_tracked = _as_int(heartbeat.get("satellites_tracked")) or 0
    satellites_used = _as_int(heartbeat.get("satellites_used")) or 0
    started_ts = _as_int(heartbeat.get("started_ts"))
    uptime_seconds = None if started_ts is None else max(0, server_ts - started_ts)

    if not heartbeat.get("fix_valid"):
        if satellites_in_view > 0 and satellites_tracked == 0:
            return {
                "level": "red",
                "headline": f"{satellites_in_view} satellites are in view from the stored almanac "
                            f"but none are being received - antenna or LNA fault.",
                "action": "Check the antenna connector and cable; try a known-good antenna.",
                "detail": "\"In view\" is computed from almanac data and needs no reception at "
                          "all, so an in-view count is never on its own evidence that the antenna "
                          "works. Only a nonzero C/N0 is.",
            }
        if 0 < satellites_tracked < 4:
            return {
                "level": "blue",
                "headline": f"{satellites_in_view} in view, {satellites_tracked} tracked - not "
                            f"enough satellites for a fix regardless of signal strength "
                            f"(3 for 2D, 4 for 3D).",
                "action": "Move the antenna somewhere with more open sky.",
                "detail": "This is a sky-view answer, not an antenna answer: the signals that are "
                          "arriving are being received fine.",
            }
        if satellites_in_view == 0:
            return {
                "level": "blue",
                "headline": f"No satellites in view yet, after {_relative_age(uptime_seconds)} "
                            f"of running.",
                "action": "Give it time and sky - a cold start downloads an almanac and can take "
                          "up to 12.5 minutes.",
                "detail": "Zero in view is ambiguous, not a verdict: it is also exactly what a "
                          "receiver powered up 90 seconds ago reports.",
            }
        last_fix_at = _as_int(heartbeat.get("last_fix_at"))
        searching_for = (uptime_seconds if last_fix_at is None
                         else max(0, server_ts - last_fix_at))
        return {
            "level": "blue",
            "headline": f"Receiver healthy and searching - {satellites_in_view} in view, "
                        f"{satellites_tracked} tracked, {satellites_used} used - no fix for "
                        f"{_relative_age(searching_for)}.",
            "action": "Move the antenna to a window or take it outdoors.",
            "detail": "This is the normal indoor state and it is not an error. Nothing here says "
                      "the hardware is broken.",
        }

    # A valid fix. The persistence check comes BEFORE declaring success even
    # though the contract lists it after: "fix good but nothing is being
    # written" is strictly more specific than "fix acquired", and first-match
    # ordering would otherwise never reach it.
    last_row_written_ts = _as_int(heartbeat.get("last_row_written_ts"))
    speed_mph = _as_float(heartbeat.get("last_fix_speed_mph"))
    is_moving = speed_mph is not None and speed_mph > 1.0
    expected_interval = 5 if is_moving else 300
    write_age = None if last_row_written_ts is None else max(0, server_ts - last_row_written_ts)
    if write_age is None or write_age > 2 * expected_interval:
        written_phrase = ("at all" if write_age is None else f"for {_relative_age(write_age)}")
        last_db_error = heartbeat.get("last_db_error")
        return {
            "level": "amber",
            "headline": f"Fix is good but no row has been written {written_phrase} "
                        f"(expected every {expected_interval}s while "
                        f"{'moving' if is_moving else 'parked'}).",
            "action": "Read the log panel below for \"database is locked\".",
            "detail": (str(last_db_error) if last_db_error else
                       "No database error has been recorded, which points at the sample-interval "
                       "gate rather than at the write itself."),
        }

    hdop = heartbeat.get("hdop")
    hdop_text = "?" if _as_float(hdop) is None else str(hdop)
    return {
        "level": "green",
        "headline": f"Fix acquired - {satellites_used} satellites used, HDOP {hdop_text} "
                    f"({_hdop_badge(hdop)}).",
        "action": "Nothing to do.",
        "detail": f"Last row written to polls {_relative_age(write_age)} ago.",
    }


def _gps_signal_layers(gps_state: dict[str, Any]) -> list[dict[str, str]]:
    """
    The five-layer signal table: service, device, port, NMEA, fix.

    Each layer carries its OWN age rather than one page-level age, because the
    values genuinely come from different moments. A void RMC resets lat/lon and
    marks the fix invalid while `satellites` and `hdop` persist from the last
    GGA, so a single page timestamp would present a healthy satellite count and
    "no fix" as if they described the same instant. Pairing every layer with
    the liveness of the layer beneath it IS this page; a bare "no fix" with
    nothing under it is the thing this table exists to replace.
    """
    heartbeat = gps_state.get("heartbeat") or {}
    server_ts = _as_int(gps_state.get("server_ts")) or int(time.time())
    heartbeat_age = _as_int(gps_state.get("age_seconds"))
    unknown = "unknown"

    def age_text(timestamp: Any) -> str:
        moment = _as_int(timestamp)
        if moment is None:
            return "never"
        return f"{_relative_age(max(0, server_ts - moment))} ago · {_format_clock(moment)}"

    if gps_state.get("exists") is False or gps_state.get("error"):
        # All five layers unknown rather than absent: an empty table reads as
        # "everything is fine and quiet", which is the opposite of the truth.
        reason = str(gps_state.get("error") or "no heartbeat has ever been written")
        return [{"name": name, "value": unknown, "age": reason}
                for name in ("service", "device", "port", "NMEA", "fix")]

    device = str(heartbeat.get("device") or "?")
    device_present = heartbeat.get("device_present")
    last_error = heartbeat.get("last_error")
    last_error_ts = heartbeat.get("last_error_ts")
    sentences_total = _as_int(heartbeat.get("sentences_total"))
    checksum_failures = _as_int(heartbeat.get("checksum_failures")) or 0
    fix_quality = _as_int(heartbeat.get("fix_quality"))
    gsa_fix_type = _as_int(heartbeat.get("gsa_fix_type"))

    quality_names = {0: "no fix", 1: "GPS", 2: "DGPS", 4: "RTK fixed", 5: "RTK float", 6: "dead reckoning"}
    fix_type_names = {1: "no fix", 2: "2D", 3: "3D"}

    return [
        {
            "name": "service",
            "value": f"{gps_state.get('service_state')} · pid {heartbeat.get('pid')} · "
                     f"up {_relative_age(max(0, server_ts - (_as_int(heartbeat.get('started_ts')) or server_ts)))}",
            "age": (f"{_relative_age(heartbeat_age)} ago · "
                    f"{_format_clock(_as_int(heartbeat.get('written_ts')))}"),
        },
        {
            "name": "device",
            "value": f"{device} · {'present' if device_present else 'ABSENT'}",
            # device_present is evaluated by the writer at write time, so its
            # age is the heartbeat's age and nothing else.
            "age": f"as of the heartbeat, {_relative_age(heartbeat_age)} ago",
        },
        {
            "name": "port",
            "value": ("open" if heartbeat.get("port_open") else "closed")
                     + (f" · {last_error}" if last_error else ""),
            "age": age_text(last_error_ts) if last_error else "no error since start",
        },
        {
            "name": "NMEA",
            "value": (f"{sentences_total if sentences_total is not None else unknown} sentences · "
                      f"{checksum_failures} failed checksum"),
            "age": age_text(heartbeat.get("last_sentence_ts")),
        },
        {
            "name": "fix",
            "value": (("valid" if heartbeat.get("fix_valid") else "no fix")
                      + f" · quality {quality_names.get(fix_quality, fix_quality if fix_quality is not None else unknown)}"
                      + f" · {fix_type_names.get(gsa_fix_type, gsa_fix_type if gsa_fix_type is not None else unknown)}"),
            "age": age_text(heartbeat.get("last_fix_at")),
        },
    ]


def _gps_page_state() -> dict[str, Any]:
    """
    The heartbeat plus the server-computed verdict and signal table.

    The verdict is computed HERE and not in the browser on purpose. Everything
    it depends on is an age, every age needs the server's clock, and the page
    must render the same verdict with JavaScript switched off as it does with
    it on. A second implementation in JS would be a second place for the
    "dead service versus parked car" logic to drift.
    """
    gps_state = _read_gps_heartbeat()
    return {
        **gps_state,
        "verdict": _gps_verdict(gps_state),
        "layers": _gps_signal_layers(gps_state),
    }


@app.get("/api/gps")
def api_gps(response: Response):
    """
    The GPS heartbeat with server-computed age and state.

    This is the only thing the page's 2s poll touches and it opens no sqlite
    connection - see _read_gps_heartbeat for why that matters.
    """
    _apply_location_headers(response)
    return _gps_page_state()


@app.get("/api/gps/logs")
def api_gps_logs(response: Response, lines: int = 200):
    """
    Tail of ${LOGS_DIR}/gps.log, the stdout mirror written by gps.py.

    A near-copy of api_natix_logs, including its clamp. The clamp is the point:
    api_polls and api_plate_detections take an unclamped `limit` and this file
    is a location history, so an unbounded `lines` here would let one request
    pull the whole thing.
    """
    from src.common import LOGS_DIR

    _apply_location_headers(response)

    lines = max(1, min(lines, 2000))
    log_path = LOGS_DIR / "gps.log"
    if not log_path.exists():
        return {
            "lines": [],
            "exists": False,
            "note": "the GPS service has not written a log yet - it may not "
                    "be installed (sudo ./scripts/install_gps.sh)",
        }

    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as handle:
            if size > 262_144:
                handle.seek(-262_144, os.SEEK_END)
            blob = handle.read().decode("utf-8", errors="replace")
    except OSError as exception_object:
        return {"lines": [], "exists": True, "error": str(exception_object)}

    tail = blob.splitlines()[-lines:]
    return {
        "lines": tail,
        "exists": True,
        "size": size,
        "modified": int(log_path.stat().st_mtime),
    }


def _findmy_band(age_seconds: int | None) -> str:
    """
    Which staleness band a position falls into. Never clamped into "just now".

    The negative branch is a real path, not a hypothetical: GPS row timestamps
    come from the SATELLITES (gps.py documents this as deliberate, because a
    headless box in a car can boot with no network and a clock hours wrong)
    while the age is computed against this container's system clock. Silently
    clamping a negative age to zero would upgrade a stale position to "Live",
    which is precisely the lie this page must not tell, so it gets its own
    band and its own sentence instead.
    """
    if age_seconds is None:
        return "unreliable"
    if age_seconds < FINDMY_CLOCK_SKEW_SECONDS:
        return "clock_skew"
    if age_seconds <= FINDMY_LIVE_SECONDS:
        return "live"
    if age_seconds <= FINDMY_RECENT_SECONDS:
        return "recent"
    if age_seconds <= FINDMY_STALE_SECONDS:
        return "stale"
    return "unreliable"


# Fixed vocabulary. Never interpolated, never generated - only the number in
# the middle changes. Prose assembled on the fly is how "Live" ends up over a
# four-hour-old pin.
FINDMY_BAND_LABELS = {
    "clock_skew": "Position timestamp is in the future",
    "live": "Live",
    "recent": "Last seen",
    "stale": "Last known position",
    "unreliable": "No position since",
}

# The age at which each band stops being true. The browser ticks the age upward
# between polls, so it needs to know where its own band ends - otherwise the
# word "Live" sits over a position that stopped being live four minutes ago,
# which is the one sentence this page may never print.
FINDMY_BAND_UPPER_BOUND = {
    # None, not FINDMY_CLOCK_SKEW_SECONDS. A NEGATIVE upper bound could never
    # be exceeded by the `age > bound` test the browser applies, so the band
    # instead flipped the moment the ticking age crossed zero - turning "in the
    # future" into "0 seconds ago" and then into a confident past-tense age
    # derived from a clock nobody trusts. A skewed clock does not become
    # trustworthy by waiting; the band holds until the server says otherwise.
    "clock_skew": None,
    "live": FINDMY_LIVE_SECONDS,
    "recent": FINDMY_RECENT_SECONDS,
    "stale": FINDMY_STALE_SECONDS,
    # Nothing above it. An unreliable position can only get older.
    "unreliable": None,
}

# Colour belongs to the AGE, not to the verdict. This line's entire subject is
# how old the position is; colouring it from verdict["level"] let a green
# verdict paint a twelve-hour-old pin green, which is the contradiction the
# band exists to prevent.
FINDMY_BAND_LEVELS = {
    "clock_skew": "amber",
    "live": "green",
    "recent": "blue",
    "stale": "amber",
    "unreliable": "grey",
}


def _findmy_band_level(band: str) -> str:
    return FINDMY_BAND_LEVELS.get(band, "grey")


def _findmy_band_template(band: str, timestamp: Any, position_exists: bool) -> str:
    """
    The band sentence with `{age}` left as the only hole.

    The template, rather than a finished string, is what the browser is handed:
    it can keep the number honest between polls without ever choosing a word.
    Every band label still comes from here and from nowhere else.
    """
    clock = _format_clock(timestamp)
    if not clock:
        # The old sentence rendered "No position since  · an unknown time ago"
        # here: an empty clock, a doubled space, and an age for a thing that
        # has no timestamp to be old relative to.
        return ("The newest position row carries no usable timestamp."
                if position_exists else "No position has ever been recorded.")
    if band == "clock_skew":
        return ("Position timestamp is {age} in the future - check the Jetson clock. "
                f"Row says {clock}.")
    if band == "live":
        return f"Live · updated {{age}} ago · {clock}"
    if band == "recent":
        return f"Last seen {{age}} ago · {clock}"
    if band == "stale":
        return f"Last known position · {{age}} ago · {clock}"
    return f"No position since {clock} · {{age}} ago"


def _findmy_band_expired_template(timestamp: Any, position_exists: bool) -> str:
    """
    What the browser says once its ticking age has left the server's band.

    Deliberately bandless. Re-deriving the band in the browser would be a
    second implementation of the one judgement this page exists to get right;
    leaving the server's word up would keep saying "Live" over a position that
    is no longer live. This third answer asserts only what is still true - a
    time, a lower bound on the age, and that nothing newer has arrived.
    """
    clock = _format_clock(timestamp)
    if not clock:
        return ("The newest position row carries no usable timestamp."
                if position_exists else "No position has ever been recorded.")
    return f"Position from {clock} · at least {{age}} ago · nothing fresher has arrived."


def _findmy_band_sentence(band: str, age_seconds: int | None, timestamp: Any,
                          position_exists: bool) -> str:
    """The one line under the headline, absolute time always attached."""
    return _findmy_band_template(band, timestamp, position_exists).replace(
        "{age}", _relative_age(age_seconds)
    )


def _resolve_address_cache_only(connection, position: dict[str, Any]) -> dict[str, Any] | None:
    """
    Street address for a position, from caches only, in the contract's order.

    NEVER calls reverse_geocode with allow_network=True. That path sleeps up to
    1.1 seconds enforcing Nominatim's rate limit and then allows an 8 second
    socket timeout, on a uvicorn threadpool worker, inside a request a phone is
    waiting on. A missing street name costs the reader nothing - the
    coordinates and the map answer the question - while a nine second page
    load in a parking garage costs them the answer entirely.
    """
    from src import geo

    if position.get("street") or position.get("city"):
        return {
            "street": position.get("street"),
            "city": position.get("city"),
            "display_name": None,
            "source": "poll",
        }

    geocode_id = position.get("geocode_id")
    if geocode_id:
        row = connection.execute(
            "SELECT display_name, road, city FROM geocodes WHERE id = ?", (geocode_id,)
        ).fetchone()
        if row:
            return {
                "street": row["road"],
                "city": row["city"],
                "display_name": row["display_name"],
                "source": "geocode_cache",
            }

    cached = geo.reverse_geocode(connection, position.get("lat"), position.get("lon"),
                                 allow_network=False)
    if cached:
        return {
            "street": cached.get("road"),
            "city": cached.get("city"),
            "display_name": cached.get("display_name"),
            "source": "geocode_cache",
        }

    return None


def _findmy_links(latitude: float, longitude: float, label: str) -> dict[str, str]:
    """
    Deep links into Google and Apple Maps, coordinates only.

    Both buttons are always shown and the user agent is never sniffed: a phone
    with Google Maps installed and a desktop opening Apple Maps in a browser
    are both normal, and guessing wrong hands somebody a dead link at the exact
    moment they are trying to walk to their car.

    The coordinate-only rule is load-bearing. Putting a place name in Google's
    `query=` or using Apple's `q=` without `ll=` triggers a TEXT geocode, and
    the pin lands somewhere plausible but wrong - which is worse than no link,
    because the reader has no way to notice.
    """
    point = f"{latitude:.6f},{longitude:.6f}"
    return {
        "google": f"https://www.google.com/maps/search/?api=1&query={point}",
        "google_walk": f"https://www.google.com/maps/dir/?api=1&destination={point}"
                       f"&travelmode=walking",
        "apple": f"https://maps.apple.com/?ll={point}&q={quote(label or 'Car')}&z=17",
        "apple_walk": f"https://maps.apple.com/?daddr={point}&dirflg=w",
    }


def _position_writer(position: dict[str, Any], gps_state: dict[str, Any]) -> str:
    """
    Which process wrote this row: "tesla", "gps" or "unknown".

    The old test - `shift_state is not None or odometer is not None` - got this
    wrong in both directions. A parked Tesla reports shift_state null, and a
    response with no vehicle_state block carries no odometer either, so a
    perfectly ordinary parked Tesla row scored as GPS. In the other direction
    the fallback claimed "gps" from `exists is True` alone, so a gps.json that
    failed to parse, or one written by another schema, still had its source
    reported as gps on a page whose whole subject is what is still watching the
    car.

    So: positive evidence first, and "unknown" when there is none. Charging is
    the strongest tell there is, because a GPS receiver cannot see a charge
    port at all.
    """
    if position.get("status") == "C":
        return "tesla"
    if (position.get("shift_state") is not None
            or position.get("odometer") is not None
            or position.get("power") is not None):
        return "tesla"
    # Every Tesla-only field is null. gps.py writes exactly that by
    # construction (None, never 0.0, so this test can exist) - but so does a
    # Tesla poll that came back without drive_state or vehicle_state, so a
    # heartbeat that actually PARSED has to vouch for the row before we name
    # the GPS service as its writer.
    if gps_state.get("service_state") in ("live", "uncertain", "stopped"):
        return "gps"
    return "unknown"


def _findmy_state() -> dict[str, Any]:
    """
    Everything /findmy needs, in one payload, with the dead-service versus
    parked-car question already answered.

    Deliberately not built on /api/map or /api/polls. /api/map is ORDER BY ts
    ASC LIMIT 5000, so the moment polls passes 5000 rows it returns the OLDEST
    points and the car's CURRENT position is simply absent from the payload -
    a page built on it would go quietly and permanently wrong at a row count
    nobody is watching. /api/polls returns 500 rows to answer a one-row
    question. This opens one connection and asks one-row questions.
    """
    from src.poller import TeslaFleetClient

    server_ts = int(time.time())
    gps_state = _read_gps_heartbeat()
    heartbeat = gps_state.get("heartbeat") or {}
    service_state = gps_state.get("service_state") or "unknown"
    fix_valid = bool(heartbeat.get("fix_valid"))

    # Reading the env the same way the poller does, so "configured" here means
    # exactly what it means on the Settings page.
    tesla_configured = TeslaFleetClient().configured

    connection = get_connection()
    try:
        position_row = connection.execute(
            "SELECT id, ts, lat, lon, heading, speed, status, shift_state, street, city, "
            "drive_id, geocode_id, odometer, power FROM polls "
            "WHERE lat IS NOT NULL ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        # `power` is selected only to tell the writers apart: gps.py writes it
        # None and the Tesla poller writes drive_state.power, which is present
        # (often 0) even on a parked car whose shift_state is null.
        # `WHERE lat IS NOT NULL` is written this way on purpose: SQLite only
        # uses the partial index idx_polls_fix when the predicate matches
        # TEXTUALLY, so the semantically identical `loc_available=1` silently
        # drops back to a full index walk.

        # Separates "the writer is alive" from "a position is known": the Tesla
        # path writes lat-NULL rows whenever drive_state is missing, so the
        # newest row is very often not the newest LOCATED row.
        newest_poll_ts = _as_int(connection.execute("SELECT MAX(ts) FROM polls").fetchone()[0])

        # Free cross-check, no schema change: dashcam clips newer than the
        # newest poll is positive evidence that the car was powered and
        # recording while the location writer produced nothing. That is a dead
        # GPS, not a quiet car, and no amount of staring at `polls` shows it.
        newest_clip_ts = _as_int(
            connection.execute("SELECT MAX(captured_ts) FROM clips").fetchone()[0]
        )

        position = dict(position_row) if position_row is not None else None
        address = None
        drive = None
        if position is not None:
            address = _resolve_address_cache_only(connection, position)
            if position.get("drive_id"):
                drive_row = connection.execute(
                    "SELECT id, start_ts, end_ts, distance_miles, is_open FROM drives "
                    "WHERE id = ?", (position["drive_id"],)
                ).fetchone()
                drive = dict(drive_row) if drive_row is not None else None
    finally:
        connection.close()

    position_payload = None
    age_seconds = None
    band = "unreliable"
    if position is not None:
        position_ts = _as_int(position.get("ts"))
        # NOT clamped. A negative age means the two clocks disagree and the
        # reader has to be told that, not shielded from it.
        age_seconds = None if position_ts is None else server_ts - position_ts
        band = _findmy_band(age_seconds)

        source = _position_writer(position, gps_state)

        position_payload = {
            "lat": _as_float(position.get("lat")),
            "lon": _as_float(position.get("lon")),
            "ts": position_ts,
            "ts_display": format_timestamp(position_ts),
            "ts_clock": _format_clock(position_ts),
            "age_seconds": age_seconds,
            "heading": _as_float(position.get("heading")),
            "speed_mph": _as_float(position.get("speed")),
            "status": position.get("status"),
            "shift_state": position.get("shift_state"),
            "source": source,
        }
    else:
        source = "unknown"

    # MOTION is never inferred from recency. A three-hour-old status=='D' row
    # means the writer died mid-drive; calling that "Moving" would put a
    # reassuring word on the single most alarming state this page can show.
    if position is None:
        motion = "unknown"
    elif position.get("status") == "C" and source == "tesla":
        motion = "charging"
    elif band == "live" and position.get("status") == "D":
        motion = "moving"
    elif source == "tesla":
        motion = "parked"
    else:
        # Never "Parked" on the GPS path: a GPS cannot see a charge port, so a
        # charging car reads as P to it and "Parked" would be an assertion the
        # data does not support.
        motion = "stationary"

    verdict = _findmy_verdict(
        position=position,
        position_payload=position_payload,
        band=band,
        age_seconds=age_seconds,
        motion=motion,
        source=source,
        gps_state=gps_state,
        heartbeat=heartbeat,
        service_state=service_state,
        fix_valid=fix_valid,
        tesla_configured=tesla_configured,
        server_ts=server_ts,
    )

    # The cross-check only earns a sentence when it actually disagrees. A clip
    # ten minutes newer than the newest poll is noise; an hour is evidence.
    cross_check = None
    if newest_clip_ts and newest_poll_ts and newest_clip_ts > newest_poll_ts + 3600:
        cross_check = (
            f"Dashcam clips are still arriving ({_format_clock(newest_clip_ts)}) but no position "
            f"has been written since {_format_clock(newest_poll_ts)}. The car was powered and "
            f"recording while the location writer produced nothing."
        )

    label = ""
    if address:
        label = address.get("street") or address.get("city") or address.get("display_name") or ""
    links = None
    if position_payload and position_payload["lat"] is not None and position_payload["lon"] is not None:
        links = _findmy_links(position_payload["lat"], position_payload["lon"], label or "Car")

    return {
        "exists": position is not None,
        "service_state": service_state,
        "position_state": verdict["position_state"],
        "position": position_payload,
        "address": address,
        "staleness_band": band,
        "band_level": _findmy_band_level(band),
        "band_sentence": _findmy_band_sentence(
            band, age_seconds, position_payload["ts"] if position_payload else None,
            position is not None,
        ),
        # What the browser needs to keep that sentence's age honest between
        # polls without ever picking a band word of its own: the template with
        # one hole, the age at which this band stops being true, and the
        # bandless sentence to fall back to past that point.
        "band_ticker": {
            "template": _findmy_band_template(
                band, position_payload["ts"] if position_payload else None,
                position is not None,
            ),
            "expired_template": _findmy_band_expired_template(
                position_payload["ts"] if position_payload else None, position is not None
            ),
            "valid_until_age": FINDMY_BAND_UPPER_BOUND.get(band),
        },
        "motion": motion,
        "newest_poll_ts": newest_poll_ts,
        "newest_poll_with_position_ts": position_payload["ts"] if position_payload else None,
        "newest_clip_ts": newest_clip_ts,
        "cross_check": cross_check,
        "links": links,
        "drive": drive,
        "verdict": verdict,
        "note": verdict["note"],
        "server_ts": server_ts,
        "tesla_configured": tesla_configured,
    }


def _findmy_verdict(*, position, position_payload, band, age_seconds, motion, source,
                    gps_state, heartbeat, service_state, fix_valid, tesla_configured,
                    server_ts) -> dict[str, str]:
    """
    The banner above the position: which of the seven states we are in, and the
    sentence that separates a dead service from a parked car.

    Everything here exists because `polls` cannot answer the question on its
    own. "No new rows for three hours" is produced identically by a receiver
    with no sky, a crashed service, an unplugged antenna, a failing database
    write, and a car that simply has not moved. Only the heartbeat joins those
    apart, and getting it wrong sends somebody to reboot a Jetson when the car
    is fine, or reassures them the car is fine when nothing has been watching
    it since Tuesday.
    """
    heartbeat_age = _as_int(gps_state.get("age_seconds"))
    written_ts = _as_int(heartbeat.get("written_ts"))
    last_fix_at = _as_int(heartbeat.get("last_fix_at"))
    started_ts = _as_int(heartbeat.get("started_ts"))
    ts_clock = position_payload["ts_clock"] if position_payload else ""

    if position is None:
        if service_state == "never_installed" and not tesla_configured:
            return {
                "position_state": "no_writer",
                "level": "grey",
                "headline": "Location tracking is not set up.",
                "note": "Neither the GPS service nor the Tesla poller is configured, so nothing "
                        "has ever recorded a position. This is not \"car not found\" - nothing "
                        "has ever been looking.",
                "action": "Install the GPS service (sudo ./scripts/install_gps.sh), or add "
                          "TESLA_* credentials - see Settings.",
            }
        if service_state == "live":
            return {
                "position_state": "no_position_yet",
                "level": "blue",
                "headline": "No position has ever been recorded.",
                "note": "The receiver is running and searching, so this is a sky-view problem "
                        "rather than a setup problem.",
                "action": "Give the antenna a view of the sky and watch /gps.",
            }
        if service_state == "uncertain":
            return {
                "position_state": "writer_restarting",
                "level": "amber",
                "headline": "No position has ever been recorded, and the GPS heartbeat is late.",
                "note": f"Last heartbeat {_relative_age(heartbeat_age)} ago "
                        f"({_format_clock(written_ts)}) - inside the window a restart takes, so "
                        f"not yet an outage.",
                "action": "Reload in a minute; if it is still late, systemctl status gps.",
            }
        if service_state == "stopped":
            return {
                "position_state": "writer_down",
                "level": "red",
                "headline": "No position has ever been recorded, and the GPS service is not running.",
                "note": f"Last heartbeat {_relative_age(heartbeat_age)} ago "
                        f"({_format_clock(written_ts)}).",
                "action": "systemctl status gps",
            }
        if service_state in ("unknown", "unknown_version"):
            return {
                "position_state": "writer_unreadable",
                "level": "amber",
                "headline": "No position has ever been recorded, and the GPS heartbeat cannot "
                            "be read.",
                "note": "So there is no way to tell from here whether the receiver is running "
                        "and finding nothing, or not running at all.",
                "action": "Check /gps - it says which of the two the heartbeat file is.",
            }
        return {
            "position_state": "no_position_yet",
            "level": "grey",
            "headline": "No position has ever been recorded.",
            "note": "No writer has reported one yet.",
            "action": "Check /gps for the state of the receiver.",
        }

    # ORDER MATTERS, and this is the order: what the position itself shows,
    # then how old it is, then what the heartbeat says about the writer.
    #
    # It used to run the other way round. `service_state == "never_installed"`
    # was tested before motion and carried no age or motion qualifier, so on
    # any Tesla-only deployment - no gps.json, because the GPS service was
    # never installed - a twenty-second-old row with status "D" at 61 mph
    # rendered the headline "Car is asleep", in the same card as a motion line
    # reading "Moving" and a band reading "Live · updated 20 seconds ago". The
    # "someone is driving my car" state was unreachable, and the page actively
    # reassured the reader while the car was being driven away.
    #
    # "Asleep" is also gone as a word. Nothing here reads a vehicle_state or a
    # poller heartbeat, so it was a present-tense assertion about the car built
    # out of the absence of a file. Where the only evidence is "no GPS
    # heartbeat exists", the sentence now says that and names where the
    # position came from instead.
    position_is_live = band == "live"

    # Named once and appended to every verdict whose subject is the CAR rather
    # than the writer, so a sentence about a healthy position never silently
    # implies a healthy logger behind it. Worded to the strength of the
    # evidence: a missing or stopped heartbeat is a fact, a late or unreadable
    # one is only an absence of confirmation.
    writer_footnote = ""
    if source == "tesla" and service_state in ("never_installed", "stopped"):
        writer_footnote = (" GPS logging is not running; this position came from the "
                           "Tesla API.")
    elif source == "tesla" and service_state != "live":
        writer_footnote = (" GPS logging cannot be confirmed running; this position came "
                           "from the Tesla API.")
    elif source == "unknown":
        writer_footnote = " Which process wrote this row cannot be determined from it."

    if motion == "moving":
        speed = position_payload.get("speed_mph")
        heading = position_payload.get("heading")
        speed_text = "unknown speed" if speed is None else f"{round(speed)} mph"
        heading_text = "" if heading is None else f", heading {_compass_point(heading)}"
        return {
            "position_state": "healthy_moving",
            "level": "green",
            "headline": f"Moving - {speed_text}{heading_text}, "
                        f"{_relative_age(age_seconds)} ago.",
            "note": "If nobody should be driving the car right now, this is that signal."
                    + writer_footnote,
            "action": "None.",
        }

    if not position_is_live and service_state == "stopped":
        return {
            "position_state": "writer_down",
            "level": "red",
            "headline": "This is a last known position, not a current one.",
            "note": f"GPS logging stopped at {_format_clock(written_ts)} "
                    f"({_relative_age(heartbeat_age)} ago). The car may have moved since.",
            "action": "systemctl status gps - and treat the pin as where the car was when "
                      "logging stopped.",
        }

    if not position_is_live and service_state == "uncertain":
        # Amber, not red, and no "not running". A 30-second-old heartbeat and a
        # 90-second-old one used to render identically, which put an outage
        # banner over the car's position for the 15 to 60 seconds after every
        # `systemctl restart gps`. See the same split in _gps_verdict.
        return {
            "position_state": "writer_restarting",
            "level": "amber",
            "headline": "GPS logging has a late heartbeat - most likely a restart.",
            "note": f"Last heartbeat {_format_clock(written_ts)} "
                    f"({_relative_age(heartbeat_age)} ago), which is inside the window a "
                    f"restart takes. If it is still late in a minute, treat this as a last "
                    f"known position rather than a current one.",
            "action": "Reload in a minute; if it persists, systemctl status gps.",
        }

    if service_state == "live" and not fix_valid and (age_seconds or 0) > FINDMY_RECENT_SECONDS:
        searching_for = (server_ts - last_fix_at) if last_fix_at is not None else (
            (server_ts - started_ts) if started_ts is not None else None)
        return {
            "position_state": "no_fix_service_alive",
            "level": "amber",
            "headline": "Receiver has no satellite lock (likely indoors or under cover).",
            "note": f"The GPS service is alive and has been searching for "
                    f"{_relative_age(searching_for)} - the car is most likely under cover, and "
                    f"this is where it went under. Showing the last known position, from "
                    f"{_relative_age(age_seconds)} ago.",
            "action": "Nothing to do. Check /gps to confirm the receiver is healthy.",
        }

    if service_state == "live" and fix_valid and (age_seconds or 0) > FINDMY_RECENT_SECONDS:
        last_db_error = heartbeat.get("last_db_error")
        return {
            "position_state": "fix_not_persisting",
            "level": "amber",
            "headline": "The fix is good but no position is being written.",
            "note": f"The receiver has had a valid fix but no row has been written for "
                    f"{_relative_age(age_seconds)}, which means the database write is failing."
                    + (f" Last database error: {last_db_error}" if last_db_error else ""),
            "action": "Check /gps for the database error - gps.py otherwise swallows it into a "
                      "printed line nobody reads.",
        }

    # An unreadable or wrong-schema gps.json used to fall straight through to
    # "This is where your car was.", with nothing on the page saying whether
    # anything was still watching the car - the single question this page
    # exists to answer, silently answered wrong.
    if not position_is_live and service_state == "unknown":
        return {
            "position_state": "writer_unreadable",
            "level": "amber",
            "headline": "Nothing here can confirm the car is still being watched.",
            "note": f"The GPS heartbeat exists but could not be read"
                    + (f" ({gps_state.get('error')})" if gps_state.get("error") else "")
                    + f", so a parked car and a dead logger look identical from here. The "
                      f"position below is the newest row in the database, from {ts_clock}."
                    + writer_footnote,
            "action": "Check /gps, then ownership and permissions on the logs directory.",
        }

    if not position_is_live and service_state == "unknown_version":
        return {
            "position_state": "writer_unknown_version",
            "level": "amber",
            "headline": "Nothing here can confirm the car is still being watched.",
            "note": f"The GPS heartbeat was written by a different version of gps.py "
                    f"(schema {heartbeat.get('schema')!r}, expected 1), so nothing in it can be "
                    f"read as a liveness signal. The position below is the newest row in the "
                    f"database, from {ts_clock}." + writer_footnote,
            "action": "Redeploy the GPS service so the writer and this dashboard agree.",
        }

    if band == "clock_skew":
        # Its own verdict, because falling through to "This is where your car
        # was." presented a row stamped in the FUTURE as an ordinary old one.
        return {
            "position_state": "clock_skew",
            "level": "amber",
            "headline": "The newest position is stamped in the future.",
            "note": f"The row says {ts_clock}, which is {_relative_age(age_seconds)} ahead of "
                    f"this server's clock. Row timestamps come from the satellites and this age "
                    f"is measured against the container clock, so until the two agree no age on "
                    f"this page means anything." + writer_footnote,
            "action": "Check the Jetson's clock (timedatectl), then reload.",
        }

    if not position_is_live and service_state == "never_installed":
        if source == "tesla":
            return {
                "position_state": "gps_never_ran",
                "level": "amber",
                "headline": f"GPS logging is not running. Position came from the Tesla API "
                            f"at {ts_clock}.",
                "note": "No GPS heartbeat has ever been written on this machine, so the Tesla "
                        "poller is the only thing recording positions - and it records only "
                        "while the car is awake and online. A gap here is not evidence that "
                        "the car has not moved.",
                "action": "Install the GPS service (sudo ./scripts/install_gps.sh) for logging "
                          "that does not depend on the car being awake.",
            }
        return {
            "position_state": "gps_never_ran",
            "level": "amber",
            "headline": f"GPS logging has not run since {ts_clock}.",
            "note": "There is no GPS heartbeat at all, so nothing has been watching the car "
                    "since that position was written." + writer_footnote,
            "action": "See /timeline for the last complete track, and /gps to install the service.",
        }

    return {
        "position_state": "healthy_stationary",
        # Coloured by the AGE of the position, from the one table that owns
        # that mapping - see FINDMY_BAND_LEVELS.
        "level": _findmy_band_level(band),
        "headline": ("Your car is here." if band == "live"
                     else "This is where your car was."),
        "note": ("Charging." if motion == "charging" else
                 "Parked (in P)." if motion == "parked" else
                 "Stationary. The GPS path cannot see a charge port, so it cannot tell parked "
                 "from charging.") + writer_footnote,
        "action": "Tap a maps link to walk to it.",
    }


def _compass_point(heading: Any) -> str:
    """Heading in degrees to one of eight compass points, for the moving banner."""
    degrees = _as_float(heading)
    if degrees is None:
        return "?"
    points = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return points[int((degrees % 360) / 45 + 0.5) % 8]


@app.get("/api/findmy")
def api_findmy(response: Response):
    """One small JSON object answering the whole "where is my car" question."""
    _apply_location_headers(response)
    return _findmy_state()


# Page-scoped CSS for /gps and /findmy. Kept out of PAGE_STYLES on purpose:
# these two pages are the only ones with staleness bands and satellite bars,
# and adding them to the shared sheet would put a `.stale` rule that greys
# every descendant into the path of every other page in the dashboard.
LOCATION_PAGE_HEAD = """
<style>
  .verdict { border-left:5px solid var(--line); }
  .verdict.green { border-left-color:var(--low); }
  .verdict.blue  { border-left-color:var(--accent); }
  .verdict.amber { border-left-color:var(--medium); }
  .verdict.red   { border-left-color:var(--high); }
  .verdict.grey  { border-left-color:var(--muted); }
  .verdict h2 { margin:0 0 6px 0; font-size:19px; letter-spacing:-0.01em; }
  .verdict .action { margin-top:10px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                     font-size:13px; background:var(--panel-2); padding:8px 10px;
                     border-radius:8px; display:inline-block; }
  .band { font-size:29px; font-weight:800; letter-spacing:-0.02em; margin:2px 0 6px 0;
          line-height:1.15; }
  .band.green { color:var(--low); }
  .band.blue  { color:var(--accent); }
  .band.amber { color:var(--medium); }
  .band.red   { color:var(--high); }
  .band.grey  { color:var(--muted); }
  .address { font-size:19px; font-weight:600; margin:2px 0; }
  .stale, .stale * { color:var(--muted) !important; border-left-color:var(--muted) !important; }
  /* Colour alone was not enough. While the page was greyed for a failed fetch
     the map pin kept full-opacity styling and the maps buttons kept live
     hrefs, so the one click this page must never encourage - navigating to a
     position that may be hours out of date - looked exactly as it does when
     the position is current. */
  .stale .maplinks a { opacity:0.45; font-weight:400; }
  .stale #map { opacity:0.35; filter:grayscale(1); }
  .snr { display:inline-block; height:9px; min-width:2px; background:var(--low);
         border-radius:3px; vertical-align:middle; }
  .snr.weak { background:var(--medium); }
  .snr.none { background:var(--high); }
  pre.tail { background:var(--panel-2); padding:12px; border-radius:8px; max-height:420px;
             overflow:auto; font-size:12px; line-height:1.45; white-space:pre-wrap;
             word-break:break-word; margin:0; }
  .maplinks a { margin-right:8px; margin-bottom:8px; display:inline-block; }
  .maplinks.demoted a { opacity:0.6; font-weight:400; }
  #map { height:420px; }
</style>
"""


def _snr_bar_html(snr: Any) -> str:
    """One C/N0 bar. Blank and 0 are both real values and both mean "heard nothing"."""
    value = _as_int(snr)
    if value is None:
        return '<span class="snr none" style="width:2px;"></span> <span class="muted">no signal</span>'
    width = max(2, min(100, value))
    css_class = "snr none" if value == 0 else ("snr weak" if value < 10 else "snr")
    return (f'<span class="{css_class}" style="width:{width}px;"></span> '
            f'<span class="muted">{esc(value)} dB-Hz</span>')


@app.get("/gps", response_class=HTMLResponse)
def gps_view():
    """
    The GPS receiver debug page.

    Its whole reason to exist is separating "receiver healthy, no sky" from
    "receiver dead", which no fix-level check can do: both produce exactly no
    fix and exactly no rows. So the page never renders a bare "no fix" - every
    layer is shown with the liveness of the layer beneath it.

    Server-rendered first, refreshed second. A car with no signal is the NORMAL
    case for this page, so it has to be fully readable with JavaScript disabled
    and with nothing but the HTML itself on the wire.
    """
    state = _gps_page_state()
    verdict = state["verdict"]
    heartbeat = state.get("heartbeat") or {}

    layer_rows = "".join(
        f"<tr><td><b>{esc(layer['name'])}</b></td>"
        f"<td class=\"mono\">{esc(layer['value'])}</td>"
        f"<td class=\"muted\">{esc(layer['age'])}</td></tr>"
        for layer in state["layers"]
    )

    sky = heartbeat.get("sky") or []
    sky_rows = "".join(
        f"<tr><td class=\"mono\">{esc(entry.get('talker') or '?')} {esc(entry.get('prn'))}</td>"
        f"<td>{esc('—' if entry.get('elevation') is None else str(entry.get('elevation')) + '°')}</td>"
        f"<td>{esc('—' if entry.get('azimuth') is None else str(entry.get('azimuth')) + '°')}</td>"
        f"<td>{_snr_bar_html(entry.get('snr'))}</td></tr>"
        for entry in sky
    ) or '<tr><td colspan="4" class="muted">no satellites reported</td></tr>'

    constellations = heartbeat.get("constellations") or {}
    constellation_text = " · ".join(
        f"{talker} {(counts or {}).get('in_view', 0)} in view / "
        f"{(counts or {}).get('tracked', 0)} tracked"
        for talker, counts in sorted(constellations.items())
    ) or "no constellation reports yet"

    recent_sentences = heartbeat.get("recent_sentences") or []
    raw_text = "\n".join(str(sentence) for sentence in recent_sentences) or "(nothing yet)"

    def number(value: Any) -> str:
        return "—" if value is None else esc(value)

    # The server's age as a machine-readable number, not only as prose. The
    # ticker seeds itself from this, so the line keeps counting even when the
    # first refresh() never lands - a garage, no signal, a bfcache restore,
    # which is the normal case for this page.
    seeded_age = state.get("age_seconds")
    seeded_age_attribute = str(seeded_age) if isinstance(seeded_age, int) else ""

    body = f"""
      <h1>GPS receiver</h1>
      <div class="sub">Why there is, or is not, a position - one layer at a time.</div>

      <div class="card verdict {esc(verdict['level'])}" id="gps-banner">
        <h2 id="gps-headline">{esc(verdict['headline'])}</h2>
        <div class="muted" id="gps-detail">{esc(verdict['detail'])}</div>
        <div class="action" id="gps-action">{esc(verdict['action'])}</div>
      </div>

      <div class="card" id="gps-crashloop" hidden></div>

      <div id="gps-data">
        <div class="muted" style="margin-bottom:10px;">
          <span id="gps-age" data-server-age="{esc(seeded_age_attribute)}" data-server-state="{esc(state.get('service_state') or '')}">heartbeat written {esc(_relative_age(state.get('age_seconds')))} ago</span>
          · <span id="gps-status">{esc(format_timestamp(_as_int(heartbeat.get('written_ts'))) or 'no heartbeat')}</span>
          <button type="button" id="gps-resume" hidden>Paused - tap to resume</button>
        </div>

        <div class="row">
          <div class="stat"><div class="n" id="sat-view">{number(heartbeat.get('satellites_in_view'))}</div>
               <div class="l">in view</div></div>
          <div class="stat"><div class="n" id="sat-tracked">{number(heartbeat.get('satellites_tracked'))}</div>
               <div class="l">tracked (C/N0 &ge; 10)</div></div>
          <div class="stat"><div class="n" id="sat-used">{number(heartbeat.get('satellites_used'))}</div>
               <div class="l">used in the fix</div></div>
          <div class="stat"><div class="n" id="sat-hdop">{number(heartbeat.get('hdop'))}</div>
               <div class="l">HDOP ({esc(_hdop_badge(heartbeat.get('hdop')))})</div></div>
          <div class="stat"><div class="n" id="sat-pdop">{number(heartbeat.get('pdop'))}</div>
               <div class="l">PDOP</div></div>
        </div>
        <div class="muted" style="margin-top:8px;" id="gps-constellations">{esc(constellation_text)}</div>
        <div class="muted" style="margin-top:4px;">
          &ldquo;In view&rdquo; comes from the receiver's stored almanac and needs no reception at
          all, so it is never on its own evidence that the antenna works. Only a nonzero C/N0 is.
        </div>

        <h2>Signal path</h2>
        <table>
          <thead><tr><th>layer</th><th>value</th><th>as of</th></tr></thead>
          <tbody id="gps-layers">{layer_rows}</tbody>
        </table>

        <h2>Satellites</h2>
        <table>
          <thead><tr><th>satellite</th><th>elevation</th><th>azimuth</th><th>C/N0</th></tr></thead>
          <tbody id="gps-sky">{sky_rows}</tbody>
        </table>

        <h2>Raw sentences</h2>
        <div class="muted" style="margin-bottom:6px;">
          Last {esc(len(recent_sentences))} lines the receiver sent, newest at the bottom.
          Latitude and longitude are redacted by the writer: watching the framing, the talker,
          the status letter and the checksums scroll past proves the whole chain works, and the
          coordinates prove nothing while turning a debug page into a location leak.
        </div>
        <pre class="tail" id="gps-raw">{esc(raw_text)}</pre>

        <h2>Service log</h2>
        <pre class="tail" id="gps-log">loading…</pre>
      </div>
    """

    # Polling, not SSE: this is served through a Cloudflare tunnel, where a
    # long-lived streaming response is the thing most likely to be buffered or
    # cut. 2s while there is something to watch, 5s once a fix is held.
    script = """
<script>
(function () {
  "use strict";
  // Every age rendered here was computed by the SERVER against the heartbeat's
  // own written_ts. The browser never calls Date.now() for an age. Three
  // clocks are in play - satellite, host and phone - and the phone's is the
  // one we have least reason to trust. Between polls the counter ticks upward
  // from the server's number using performance.now(), which measures ELAPSED
  // time and so cannot be rewritten by an NTP step or a laptop lid closing.
  var FAST_MS = 2000, SLOW_MS = 5000, MAX_BACKOFF_MS = 60000;
  // Two restarts inside five minutes is a loop; one is a deploy.
  var CRASH_WINDOW_MS = 300000, CRASH_MIN_RESTARTS = 2;
  var baseMs = FAST_MS;
  var failures = 0, inFlight = false, paused = false, timer = null, isStale = false;
  var serverAge = null, serverAgeAt = 0, lastGoodClock = "", serviceState = null;
  var identities = [];

  function el(id) { return document.getElementById(id); }
  function setText(id, value) {
    var node = el(id);
    if (node) { node.textContent = (value === null || value === undefined) ? "\\u2014" : String(value); }
  }
  function clearChildren(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

  // Magnitude only - the SIGN is handled by agedPhrase, never swallowed here.
  // Math.abs() on the whole thing is how a clock-skew age of -3599 rendered as
  // "59 minutes ago" and then ticked DOWN through "29 minutes ago" to "0
  // seconds ago", walking a future-stamped row all the way to "just now" while
  // the band directly above it said the opposite.
  function humanAge(seconds) {
    seconds = Math.floor(Math.abs(seconds));
    if (seconds < 90) { return seconds + (seconds === 1 ? " second" : " seconds"); }
    var minutes = Math.floor(seconds / 60);
    if (minutes < 90) { return minutes + (minutes === 1 ? " minute" : " minutes"); }
    var hours = Math.floor(seconds / 3600);
    if (hours < 48) { return hours + (hours === 1 ? " hour" : " hours"); }
    var days = Math.floor(seconds / 86400);
    return days + (days === 1 ? " day" : " days");
  }

  function agedPhrase(seconds, prefix) {
    if (seconds < 0) { return prefix + humanAge(seconds) + " in the future"; }
    return prefix + humanAge(seconds) + " ago";
  }

  function cell(text, className) {
    var td = document.createElement("td");
    // .textContent, never .innerHTML. Everything below is written into a file
    // by a separate process and read back here; a debug page that executes its
    // own log is a debug page that hands the operator's session to whatever
    // ended up in the log.
    td.textContent = (text === null || text === undefined) ? "\\u2014" : String(text);
    if (className) { td.className = className; }
    return td;
  }

  function paint(data) {
    var verdict = data.verdict || {};
    var banner = el("gps-banner");
    if (banner) { banner.className = "card verdict " + (verdict.level || "grey"); }
    setText("gps-headline", verdict.headline || "");
    setText("gps-detail", verdict.detail || "");
    setText("gps-action", verdict.action || "");

    var layerBody = el("gps-layers");
    var layers = data.layers || [];
    if (layerBody) {
      clearChildren(layerBody);
      for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        var layerRow = document.createElement("tr");
        var nameCell = document.createElement("td");
        var bold = document.createElement("b");
        bold.textContent = String(layers[layerIndex].name);
        nameCell.appendChild(bold);
        layerRow.appendChild(nameCell);
        layerRow.appendChild(cell(layers[layerIndex].value, "mono"));
        layerRow.appendChild(cell(layers[layerIndex].age, "muted"));
        layerBody.appendChild(layerRow);
      }
    }

    var hb = data.heartbeat || {};
    setText("sat-view", hb.satellites_in_view);
    setText("sat-tracked", hb.satellites_tracked);
    setText("sat-used", hb.satellites_used);
    setText("sat-hdop", hb.hdop);
    setText("sat-pdop", hb.pdop);

    var constellations = hb.constellations || {};
    var names = Object.keys(constellations).sort();
    var parts = [];
    for (var nameIndex = 0; nameIndex < names.length; nameIndex++) {
      var counts = constellations[names[nameIndex]] || {};
      parts.push(names[nameIndex] + " " + (counts.in_view || 0) + " in view / "
        + (counts.tracked || 0) + " tracked");
    }
    setText("gps-constellations", parts.join(" \\u00b7 ") || "no constellation reports yet");

    var skyBody = el("gps-sky");
    if (skyBody) {
      clearChildren(skyBody);
      var sats = hb.sky || [];
      for (var satelliteIndex = 0; satelliteIndex < sats.length; satelliteIndex++) {
        var sat = sats[satelliteIndex];
        var satRow = document.createElement("tr");
        satRow.appendChild(cell((sat.talker || "?") + " " + sat.prn, "mono"));
        satRow.appendChild(cell(sat.elevation === null || sat.elevation === undefined
          ? "\\u2014" : sat.elevation + "\\u00b0"));
        satRow.appendChild(cell(sat.azimuth === null || sat.azimuth === undefined
          ? "\\u2014" : sat.azimuth + "\\u00b0"));
        var snrCell = document.createElement("td");
        var bar = document.createElement("span");
        var snr = sat.snr;
        var missing = (snr === null || snr === undefined);
        // Blank and "00" are both valid and both mean "in view, heard nothing".
        // That is the strongest hardware signal this page has, so neither is
        // allowed to be dropped or rounded into the other.
        bar.className = "snr" + ((missing || snr === 0) ? " none" : (snr < 10 ? " weak" : ""));
        bar.style.width = (missing ? 2 : Math.max(2, Math.min(100, snr))) + "px";
        snrCell.appendChild(bar);
        var label = document.createElement("span");
        label.className = "muted";
        label.textContent = " " + (missing ? "no signal" : snr + " dB-Hz");
        snrCell.appendChild(label);
        satRow.appendChild(snrCell);
        skyBody.appendChild(satRow);
      }
      if (!sats.length) {
        var emptyRow = document.createElement("tr");
        var emptyCell = cell("no satellites reported", "muted");
        emptyCell.colSpan = 4;
        emptyRow.appendChild(emptyCell);
        skyBody.appendChild(emptyRow);
      }
    }

    var raw = el("gps-raw");
    if (raw) { raw.textContent = (hb.recent_sentences || []).join("\\n") || "(nothing yet)"; }

    // CRASH LOOP. Restart=always with RestartSec=10 means a service that dies
    // every ten seconds writes a fresh heartbeat every ten seconds and looks
    // permanently healthy to any freshness check. Only pid/started_ts churn
    // across consecutive polls reveals it, so it is detected here and nowhere
    // else.
    if (hb.pid !== undefined || hb.started_ts !== undefined) {
      var identity = String(hb.pid) + ":" + String(hb.started_ts);
      var nowMs = performance.now();
      if (!identities.length || identities[identities.length - 1].identity !== identity) {
        identities.push({identity: identity, at: nowMs});
      }
      // A SLIDING window, and two changes rather than one. Restarting once is
      // what `systemctl restart gps` looks like, and the old test fired on
      // that single change and then had no way to clear for the life of the
      // page - so a page left open all afternoon accused a service that had
      // been stable since lunchtime.
      while (identities.length > 1 && nowMs - identities[0].at > CRASH_WINDOW_MS) {
        identities.shift();
      }
      while (identities.length > 8) { identities.shift(); }
    }
    var crash = el("gps-crashloop");
    if (crash) {
      var restarts = identities.length - 1;
      if (restarts >= CRASH_MIN_RESTARTS) {
        crash.hidden = false;
        crash.className = "card verdict red";
        crash.textContent = "GPS service is restarting: its pid or start time has changed "
          + restarts + " time(s) in the last "
          + Math.round(CRASH_WINDOW_MS / 60000) + " minutes. Suspect "
          + "MemoryMax=128M in deploy/gps.service, or a database lock - read the service log below.";
      } else {
        crash.hidden = true;
      }
    }

    serverAge = (typeof data.age_seconds === "number") ? data.age_seconds : null;
    serverAgeAt = performance.now();
    serviceState = data.service_state || null;
    lastGoodClock = new Date().toLocaleTimeString();
    // 2s while there is something changing to watch, 5s once a fix is held and
    // the only thing moving is the clock.
    baseMs = (data.service_state === "live" && hb.fix_valid) ? SLOW_MS : FAST_MS;
  }

  function paintLog(logs) {
    var box = el("gps-log");
    if (!box) { return; }
    if (!logs) { box.textContent = "could not read the log"; return; }
    if (logs.exists === false) { box.textContent = logs.note || "no log yet"; return; }
    if (logs.error) { box.textContent = "could not read the log: " + logs.error; return; }
    var stick = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    box.textContent = (logs.lines || []).join("\\n") || "(log is empty)";
    if (stick) { box.scrollTop = box.scrollHeight; }
  }

  function tick() {
    if (serverAge === null) {
      // Said out loud rather than left frozen. This line is written ONLY here,
      // so returning early left whatever the HTML was born with on screen,
      // ageing silently, with nothing to tell the reader it had stopped.
      setText("gps-age", "heartbeat age unknown");
      return;
    }
    // Frozen while the host clock is not trusted, for the same reason as the
    // /findmy tick: ticking a negative age forward walks it through zero and
    // reports a stepped clock as a healthy one.
    var elapsed = serviceState === "clock_skew"
      ? 0
      : Math.floor((performance.now() - serverAgeAt) / 1000);
    setText("gps-age", agedPhrase(serverAge + elapsed,
      isStale ? "heartbeat written at least " : "heartbeat written "));
  }

  function clearStale() {
    isStale = false;
    var region = el("gps-data");
    if (region) { region.classList.remove("stale"); }
    setText("gps-status", "refreshed " + lastGoodClock);
  }

  function markStale(error) {
    isStale = true;
    // EVERY field goes grey, not just a status dot. Leaving satellite counts
    // and a fix age in normal styling while nothing at all is arriving is how
    // a monitoring page reports a dead service as maximally healthy.
    var region = el("gps-data");
    if (region) { region.classList.add("stale"); }
    var banner = el("gps-banner");
    if (banner) { banner.classList.add("stale"); }
    var crash = el("gps-crashloop");
    if (crash) { crash.classList.add("stale"); }
    setText("gps-status", "dashboard unreachable ("
      + ((error && error.message) ? error.message : "no response")
      + ") \\u00b7 frozen at " + (lastGoodClock || "page load") + " \\u00b7 not refreshing");
    tick();
  }

  function intervalMs() {
    if (failures === 0) { return baseMs; }
    // Network loss is the normal case in a car. Back off to a minute rather
    // than hammering a tunnel that is already down.
    return Math.min(MAX_BACKOFF_MS, baseMs * Math.pow(2, failures - 1));
  }

  function schedule() {
    if (timer) { clearTimeout(timer); }
    if (paused) { return; }
    timer = setTimeout(refresh, intervalMs());
  }

  function refresh() {
    if (inFlight || paused) { return; }
    inFlight = true;
    var controller = new AbortController();
    // Abort at twice the interval. Without a signal, a stalled tunnel lets
    // requests overlap and accumulate silently - no error, no symptom, just a
    // growing pile of sockets.
    var abortTimer = setTimeout(function () { controller.abort(); }, intervalMs() * 2);
    Promise.all([
      fetch("/api/gps", {cache: "no-store", signal: controller.signal}),
      fetch("/api/gps/logs?lines=400", {cache: "no-store", signal: controller.signal})
    ]).then(function (responses) {
      // ok is checked BEFORE .json(). A 401 after a Cloudflare Access session
      // expires returns HTML, and letting .json() throw the parse error would
      // report an expired login as a dead network.
      if (!responses[0].ok) { throw new Error("HTTP " + responses[0].status); }
      return Promise.all([
        responses[0].json(),
        responses[1].ok ? responses[1].json() : null
      ]);
    }).then(function (payloads) {
      failures = 0;
      paint(payloads[0]);
      paintLog(payloads[1]);
      clearStale();
      tick();
    }).catch(function (error) {
      failures += 1;
      markStale(error);
    }).then(function () {
      clearTimeout(abortTimer);
      inFlight = false;
      schedule();
    });
  }

  var resume = el("gps-resume");
  if (resume) {
    resume.addEventListener("click", function () {
      paused = false;
      resume.hidden = true;
      refresh();
    });
  }

  // Left alone for eight hours a naive interval either burns thousands of
  // pointless requests on a throttled background tab, or gets evicted by iOS
  // Safari and comes back showing a frozen page with no sign the values are
  // old. Pause when hidden, refresh once on return.
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      paused = true;
      if (timer) { clearTimeout(timer); timer = null; }
      if (resume) { resume.hidden = false; }
    } else {
      paused = false;
      if (resume) { resume.hidden = true; }
      refresh();
    }
  });

  // Seeded from the server-rendered number BEFORE the first fetch. Left null
  // until a fetch succeeded, tick() returned early and the server-rendered
  // "heartbeat written 20 seconds ago" stayed on screen forever if that fetch
  // threw - which is the ordinary outcome in a garage.
  var ageNode = el("gps-age");
  var seededAge = ageNode ? parseInt(ageNode.getAttribute("data-server-age"), 10) : NaN;
  if (!isNaN(seededAge)) {
    serverAge = seededAge;
    serverAgeAt = performance.now();
    // Seeded too, or the clock-skew freeze in tick() would not apply until the
    // first successful fetch - which is exactly the fetch that fails in a
    // garage, leaving a skewed age ticking toward plausible.
    serviceState = ageNode.getAttribute("data-server-state") || null;
  }

  setInterval(tick, 1000);
  tick();
  refresh();
})();
</script>
"""
    page = render_page("GPS", body + script, "/gps", LOCATION_PAGE_HEAD)
    # Set on the response object directly rather than through a `response:
    # Response` parameter: FastAPI only merges that parameter's headers when
    # the handler returns data it has to serialize. render_page hands back a
    # finished HTMLResponse, which FastAPI passes straight through, so headers
    # set on the injected sub-response would be silently dropped.
    _apply_location_headers(page)
    return page


# Leaflet is loaded from MEDIA_DIR, which the app already mounts at /media, so
# the map is never a network dependency on a CDN that a parked car cannot
# reach. If the files are not installed the onerror handler fires, the map area
# is replaced with a plain sentence, and the position, address and timestamps
# above it are untouched - a car with no signal is the NORMAL case here, so a
# map that cannot load must never be able to remove the answer.
#
# Honest caveat, stated because it is not obvious: the tile requests themselves
# tell OpenStreetMap's CDN which tile the car is in. Referrer-Policy:
# no-referrer keeps this dashboard's URL out of them, but the tile coordinates
# are the request. The existing /timeline page already makes that trade; if it
# ever stops being acceptable, the fix is a local tile cache, not a smaller z.
def _findmy_map_head() -> str:
    """
    Load Leaflet locally if it is actually vendored, and from the CDN if not.

    The local paths were the whole story here, and MEDIA_DIR/leaflet does not
    exist and nothing in this repository creates it - so every request 404ed,
    `L` was never defined, and the map simply never appeared. Falling back to
    the same CDN /timeline already uses means the map works out of the box; the
    source is stamped into the page so the failure message can name what was
    tried instead of blaming the network for a missing directory.
    """
    vendored_script = MEDIA_DIR / "leaflet" / "leaflet.js"
    vendored_style = MEDIA_DIR / "leaflet" / "leaflet.css"
    if vendored_script.exists() and vendored_style.exists():
        source = "vendored under /media/leaflet"
        style_href = "/media/leaflet/leaflet.css"
        script_src = "/media/leaflet/leaflet.js"
    else:
        source = "the unpkg CDN"
        style_href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        script_src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"

    return f"""
<link rel="stylesheet" href="{style_href}" />
<script>window.__leafletSource = "{esc(source)}";</script>
<script defer src="{script_src}"
        onerror="window.__leafletUnavailable = true;"></script>
"""


@app.get("/findmy", response_class=HTMLResponse)
def findmy_view():
    """
    Find My Car.

    The hard requirement here is that the coordinates, the address, the
    absolute fix time and the staleness band are TEXT in the served HTML before
    any JavaScript runs. Someone opening this page is usually standing in a
    car park on one bar of signal; the map is an enhancement layered on top and
    its failure must never take the answer with it.
    """
    state = _findmy_state()
    verdict = state["verdict"]
    position = state.get("position") or {}
    address = state.get("address") or {}
    band = state.get("staleness_band") or "unreliable"
    links = state.get("links") or {}

    if position.get("lat") is not None and position.get("lon") is not None:
        coordinate_text = f"{position['lat']:.6f}, {position['lon']:.6f}"
    else:
        coordinate_text = "no coordinates recorded"

    address_text = (
        " · ".join(part for part in (address.get("street"), address.get("city")) if part)
        or address.get("display_name")
        or coordinate_text
    )

    motion_labels = {
        "moving": "Moving",
        "charging": "Charging",
        "parked": "Parked (in P)",
        # Never "Parked" on the GPS path: gps.py cannot see a charge port, so a
        # charging car reads as P to it and the stronger word would be a claim
        # the data does not support.
        "stationary": "Stationary",
        "unknown": "Unknown",
    }

    if links:
        confirm_note = (
            '<div class="muted" style="margin-top:6px;">This position is more than 12 hours '
            'old. Following one of these is the exact tap that sends somebody driving to the '
            'wrong address.</div>' if band == "unreliable" else ""
        )
        links_html = f"""
          <div class="maplinks{' demoted' if band == 'unreliable' else ''}" id="findmy-links">
            <a class="btn" id="link-google" href="{esc(links['google'])}"
               rel="noreferrer noopener" target="_blank">Google Maps</a>
            <a class="btn" id="link-google-walk" href="{esc(links['google_walk'])}"
               rel="noreferrer noopener" target="_blank">Walk there (Google)</a>
            <a class="btn" id="link-apple" href="{esc(links['apple'])}"
               rel="noreferrer noopener" target="_blank">Apple Maps</a>
            <a class="btn" id="link-apple-walk" href="{esc(links['apple_walk'])}"
               rel="noreferrer noopener" target="_blank">Walk there (Apple)</a>
            {confirm_note}
          </div>
        """
    else:
        links_html = ""

    cross_check = state.get("cross_check")
    cross_check_html = (
        f'<div class="card verdict amber" id="findmy-crosscheck">{esc(cross_check)}</div>'
        if cross_check else '<div class="card verdict amber" id="findmy-crosscheck" hidden></div>'
    )

    drive = state.get("drive") or {}
    drive_html = ""
    if drive:
        drive_html = (
            f'<div class="muted" style="margin-top:6px;" id="findmy-drive">'
            f'Part of a drive that started {esc(_format_clock(drive.get("start_ts")))}'
            f' · {esc(round(drive.get("distance_miles") or 0, 1))} miles'
            f'{" · still open" if drive.get("is_open") else ""}</div>'
        )

    # Everything the ticker needs before the first fetch, as data rather than
    # as prose it would have to parse back out of the page. "<" is escaped
    # because this lands inside a <script> element, where the parser is looking
    # for a closing tag and not for JSON.
    seed_json = json.dumps({
        "age_seconds": position.get("age_seconds"),
        "band": band,
        "band_level": state.get("band_level"),
        "band_ticker": state.get("band_ticker"),
        "ts_display": position.get("ts_display") or "",
    }).replace("<", "\\u003c")

    body = f"""
      <h1>Find My Car</h1>
      <div class="sub">Where the car is, how old that answer is, and whether anything is
        still watching it.</div>

      <div class="card verdict {esc(verdict['level'])}" id="findmy-banner">
        <h2 id="findmy-headline">{esc(verdict['headline'])}</h2>
        <div class="muted" id="findmy-note">{esc(verdict['note'])}</div>
        <div class="action" id="findmy-action">{esc(verdict['action'])}</div>
      </div>

      {cross_check_html}

      <div id="findmy-data">
        <script type="application/json" id="findmy-seed">{seed_json}</script>

        <div class="card">
          <div class="band {esc(state['band_level'])}" id="findmy-band">{esc(state['band_sentence'])}</div>
          <div class="muted" id="findmy-absolute">
            {esc(position.get('ts_display') or 'no position row')}
          </div>
          <div class="address" id="findmy-address">{esc(address_text)}</div>
          <div class="mono muted" id="findmy-coords">{esc(coordinate_text)}</div>
          <div class="muted" id="findmy-motion">
            {esc(motion_labels.get(state.get('motion'), 'Unknown'))}
            · source {esc(position.get('source') or 'none')}
          </div>
          {drive_html}
          {links_html}
        </div>

        <div class="card" id="map"></div>
        <div class="empty" id="map-fallback" hidden></div>

        <div class="card">
          <div class="muted">
            GPS service: <b id="findmy-service">{esc(state.get('service_state'))}</b> ·
            newest position row
            <span id="findmy-newest">{esc(_format_clock(state.get('newest_poll_with_position_ts')) or 'never')}</span> ·
            newest poll of any kind
            <span id="findmy-newest-poll">{esc(_format_clock(state.get('newest_poll_ts')) or 'never')}</span> ·
            newest dashcam clip
            <span id="findmy-newest-clip">{esc(_format_clock(state.get('newest_clip_ts')) or 'never')}</span>.
            <a href="/gps">Receiver detail →</a>
          </div>
          <div class="muted" style="margin-top:6px;" id="findmy-status">
            served {esc(format_timestamp(state.get('server_ts')))}
            <button type="button" id="findmy-resume" hidden>Paused - tap to resume</button>
          </div>
        </div>
      </div>
    """

    script = """
<script>
(function () {
  "use strict";
  // The age shown here is the SERVER's, computed against the position row's
  // own timestamp, and only ticked upward locally with performance.now()
  // between polls. Recomputing it from Date.now() would let the phone's clock
  // decide how old the car's position is, and GPS row timestamps come from the
  // satellites while the age is measured against the container's clock - two
  // clocks that on a Jetson booted without NTP can differ by hours.
  var STATIONARY_MS = 30000, MOVING_MS = 5000, MAX_BACKOFF_MS = 60000;
  var baseMs = STATIONARY_MS;
  var failures = 0, inFlight = false, paused = false, timer = null, isStale = false;
  var serverAge = null, serverAgeAt = 0, lastGoodClock = "", band = "unreliable";
  var bandLabel = "No position since";
  var bandLevel = "grey";
  // {template, expired_template, valid_until_age} - the band sentence with one
  // hole in it, so the age can be re-rendered every second without this file
  // ever choosing a band word of its own.
  var ticker = null;
  var absoluteClock = "";
  var map = null, marker = null, tileErrors = 0;
  var pending = null;

  function el(id) { return document.getElementById(id); }
  function setText(id, value) {
    var node = el(id);
    if (node) { node.textContent = (value === null || value === undefined) ? "\\u2014" : String(value); }
  }

  // Magnitude only. The SIGN is handled by agedPhrase and by the clock_skew
  // template, never swallowed here: with a poll stamped an hour in the future
  // the server correctly ships age_seconds -3599 and band "clock_skew", and
  // Math.abs() on the whole age made this line count DOWN - "59 minutes ago",
  // "29 minutes ago", "0 seconds ago" - until it flatly contradicted the band
  // printed directly above it. A clamp may never silently upgrade a stale
  // position to "just now"; neither may a missing sign.
  function humanAge(seconds) {
    seconds = Math.floor(Math.abs(seconds));
    if (seconds < 90) { return seconds + (seconds === 1 ? " second" : " seconds"); }
    var minutes = Math.floor(seconds / 60);
    if (minutes < 90) { return minutes + (minutes === 1 ? " minute" : " minutes"); }
    var hours = Math.floor(seconds / 3600);
    if (hours < 48) { return hours + (hours === 1 ? " hour" : " hours"); }
    var days = Math.floor(seconds / 86400);
    return days + (days === 1 ? " day" : " days");
  }

  function agedPhrase(seconds) {
    if (seconds < 0) { return humanAge(seconds) + " in the future"; }
    return humanAge(seconds) + " ago";
  }

  // A link href is the one thing on this page that .textContent cannot make
  // safe, so the scheme is checked before it is ever assigned. These URLs are
  // built server-side out of floats, but the check costs nothing and it is the
  // difference between a broken link and a javascript: URL.
  function setLink(id, url) {
    var node = el(id);
    if (!node) { return; }
    if (typeof url === "string" && url.indexOf("https://") === 0) {
      node.href = url;
      node.hidden = false;
    } else {
      node.removeAttribute("href");
      node.hidden = true;
    }
  }

  function paint(data) {
    pending = data;
    var verdict = data.verdict || {};
    var banner = el("findmy-banner");
    if (banner) { banner.className = "card verdict " + (verdict.level || "grey"); }
    setText("findmy-headline", verdict.headline || "");
    setText("findmy-note", verdict.note || "");
    setText("findmy-action", verdict.action || "");

    var crosscheck = el("findmy-crosscheck");
    if (crosscheck) {
      if (data.cross_check) {
        crosscheck.hidden = false;
        crosscheck.textContent = data.cross_check;
      } else {
        crosscheck.hidden = true;
        crosscheck.textContent = "";
      }
    }

    band = data.staleness_band || "unreliable";
    // Coloured by the age it describes, not by the verdict beside it: taking
    // the class from verdict.level let a green verdict about something else
    // paint a twelve-hour-old band green.
    bandLevel = data.band_level || "grey";
    ticker = data.band_ticker || null;
    // Every band word still comes from the server, from a fixed vocabulary.
    // What arrives here is that sentence with the age left as a hole, so
    // "Live" and "Last known position" can never be assembled out of a number
    // and a guess - only the number moves.
    var bandNode = el("findmy-band");
    if (bandNode) { bandNode.className = "band " + bandLevel; }
    setText("findmy-band", data.band_sentence || "");
    bandLabel = data.band_sentence || "";

    var position = data.position || {};
    absoluteClock = position.ts_display || "no position row";
    setText("findmy-absolute", absoluteClock);
    setText("findmy-coords", (position.lat === null || position.lat === undefined)
      ? "no coordinates recorded"
      : position.lat.toFixed(6) + ", " + position.lon.toFixed(6));

    var address = data.address || {};
    var addressParts = [];
    if (address.street) { addressParts.push(address.street); }
    if (address.city) { addressParts.push(address.city); }
    var addressText = addressParts.join(" \\u00b7 ") || address.display_name;
    if (!addressText) {
      addressText = (position.lat === null || position.lat === undefined)
        ? "no coordinates recorded"
        : position.lat.toFixed(6) + ", " + position.lon.toFixed(6);
    }
    setText("findmy-address", addressText);

    var motionLabels = {
      moving: "Moving", charging: "Charging", parked: "Parked (in P)",
      stationary: "Stationary", unknown: "Unknown"
    };
    setText("findmy-motion", (motionLabels[data.motion] || "Unknown")
      + " \\u00b7 source " + (position.source || "none"));

    var links = data.links || {};
    setLink("link-google", links.google);
    setLink("link-google-walk", links.google_walk);
    setLink("link-apple", links.apple);
    setLink("link-apple-walk", links.apple_walk);
    var linkBox = el("findmy-links");
    if (linkBox) { linkBox.className = "maplinks" + (band === "unreliable" ? " demoted" : ""); }

    setText("findmy-service", data.service_state);

    serverAge = (position && typeof position.age_seconds === "number") ? position.age_seconds : null;
    serverAgeAt = performance.now();
    lastGoodClock = new Date().toLocaleTimeString();
    // Matched to the source cadence, not to what feels responsive:
    // GPS_PARKED_SAMPLE_SECONDS is 300, so 5s polling while parked would burn
    // sixty requests per actual update. Moving, the source really is 5s.
    baseMs = (data.motion === "moving") ? MOVING_MS : STATIONARY_MS;

    updateMap(position);
  }

  function paintBand(age) {
    var bandNode = el("findmy-band");
    if (!bandNode || !ticker || !ticker.template) { return; }
    // Once the ticking age passes the top of the band the server assigned, its
    // word stops being true and is dropped rather than repeated: this is the
    // largest text on the page, and "Live" over a five-minute-old position is
    // the one thing it may never say. A negative age keeps its own template,
    // which already reads "in the future".
    var bound = ticker.valid_until_age;
    var expired = (bound !== null && bound !== undefined && age >= 0 && age > bound);
    var template = expired ? (ticker.expired_template || ticker.template) : ticker.template;
    bandNode.textContent = template.replace("{age}", humanAge(age));
    bandNode.className = "band " + (expired ? "grey" : bandLevel);
    bandLabel = bandNode.textContent;
  }

  function tick() {
    if (serverAge === null) {
      // No age to count from. Say which absolute moment is on screen rather
      // than leaving a relative phrase in place that has stopped moving.
      setText("findmy-absolute", absoluteClock || "no position row");
      return;
    }
    // A skewed clock is FROZEN, not ticked. Adding elapsed time to a negative
    // age walks it up through zero and out the other side, so a position
    // stamped an hour in the future silently became "0 seconds ago" and then a
    // confident past-tense age - the exact upgrade-to-plausible this page may
    // never perform. Waiting does not make an untrusted clock trustworthy, so
    // the age holds at what the server last said until the server says
    // something else.
    var elapsed = band === "clock_skew"
      ? 0
      : Math.floor((performance.now() - serverAgeAt) / 1000);
    var age = serverAge + elapsed;
    // The absolute time is ALWAYS beside the relative one. A relative age
    // alone, recomputed on a page a phone woke up holding, is the single most
    // common way a Find My page misleads.
    setText("findmy-absolute", absoluteClock + " \\u00b7 " + agedPhrase(age)
      + (isStale ? " (at least - not refreshing)" : ""));
    // The band sentence carries an age too, and it used to be a static string
    // written once per poll - so 30 seconds of stationary polling, or an hour
    // of failed ones, left it asserting an age that had stopped being true.
    paintBand(age);
  }

  function clearStale() {
    isStale = false;
    var region = el("findmy-data");
    if (region) { region.classList.remove("stale"); }
    var banner = el("findmy-banner");
    if (banner) { banner.classList.remove("stale"); }
    var links = el("findmy-links");
    if (links) { links.className = "maplinks" + (band === "unreliable" ? " demoted" : ""); }
    styleMarker();
    setText("findmy-status", "refreshed " + lastGoodClock);
  }

  function markStale(error) {
    isStale = true;
    // EVERY field, not just a status line. natix_view's catch touches only the
    // dot and the age text and leaves the rest on screen in normal styling;
    // here that would render an hour-old position as though it were current,
    // which is the one failure this page cannot be allowed to have.
    var region = el("findmy-data");
    if (region) { region.classList.add("stale"); }
    var banner = el("findmy-banner");
    if (banner) { banner.classList.add("stale"); }
    // The deep links and the pin are ACTIONS, and greying the words around
    // them left both fully live: full-opacity pin, live hrefs, one tap from
    // driving to an address that may be hours out of date. Demote them here,
    // and the click handler below asks before navigating on an answer this
    // page can no longer stand behind.
    var links = el("findmy-links");
    if (links) { links.className = "maplinks demoted"; }
    styleMarker();
    setText("findmy-status", "dashboard unreachable ("
      + ((error && error.message) ? error.message : "no response")
      + ") \\u00b7 frozen at " + (lastGoodClock || "page load") + " \\u00b7 not refreshing");
    tick();
  }

  function mapUnavailable(message) {
    var mapNode = el("map");
    var fallback = el("map-fallback");
    if (mapNode) { mapNode.hidden = true; }
    if (fallback) {
      fallback.hidden = false;
      fallback.textContent = message;
    }
  }

  function updateMap(position) {
    if (!map || position.lat === null || position.lat === undefined) { return; }
    var point = [position.lat, position.lon];
    // Zoomed out at city level when the position is older than twelve hours,
    // so the geometry itself refuses to imply a precision the age does not
    // support.
    map.setView(point, band === "unreliable" ? 12 : 17);
    if (marker) {
      marker.setLatLng(point);
    } else {
      marker = L.circleMarker(point, {
        radius: 9, color: "#7aa2ff", fillColor: "#7aa2ff",
        fillOpacity: band === "unreliable" ? 0 : 0.85, weight: 3,
        dashArray: band === "unreliable" ? "4 4" : null
      }).addTo(map);
    }
    styleMarker();
  }

  function styleMarker() {
    if (!marker) { return; }
    // Hollow and dashed for a position the page will not vouch for - either
    // because it is older than twelve hours, or because nothing has refreshed
    // it since the network went away.
    var hollow = (band === "unreliable") || isStale;
    marker.setStyle({
      color: isStale ? "#9a9ab0" : "#7aa2ff",
      fillColor: isStale ? "#9a9ab0" : "#7aa2ff",
      fillOpacity: hollow ? 0 : 0.85,
      dashArray: hollow ? "4 4" : null
    });
  }

  function initMap() {
    if (window.__leafletUnavailable || typeof L === "undefined") {
      // Names the source it tried. "unavailable offline" was the message shown
      // when the vendored files simply did not exist, which blamed the network
      // for a directory nothing in this repository ever created.
      mapUnavailable("The map library did not load from "
        + (window.__leafletSource || "its configured source")
        + ". The position, address and time above are the complete answer and do not "
        + "depend on it.");
      return;
    }
    try {
      map = L.map("map").setView([0, 0], 2);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap", maxZoom: 19
      }).addTo(map).on("tileerror", function () {
        tileErrors += 1;
        // The common offline case is Leaflet served from the browser's HTTP
        // cache with the tiles blocked, which passes a `typeof L` check
        // cleanly and then renders a featureless grey rectangle. Say so
        // instead of showing the grey.
        if (tileErrors > 4 && map) {
          map.remove();
          map = null;
          marker = null;
          mapUnavailable("Map tiles unavailable offline. The position, address and time above "
            + "are the complete answer and do not depend on them.");
        }
      });
      if (pending) { updateMap(pending.position || {}); }
    } catch (error) {
      mapUnavailable("The map could not start (" + error.message + "). Everything above still "
        + "answers the question.");
    }
  }

  function intervalMs() {
    if (failures === 0) { return baseMs; }
    return Math.min(MAX_BACKOFF_MS, baseMs * Math.pow(2, failures - 1));
  }

  function schedule() {
    if (timer) { clearTimeout(timer); }
    if (paused) { return; }
    timer = setTimeout(refresh, intervalMs());
  }

  function refresh() {
    if (inFlight || paused) { return; }
    inFlight = true;
    var controller = new AbortController();
    var abortTimer = setTimeout(function () { controller.abort(); }, intervalMs() * 2);
    fetch("/api/findmy", {cache: "no-store", signal: controller.signal})
      .then(function (response) {
        if (!response.ok) { throw new Error("HTTP " + response.status); }
        return response.json();
      })
      .then(function (data) {
        failures = 0;
        paint(data);
        clearStale();
        tick();
      })
      .catch(function (error) {
        failures += 1;
        markStale(error);
      })
      .then(function () {
        clearTimeout(abortTimer);
        inFlight = false;
        schedule();
      });
  }

  var linkBox = el("findmy-links");
  if (linkBox) {
    linkBox.addEventListener("click", function (event) {
      if (band !== "unreliable" && !isStale) { return; }
      var anchor = event.target && event.target.closest ? event.target.closest("a") : null;
      if (!anchor) { return; }
      var reason = isStale
        ? "This page has lost contact with the dashboard, so this position is at least as "
          + "old as the last refresh and may be much older (" + bandLabel + ")."
        : "This position is more than 12 hours old (" + bandLabel + ").";
      if (!window.confirm(reason + " Navigate to it anyway?")) {
        event.preventDefault();
      }
    });
  }

  var resume = el("findmy-resume");
  if (resume) {
    resume.addEventListener("click", function () {
      paused = false;
      resume.hidden = true;
      refresh();
    });
  }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      paused = true;
      if (timer) { clearTimeout(timer); timer = null; }
      if (resume) { resume.hidden = false; }
    } else {
      paused = false;
      if (resume) { resume.hidden = true; }
      refresh();
    }
  });

  // Seeded from what the server already rendered, BEFORE the first fetch.
  // serverAge stayed null until a fetch succeeded, so tick() returned early -
  // and a failed initial refresh (a garage, no signal, a bfcache restore, the
  // exact scenario this page exists for) left "Live - updated 20 seconds
  // ago" on screen indefinitely: greyed, still the largest text on the page,
  // and still the word "Live".
  (function seedFromServer() {
    var node = el("findmy-seed");
    if (!node) { return; }
    var seed;
    try { seed = JSON.parse(node.textContent); } catch (error) { return; }
    band = seed.band || band;
    bandLevel = seed.band_level || bandLevel;
    ticker = seed.band_ticker || null;
    absoluteClock = seed.ts_display || "no position row";
    if (typeof seed.age_seconds === "number") {
      serverAge = seed.age_seconds;
      serverAgeAt = performance.now();
    }
  })();

  // Leaflet is a deferred script, so it has not run yet when this inline
  // script does. Waiting for DOMContentLoaded is what makes the onerror
  // fallback reachable instead of racing it.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initMap);
  } else {
    initMap();
  }

  setInterval(tick, 1000);
  tick();
  refresh();
})();
</script>
"""
    page = render_page("Find My", body + script, "/findmy",
                       LOCATION_PAGE_HEAD + _findmy_map_head())
    # See the note in gps_view: a handler that returns a finished Response gets
    # it passed straight through, so the headers have to be set on it here.
    _apply_location_headers(page)
    return page


@app.get("/settings", response_class=HTMLResponse)
def settings_view():
    """
    Scout's Settings.vue - what's running, what isn't, and why.

    Every optional stage self-reports here, so a missing model file or absent
    Tesla credential is visible rather than showing up as silently empty pages.
    """
    from src.common import MODELS_DIR

    # Each import is guarded: the web container should render this page even if
    # a model runtime is broken.
    def probe(loader):
        try:
            return loader()
        except Exception as exception_object:
            return f"error: {exception_object}"

    def alpr_status():
        from src.alpr import PlateReader
        return PlateReader().status_text()

    def face_status():
        from src.faces import FaceEngine
        return FaceEngine().status_text()

    def tesla_status():
        from src.poller import TeslaFleetClient
        return TeslaFleetClient().status_text()

    def notify_status():
        from src import notify
        return notify.status_text()

    stats = api_stats()

    model_files = []
    for name in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx", "plate_detector.pt"):
        path = MODELS_DIR / name
        state = f"{path.stat().st_size // 1024} KB" if path.exists() else "missing"
        model_files.append(f'<tr><td class="mono">{esc(name)}</td><td>{esc(state)}</td></tr>')

    stage_rows = "".join(
        f'<tr><td><b>{esc(name)}</b></td><td>{esc(probe(loader))}</td></tr>'
        for name, loader in [
            ("ALPR (plates)", alpr_status),
            ("Face recognition", face_status),
            ("Tesla telemetry", tesla_status),
            ("Notifications", notify_status),
        ]
    )

    stat_rows = "".join(
        f"<tr><td>{esc(key.replace('_', ' '))}</td><td>{esc(value)}</td></tr>"
        for key, value in stats.items()
    )

    body = f"""
      <h1>Settings &amp; status</h1>
      <div class="sub">What each stage of the pipeline is doing right now.</div>

      <div class="card">
        <div style="font-weight:700; margin-bottom:10px;">Pipeline stages</div>
        <table>{stage_rows}</table>
      </div>

      <div class="card">
        <div style="font-weight:700; margin-bottom:10px;">Model weights</div>
        <table><tr><th>file</th><th>state</th></tr>{"".join(model_files)}</table>
        <div class="muted" style="margin-top:10px;">
          Missing weights? Run <span class="mono">./scripts/fetch_models.sh</span> on the host.
          Directory: <span class="mono">{esc(MODELS_DIR)}</span>
        </div>
      </div>

      <div class="card">
        <div style="font-weight:700; margin-bottom:10px;">Database</div>
        <table>{stat_rows}</table>
      </div>

      <div class="card">
        <div style="font-weight:700; margin-bottom:10px;">API</div>
        <div class="muted">
          <a href="/api/findings">/api/findings</a> ·
          <a href="/api/plates">/api/plates</a> ·
          <a href="/api/faces">/api/faces</a> ·
          <a href="/api/polls">/api/polls</a> ·
          <a href="/api/drives">/api/drives</a> ·
          <a href="/api/map">/api/map</a> ·
          <a href="/alerts">/alerts</a>
        </div>
      </div>
    """
    return render_page("Settings", body, "/settings")
