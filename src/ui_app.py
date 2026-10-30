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
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse

from src.common import (
    BASE_DIR,
    MEDIA_DIR,
    ALERTS_DIR,
    ensure_dirs,
    sighting_frame_name,
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

    # Off the event loop: it loads a YOLO model, and startup must not wait.
    import threading
    threading.Thread(target=_warm_stage_status, name="warm-settings", daemon=True).start()

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

# --- Path prefix -------------------------------------------------------------
# azula.tedebyte.dev is shared: '/' and most of /api/* belong to the spv-server,
# so the tunnel hands this dashboard everything under /tesla instead. Every
# link a page emits starts with URL_ROOT, and _StripUrlRoot below removes it
# again before routing, so the routes themselves never mention it. Unprefixed
# paths keep working too, which is what the LAN and tailnet use at :8080.
# Empty (the default) means the dashboard owns its whole host.
URL_ROOT = os.environ.get("URL_ROOT", "").strip().rstrip("/")

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


class _StripUrlRoot:
    """
    Pure ASGI, not @app.middleware: it must run OUTSIDE require_basic_auth so
    the auth gate, and LOCATION_ROUTES in particular, see /findmy and not
    /tesla/findmy. Starlette wraps later-added middleware around earlier, so
    adding this after the auth middleware makes it the outer layer.
    """

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if URL_ROOT and scope["type"] in ("http", "websocket"):
            path = scope["path"]
            if path == URL_ROOT:
                # /tesla -> /tesla/, or every relative fetch on the page
                # would resolve one level too high.
                query = scope.get("query_string", b"").decode("latin-1")
                location = URL_ROOT + "/" + (f"?{query}" if query else "")
                await RedirectResponse(location, status_code=308)(scope, receive, send)
                return
            if path.startswith(URL_ROOT + "/"):
                scope = dict(scope, path=path[len(URL_ROOT):])
                raw_path = scope.get("raw_path")
                if raw_path and raw_path.startswith(URL_ROOT.encode() + b"/"):
                    scope["raw_path"] = raw_path[len(URL_ROOT):]
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return

        # Server-Timing: how long the Orin spent before the first byte. The
        # loading screen reads it from the browser's navigation and resource
        # timing, which is how it can say "the Orin built this page in 9 ms"
        # separately from the network time around it.
        started = time.perf_counter()

        async def timed_send(message):
            if message["type"] == "http.response.start":
                elapsed_ms = (time.perf_counter() - started) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"server-timing", f"app;dur={elapsed_ms:.1f};desc=\"Orin\"".encode()))
                message = dict(message, headers=headers)
            await send(message)

        await self.inner(scope, receive, timed_send)


app.add_middleware(_StripUrlRoot)


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
  .media-link { display:inline-block; padding:4px 9px; font-size:12px; margin:2px 4px 2px 0;
                white-space:nowrap; }
  input[type=text] { background:var(--panel-2); border:1px solid var(--line); color:var(--text);
                     border-radius:9px; padding:7px 11px; font-family:inherit; font-size:13px; }
  #map { height:520px; border-radius:14px; border:1px solid var(--line); }
  .reason { padding:3px 0; font-size:13px; }
  .reason::before { content:"▸ "; color:var(--accent); }
  .reason.alarm { color:var(--high); font-weight:700; }
  .reason.alarm::before { content:"⚠ "; }
  /* Pinned to the bottom of every page. --footer-h is measured by the
     script below (the footer wraps differently at every width) so the page
     can pad itself and nothing ends up hidden behind it. */
  :root { --footer-h: 64px; }
  body { padding-bottom:calc(var(--footer-h) + 12px); }
  footer.sysbar { position:fixed; left:0; right:0; bottom:0; z-index:20;
                  background:rgba(11,11,15,0.94); backdrop-filter:blur(8px);
                  border-top:1px solid var(--line); padding:8px 20px calc(8px + env(safe-area-inset-bottom));
                  color:var(--muted); font-size:12px; }
  footer.sysbar .inner { max-width:1400px; margin:0 auto; display:flex; flex-wrap:wrap;
                         gap:6px 18px; align-items:center; }
  footer.sysbar .label { font-weight:700; color:var(--text); margin-right:4px; }
  footer.sysbar .metric b { color:var(--text); font-weight:600;
                            font-variant-numeric:tabular-nums; }
  footer.sysbar .metric.warn b { color:var(--medium); }
  footer.sysbar .metric.high b { color:var(--high); }
  footer.sysbar .metric.good b { color:var(--low); }
  footer.sysbar .inner + .inner { margin-top:4px; }
  @media (max-width: 700px) {
    /* One scrollable line per row on a phone, not a wall of wrapped metrics.
       After the base .inner rules, which would otherwise win. */
    footer.sysbar { padding-left:12px; padding-right:12px; }
    footer.sysbar .inner { flex-wrap:nowrap; overflow-x:auto; white-space:nowrap;
                           scrollbar-width:none; -webkit-overflow-scrolling:touch; }
    footer.sysbar .inner::-webkit-scrollbar { display:none; }
  }
  .pipeline-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }
  .pipe-head { display:flex; justify-content:space-between; align-items:center; gap:10px;
               flex-wrap:wrap; margin-bottom:12px; }
  .demo-buttons { display:flex; gap:6px; flex-wrap:wrap; }
  .lag-wrap { display:flex; align-items:center; gap:14px; }
  .lag { font-size:30px; font-weight:800; font-variant-numeric:tabular-nums; letter-spacing:-0.02em;
         min-width:4.2em; }
  .lag.good { color:var(--low); } .lag.warn { color:var(--medium); } .lag.high { color:var(--high); }
  .lag.off { color:var(--muted); font-size:16px; font-weight:600; }
  .demo-buttons .btn { cursor:pointer; font:inherit; font-size:12px; padding:6px 10px; }
  .demo-banner { background:rgba(122,162,255,0.10); border:1px solid rgba(122,162,255,0.35);
                 border-radius:10px; padding:8px 12px; margin-bottom:12px; font-size:13px; }
  .flow { display:flex; align-items:stretch; gap:6px; margin-bottom:16px; }
  .flow .node { background:var(--panel-2); border:1px solid var(--line); border-radius:12px;
                padding:10px 12px; min-width:0; flex:1 1 0; transition:border-color .3s; }
  .flow .node.wide { flex:2.6 1 0; }
  .flow .outs { display:flex; flex-direction:column; gap:6px; flex:1.1 1 0; min-width:0; }
  .flow .outs .node { flex:1 1 auto; padding:7px 10px; }
  .flow .arrow { display:flex; align-items:center; color:var(--muted); font-size:18px; flex:0 0 auto; }
  .node .nh { display:flex; align-items:center; gap:7px; font-size:12px; font-weight:700;
              text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:6px; }
  .node .nv { font-size:20px; font-weight:800; font-variant-numeric:tabular-nums; }
  .node .ns { font-size:12px; color:var(--muted); margin-top:3px; overflow-wrap:anywhere; }
  .node .dot { width:9px; height:9px; border-radius:50%; background:var(--line); flex:0 0 auto; }
  .node.ok   { border-color:rgba(74,208,122,0.45); }   .node.ok .dot   { background:var(--low); }
  .node.warn { border-color:rgba(255,176,32,0.55); }   .node.warn .dot { background:var(--medium); }
  .node.bad  { border-color:rgba(255,92,92,0.6); }     .node.bad .dot  { background:var(--high); }
  .node.active { border-color:var(--accent); }
  .node.active .dot { background:var(--accent); animation:pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { box-shadow:0 0 0 0 rgba(122,162,255,.7); }
                     50% { box-shadow:0 0 0 5px rgba(122,162,255,0); } }
  @media (prefers-reduced-motion: reduce) { .node.active .dot { animation:none; } }
  .states { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px; }
  .states span { font-size:11px; padding:2px 8px; border-radius:999px; border:1px solid var(--line);
                 color:var(--muted); }
  .states span.on { background:var(--accent); border-color:var(--accent); color:#0b0b0f; font-weight:700; }
  .states span.on.bad { background:var(--high); border-color:var(--high); }
  .states span.on.warn { background:var(--medium); border-color:var(--medium); }
  .steps { display:grid; grid-template-columns:repeat(4,1fr); gap:4px; margin:8px 0 6px; }
  .steps div { border:1px solid var(--line); border-radius:8px; padding:5px 6px; text-align:center; }
  .steps b { display:block; font-size:11px; font-weight:700; }
  .steps small { display:block; font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .steps div.done { border-color:rgba(74,208,122,0.45); }
  .steps div.now { border-color:var(--accent); background:rgba(122,162,255,0.12); }
  .bar { height:5px; background:var(--line); border-radius:3px; overflow:hidden; }
  .bar div { height:100%; width:0; background:var(--accent); transition:width .8s linear; }
  @media (max-width: 900px) {
    .flow { flex-direction:column; }
    .flow .arrow { justify-content:center; transform:rotate(90deg); height:14px; }
    .flow .node.wide, .flow .node, .flow .outs { flex:1 1 auto; }
    .flow .outs { flex-direction:row; flex-wrap:wrap; }
    .flow .outs .node { flex:1 1 140px; }
  }
  .pipeline-grid table td { padding:5px 8px; font-size:13px; }
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
    ("/storage", "Storage"),
    ("/settings", "Settings"),
]


# ---------------------------------------------------------------------------
# Activity layer: the loading screen, and anything loading after it
# ---------------------------------------------------------------------------
# On every page: a loading screen built from the browser's own measurements
# (navigation timing, the Server-Timing header _StripUrlRoot adds, and every
# fetch the page makes), shown again the moment a link is followed; then a
# progress bar and pill for any request still running after 250 ms, a
# spinner on the button that started it, red for failures, and a log of all
# of it behind the footer's "Loaded in" badge. Repeated GETs to one URL are
# the pages' polling and stay out of sight unless slow or failing.
ACTIVITY_STYLES = """
  /* --- Activity layer: loading screen, progress bar, pill, log ---------- */
  #tv-boot { position:fixed; inset:0; z-index:1000; background:rgba(11,11,15,0.97);
             display:flex; align-items:center; justify-content:center; padding:16px;
             transition:opacity .35s ease; }
  #tv-boot.fade { opacity:0; pointer-events:none; }
  #tv-boot[hidden] { display:none; }
  #tv-boot .panel { width:min(640px, 100%); max-height:calc(100vh - 32px); overflow:auto;
                    background:var(--panel); border:1px solid var(--line); border-radius:16px;
                    padding:18px 18px 14px; box-shadow:0 20px 60px rgba(0,0,0,.5); }
  #tv-boot .top { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
  #tv-boot .title { font-weight:800; font-size:17px; letter-spacing:-0.01em; }
  #tv-boot .elapsed { margin-left:auto; font-variant-numeric:tabular-nums; color:var(--muted);
                      font-size:13px; }
  #tv-boot .close { background:none; border:0; color:var(--muted); font-size:18px; cursor:pointer;
                    padding:0 2px; }
  #tv-boot .where { color:var(--muted); font-size:12px; margin-bottom:12px; overflow-wrap:anywhere; }
  #tv-boot .meter { height:4px; background:var(--line); border-radius:2px; overflow:hidden;
                    margin-bottom:12px; }
  #tv-boot .meter div { height:100%; width:0; background:var(--accent); transition:width .25s ease; }
  #tv-boot ol { list-style:none; margin:0; padding:0; font-size:13px; }
  #tv-boot li { display:grid; grid-template-columns:18px 1fr auto; gap:8px; align-items:baseline;
                padding:4px 0; border-bottom:1px solid rgba(42,42,56,.6); }
  #tv-boot li .what { overflow-wrap:anywhere; }
  #tv-boot li .what small { display:block; color:var(--muted); font-size:11px; }
  #tv-boot li .ms { font-variant-numeric:tabular-nums; color:var(--muted); font-size:12px;
                    white-space:nowrap; }
  #tv-boot li.bg { opacity:.6; }
  #tv-boot .ico { width:14px; height:14px; display:inline-block; text-align:center; line-height:14px;
                  font-size:12px; }
  #tv-boot .ok   .ico { color:var(--low); }
  #tv-boot .fail .ico { color:var(--high); }
  #tv-boot .fail .what { color:var(--high); }
  #tv-boot .run  .ico { border:2px solid var(--line); border-top-color:var(--accent);
                        border-radius:50%; animation:tvspin .8s linear infinite; width:10px; height:10px; }
  #tv-boot .foot { color:var(--muted); font-size:11px; margin-top:10px; }
  @keyframes tvspin { to { transform:rotate(360deg); } }

  #tv-bar { position:fixed; top:0; left:0; right:0; height:3px; z-index:999; pointer-events:none;
            opacity:0; transition:opacity .2s; overflow:hidden; }
  #tv-bar.on { opacity:1; }
  #tv-bar div { position:absolute; top:0; bottom:0; width:40%; background:var(--accent);
                animation:tvslide 1.1s ease-in-out infinite; }
  #tv-bar.err div { background:var(--high); animation:none; width:100%; }
  @keyframes tvslide { from { left:-40%; } to { left:100%; } }

  #tv-pill { position:fixed; right:14px; bottom:calc(var(--footer-h, 0px) + 10px); z-index:998; max-width:min(420px, calc(100vw - 28px));
             background:var(--panel); border:1px solid var(--line); border-radius:999px;
             padding:7px 14px; font-size:12px; display:flex; gap:8px; align-items:center;
             box-shadow:0 8px 24px rgba(0,0,0,.4); cursor:pointer; transition:opacity .2s, transform .2s; }
  #tv-pill[hidden] { display:none; }
  #tv-pill.err { border-color:var(--high); color:var(--high); }
  #tv-pill .spin { width:10px; height:10px; border:2px solid var(--line); border-top-color:var(--accent);
                   border-radius:50%; animation:tvspin .8s linear infinite; flex:0 0 auto; }
  #tv-pill span.t { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  button.tv-busy { position:relative; color:transparent !important; pointer-events:none; }
  button.tv-busy::after { content:""; position:absolute; left:50%; top:50%; width:12px; height:12px;
                          margin:-8px 0 0 -8px; border:2px solid var(--muted); border-top-color:var(--accent);
                          border-radius:50%; animation:tvspin .8s linear infinite; }
  .tv-loadbadge { cursor:pointer; text-decoration:underline dotted; }
  @media (prefers-reduced-motion: reduce) {
    #tv-bar div, #tv-pill .spin, #tv-boot .run .ico, button.tv-busy::after { animation:none; }
  }
"""

ACTIVITY_HTML = """
<noscript><style>#tv-boot { display:none !important; }</style></noscript>
<div id="tv-boot" role="status" aria-live="polite">
  <div class="panel">
    <div class="top"><span class="title" id="tv-boot-title">Loading Tesla Vision</span>
      <span class="elapsed" id="tv-boot-elapsed">0 ms</span>
      <button class="close" id="tv-boot-close" title="Hide (Esc)" hidden>✕</button></div>
    <div class="where" id="tv-boot-where"></div>
    <div class="meter"><div id="tv-boot-meter"></div></div>
    <ol id="tv-boot-steps"></ol>
    <div class="foot" id="tv-boot-foot">Every line is measured by this browser, not estimated.</div>
  </div>
</div>
<div id="tv-bar"><div></div></div>
<div id="tv-pill" hidden title="Open the activity log"><span class="spin"></span><span class="t" id="tv-pill-text"></span></div>
"""

# Runs in <head>, before any page script, so its fetch wrapper sees every
# request a page makes. Plain string (JavaScript braces). It only touches the
# DOM once the body exists.
ACTIVITY_SCRIPT = """
<script>
(function () {
  "use strict";
  const T0 = performance.now();   // ms since this navigation began
  const log = [];                 // every request this page has made
  const inflight = new Set();     // foreground requests still running
  const seenGets = new Set();     // a GET URL seen before is a poll -> background
  const BG_SLOW_MS = 3000, FG_SHOW_MS = 250;
  let booted = false, bootDone = null, mode = "boot";   // boot | nav | log | hidden
  let lastButton = null, lastButtonAt = 0;
  const $ = (id) => document.getElementById(id);

  function fmtMs(ms) { return ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + " s"; }
  function fmtKB(bytes) { return bytes == null ? "" : bytes < 1024 ? bytes + " B" : (bytes / 1024).toFixed(bytes < 10240 ? 1 : 0) + " KB"; }
  function shortPath(url) {
    try { const u = new URL(url, location.href); return (u.origin === location.origin ? "" : u.host) + u.pathname + (u.search.length > 1 && u.search.length < 30 ? u.search : ""); }
    catch (e) { return String(url); }
  }
  function serverMs(entry) {
    const t = entry && entry.serverTiming && entry.serverTiming.find((s) => s.name === "app");
    return t ? t.duration : null;
  }
  function timingFor(url) {
    try {
      const list = performance.getEntriesByName(new URL(url, location.href).href);
      return list.length ? list[list.length - 1] : null;
    } catch (e) { return null; }
  }

  // ---- The step list (loading screen and log view share it) -------------
  const steps = [];
  // Display order, whatever order things happen in: the footer's first
  // fetches start before DOMContentLoaded, but connecting came first.
  const RANK = {conn: 1, ttfb: 2, dl: 3, dom: 4, assets: 5, img: 30, ready: 99};
  function step(key, what, detail) {
    let s = steps.find((x) => x.key === key);
    if (!s) {
      s = {key, what, detail: detail || "", state: "run", ms: null, bg: false,
           rank: RANK[key] || 10, seq: steps.length};
      steps.push(s);
      steps.sort((a, b) => a.rank - b.rank || a.seq - b.seq);
    }
    else { s.what = what; if (detail != null) s.detail = detail; }
    render(); return s;
  }
  function settle(key, state, ms, detail) {
    const s = steps.find((x) => x.key === key);
    if (!s) return;
    s.state = state; s.ms = ms; if (detail != null) s.detail = detail;
    render();
  }
  function fillSizes() {
    // A resource's timing entry (size, Server-Timing) appears only once its
    // body has arrived - after fetch() resolves - so look it up late.
    for (const r of log) {
      if (r.state === "run" || r.sized) continue;
      const t = timingFor(r.url);
      if (!t) continue;
      r.sized = true;
      r.size = t.encodedBodySize || t.transferSize || null;
      r.server = serverMs(t);
      const s = steps.find((x) => x.key === "f" + r.id);
      if (s) s.detail = (r.status ? "HTTP " + r.status : "") + (r.size ? " · " + fmtKB(r.size) : "") +
                        (r.server != null ? " · Orin " + fmtMs(r.server) : "");
    }
  }
  function render() {
    const list = $("tv-boot-steps");
    if (!list || mode === "hidden") return;
    fillSizes();
    const shown = mode === "log" ? steps.concat(log.filter((r) => r.after).map(logStep)) : steps;
    list.replaceChildren(...shown.map((s) => {
      const li = document.createElement("li");
      li.className = s.state + (s.bg ? " bg" : "");
      const ico = document.createElement("span"); ico.className = "ico";
      ico.textContent = s.state === "ok" ? "✓" : s.state === "fail" ? "✗" : s.state === "info" ? "·" : "";
      const what = document.createElement("span"); what.className = "what"; what.textContent = s.what;
      if (s.detail) { const small = document.createElement("small"); small.textContent = s.detail; what.appendChild(small); }
      const ms = document.createElement("span"); ms.className = "ms";
      ms.textContent = s.ms != null ? fmtMs(s.ms) : s.state === "run" ? "…" : "";
      li.append(ico, what, ms); return li;
    }));
    const done = steps.filter((s) => s.state !== "run").length;
    const meter = $("tv-boot-meter");
    if (meter) meter.style.width = (mode === "nav" ? 30 : Math.round(100 * done / Math.max(steps.length, 1))) + "%";
  }
  function logStep(r) {
    return {key: "log" + r.id, what: r.method + " " + r.path,
            detail: (r.status ? r.status + " · " : "") + (r.size != null ? fmtKB(r.size) + " · " : "") +
                    (r.server != null ? "Orin " + fmtMs(r.server) + " · " : "") + (r.bg ? "background refresh · " : "") +
                    new Date(r.wall).toLocaleTimeString(),
            state: r.state, ms: r.ms, bg: r.bg};
  }

  // ---- fetch: every request, logged; foreground ones shown ----------------
  const realFetch = window.fetch.bind(window);
  let nextId = 1;
  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || String(input);
    const method = ((init && init.method) || (input && input.method) || "GET").toUpperCase();
    const path = shortPath(url);
    const key = method + " " + path.split("?")[0];
    const bg = (init && init.tvBackground) || (method === "GET" && seenGets.has(key));
    if (method === "GET") seenGets.add(key);
    const r = {id: nextId++, method, path, url, bg, state: "run", start: performance.now(), wall: Date.now(),
               after: booted, status: null, ms: null, size: null, server: null,
               quiet: Boolean(init && init.tvQuiet)};   // an expected 404 (e.g. no boxes yet)
    log.push(r); if (log.length > 150) log.shift();
    if (!booted) step("f" + r.id, method + " " + path, "requesting…");
    if (!bg) { inflight.add(r); scheduleActivity(); }
    if (method !== "GET" && lastButton && performance.now() - lastButtonAt < 400 && document.contains(lastButton)) {
      r.button = lastButton; r.button.classList.add("tv-busy");
    }
    return realFetch(input, init).then((response) => {
      finish(r, response.ok ? "ok" : "fail", response.status);
      return response;
    }, (error) => {
      finish(r, "fail", null, error && error.name === "AbortError" ? "cancelled" : "network error");
      throw error;
    });
  };
  function finish(r, state, status, why) {
    r.ms = performance.now() - r.start; r.state = state; r.status = status || why || null;
    inflight.delete(r);
    setTimeout(render, 300);   // by then the body is usually in: sizes appear
    if (r.button) r.button.classList.remove("tv-busy");
    if (!r.after) {
      settle("f" + r.id, state, r.ms,
        (status ? "HTTP " + status : why || "") + (r.size ? " · " + fmtKB(r.size) : "") +
        (r.server != null ? " · Orin " + fmtMs(r.server) : ""));
    }
    if (state === "fail" && !r.quiet && (r.after || booted) && !(r.bg && why === "cancelled")) flashError(r);
    else if (r.bg && r.ms > BG_SLOW_MS) flashSlow(r);
    scheduleActivity();
    if (mode === "log") render();
    if (!booted) maybeBootDone();
  }
  document.addEventListener("click", (event) => {
    const button = event.target.closest && event.target.closest("button");
    if (button) { lastButton = button; lastButtonAt = performance.now(); }
  }, true);

  // ---- After load: top bar + pill for anything foreground -----------------
  let activityTimer = null, pillHold = null;
  function scheduleActivity() {
    clearTimeout(activityTimer);
    activityTimer = setTimeout(showActivity, inflight.size ? FG_SHOW_MS : 0);
  }
  function showActivity() {
    if (!booted || mode === "nav") return;
    const bar = $("tv-bar"), pill = $("tv-pill"), text = $("tv-pill-text");
    if (!bar || !pill) return;
    const running = [...inflight].filter((r) => performance.now() - r.start >= FG_SHOW_MS);
    if (running.length) {
      bar.className = "on";
      const first = running[0];
      text.textContent = first.method + " " + first.path + " · " + fmtMs(performance.now() - first.start) +
                         (running.length > 1 ? " · +" + (running.length - 1) + " more" : "");
      pill.className = ""; pill.hidden = false; pill.querySelector(".spin").hidden = false;
      clearTimeout(activityTimer); activityTimer = setTimeout(showActivity, 200);
    } else if (!pillHold) {
      bar.className = ""; pill.hidden = true;
    }
  }
  function flashError(r) {
    const bar = $("tv-bar"), pill = $("tv-pill"), text = $("tv-pill-text");
    if (!pill) return;
    text.textContent = "✗ " + r.method + " " + r.path + " failed" + (r.status ? " (" + r.status + ")" : "");
    pill.className = "err"; pill.hidden = false; pill.querySelector(".spin").hidden = true;
    bar.className = "on err";
    clearTimeout(pillHold);
    pillHold = setTimeout(() => { pillHold = null; bar.className = ""; showActivity(); }, 6000);
  }
  function flashSlow(r) {
    const pill = $("tv-pill"), text = $("tv-pill-text");
    if (!pill || inflight.size) return;
    text.textContent = "slow: " + r.method + " " + r.path + " took " + fmtMs(r.ms);
    pill.className = ""; pill.hidden = false; pill.querySelector(".spin").hidden = true;
    clearTimeout(pillHold);
    pillHold = setTimeout(() => { pillHold = null; showActivity(); }, 4000);
  }

  // ---- The loading screen itself -------------------------------------------
  function connectionLine() {
    const host = location.hostname;
    const direct = /^(\\d+\\.){3}\\d+$/.test(host) || host === "localhost";
    const route = direct ? (host.startsWith("100.") ? "direct over the tailnet" : "direct on the local network")
                         : "through Cloudflare (" + host + ")";
    const net = navigator.connection && navigator.connection.effectiveType ? " · network " + navigator.connection.effectiveType : "";
    return location.pathname + " · " + route + net;
  }
  function navigationSteps() {
    const nav = performance.getEntriesByType("navigation")[0];
    if (!nav) { step("nav", "Page received", "no navigation timing in this browser"); settle("nav", "info", null); return; }
    const dns = nav.domainLookupEnd - nav.domainLookupStart;
    const tcp = nav.connectEnd - nav.connectStart;
    const tls = nav.secureConnectionStart > 0 ? nav.connectEnd - nav.secureConnectionStart : 0;
    const wait = nav.responseStart - nav.requestStart;
    const orin = serverMs(nav);
    const reused = tcp <= 0 && dns <= 0;
    step("conn", "Connected", reused ? "reused an open connection"
      : "DNS " + fmtMs(dns) + " · TCP " + fmtMs(Math.max(0, tcp - tls)) + (tls ? " · TLS " + fmtMs(tls) : ""));
    settle("conn", "ok", reused ? 0 : nav.connectEnd - nav.startTime);
    step("ttfb", "Orin built the page", orin != null
      ? "server " + fmtMs(orin) + " · network and tunnel " + fmtMs(Math.max(0, wait - orin))
      : "first byte after " + fmtMs(wait));
    settle("ttfb", "ok", wait);
    const size = nav.encodedBodySize || nav.transferSize;
    step("dl", "Page downloaded", nav.transferSize === 0 && nav.encodedBodySize > 0 ? "from browser cache" : fmtKB(size));
    settle("dl", "ok", nav.responseEnd - nav.responseStart);
  }
  function parsedStep() {
    const nav = performance.getEntriesByType("navigation")[0];
    step("dom", "Page parsed", document.querySelectorAll("*").length + " elements");
    // Called from DOMContentLoaded itself, when the browser has not yet
    // stamped domContentLoadedEventEnd - so measure to now.
    settle("dom", "ok", Math.max(0, performance.now() - (nav ? nav.responseEnd : T0)));
  }
  function assetSteps() {
    const assets = performance.getEntriesByType("resource")
      .filter((e) => e.initiatorType === "script" || e.initiatorType === "link" || e.initiatorType === "css");
    if (!assets.length) return;
    const slowest = Math.max(...assets.map((e) => e.duration));
    step("assets", "Scripts and styles", assets.length + " file(s): " +
         assets.map((e) => shortPath(e.name).split("/").pop()).join(", "));
    settle("assets", "ok", slowest);
  }
  const images = [];
  function watchImages() {
    const visible = [...document.images].filter((img) => {
      if (img.loading !== "lazy") return true;
      const box = img.getBoundingClientRect();
      return box.top < innerHeight && box.bottom > 0;
    });
    if (!visible.length) return;
    images.push(...visible);
    const started = performance.now();
    const update = () => {
      const done = images.filter((img) => img.complete).length;
      const broken = images.filter((img) => img.complete && img.naturalWidth === 0).length;
      step("img", "Images", done + " of " + images.length + " on screen" + (broken ? " · " + broken + " failed" : ""));
      if (done === images.length) { settle("img", broken ? "fail" : "ok", performance.now() - started); maybeBootDone(); }
    };
    images.forEach((img) => { if (!img.complete) { img.addEventListener("load", update); img.addEventListener("error", update); } });
    update();
  }
  let loadFired = false;
  function maybeBootDone() {
    if (booted || !loadFired) return;
    if (steps.some((s) => s.state === "run")) return;
    completeBoot(false);
  }
  function completeBoot(timedOut) {
    if (booted) return;
    booted = true;
    bootDone = performance.now();   // performance.now() counts from the navigation's start
    step("ready", timedOut ? "Shown before everything finished" : "Ready",
         timedOut ? "some requests are still running; they continue in the background" : "");
    settle("ready", timedOut ? "info" : "ok", performance.now());
    const badge = document.querySelector(".tv-loadbadge b");
    if (badge) badge.textContent = fmtMs(performance.now());
    // ...#activity on the URL keeps the log open instead (handy for sharing
    // what a slow load looked like).
    setTimeout(() => (location.hash === "#activity" ? openLog() : hideOverlay()), timedOut ? 900 : 350);
  }
  function hideOverlay() {
    const boot = $("tv-boot");
    if (!boot || mode === "nav") return;
    boot.classList.add("fade");
    setTimeout(() => { if (mode !== "nav" && mode !== "log") { boot.hidden = true; boot.classList.remove("fade"); mode = "hidden"; } }, 380);
  }
  let elapsedTimer = null;
  function tickElapsed(from) {
    clearInterval(elapsedTimer);
    elapsedTimer = setInterval(() => {
      const el = $("tv-boot-elapsed");
      if (el && (mode === "boot" || mode === "nav")) el.textContent = fmtMs(performance.now() - from);
      if (mode === "hidden") clearInterval(elapsedTimer);
    }, 100);
  }

  // ---- Leaving the page: show it from the very first moment ---------------
  function showNavigating(target, label) {
    const boot = $("tv-boot");
    if (!boot) return;
    mode = "nav";
    steps.length = 0;
    $("tv-boot-title").textContent = label;
    $("tv-boot-where").textContent = target + " · waiting for the Orin to answer";
    $("tv-boot-close").hidden = true;
    step("req", "Request sent", "the next page appears as soon as its first byte arrives");
    boot.hidden = false; boot.classList.remove("fade");
    render(); tickElapsed(performance.now());
  }
  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const a = event.target.closest && event.target.closest("a[href]");
    if (!a || a.target === "_blank" || a.hasAttribute("download")) return;
    const url = new URL(a.href, location.href);
    if (url.origin !== location.origin || (url.pathname === location.pathname && url.hash)) return;
    if (url.pathname.indexOf("/media/") >= 0 || url.pathname.indexOf("/api/") >= 0) return;
    showNavigating(url.pathname + url.search, "Loading " + (a.textContent.trim() || url.pathname).slice(0, 40));
  });
  window.addEventListener("beforeunload", () => { if (mode !== "nav") showNavigating(location.pathname, "Reloading"); });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) { mode = "hidden"; const boot = $("tv-boot"); if (boot) boot.hidden = true; }
  });

  // ---- The log view (footer badge) -----------------------------------------
  function openLog() {
    const boot = $("tv-boot");
    if (!boot) return;
    mode = "log";
    $("tv-boot-title").textContent = "Activity log";
    $("tv-boot-where").textContent = connectionLine() + " · page ready after " + fmtMs(bootDone || 0) +
                                     " · " + log.length + " request(s) since";
    $("tv-boot-elapsed").textContent = "";
    $("tv-boot-close").hidden = false;
    $("tv-boot-foot").textContent = "Faded rows are the automatic 5-second refreshes. Esc or ✕ to close.";
    boot.hidden = false; boot.classList.remove("fade");
    render();
  }
  function closeLog() {
    if (mode !== "log") return;
    mode = "hidden"; $("tv-boot").hidden = true;
  }
  window.tvActivity = {open: openLog, log};

  document.addEventListener("DOMContentLoaded", () => {
    $("tv-boot-where").textContent = connectionLine();
    tickElapsed(0);
    navigationSteps();
    parsedStep();
    $("tv-boot-close").addEventListener("click", closeLog);
    $("tv-pill").addEventListener("click", openLog);
    $("tv-boot").addEventListener("click", (event) => { if (event.target.id === "tv-boot") closeLog(); });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeLog(); });
    for (const badge of document.querySelectorAll(".tv-loadbadge")) badge.addEventListener("click", openLog);
    watchImages();
  });
  window.addEventListener("load", () => {
    loadFired = true;
    assetSteps();
    // Requests a page fires right on load get a moment to start.
    setTimeout(maybeBootDone, 60);
  });
  // Never hold the page hostage: after 8 s, show it regardless.
  setTimeout(() => { if (!booted) { loadFired = true; completeBoot(true); } }, 8000);
})();
</script>
"""


# ---------------------------------------------------------------------------
# Media viewer: click any snapshot or GIF
# ---------------------------------------------------------------------------
VIEWER_STYLES = """
  /* --- Media viewer: zoom for snapshots, play/scrub/zoom for GIFs -------- */
  img.evidence { cursor:zoom-in; }
  #vw { position:fixed; inset:0; z-index:1100; background:#050508; display:flex;
        flex-direction:column; color:var(--text); user-select:none; -webkit-user-select:none; }
  #vw[hidden] { display:none; }
  /* The display rules below would otherwise beat the browser's [hidden]. */
  #vw [hidden] { display:none !important; }
  #vw .vw-top, #vw .vw-bottom { display:flex; align-items:center; gap:8px; padding:8px 12px;
        background:rgba(21,21,29,0.92); border-color:var(--line); flex-wrap:wrap; }
  #vw .vw-top { border-bottom:1px solid var(--line); }
  #vw .vw-bottom { border-top:1px solid var(--line); }
  #vw .vw-caption { font-size:13px; min-width:0; flex:1 1 200px; overflow:hidden; text-overflow:ellipsis;
        white-space:nowrap; }
  #vw .vw-caption small { color:var(--muted); margin-left:8px; }
  #vw button, #vw select, #vw a.vw-btn { background:var(--panel-2); color:var(--text); border:1px solid var(--line);
        border-radius:8px; padding:6px 10px; font:inherit; font-size:13px; cursor:pointer; line-height:1;
        text-decoration:none; min-width:34px; text-align:center; }
  #vw button:hover, #vw a.vw-btn:hover { border-color:var(--accent); }
  #vw button.on { background:var(--accent); color:#0b0b0f; border-color:var(--accent); }
  #vw button:disabled { opacity:.4; cursor:default; }
  #vw .vw-zoom { font-variant-numeric:tabular-nums; font-size:12px; color:var(--muted); min-width:44px;
        text-align:center; }
  #vw .vw-stage { position:relative; flex:1 1 auto; overflow:hidden; touch-action:none; cursor:grab; }
  #vw .vw-stage.dragging { cursor:grabbing; }
  #vw .vw-media { position:absolute; left:0; top:0; transform-origin:0 0; max-width:none;
        will-change:transform; image-rendering:auto; }
  #vw .vw-media.pixels { image-rendering:pixelated; }
  #vw .vw-nav { position:absolute; top:50%; transform:translateY(-50%); z-index:2; font-size:22px;
        padding:14px 10px; opacity:.75; }
  #vw .vw-nav.prev { left:10px; } #vw .vw-nav.next { right:10px; }
  #vw .vw-msg { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
        color:var(--muted); font-size:14px; pointer-events:none; text-align:center; padding:20px; }
  #vw .vw-scrub { flex:1 1 220px; accent-color:var(--accent); min-width:120px; }
  #vw .vw-time { font-variant-numeric:tabular-nums; font-size:12px; color:var(--muted); min-width:120px; }
  #vw .vw-help { color:var(--muted); font-size:11px; }
  #vw .vw-overlay { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:1; }
  #vw .vw-boxinfo { font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
  @media (max-width: 700px) {
    #vw .vw-help, #vw .vw-hide-narrow, #vw .vw-boxinfo { display:none; }
    #vw .vw-nav { padding:10px 6px; font-size:18px; }
  }
"""

VIEWER_HTML = """
<div id="vw" hidden role="dialog" aria-modal="true" aria-label="Media viewer">
  <div class="vw-top">
    <div class="vw-caption" id="vw-caption"></div>
    <span class="vw-boxinfo" id="vw-boxinfo"></span>
    <button data-vw="boxes" id="vw-boxes" class="on" title="Detection boxes (B)">▣ Boxes</button>
    <select id="vw-minconf" title="Only boxes at or above this confidence">
      <option value="0">all</option><option value="0.5">≥ 0.50</option><option value="0.7">≥ 0.70</option>
    </select>
    <button data-vw="zoomout" title="Zoom out (-)">−</button>
    <span class="vw-zoom" id="vw-zoom">100%</span>
    <button data-vw="zoomin" title="Zoom in (+)">+</button>
    <button data-vw="fit" title="Fit to screen (0)">Fit</button>
    <button data-vw="actual" class="vw-hide-narrow" title="Actual pixels (1)">1:1</button>
    <a class="vw-btn vw-hide-narrow" id="vw-open" target="_blank" title="Open the original file">↗</a>
    <button data-vw="close" title="Close (Esc)">✕</button>
  </div>
  <div class="vw-stage" id="vw-stage">
    <button class="vw-nav prev" data-vw="prev" title="Previous (PageUp)">‹</button>
    <button class="vw-nav next" data-vw="next" title="Next (PageDown)">›</button>
    <canvas class="vw-overlay" id="vw-overlay"></canvas>
    <div class="vw-msg" id="vw-msg"></div>
  </div>
  <div class="vw-bottom" id="vw-gifbar" hidden>
    <button data-vw="first" title="First frame (Home)">⏮</button>
    <button data-vw="back" title="Previous frame (←)">◀︎❙</button>
    <button data-vw="play" id="vw-play" title="Play / pause (space)">▶</button>
    <button data-vw="fwd" title="Next frame (→)">❙▶︎</button>
    <input type="range" class="vw-scrub" id="vw-scrub" min="0" max="0" value="0" aria-label="Scrub frames">
    <span class="vw-time" id="vw-time">–</span>
    <select id="vw-speed" title="Playback speed">
      <option value="0.25">¼×</option><option value="0.5">½×</option>
      <option value="1" selected>1×</option><option value="2">2×</option>
    </select>
    <button data-vw="loop" id="vw-loop" class="on" title="Loop">⟲</button>
    <span class="vw-help">space play · ←/→ frame · B boxes · scroll or pinch to zoom · drag to pan · double-click to zoom in</span>
  </div>
</div>
"""

# A self-contained viewer. GIFs are decoded here (LZW, disposal, transparency,
# interlacing) because no browser lets a page pause or seek an <img> GIF; the
# frames go to a canvas, which is what makes play/pause, scrubbing and frame
# stepping possible. No library and no CDN: it has to work in the car.
VIEWER_SCRIPT = """
<script>
(function () {
  "use strict";

  // ---- GIF decoding ------------------------------------------------------
  function lzw(minCodeSize, data, pixelCount) {
    const out = new Uint8Array(pixelCount);
    const clear = 1 << minCodeSize, eoi = clear + 1;
    let codeSize = minCodeSize + 1, codeMask = (1 << codeSize) - 1, next = eoi + 1;
    const prefix = new Int16Array(4096), suffix = new Uint8Array(4096), stack = new Uint8Array(4097);
    for (let i = 0; i < clear; i++) { prefix[i] = -1; suffix[i] = i; }
    let datum = 0, bits = 0, pos = 0, old = -1, first = 0, top = 0, o = 0;
    while (o < pixelCount) {
      if (top === 0) {
        if (bits < codeSize) {
          if (pos >= data.length) break;
          datum |= data[pos++] << bits; bits += 8; continue;
        }
        let code = datum & codeMask; datum >>= codeSize; bits -= codeSize;
        if (code === clear) { codeSize = minCodeSize + 1; codeMask = (1 << codeSize) - 1; next = eoi + 1; old = -1; continue; }
        if (code === eoi) break;
        if (old === -1) { stack[top++] = suffix[code]; old = code; first = code; continue; }
        const inCode = code;
        if (code >= next) { stack[top++] = first; code = old; }
        while (code >= clear) { stack[top++] = suffix[code]; code = prefix[code]; }
        first = suffix[code]; stack[top++] = first;
        if (next < 4096) {
          prefix[next] = old; suffix[next] = first; next++;
          if ((next & codeMask) === 0 && next < 4096) { codeSize++; codeMask += next; }
        }
        old = inCode;
      }
      out[o++] = stack[--top];
    }
    return out;
  }

  function decodeGif(buffer) {
    const b = new Uint8Array(buffer);
    const sig = String.fromCharCode(b[0], b[1], b[2], b[3], b[4], b[5]);
    if (sig !== "GIF87a" && sig !== "GIF89a") throw new Error("not a GIF");
    const W = b[6] | (b[7] << 8), H = b[8] | (b[9] << 8), flags = b[10];
    let p = 13, gct = null;
    if (flags & 0x80) { const n = 3 * (1 << ((flags & 7) + 1)); gct = b.subarray(p, p + n); p += n; }
    const frames = [];
    let pixels = new Uint8ClampedArray(W * H * 4);
    let gce = {delay: 0, disposal: 0, trans: -1};
    const skipBlocks = () => { while (p < b.length && b[p] !== 0) p += b[p] + 1; p++; };
    while (p < b.length) {
      const block = b[p++];
      if (block === 0x3B) break;
      if (block === 0x21) {
        const label = b[p++];
        if (label === 0xF9 && b[p] >= 4) {
          const packed = b[p + 1];
          gce = {delay: (b[p + 2] | (b[p + 3] << 8)) * 10, disposal: (packed >> 2) & 7,
                 trans: packed & 1 ? b[p + 4] : -1};
        }
        skipBlocks();
        continue;
      }
      if (block !== 0x2C) throw new Error("bad GIF block 0x" + block.toString(16) + " at " + (p - 1));
      const x = b[p] | (b[p + 1] << 8), y = b[p + 2] | (b[p + 3] << 8);
      const w = b[p + 4] | (b[p + 5] << 8), h = b[p + 6] | (b[p + 7] << 8), f = b[p + 8];
      p += 9;
      let ct = gct;
      if (f & 0x80) { const n = 3 * (1 << ((f & 7) + 1)); ct = b.subarray(p, p + n); p += n; }
      const minCode = b[p++];
      let total = 0;
      for (let q = p; q < b.length && b[q] !== 0; q += b[q] + 1) total += b[q];
      const data = new Uint8Array(total);
      for (let o = 0; p < b.length && b[p] !== 0;) { const n = b[p++]; data.set(b.subarray(p, p + n), o); o += n; p += n; }
      p++;
      const indices = lzw(minCode, data, w * h);
      // Interlaced images store rows in four passes; map stored -> actual.
      const rows = [];
      if (f & 0x40) {
        for (const [start, step] of [[0, 8], [4, 8], [2, 4], [1, 2]]) for (let r = start; r < h; r += step) rows.push(r);
      } else for (let r = 0; r < h; r++) rows.push(r);
      const before = gce.disposal === 3 ? pixels.slice() : null;
      if (ct) {
        for (let k = 0; k < h; k++) {
          const py = y + rows[k];
          if (py >= H) continue;
          for (let col = 0; col < w; col++) {
            const idx = indices[k * w + col];
            if (idx === gce.trans) continue;
            const px = x + col;
            if (px >= W) continue;
            const d = (py * W + px) * 4, c = idx * 3;
            pixels[d] = ct[c]; pixels[d + 1] = ct[c + 1]; pixels[d + 2] = ct[c + 2]; pixels[d + 3] = 255;
          }
        }
      }
      // Browsers play delays under 20 ms at 100 ms; match them.
      frames.push({data: new ImageData(pixels.slice(), W, H), delay: gce.delay >= 20 ? gce.delay : 100});
      if (gce.disposal === 2) {
        for (let r = y; r < Math.min(y + h, H); r++) pixels.fill(0, (r * W + x) * 4, (r * W + Math.min(x + w, W)) * 4);
      } else if (gce.disposal === 3 && before) {
        pixels = before;
      }
      gce = {delay: 0, disposal: 0, trans: -1};
    }
    if (!frames.length) throw new Error("GIF has no frames");
    return {width: W, height: H, frames};
  }
  window.tvDecodeGif = decodeGif;   // exposed for testing

  // ---- The viewer ---------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  let gallery = [], index = 0, media = null, natW = 0, natH = 0;
  let scale = 1, tx = 0, ty = 0, fitScale = 1;
  let gif = null, frame = 0, playing = false, speed = 1, loop = true, timer = null, loadToken = 0;
  let boxes = null, boxesState = "", showBoxes = true, minConf = 0;
  const MAX_SCALE = 16;
  const BOX_COLOURS = {vehicle: "#7aa2ff", person: "#ffb020", plate: "#4ad07a", face: "#ff7ab6", other: "#c8c8d8"};

  function stageBox() { return $("vw-stage").getBoundingClientRect(); }
  function apply() {
    if (!media) return;
    media.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    media.classList.toggle("pixels", scale >= 2.5);
    $("vw-zoom").textContent = Math.round(scale * 100) + "%";
    drawBoxes();
  }

  // ---- Detection boxes ---------------------------------------------------
  // Drawn on a screen-sized canvas over the picture rather than scaled with
  // it, so lines and labels stay crisp and readable at any zoom.
  function currentBoxes() {
    if (!boxes || !boxes.frames || !boxes.frames.length) return null;
    const i = gif ? Math.min(frame, boxes.frames.length - 1) : 0;
    return (boxes.frames[i] || []).filter((b) => b.conf >= minConf);
  }
  function drawBoxes() {
    const canvas = $("vw-overlay");
    if (!canvas) return;
    const box = stageBox(), ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(box.width * ratio) || canvas.height !== Math.round(box.height * ratio)) {
      canvas.width = Math.round(box.width * ratio); canvas.height = Math.round(box.height * ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, box.width, box.height);
    const list = currentBoxes();
    updateBoxInfo(list);
    if (!showBoxes || !list || !media) return;
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.textBaseline = "top";
    for (const b of list) {
      const x = tx + b.box[0] * natW * scale, y = ty + b.box[1] * natH * scale;
      const w = b.box[2] * natW * scale, h = b.box[3] * natH * scale;
      const colour = BOX_COLOURS[b.kind] || BOX_COLOURS.other;
      ctx.lineWidth = 2; ctx.strokeStyle = colour;
      ctx.strokeRect(x, y, w, h);
      const text = b.label + " " + b.conf.toFixed(2);
      const tw = ctx.measureText(text).width + 8, th = 17;
      const ly = y - th >= 0 ? y - th : y;   // above the box unless it would leave the screen
      ctx.fillStyle = colour; ctx.fillRect(x - 1, ly, tw, th);
      ctx.fillStyle = "#0b0b0f"; ctx.fillText(text, x + 3, ly + 2);
    }
  }
  function updateBoxInfo(list) {
    const info = $("vw-boxinfo");
    if (!info) return;
    if (!boxes) { info.textContent = boxesState; return; }
    if (!list || !list.length) { info.textContent = gif ? "no detections this frame" : "no detections"; return; }
    const best = list.reduce((a, b) => (b.conf > a.conf ? b : a));
    info.textContent = list.length + " detection" + (list.length === 1 ? "" : "s") +
                       " · best " + best.label + " " + best.conf.toFixed(2);
  }
  async function loadBoxes(src, token) {
    boxes = null;
    boxesState = "";
    try {
      const response = await fetch(src + ".boxes.json", {cache: "no-store", tvQuiet: true});
      if (token !== loadToken) return;
      if (response.ok) boxes = await response.json();
      else boxesState = /\\.gif(\\?|$)/i.test(src) ? "boxes pending: added when the processor is idle"
                                                  : "no boxes recorded for this image";
    } catch (error) {
      boxesState = "";
    }
    if (token === loadToken) drawBoxes();
  }
  function clampPan() {
    const box = stageBox(), w = natW * scale, h = natH * scale;
    // Keep at least a quarter of the picture on screen.
    tx = Math.min(box.width - w * 0.25, Math.max(-w * 0.75, tx));
    ty = Math.min(box.height - h * 0.25, Math.max(-h * 0.75, ty));
  }
  function fit() {
    const box = stageBox();
    if (!natW || !natH) return;
    // Small crops (plates, faces) are worth enlarging - up to 4x.
    fitScale = Math.min(box.width / natW, box.height / natH, 4);
    scale = fitScale;
    tx = (box.width - natW * scale) / 2; ty = (box.height - natH * scale) / 2;
    apply();
  }
  function zoomAt(factor, cx, cy) {
    const box = stageBox();
    if (cx == null) { cx = box.width / 2; cy = box.height / 2; }
    const next = Math.min(MAX_SCALE, Math.max(fitScale * 0.5, scale * factor));
    tx = cx - (cx - tx) * (next / scale); ty = cy - (cy - ty) * (next / scale);
    scale = next; clampPan(); apply();
  }
  function actualSize() { zoomAt(1 / scale); }

  function caption(img) {
    const src = img.currentSrc || img.src;
    const name = decodeURIComponent(src.split("/").pop().split("?")[0]);
    const row = img.closest("tr, .card, .grid > *");
    const context = row ? row.textContent.replace(/\\s+/g, " ").trim().slice(0, 90) : "";
    return {name, context};
  }

  function setMedia(element) {
    const stage = $("vw-stage");
    if (media) media.remove();
    media = element; media.classList.add("vw-media");
    stage.insertBefore(media, $("vw-msg"));
  }

  function open(img) {
    gallery = [...document.querySelectorAll("img.evidence, a[data-viewer]")].filter((el) => el.offsetParent !== null || el === img);
    index = Math.max(0, gallery.indexOf(img));
    $("vw").hidden = false;
    document.documentElement.style.overflow = "hidden";
    show(index);
  }
  function close() {
    stopPlay(); loadToken++;
    $("vw").hidden = true;
    document.documentElement.style.overflow = "";
    if (media) { media.remove(); media = null; }
    gif = null; boxes = null; drawBoxes();
  }
  function srcOf(el) { return el.tagName === "A" ? el.href : (el.currentSrc || el.src); }

  function show(i) {
    stopPlay(); gif = null; frame = 0;
    index = (i + gallery.length) % gallery.length;
    const el = gallery[index], src = srcOf(el);
    const token = ++loadToken;
    const info = el.tagName === "A" ? {name: decodeURIComponent(src.split("/").pop()), context: el.title || ""} : caption(el);
    $("vw-caption").textContent = info.name;
    if (info.context) { const small = document.createElement("small"); small.textContent = info.context; $("vw-caption").appendChild(small); }
    $("vw-open").href = src;
    $("vw-msg").textContent = "Loading…";
    for (const nav of document.querySelectorAll("#vw .vw-nav")) nav.hidden = gallery.length < 2;
    const isGif = /\\.gif(\\?|$)/i.test(src);
    $("vw-gifbar").hidden = !isGif;
    loadBoxes(src, token);
    if (isGif) loadGif(src, token); else loadImage(src, token);
  }

  function loadImage(src, token) {
    const img = new Image();
    img.onload = () => {
      if (token !== loadToken) return;
      natW = img.naturalWidth; natH = img.naturalHeight;
      setMedia(img); $("vw-msg").textContent = ""; fit();
      appendDims();
    };
    img.onerror = () => { if (token === loadToken) $("vw-msg").textContent = "Could not load " + src; };
    img.src = src; img.alt = "";
  }
  function appendDims(extra) {
    const small = document.createElement("small");
    small.textContent = natW + "×" + natH + (extra ? " · " + extra : "");
    $("vw-caption").appendChild(small);
  }

  async function loadGif(src, token) {
    $("vw-msg").textContent = "Decoding GIF frames…";
    try {
      const response = await fetch(src);
      if (!response.ok) throw new Error("HTTP " + response.status);
      const buffer = await response.arrayBuffer();
      if (token !== loadToken) return;
      const decoded = decodeGif(buffer);
      if (token !== loadToken) return;
      gif = decoded; natW = gif.width; natH = gif.height;
      gif.starts = []; let t = 0;
      for (const f of gif.frames) { gif.starts.push(t); t += f.delay; }
      gif.duration = t;
      const canvas = document.createElement("canvas");
      canvas.width = natW; canvas.height = natH;
      gif.ctx = canvas.getContext("2d");
      setMedia(canvas); $("vw-msg").textContent = "";
      $("vw-scrub").max = String(gif.frames.length - 1);
      fit(); appendDims(gif.frames.length + " frames · " + (gif.duration / 1000).toFixed(1) + " s");
      draw(0); play();
    } catch (error) {
      if (token !== loadToken) return;
      // Fall back to the browser's own (unscrubbable) playback.
      $("vw-gifbar").hidden = true;
      loadImage(src, token);
      setTimeout(() => { if (token === loadToken) $("vw-msg").textContent = ""; }, 10);
      console.warn("GIF decode failed, showing it plainly:", error);
    }
  }
  function draw(i) {
    if (!gif) return;
    frame = Math.max(0, Math.min(gif.frames.length - 1, i));
    gif.ctx.putImageData(gif.frames[frame].data, 0, 0);
    drawBoxes();
    $("vw-scrub").value = String(frame);
    $("vw-time").textContent = "frame " + (frame + 1) + "/" + gif.frames.length + " · " +
      (gif.starts[frame] / 1000).toFixed(2) + " / " + (gif.duration / 1000).toFixed(2) + " s";
  }
  function tick() {
    if (!playing || !gif) return;
    let nextFrame = frame + 1;
    if (nextFrame >= gif.frames.length) {
      if (!loop) { stopPlay(); return; }
      nextFrame = 0;
    }
    draw(nextFrame);
    timer = setTimeout(tick, gif.frames[frame].delay / speed);
  }
  function play() {
    if (!gif) return;
    if (frame >= gif.frames.length - 1 && !loop) draw(0);
    playing = true; $("vw-play").textContent = "❚❚";
    clearTimeout(timer); timer = setTimeout(tick, gif.frames[frame].delay / speed);
  }
  function stopPlay() { playing = false; clearTimeout(timer); const b = $("vw-play"); if (b) b.textContent = "▶"; }
  function step(delta) { stopPlay(); draw(frame + delta); }

  // ---- Input: buttons, keys, wheel, drag, pinch, double-tap ----------------
  const actions = {
    close, fit, actual: actualSize,
    zoomin: () => zoomAt(1.4), zoomout: () => zoomAt(1 / 1.4),
    prev: () => show(index - 1), next: () => show(index + 1),
    play: () => (playing ? stopPlay() : play()), back: () => step(-1), fwd: () => step(1),
    first: () => step(-1e9),
    boxes: () => { showBoxes = !showBoxes; $("vw-boxes").classList.toggle("on", showBoxes); drawBoxes(); },
    loop: () => { loop = !loop; $("vw-loop").classList.toggle("on", loop); },
  };
  document.addEventListener("click", (event) => {
    const button = event.target.closest && event.target.closest("[data-vw]");
    if (button && !$("vw").hidden) { event.preventDefault(); actions[button.dataset.vw](); return; }
    const target = event.target.closest && event.target.closest("img.evidence, a[data-viewer]");
    if (!target || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
    event.preventDefault(); event.stopPropagation();
    open(target);
  }, true);

  document.addEventListener("keydown", (event) => {
    if ($("vw").hidden) return;
    const k = event.key;
    const handled = {
      "Escape": close, "+": actions.zoomin, "=": actions.zoomin, "-": actions.zoomout, "_": actions.zoomout,
      "0": fit, "1": actualSize, "PageUp": actions.prev, "PageDown": actions.next,
      "b": actions.boxes, "B": actions.boxes,
      " ": () => gif && actions.play(), "Home": () => gif && actions.first(),
      "ArrowLeft": () => (gif ? step(-1) : actions.prev()), "ArrowRight": () => (gif ? step(1) : actions.next()),
    }[k];
    if (handled) { event.preventDefault(); event.stopPropagation(); handled(); }
  }, true);

  document.addEventListener("DOMContentLoaded", () => {
    const stage = $("vw-stage");
    $("vw-scrub").addEventListener("input", (event) => { stopPlay(); draw(Number(event.target.value)); });
    $("vw-speed").addEventListener("change", (event) => { speed = Number(event.target.value); if (playing) play(); });
    $("vw-minconf").addEventListener("change", (event) => { minConf = Number(event.target.value); drawBoxes(); });
    stage.addEventListener("wheel", (event) => {
      event.preventDefault();
      const box = stageBox();
      zoomAt(Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0015)), event.clientX - box.left, event.clientY - box.top);
    }, {passive: false});

    const pointers = new Map();
    let pinch = null, lastTap = 0, moved = false;
    stage.addEventListener("pointerdown", (event) => {
      if (event.target.closest("button")) return;
      stage.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, {x: event.clientX, y: event.clientY});
      moved = false;
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = {dist: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2};
      }
      stage.classList.add("dragging");
    });
    stage.addEventListener("pointermove", (event) => {
      const prev = pointers.get(event.pointerId);
      if (!prev) return;
      const now = {x: event.clientX, y: event.clientY};
      pointers.set(event.pointerId, now);
      if (pointers.size === 2 && pinch) {
        const [a, b] = [...pointers.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y), mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const box = stageBox();
        tx += mx - pinch.mx; ty += my - pinch.my;
        zoomAt(dist / pinch.dist, mx - box.left, my - box.top);
        pinch = {dist, mx, my}; moved = true;
      } else if (pointers.size === 1) {
        const dx = now.x - prev.x, dy = now.y - prev.y;
        if (Math.abs(dx) + Math.abs(dy) > 1) moved = true;
        tx += dx; ty += dy; clampPan(); apply();
      }
    });
    const release = (event) => {
      pointers.delete(event.pointerId);
      if (pointers.size < 2) pinch = null;
      if (!pointers.size) stage.classList.remove("dragging");
      if (event.type === "pointerup" && !moved && !event.target.closest("button")) {
        const now = performance.now();
        if (now - lastTap < 320) {           // double-click or double-tap
          const box = stageBox();
          if (scale > fitScale * 1.05) fit();
          else zoomAt(Math.max(2.5, 3 * fitScale) / scale, event.clientX - box.left, event.clientY - box.top);
          lastTap = 0;
        } else lastTap = now;
      }
    };
    stage.addEventListener("pointerup", release);
    stage.addEventListener("pointercancel", release);
    window.addEventListener("resize", () => { if (!$("vw").hidden) fit(); });
  });
  // ...#view=N opens the Nth snapshot on the page (counting from 0): a link
  // straight to one piece of evidence.
  window.addEventListener("load", () => {
    const match = /^#view=(\\d+)$/.exec(location.hash);
    if (!match) return;
    const items = [...document.querySelectorAll("img.evidence, a[data-viewer]")];
    if (items[Number(match[1])]) open(items[Number(match[1])]);
  });
})();
</script>
"""


# ---------------------------------------------------------------------------
# Orin box monitoring (the footer on every page)
# ---------------------------------------------------------------------------
# A container shares the host kernel, so /proc and /sys here describe the whole
# Jetson, not just this container - which is exactly what the footer wants.
# Every reader returns None rather than raising: a missing sensor should blank
# one figure, never take a page down with it.

_cpu_previous_sample: tuple[float, int, int] | None = None   # (monotonic, busy, total)

# Junction temperature is the one the Orin throttles on; the others are fallbacks
# for boards that name their zones differently.
_THERMAL_PREFERENCE = ("tj-thermal", "cpu-thermal", "gpu-thermal", "soc0-thermal")


def _read_cpu_times() -> tuple[int, int] | None:
    try:
        fields = [int(v) for v in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    except (OSError, ValueError):
        return None
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)   # idle + iowait
    total = sum(fields[:8])                                    # guest is already in user
    return total - idle, total


def _cpu_percent() -> float | None:
    """Busy share since the previous call; a fresh 200 ms sample if that is stale."""
    global _cpu_previous_sample
    now = time.monotonic()
    current = _read_cpu_times()
    if current is None:
        return None
    previous = _cpu_previous_sample
    if previous is None or now - previous[0] > 60:
        time.sleep(0.2)
        previous = (now, *current)
        now = time.monotonic()
        current = _read_cpu_times() or current
    _cpu_previous_sample = (now, *current)
    busy_delta, total_delta = current[0] - previous[1], current[1] - previous[2]
    return round(100 * busy_delta / total_delta, 1) if total_delta > 0 else None


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return values


def _read_number(path: str) -> float | None:
    try:
        return float(Path(path).read_text().strip())
    # TypeError: a sysfs read that returns EAGAIN (the idle cv*-thermal zones)
    # reaches Python 3.10's text decoder as None instead of an OSError.
    except (OSError, ValueError, TypeError):
        return None


def _temperature_c() -> tuple[str, float] | None:
    readings: dict[str, float] = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            name = (zone / "type").read_text().strip()
        except OSError:
            continue
        millidegrees = _read_number(str(zone / "temp"))
        if millidegrees is not None and millidegrees > 0:
            readings[name] = millidegrees / 1000
    for name in _THERMAL_PREFERENCE:
        if name in readings:
            return name.removesuffix("-thermal"), round(readings[name], 1)
    if readings:
        name, value = max(readings.items(), key=lambda item: item[1])
        return name.removesuffix("-thermal"), round(value, 1)
    return None


def system_stats() -> dict[str, Any]:
    memory = _meminfo()
    mem_total = memory.get("MemTotal")
    mem_used = mem_total - memory["MemAvailable"] if mem_total and "MemAvailable" in memory else None
    swap_total = memory.get("SwapTotal")
    swap_used = swap_total - memory.get("SwapFree", 0) if swap_total else None

    try:
        disk = os.statvfs(BASE_DIR)
        disk_total = disk.f_blocks * disk.f_frsize
        disk_used = disk_total - disk.f_bavail * disk.f_frsize   # what we can still write
    except OSError:
        disk_total = disk_used = None

    try:
        load = [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        load = None

    gpu_load = _read_number("/sys/devices/platform/gpu.0/load")   # per mille
    uptime = None
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    temperature = _temperature_c()

    return {
        "ts": int(time.time()),
        "cpu_percent": _cpu_percent(),
        "cpu_count": os.cpu_count(),
        "load": load,
        "mem_used": mem_used,
        "mem_total": mem_total,
        "swap_used": swap_used,
        "swap_total": swap_total,
        "disk_used": disk_used,
        "disk_total": disk_total,
        "disk_path": str(BASE_DIR),
        "gpu_percent": round(gpu_load / 10, 1) if gpu_load is not None else None,
        "temp_c": temperature[1] if temperature else None,
        "temp_zone": temperature[0] if temperature else None,
        "uptime_seconds": uptime,
    }


# ---------------------------------------------------------------------------
# Pipeline status (the queue, and how far behind the processor is)
# ---------------------------------------------------------------------------
# The processor rewrites logs/processor.json when a clip starts, when it ends
# and every 5 s while idle; ingest writes logs/ingest.json every pass. Neither
# goes through sqlite, for the reason _read_gps_heartbeat gives.

# A heartbeat this old while idle means the processor is not running.
PIPELINE_IDLE_STALE_SECONDS = 60
# One clip taking this long means it is wedged, not slow: the worst case, a
# 5 MP clip on the CPU fallback, is a couple of minutes.
PIPELINE_CLIP_STUCK_SECONDS = 900

# "Behind real time" is only meaningful while footage is arriving. With the
# car away, now minus the newest analysed footage is just "when did I last
# drive" - so past this, or with no car volume attached, it is not shown.
LIVE_EDGE_MAX_AGE_SECONDS = 900


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def pipeline_status() -> dict[str, Any]:
    from src.common import GIF_QUEUE_DIR, INBOX_DIR, LOGS_DIR
    from src import clipqueue

    now = time.time()
    beat = _read_json(LOGS_DIR / "processor.json")
    ingest = _read_json(LOGS_DIR / "ingest.json")

    # The queue is read live from inbox/ rather than taken from the
    # heartbeat: it is what is actually waiting, even mid-clip. A running
    # demo's clips exist only in the processor, so during a demo the
    # processor's own view (rewritten every second then) is used instead.
    entries = clipqueue.queue_entries(INBOX_DIR)
    current_name = ((beat or {}).get("current") or {}).get("name")
    pending = [entry for entry in entries if entry["name"] != current_name]
    demo = (beat or {}).get("demo")

    result: dict[str, Any] = {"ts": int(now), "queue": clipqueue.summarize(pending, now)}
    if demo and demo.get("active") and isinstance((beat or {}).get("queue"), dict):
        result["queue"] = beat["queue"]
    try:
        result["gif_queue"] = sum(1 for _ in GIF_QUEUE_DIR.glob("*.json"))
    except OSError:
        result["gif_queue"] = None

    if beat is None:
        result.update(state="unknown", detail="the processor has not reported yet")
    else:
        age = now - (beat.get("ts") or 0)
        current = beat.get("current")
        if current:
            current = dict(current)
            current["elapsed_seconds"] = round(now - (current.get("started_ts") or now), 1)
        if current and current["elapsed_seconds"] > PIPELINE_CLIP_STUCK_SECONDS:
            state, detail = "stuck", f"one clip has run {int(current['elapsed_seconds'])}s"
        elif not current and age > PIPELINE_IDLE_STALE_SECONDS:
            state, detail = "down", f"no heartbeat for {int(age)}s"
        elif current:
            state, detail = "processing", None
        elif beat.get("state") == "correlating" and age <= PIPELINE_IDLE_STALE_SECONDS:
            state, detail = "correlating", None
        elif beat.get("state") == "annotating" and age <= PIPELINE_IDLE_STALE_SECONDS:
            state, detail = "annotating", f"idle-time detection boxes: {beat.get('annotating')}"
        else:
            state, detail = ("queued", "waiting for the next pass") if result["queue"]["depth"] else ("idle", None)
        by_camera = beat.get("seconds_by_camera") or {}
        overall = (sum(by_camera.values()) / len(by_camera)) if by_camera else None
        queued_now = (result["queue"].get("next") or []) if demo and demo.get("active") else pending
        result.update(
            state=state, detail=detail, heartbeat_age_seconds=int(age),
            current=current,
            eta_seconds=(beat.get("eta_seconds") if demo and demo.get("active")
                         else clipqueue.estimate_seconds(queued_now, by_camera, overall)),
            **{key: beat.get(key) for key in (
                "device", "has_gpu", "gpu_failures", "cpu_fallbacks", "gpu_retry_after",
                "last_gpu_error", "realtime_factor", "seconds_by_camera", "last", "recent",
                "processed_total", "errors_total", "last_error", "started_ts",
                "alerts_total", "last_alert_ts", "last_correlation_ts", "last_correlation_raised",
                "demo", "annotating", "boxes_pending",
            )},
        )

    if ingest is None:
        result["ingest"] = {"state": "unknown", "reason": "ingest has not reported"}
    else:
        result["ingest"] = {
            "state": ingest.get("state"), "reason": ingest.get("reason"),
            "age_seconds": int(now - (ingest.get("last_pass_ts") or 0)),
            "copied_total": ingest.get("copied_total"), "last_copied": ingest.get("last_copied"),
        }

    # "Behind real time": which edge to count from, if any. The page ticks
    # it every second between fetches, from edge_ts and the server clock.
    demo_edge = (beat or {}).get("demo_edge_ts")
    live_edge = (beat or {}).get("live_edge_ts")
    car_attached = (result.get("ingest") or {}).get("state") in ("exported", "ready")
    if demo and demo.get("active"):
        result["lag"] = {"mode": "demo", "edge_ts": demo_edge}
    elif live_edge and (car_attached or now - live_edge <= LIVE_EDGE_MAX_AGE_SECONDS):
        result["lag"] = {"mode": "live", "edge_ts": live_edge}
    elif car_attached:
        result["lag"] = {"mode": "live", "edge_ts": None}
    else:
        result["lag"] = {"mode": "offline", "edge_ts": None}

    # Path objects do not serialise; the queue summary only carries names.
    return result


# Plain string, not an f-string: it is JavaScript, and doubling every brace
# would make it unreadable. Fills the footer render_page emits and refreshes
# it every 5 s; a failed fetch leaves the last figures and dims the bar.
SYSBAR_SCRIPT = """
<script>
(function () {
  const bar = document.getElementById("sysbar");
  if (!bar) return;
  const footer = bar.closest("footer");
  const setFooterHeight = () =>
    document.documentElement.style.setProperty("--footer-h", footer.offsetHeight + "px");
  setFooterHeight();
  if (window.ResizeObserver) new ResizeObserver(setFooterHeight).observe(footer);
  else window.addEventListener("resize", setFooterHeight);
  const gib = (bytes) => (bytes / 1073741824).toFixed(1);
  const level = (percent, warn, high) => percent >= high ? "high" : percent >= warn ? "warn" : "";
  function put(key, text, cls) {
    const el = bar.querySelector('[data-k="' + key + '"]');
    if (!el) return;
    el.className = "metric " + (cls || "");
    el.querySelector("b").textContent = text;
  }
  function pct(used, total) { return total ? 100 * used / total : null; }
  function duration(seconds) {
    const d = Math.floor(seconds / 86400), h = Math.floor(seconds % 86400 / 3600),
          m = Math.floor(seconds % 3600 / 60);
    return d ? d + "d " + h + "h" : h ? h + "h " + m + "m" : m + "m";
  }
  async function refresh() {
    let s;
    try {
      s = await (await fetch(URL_ROOT + "/api/system", {cache: "no-store"})).json();
    } catch (error) {
      bar.style.opacity = 0.5;
      return;
    }
    bar.style.opacity = 1;
    if (s.cpu_percent != null) {
      const load = s.load ? " · load " + s.load[0].toFixed(2) : "";
      put("cpu", s.cpu_percent.toFixed(0) + "%" + load, level(s.cpu_percent, 80, 95));
    }
    if (s.gpu_percent != null) put("gpu", s.gpu_percent.toFixed(0) + "%", level(s.gpu_percent, 80, 95));
    const mem = pct(s.mem_used, s.mem_total);
    if (mem != null) put("mem", gib(s.mem_used) + " / " + gib(s.mem_total) + " GB", level(mem, 80, 92));
    const swap = pct(s.swap_used, s.swap_total);
    if (swap != null) put("swap", gib(s.swap_used) + " / " + gib(s.swap_total) + " GB", level(swap, 60, 85));
    const disk = pct(s.disk_used, s.disk_total);
    if (disk != null) put("disk", disk.toFixed(0) + "% · " + gib(s.disk_total - s.disk_used) + " GB free",
                          level(disk, 80, 90));
    if (s.temp_c != null) put("temp", s.temp_c.toFixed(0) + "°C " + s.temp_zone, level(s.temp_c, 75, 90));
    if (s.uptime_seconds != null) put("up", duration(s.uptime_seconds));
  }
  refresh();
  setInterval(refresh, 5000);

  // ---- Pipeline: the queue and how far behind the processor is ----------
  const pipe = document.getElementById("pipebar");
  const card = document.getElementById("pipeline-card");
  const STATE_CLASS = {processing: "good", annotating: "good", idle: "", queued: "warn", down: "high",
                       stuck: "high", unknown: "warn"};
  function putPipe(key, text, cls, title) {
    const el = pipe.querySelector('[data-k="' + key + '"]');
    if (!el) return;
    el.className = "metric " + (cls || "");
    el.querySelector("b").textContent = text;
    el.title = title || "";
  }
  function clipTime(ts) {
    if (ts == null) return "?";
    const when = new Date(ts * 1000), age = Date.now() / 1000 - ts;
    return age < 86400 ? when.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"})
                       : when.toLocaleString([], {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});
  }
  function ago(seconds) {
    if (seconds == null) return "–";
    if (seconds < 90) return Math.max(0, Math.round(seconds)) + "s ago";
    return duration(seconds) + " ago";
  }
  function cameraLabel(c) { return (c || "?").replace("_", " "); }
  function row(cells) {
    const tr = document.createElement("tr");
    for (const [text, cls] of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      if (cls) td.className = cls;
      tr.appendChild(td);
    }
    return tr;
  }
  function fill(tbody, rows, empty) {
    tbody.replaceChildren(...(rows.length ? rows : [row([[empty, "muted"]])]));
  }
  async function refreshPipeline() {
    let p;
    try {
      p = await (await fetch(URL_ROOT + "/api/pipeline", {cache: "no-store"})).json();
    } catch (error) {
      pipe.style.opacity = 0.5;
      return;
    }
    pipe.style.opacity = 1;
    // skew: how far this browser's clock is from the Orin's, so the counter
    // is right even when the phone and the Jetson disagree about the time.
    lagState = p.lag ? {mode: p.lag.mode, edge_ts: p.lag.edge_ts, skew: Date.now() / 1000 - p.ts} : null;
    tickLag();
    const q = p.queue || {depth: 0, events: 0, next: []};
    putPipe("pstate", "● " + p.state, STATE_CLASS[p.state] || "", p.detail || "");
    if (p.current) {
      putPipe("pnow", cameraLabel(p.current.camera) + " " + clipTime(p.current.captured_ts) +
              " · " + Math.round(p.current.elapsed_seconds) + "s", "");
    } else {
      putPipe("pnow", "–", "");
    }
    putPipe("pqueue", q.depth + (q.events ? " (" + q.events + " events)" : ""),
            q.depth > 30 ? "high" : q.depth > 6 ? "warn" : "");
    putPipe("peta", q.depth === 0 ? "clear" : p.eta_seconds != null ? "~" + duration(Math.max(60, p.eta_seconds)) : "?",
            p.eta_seconds > 900 ? "high" : p.eta_seconds > 120 ? "warn" : "");
    const gpuResting = p.gpu_retry_after && p.gpu_retry_after * 1000 > Date.now();
    const device = p.device === "cuda" ? "GPU" : p.has_gpu ? "CPU (GPU resting)" : "CPU";
    putPipe("pspeed", (p.realtime_factor != null ? p.realtime_factor.toFixed(2) + "× real time · " : "") + device,
            gpuResting ? "warn" : p.realtime_factor == null ? "" : p.realtime_factor < 0.9 ? "good" : p.realtime_factor < 1 ? "warn" : "high",
            (p.last_gpu_error ? "last GPU error: " + p.last_gpu_error + ". " : "") +
            (p.cpu_fallbacks ? p.cpu_fallbacks + " clip(s) re-run on the CPU" : ""));
    putPipe("pfresh", p.last ? clipTime(p.last.captured_ts) : "–", "");
    const ingest = p.ingest || {};
    putPipe("pingest", ingest.state === "ready" ? "copying" : (ingest.state || "?"),
            ingest.state === "ready" ? "good" : "", ingest.reason || "");

    if (card) renderFlow(p, q);
  }

  // ---- The flow diagram on the Recent page --------------------------------
  function setNode(id, cls, value, sub) {
    const node = document.getElementById(id);
    if (!node) return;
    node.className = "node" + (node.classList.contains("wide") ? " wide" : "") + (cls ? " " + cls : "");
    const nv = node.querySelector(".nv"), ns = node.querySelector(".ns");
    if (nv && value != null) nv.textContent = value;
    if (ns && sub != null) ns.textContent = sub;
  }
  function since(ts) { return ts ? ago(Date.now() / 1000 - ts) : "never"; }
  const INGEST_VIEW = {
    exported: ["ok", "Recording", "the car owns the drive; clips are copied when it is released"],
    ready: ["active", "Mounted", "drive readable, ingest copying"],
    absent: ["", "Not connected", "car not plugged in, or drive not mounted"],
  };

  function renderFlow(p, q) {
    const demo = p.demo && p.demo.active ? p.demo : null;
    const ingest = p.ingest || {};

    // Car drive (or the demo feed standing in for it)
    if (demo) {
      setNode("n-car", "active", "NATIX demo",
        demo.released + "/" + demo.total + " clips released" +
        (demo.next_release_in != null ? " · next minute in " + demo.next_release_in + "s" : ""));
    } else {
      const view = INGEST_VIEW[ingest.state] || ["warn", ingest.state || "unknown", ingest.reason || ""];
      setNode("n-car", view[0], view[1], view[2]);
    }

    // Ingest
    const ingestStale = ingest.age_seconds != null && ingest.age_seconds > 120;
    setNode("n-ingest",
      ingest.state === "unknown" || ingestStale ? "bad" : ingest.state === "ready" ? "active" : "",
      (ingest.copied_total || 0) + " copied",
      ingestStale ? "no heartbeat for " + duration(ingest.age_seconds)
                  : ingest.last_copied ? "last: " + ingest.last_copied : "nothing copied since it started");

    // Queue
    const eta = q.depth === 0 ? "empty" : p.eta_seconds != null ? "clears in ~" + duration(Math.max(60, p.eta_seconds)) : "";
    setNode("n-queue", q.depth > 30 ? "bad" : q.depth > 6 ? "warn" : q.depth ? "active" : "ok",
      String(q.depth),
      [q.events ? q.events + " events" : "", q.demo ? q.demo + " demo" : "", eta].filter(Boolean).join(" · "));

    // Processor: state machine, current clip, step tracker
    const resting = p.gpu_retry_after && p.gpu_retry_after * 1000 > Date.now();
    const machine = (p.state === "down" || p.state === "stuck") ? "down"
                  : (p.state === "processing" && p.current && p.current.device === "cpu" && p.has_gpu) ? "fallback"
                  : p.state;
    for (const chip of document.querySelectorAll("#proc-states span")) {
      const on = chip.dataset.s === machine;
      chip.className = on ? "on" + (machine === "down" ? " bad" : machine === "fallback" ? " warn" : "") : "";
    }
    setNode("n-proc", machine === "down" ? "bad" : machine === "fallback" ? "warn"
                     : (machine === "processing" || machine === "correlating" || machine === "annotating") ? "active" : "ok");
    document.getElementById("proc-device").textContent =
      (p.device === "cuda" ? "GPU" : p.has_gpu ? "CPU · GPU resting" : "CPU") +
      (p.realtime_factor != null ? " · " + p.realtime_factor.toFixed(2) + "× real time" : "");

    const c = p.current;
    const order = ["detect", "plates", "faces", "saving"];
    const stepText = {
      detect: c ? c.frames_done + "/" + c.frames_budget + " frames" : "–",
      plates: c ? (c.plate_budget ? c.plate_frames + "/" + c.plate_budget : "off") : "–",
      faces: c ? (c.face_budget ? c.face_frames + "/" + c.face_budget : "off") : "–",
      saving: c ? (c.demo ? "summary only" : "alerts, GIF") : "–",
    };
    const nowIndex = c ? order.indexOf(c.stage) : -1;
    for (const box of document.querySelectorAll("#proc-steps > div")) {
      const index = order.indexOf(box.dataset.step);
      // Plates and faces run inside the frame loop, so "done" is by count.
      const finished = c && (box.dataset.step === "detect" ? c.frames_done >= c.frames_budget || nowIndex === 3
                           : box.dataset.step === "saving" ? false : nowIndex === 3);
      box.className = c && index === nowIndex ? "now" : finished ? "done" : "";
      box.querySelector("small").textContent = stepText[box.dataset.step];
    }
    document.getElementById("proc-bar").style.width =
      c ? Math.min(100, Math.round(100 * (c.stage === "saving" ? 1 : c.frames_done / (c.frames_budget || 1)))) + "%" : "0%";
    document.getElementById("proc-clip").textContent = c
      ? (c.demo ? "DEMO · " : "") + cameraLabel(c.camera) + " " + clipTime(c.captured_ts) +
        (c.source ? " · " + c.source.replace("Clips", "") : "") + " · " + Math.round(c.elapsed_seconds) + "s" +
        " · " + c.vehicles + " vehicle / " + c.people + " person hits"
      : p.state === "correlating" ? "re-scoring identities…"
      : p.state === "annotating" ? "adding detection boxes to " + p.annotating + (p.boxes_pending > 1 ? " · " + (p.boxes_pending - 1) + " more" : "")
      : (p.detail || "waiting for clips");
    document.getElementById("proc-foot").textContent =
      (p.processed_total || 0) + " clips since start" +
      (p.cpu_fallbacks ? " · " + p.cpu_fallbacks + " on CPU fallback" : "") +
      (p.errors_total ? " · " + p.errors_total + " errors" : "") +
      (resting && p.last_gpu_error ? " · GPU: " + p.last_gpu_error : "");

    // Outputs
    setNode("n-gif", p.gif_queue ? "active" : "ok", null,
      p.gif_queue ? p.gif_queue + " GIF" + (p.gif_queue === 1 ? "" : "s") + " rendering" : "idle");
    setNode("n-corr", p.state === "correlating" ? "active" : p.last_correlation_raised ? "warn" : "ok", null,
      p.state === "correlating" ? "running…" : "last run " + since(p.last_correlation_ts) +
      (p.last_correlation_raised ? " · " + p.last_correlation_raised + " new finding(s)" : ""));
    setNode("n-alerts", p.alerts_total ? "warn" : "ok", null,
      (p.alerts_total || 0) + " since start · last " + since(p.last_alert_ts));

    // Headline, demo banner and buttons
    const headline = document.getElementById("pipeline-headline");
    headline.textContent = q.depth === 0
      ? (p.last ? "Caught up · newest footage analysed " + clipTime(p.last.captured_ts) : "Caught up")
      : q.depth + " waiting" + (p.eta_seconds != null ? ", clears in ~" + duration(Math.max(60, p.eta_seconds)) : "");
    const banner = document.getElementById("demo-banner");
    const d = p.demo;
    if (d && (d.active || d.finished_ts || d.stopped) && Date.now() / 1000 - (d.finished_ts || d.started_ts) < 3600) {
      banner.hidden = false;
      const lag = d.max_lag_seconds != null ? " · worst wait from arrival to result " + Math.round(d.max_lag_seconds) + "s" : "";
      banner.textContent = d.active
        ? "Demo (" + d.mode + "): " + d.done + "/" + d.total + " clips analysed in " + duration(d.elapsed_seconds) + lag +
          ". Nothing is saved: results appear below only."
        : (d.stopped ? "Demo stopped: " : "Demo finished: ") + d.done + "/" + d.total + " clips in " +
          duration(d.elapsed_seconds) + lag + (d.mode === "realtime" && d.max_lag_seconds != null
            ? (d.max_lag_seconds < 60 ? " — kept up with the car." : " — fell behind the car.") : "");
    } else {
      banner.hidden = true;
    }
    for (const button of document.querySelectorAll("[data-demo]")) {
      button.hidden = (button.dataset.demo === "stop") !== Boolean(demo);
    }

    fill(document.getElementById("pipeline-next"), (q.next || []).map((clip) => row([
      [clipTime(clip.captured_ts)], [cameraLabel(clip.camera)],
      [clip.demo ? "demo" : clip.source ? clip.source.replace("Clips", "") : "",
       clip.demo ? "pill low" : clip.source && clip.source !== "RecentClips" ? "pill medium" : "muted"],
    ])), "Nothing waiting");
    fill(document.getElementById("pipeline-recent"), (p.recent || []).map((clip) => {
      const r = clip.result;
      const found = r ? r.vehicles + " veh · " + r.people + " ppl" + (r.plate ? " · " + r.plate : "") +
                        (r.faces ? " · " + r.faces + " face" + (r.faces === 1 ? "" : "s") : "") : "";
      const tr = row([
        [clipTime(clip.captured_ts)], [cameraLabel(clip.camera)],
        [clip.seconds + "s " + (clip.device === "cuda" ? "GPU" : "CPU"), clip.device === "cuda" ? "" : "muted"],
        [found, "muted"],
        [clip.demo ? "demo" : clip.alert ? "alert" : "", clip.demo ? "pill low" : clip.alert ? "pill high" : ""],
      ]);
      if (r && r.snapshot) {
        const link = document.createElement("a");
        link.href = URL_ROOT + "/media/" + r.snapshot;
        link.target = "_blank";
        link.textContent = "📷";
        link.title = "best frame from this demo clip";
        link.dataset.viewer = "1";
        const td = document.createElement("td");
        td.appendChild(link);
        tr.appendChild(td);
      }
      return tr;
    }), "Nothing yet since the processor started");
  }

  for (const button of document.querySelectorAll("[data-demo]")) {
    button.addEventListener("click", async () => {
      const action = button.dataset.demo === "stop" ? "stop" : "start";
      button.disabled = true;
      try {
        const response = await fetch(URL_ROOT + "/api/demo", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({action: action, mode: action === "start" ? button.dataset.demo : null}),
        });
        const answer = await response.json();
        if (!answer.ok) alert("Demo: " + answer.error);
      } catch (error) {
        alert("Demo request failed: " + error);
      } finally {
        button.disabled = false;
        setTimeout(refreshPipeline, 1200);
      }
    });
  }
  // ---- "Behind real time", ticking every second between fetches --------
  let lagState = null;   // {mode, edge_ts, skew}
  function lagText(seconds) {
    seconds = Math.max(0, Math.round(seconds));
    if (seconds < 100) return "−" + seconds + " s";
    const m = Math.floor(seconds / 60), s = seconds % 60;
    if (m < 60) return "−" + m + ":" + String(s).padStart(2, "0");
    return "−" + duration(seconds);
  }
  function tickLag() {
    const big = document.getElementById("lag-big");
    const label = document.getElementById("lag-label");
    let text = "–", cls = "off", note = "behind real time", title = "";
    if (lagState) {
      if (lagState.mode === "offline") {
        text = "no live footage"; note = "car not connected";
      } else if (lagState.edge_ts == null) {
        text = "waiting"; note = lagState.mode === "demo" ? "demo: first clip in progress" : "for the first clip";
      } else {
        const behind = Date.now() / 1000 - lagState.skew - lagState.edge_ts;
        text = lagText(behind);
        cls = behind < 150 ? "good" : behind < 300 ? "warn" : "high";
        note = lagState.mode === "demo" ? "behind real time (demo clock)" : "behind real time";
      }
    }
    putPipe("pbehind", text, cls === "off" ? "" : cls, note);
    if (big) { big.textContent = text; big.className = "lag " + cls; }
    if (label) label.textContent = note;
  }
  if (pipe) {
    refreshPipeline();
    setInterval(refreshPipeline, 5000);
    setInterval(tickLag, 1000);
  }
})();
</script>
"""

PIPELINE_METRICS = [
    ("pstate", "Pipeline"), ("pbehind", "Behind real time"), ("pnow", "Now"), ("pqueue", "Queue"), ("peta", "Clears in"),
    ("pspeed", "Speed"), ("pfresh", "Newest done"), ("pingest", "Ingest"),
]

SYSBAR_METRICS = [
    ("cpu", "CPU"), ("gpu", "GPU"), ("mem", "RAM"), ("swap", "Swap"),
    ("disk", "Disk"), ("temp", "Temp"), ("up", "Up"),
]


def render_page(title: str, body: str, active_path: str = "", head_extra: str = "") -> HTMLResponse:
    """Wrap page content in the shared shell (nav, styles, viewport)."""
    tabs = "".join(
        f'<a class="tab{" active" if path == active_path else ""}" href="{URL_ROOT}{path}">{esc(label)}</a>'
        for path, label in NAV_TABS
    )
    # Page scripts that live in plain (non-f) strings build their URLs from
    # this, so it has to be defined before head_extra or the body run.
    url_root_js = json.dumps(URL_ROOT).replace("<", "\\u003c")
    sysbar = "".join(
        f'<span class="metric" data-k="{key}">{label} <b>–</b></span>'
        for key, label in SYSBAR_METRICS
    )
    pipebar = "".join(
        f'<span class="metric" data-k="{key}">{"" if key == "pstate" else label + " "}<b>–</b></span>'
        for key, label in PIPELINE_METRICS
    )

    return HTMLResponse(f"""<!doctype html>
<html><head>
  <meta charset="utf-8" />
  <title>{esc(title)} - Tesla Vision</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>{PAGE_STYLES}{ACTIVITY_STYLES}{VIEWER_STYLES}</style>
  <script>var URL_ROOT = {url_root_js};</script>
  {ACTIVITY_SCRIPT}
  {head_extra}
</head>
<body>
  {ACTIVITY_HTML}
  <header><nav>
    <span class="brand">🛰️ Tesla Vision</span>
    {tabs}
  </nav></header>
  <main>{body}</main>
  <footer class="sysbar">
    <div class="inner" id="pipebar">
      <span class="label">Pipeline</span>
      {pipebar}
    </div>
    <div class="inner" id="sysbar">
      <span class="label">Orin box</span>
      {sysbar}
      <span class="metric tv-loadbadge" title="Open the loading and activity log">⚡ Loaded in <b>…</b></span>
    </div>
  </footer>
  {SYSBAR_SCRIPT}
  {VIEWER_HTML}
  {VIEWER_SCRIPT}
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

@app.get("/api/pipeline")
def api_pipeline():
    return pipeline_status()


@app.post("/api/demo")
async def api_demo(request: Request):
    """
    Start or stop the NATIX replay. The processor owns the models and the GPU,
    so this only leaves it a request in jobs/; it picks that up within a second.

    JSON only: a cross-site HTML form cannot send application/json without a
    CORS preflight this app never answers, so a link on another site cannot
    start a demo with the viewer's credentials.
    """
    from src.common import BASE_DIR, JOBS_DIR
    if "application/json" not in (request.headers.get("content-type") or ""):
        return JSONResponse({"ok": False, "error": "expected application/json"}, status_code=415)
    try:
        body = await request.json()
    except ValueError:
        body = None
    action = body.get("action") if isinstance(body, dict) else None
    mode = body.get("mode") if isinstance(body, dict) else None
    if action not in ("start", "stop") or (action == "start" and mode not in ("realtime", "backlog")):
        return JSONResponse({"ok": False, "error": "action must be start (mode realtime|backlog) or stop"},
                            status_code=400)
    clips_dir = Path(os.environ.get("DEMO_CLIPS_DIR", str(BASE_DIR / "natix-stray")))
    if action == "start" and not any(clips_dir.rglob("*.mp4")):
        return JSONResponse({"ok": False, "error": f"no demo clips under {clips_dir}"}, status_code=409)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    target = JOBS_DIR / "demo.json"
    temporary = JOBS_DIR / ".demo.json.tmp"
    temporary.write_text(json.dumps({"action": action, "mode": mode, "ts": int(time.time())}),
                         encoding="utf-8")
    os.replace(temporary, target)
    return {"ok": True, "action": action, "mode": mode}


@app.get("/api/system")
def api_system():
    return system_stats()


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
            f'<a href="{URL_ROOT}/entity/{esc(row["entity_type"])}/{esc(row["entity_id"])}">'
            f'<b>{esc(row["entity_label"])}</b></a> '
            f'<span class="muted">scored {esc(row["score"])}/100</span></div>'
            for row in top_findings
        )
        banner = (
            '<div class="card" style="border-color:var(--high);">'
            '<div style="font-weight:800; margin-bottom:6px;">⚠️ Active surveillance findings</div>'
            f'{items}'
            f'<div style="margin-top:8px;"><a href="{URL_ROOT}/findings">See all findings →</a></div>'
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
        f'<a class="btn" href="{URL_ROOT}/?limit={limit + 50}">Show more '
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
        jpeg_tag = f'<img class="evidence" loading="lazy" src="{URL_ROOT}/media/{esc(jpeg)}" style="max-width:420px;" />' if jpeg else ""
        gif_tag = f'<img class="evidence" loading="lazy" src="{URL_ROOT}/media/{esc(gif)}" style="max-width:420px;" />' if gif else ""

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
      <div class="card" id="pipeline-card">
        <div class="pipe-head">
          <div class="lag-wrap">
            <div class="lag" id="lag-big" title="Now minus the end of the newest footage analysed. It climbs while the car records the current one-minute clip, then drops once that clip is copied and processed: about -15 s to -100 s while driving. Climbing steadily past that means a backlog is building.">–</div>
            <div><span style="font-weight:800;">Processing pipeline</span><br>
              <span class="muted" id="lag-label">behind real time</span>
              <span class="muted" id="pipeline-headline" style="margin-left:8px;"></span></div>
          </div>
          <div class="demo-buttons">
            <button class="btn" data-demo="realtime" title="Replay the NATIX clips one minute at a time, as the car would deliver them">▶ Real-time demo</button>
            <button class="btn" data-demo="backlog" title="Drop all NATIX clips in at once and watch the queue drain">⏩ Backlog demo</button>
            <button class="btn" data-demo="stop" hidden>■ Stop demo</button>
          </div>
        </div>
        <div class="demo-banner" id="demo-banner" hidden></div>
        <div class="flow">
          <div class="node" id="n-car"><div class="nh"><span class="dot"></span>Car drive</div>
            <div class="nv">–</div><div class="ns">–</div></div>
          <div class="arrow"><span>→</span></div>
          <div class="node" id="n-ingest"><div class="nh"><span class="dot"></span>Ingest</div>
            <div class="nv">–</div><div class="ns">–</div></div>
          <div class="arrow"><span>→</span></div>
          <div class="node" id="n-queue"><div class="nh"><span class="dot"></span>Queue</div>
            <div class="nv">–</div><div class="ns">–</div></div>
          <div class="arrow"><span>→</span></div>
          <div class="node wide" id="n-proc"><div class="nh"><span class="dot"></span>Processor
              <span class="muted" id="proc-device" style="margin-left:auto;"></span></div>
            <div class="states" id="proc-states">
              <span data-s="idle">Idle</span><span data-s="queued">Waiting</span><span data-s="processing">Processing</span><span data-s="correlating">Correlating</span><span data-s="annotating">Boxing GIFs</span><span data-s="fallback">CPU fallback</span><span data-s="down">Down</span>
            </div>
            <div class="ns" id="proc-clip">–</div>
            <div class="steps" id="proc-steps">
              <div data-step="detect"><b>Detect</b><small>–</small></div>
              <div data-step="plates"><b>Plates</b><small>–</small></div>
              <div data-step="faces"><b>Faces</b><small>–</small></div>
              <div data-step="saving"><b>Save</b><small>–</small></div>
            </div>
            <div class="bar"><div id="proc-bar"></div></div>
            <div class="ns" id="proc-foot">–</div>
          </div>
          <div class="arrow"><span>→</span></div>
          <div class="outs">
            <div class="node" id="n-gif"><div class="nh"><span class="dot"></span>GIF worker</div>
              <div class="ns">–</div></div>
            <div class="node" id="n-corr"><div class="nh"><span class="dot"></span>Correlation</div>
              <div class="ns">–</div></div>
            <div class="node" id="n-alerts"><div class="nh"><span class="dot"></span>Alerts</div>
              <div class="ns">–</div></div>
          </div>
        </div>
        <div class="pipeline-grid">
          <div><div class="muted" style="margin-bottom:4px;">Up next (Sentry/Saved first, then newest)</div>
            <table><tbody id="pipeline-next"><tr><td class="muted">–</td></tr></tbody></table></div>
          <div><div class="muted" style="margin-bottom:4px;">Just processed</div>
            <table><tbody id="pipeline-recent"><tr><td class="muted">–</td></tr></tbody></table></div>
        </div>
      </div>
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
                  <a href="{URL_ROOT}/entity/{esc(finding["entity_type"])}/{esc(finding["entity_id"])}">{esc(finding["entity_label"])}</a>
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


# ---------------------------------------------------------------------------
# Car storage: the drive the car records to (src/carstore.py does the work on
# the host; this page only shows its heartbeat and passes it requests)
# ---------------------------------------------------------------------------
CARSTORE_STALE_SECONDS = 30
GB_BYTES = 1000 ** 3


def storage_status() -> dict[str, Any]:
    """carstore.json, plus whether the service behind it is alive."""
    from src.common import LOGS_DIR
    beat = _read_json(LOGS_DIR / "carstore.json")
    if beat is None:
        return {"service": "missing"}
    age = int(time.time() - (beat.get("written_ts") or 0))
    beat["service"] = "stale" if age > CARSTORE_STALE_SECONDS else "running"
    beat["age_seconds"] = age
    return beat


def _gb(size: Any) -> str:
    return f"{(_as_int(size) or 0) / GB_BYTES:.0f} GB"


def _storage_summary(status: dict[str, Any]) -> str:
    """The sentences at the top of the page: is the car recording somewhere we can read?"""
    service = status.get("service")
    if service == "missing":
        return ('<div class="reason alarm">The car storage service is not running on the Jetson, so the car '
                'is not recording anywhere the Jetson can read. Install it once with '
                '<code>sudo ./scripts/install_carstore.sh</code> in the tesla-vision checkout.</div>')
    lines = []
    if service == "stale":
        lines.append(f'<div class="reason alarm">The car storage service last reported '
                     f'{esc(_relative_age(status.get("age_seconds")))} ago - it may have stopped.</div>')
    state = status.get("state")
    if state == "no_storage":
        lines.append('<div class="reason">No car storage drive yet. Format a USB drive below and the car '
                     'records to it; footage then reaches the Jetson about a minute after it is recorded.</div>')
    elif state == "error":
        lines.append(f'<div class="reason alarm">Car storage error: {esc(status.get("reason"))}</div>')
    backing = status.get("backing") or {}
    export = status.get("export") or {}
    snap = status.get("snapshot") or {}
    if state == "exporting" and backing:
        car = ("The car is connected and has the drive." if export.get("car_connected")
               else "The car is not connected right now (or is asleep).")
        lines.append(f'<div class="reason">Recording to a {esc(_gb(backing.get("image_bytes")))} image on '
                     f'{esc(backing.get("device"))} ({esc(_gb(backing.get("free_bytes")))} free). {car}</div>')
        if backing.get("below_tesla_minimum"):
            lines.append(f'<div class="reason alarm">The image is smaller than Tesla\'s '
                         f'{esc(status.get("settings", {}).get("tesla_min_image_gb", 64))} GB minimum - the car '
                         'may refuse it. Use a larger drive.</div>')
    if snap.get("last_ts"):
        ago = _relative_age(time.time() - snap["last_ts"])
        if snap.get("error"):
            lines.append(f'<div class="reason alarm">Last snapshot, {esc(ago)} ago, failed: '
                         f'{esc(snap["error"])}</div>')
        else:
            text = (f'Last snapshot {esc(ago)} ago took {esc(snap.get("seconds"))} s and picked up '
                    f'{esc(snap.get("copied_last", 0))} new clip(s); {esc(snap.get("copied_total", 0))} '
                    'since the service started.')
            if snap.get("newest_clip_ts"):
                text += (f' Newest clip received was recorded '
                         f'{esc(_relative_age(time.time() - snap["newest_clip_ts"]))} ago.')
            lines.append(f'<div class="reason">{text}</div>')
        if snap.get("copying"):
            lines.append(f'<div class="reason">Copying now: {esc(snap["copying"])} clip(s) so far '
                         'this snapshot.</div>')
        if snap.get("deferred"):
            lines.append('<div class="reason">More finished clips are waiting; they follow over the next '
                         'snapshots, newest first.</div>')
        if snap.get("paused"):
            lines.append(f'<div class="reason alarm">Copying paused: {esc(snap["paused"])}.</div>')
    return "".join(lines)


def _storage_job(status: dict[str, Any]) -> str:
    job = status.get("job")
    if not job:
        return ""
    target = job.get("partition") or "the drive"
    state = job.get("state")
    if state == "running":
        text, css = f'Formatting {esc(target)}: {esc(job.get("step"))}…', "reason"
    elif state == "done":
        text, css = f'Done: {esc(job.get("message"))}', "reason"
    else:
        text, css = f'Not formatted: {esc(job.get("message"))}', "reason alarm"
    return f'<div class="card"><div style="font-weight:700; margin-bottom:6px;">Last format request</div>' \
           f'<div class="{css}">{text}</div></div>'


def _format_button(drive: dict[str, Any], partition: dict[str, Any] | None, label: str) -> str:
    """A button that opens the confirm dialog, carrying the exact words of what gets erased."""
    if partition is None:
        names = ", ".join(f'{p["path"]} ({p.get("label") or p.get("fstype") or "unformatted"})'
                          for p in drive.get("partitions") or []) or "nothing readable"
        erases = (f'Every partition on {drive["path"]} ({drive.get("model")}, {_gb(drive.get("size_bytes"))}) '
                  f'is erased: {names}.')
    else:
        erases = (f'{partition["path"]} ({partition.get("label") or "no label"}, '
                  f'{partition.get("fstype") or "unformatted"}, {_gb(partition.get("size_bytes"))}) is erased. '
                  'Other partitions on the drive are not touched.')
        if partition.get("label") == "TESLACAM":
            if partition.get("fstype") in ("vfat", "ext4", "ext3", "ext2"):
                erases += ' Clips on it that were never processed are copied to the Jetson first.'
            else:
                erases += (f' It is {partition.get("fstype") or "unformatted"}, which the Jetson cannot read, '
                           'so its clips cannot be copied off first.')
    if drive.get("role") == "car storage":
        erases += " This drive is the car's storage now: its recordings are erased too."
    size = partition["size_bytes"] if partition else drive.get("size_bytes")
    if (_as_int(size) or 0) < 64 * GB_BYTES:
        erases += " It is smaller than Tesla's 64 GB minimum, so the car may refuse it."
    return (f'<button class="btn" data-format-drive="{esc(drive["id"])}" '
            f'data-format-partition="{esc(partition["path"] if partition else "")}" '
            f'data-format-erases="{esc(erases)}">{esc(label)}</button>')


def _storage_drives(status: dict[str, Any]) -> str:
    drives = status.get("drives") or []
    if not drives:
        return ('<div class="empty">No USB drives attached to the Jetson. Plug one into a USB-A port; '
                'it appears here within a few seconds.</div>')
    rows = []
    for drive in drives:
        parts = "<br/>".join(
            f'{esc(p["path"])}: {esc(p.get("label") or "no label")}, {esc(p.get("fstype") or "unformatted")}, '
            f'{esc(_gb(p.get("size_bytes")))}' + (f', mounted at {esc(p["mountpoint"])}' if p.get("mountpoint") else "")
            for p in drive.get("partitions") or []) or '<span class="muted">no partitions</span>'
        if drive.get("blocked"):
            actions = f'<span class="muted">Not offered: {esc(drive["blocked"])}</span>'
        else:
            buttons = [_format_button(drive, None, "Format whole drive as car storage")]
            buttons += [_format_button(drive, p, f'Format {p["path"]} only')
                        for p in drive.get("partitions") or [] if p.get("label") != "teslacam-backing"]
            actions = " ".join(buttons)
        rows.append(f"""
          <tr>
            <td>{esc(drive.get("model"))}<br/><span class="muted">{esc(drive["path"])}, {esc(_gb(drive.get("size_bytes")))}</span></td>
            <td>{esc(drive.get("role"))}</td>
            <td style="font-size:12px;">{parts}</td>
            <td>{actions}</td>
          </tr>""")
    return ('<table><tr><th>drive</th><th>role</th><th>partitions</th><th>actions</th></tr>'
            + "".join(rows) + "</table>")


def storage_fragment(status: dict[str, Any] | None = None) -> str:
    """Everything on the page that changes: refreshed in place every few seconds."""
    status = storage_status() if status is None else status
    return f"""
      <div id="storage-live" data-nonce="{esc(status.get("nonce") or "")}">
        <div class="card">{_storage_summary(status)}</div>
        {_storage_job(status)}
        <div class="card">
          <div style="font-weight:700; margin-bottom:10px;">USB drives on the Jetson</div>
          {_storage_drives(status) if status.get("service") != "missing" else
           '<div class="muted">Drives are listed by the car storage service, which is not running.</div>'}
        </div>
      </div>"""


@app.get("/api/storage")
def api_storage():
    return storage_status()


@app.get("/storage/live", response_class=HTMLResponse)
def storage_live():
    return HTMLResponse(storage_fragment())


@app.post("/api/storage/format")
async def api_storage_format(request: Request):
    """
    Ask carstore.py to format a drive as car storage. It re-checks everything
    itself; this only refuses what is plainly wrong and leaves it the request.

    JSON only, like /api/demo: a cross-site form cannot send it without a
    CORS preflight this app never answers.
    """
    from src.common import JOBS_DIR
    if "application/json" not in (request.headers.get("content-type") or ""):
        return JSONResponse({"ok": False, "error": "expected application/json"}, status_code=415)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "expected a JSON object"}, status_code=400)
    drive_id, partition = body.get("drive_id"), body.get("partition") or None
    if not (isinstance(drive_id, str) and len(drive_id) == 12 and all(c in "0123456789abcdef" for c in drive_id)):
        return JSONResponse({"ok": False, "error": "unknown drive"}, status_code=400)
    if partition is not None and not (isinstance(partition, str) and partition.startswith("/dev/")):
        return JSONResponse({"ok": False, "error": "unknown partition"}, status_code=400)
    if body.get("confirm") != "ERASE":
        return JSONResponse({"ok": False, "error": "type ERASE to confirm"}, status_code=400)
    status = storage_status()
    if status.get("service") != "running":
        return JSONResponse({"ok": False, "error": "the car storage service is not running"}, status_code=409)
    if (status.get("job") or {}).get("state") == "running":
        return JSONResponse({"ok": False, "error": "a format is already running"}, status_code=409)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = JOBS_DIR / ".carstore-request.json.tmp"
    temporary.write_text(json.dumps({
        "action": "format", "drive_id": drive_id, "partition": partition, "confirm": "ERASE",
        "nonce": str(body.get("nonce") or ""), "ts": int(time.time()),
    }), encoding="utf-8")
    os.replace(temporary, JOBS_DIR / "carstore-request.json")
    return {"ok": True}


STORAGE_SCRIPT = """
<dialog id="fmt" style="max-width:520px; background:var(--panel); color:var(--text);
        border:1px solid var(--line); border-radius:14px; padding:20px;">
  <div style="font-weight:800; font-size:17px; margin-bottom:10px;">Format as car storage?</div>
  <div id="fmt-erases" class="reason alarm" style="margin-bottom:12px;"></div>
  <div class="muted" style="margin-bottom:12px;">The drive becomes the car's recording drive: the Jetson
    creates the car's image on it and offers it to the car. This cannot be undone.</div>
  <label for="fmt-confirm">Type <b>ERASE</b> to confirm:</label>
  <input type="text" id="fmt-confirm" autocomplete="off" style="display:block; margin:8px 0 14px;" />
  <div class="row">
    <button id="fmt-go" disabled>Erase and format</button>
    <button id="fmt-cancel">Cancel</button>
  </div>
  <div id="fmt-msg" class="muted" style="margin-top:10px;"></div>
</dialog>
<script>
(() => {
  const dialog = document.getElementById("fmt"), confirmBox = document.getElementById("fmt-confirm");
  const go = document.getElementById("fmt-go"), msg = document.getElementById("fmt-msg");
  let target = null;
  document.addEventListener("click", (event) => {
    const button = event.target.closest && event.target.closest("[data-format-drive]");
    if (!button) return;
    target = {drive_id: button.dataset.formatDrive, partition: button.dataset.formatPartition || null,
              nonce: document.getElementById("storage-live").dataset.nonce};
    document.getElementById("fmt-erases").textContent = button.dataset.formatErases;
    confirmBox.value = ""; go.disabled = true; msg.textContent = "";
    dialog.showModal();
  });
  confirmBox.addEventListener("input", () => { go.disabled = confirmBox.value.trim() !== "ERASE"; });
  document.getElementById("fmt-cancel").addEventListener("click", () => dialog.close());
  go.addEventListener("click", async () => {
    go.disabled = true; msg.textContent = "Sending…";
    try {
      const response = await fetch(URL_ROOT + "/api/storage/format", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(Object.assign({confirm: confirmBox.value.trim()}, target)),
      });
      const answer = await response.json();
      if (answer.ok) { dialog.close(); refresh(); }
      else { msg.textContent = "Not sent: " + answer.error; go.disabled = false; }
    } catch (error) { msg.textContent = "Not sent: " + error; go.disabled = false; }
  });
  async function refresh() {
    if (dialog.open) return;
    try {
      const response = await fetch(URL_ROOT + "/storage/live", {cache: "no-store", tvQuiet: true});
      if (response.ok) document.getElementById("storage-live").outerHTML = await response.text();
    } catch (error) { /* keep the last view */ }
  }
  setInterval(refresh, 5000);
})();
</script>
"""


@app.get("/storage", response_class=HTMLResponse)
def storage_view():
    body = f"""
      <h1>Car storage</h1>
      <div class="sub">The car records into an image on a USB drive attached to the Jetson. Every minute the
        Jetson takes a snapshot of it and picks up the clips the car has finished, so footage reaches the
        pipeline about a minute after it is recorded - and the car keeps recording the whole time.</div>
      {storage_fragment()}
      {STORAGE_SCRIPT}
    """
    return render_page("Storage", body, "/storage")


# ---------------------------------------------------------------------------
# Sighting media: the frame a plate or face was read from, and its clip's alert
# ---------------------------------------------------------------------------

def _alerts_by_clip(connection) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Every alert's snapshot and GIF, by clip id and by clip filename.

    A sighting and its alert come from the same clip, so they meet on clip_id.
    Alerts imported from the legacy JSON files have no clip_id, so the clip's
    filename is the fallback. Read once per page rather than per sighting:
    alerts has no index on either column.
    """
    by_id: dict[str, Any] = {}
    by_file: dict[str, Any] = {}
    for alert in connection.execute(
        "SELECT clip_id, source_file, jpeg, gif FROM alerts ORDER BY timestamp"
    ).fetchall():
        if alert["clip_id"]:
            by_id[alert["clip_id"]] = alert
        if alert["source_file"]:
            by_file[alert["source_file"]] = alert
    return by_id, by_file


def _sighting_media(row, folder: str, alerts: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, str]:
    """
    What can be shown for one sighting: its frame, and its alert's snapshot and GIF.

    Only files that exist are offered. Sightings from before frames were saved
    have none; a clip that never raised an alert has no snapshot; a GIF is
    queued after the alert and only appears once the worker has written it.
    """
    media: dict[str, str] = {}
    frame = sighting_frame_name(folder, row["id"])
    if (MEDIA_DIR / frame).is_file():
        media["frame"] = frame
    by_id, by_file = alerts
    alert = by_id.get(row["clip_id"]) or by_file.get(row["source_file"])
    if alert:
        for kind, column in (("snapshot", "jpeg"), ("gif", "gif")):
            if alert[column] and (MEDIA_DIR / alert[column]).is_file():
                media[kind] = alert[column]
    return media


SIGHTING_MEDIA_LINKS = (
    ("frame", "🎯 frame", "the frame this was read from, with its box"),
    ("snapshot", "📷 snapshot", "the alert snapshot from this clip"),
    ("gif", "🎞️ GIF", "the alert GIF from this clip"),
)


def _media_links(media: dict[str, str]) -> str:
    """Links that open a sighting's media in the page's viewer."""
    return " ".join(
        f'<a class="btn media-link" href="{URL_ROOT}/media/{esc(media[kind])}" target="_blank" '
        f'data-viewer="1" title="{esc(title)}">{label}</a>'
        for kind, label, title in SIGHTING_MEDIA_LINKS if kind in media
    )


def _latest_media_by_entity(table: str, entity_column: str, folder: str) -> dict[str, dict[str, str]]:
    """
    Each identity's most recent sighting that has something to show.

    Newest first, stopping at the first sighting with media, so an identity's
    card links to its latest frame, snapshot or GIF.
    """
    connection = get_connection()
    try:
        alerts = _alerts_by_clip(connection)
        rows = connection.execute(
            f"SELECT id, {entity_column} AS entity_id, clip_id, source_file FROM {table} "
            "ORDER BY ts DESC"
        ).fetchall()
    finally:
        connection.close()

    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["entity_id"] in latest:
            continue
        media = _sighting_media(row, folder, alerts)
        if media:
            latest[row["entity_id"]] = media
    return latest


def _latest_media_block(media: dict[str, str] | None) -> str:
    if not media:
        return ""
    return (f'<div style="margin-top:10px;"><div class="muted" style="margin-bottom:4px;">'
            f'latest sighting</div>{_media_links(media)}</div>')


@app.get("/plates", response_class=HTMLResponse)
def plates_view():
    """Scout's AllPlates.vue."""
    plates = api_plates()
    latest_media = _latest_media_by_entity("plate_detections", "plate_id", "plates")

    cards = []
    for plate in plates:
        image = (
            f'<img class="evidence" loading="lazy" src="{URL_ROOT}/media/{esc(plate["best_image"])}" style="max-height:96px;" />'
            if plate.get("best_image") else '<div class="muted">no crop saved</div>'
        )
        known_badge = '<span class="pill low">known</span>' if plate["is_known"] else ""
        cards.append(f"""
          <div class="card">
            <div style="margin-bottom:10px;">
              <a href="{URL_ROOT}/entity/plate/{esc(plate["id"])}"><span class="plate">{esc(plate["plate_text"])}</span></a>
              {known_badge}
            </div>
            {image}
            <div class="muted" style="margin-top:10px;">
              {esc(plate["detection_count"])} sightings<br/>
              first {esc(format_timestamp(plate["first_seen_ts"]))}<br/>
              last {esc(format_timestamp(plate["last_seen_ts"]))}
            </div>
            <div style="margin-top:8px;">threat {esc(plate["threat_score"])}/100</div>
            {_latest_media_block(latest_media.get(plate["id"]))}
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
    latest_media = _latest_media_by_entity("face_detections", "face_id", "faces")

    cards = []
    for face in faces:
        image = (
            f'<img class="evidence" loading="lazy" src="{URL_ROOT}/media/{esc(face["best_image"])}" style="max-height:150px;" />'
            if face.get("best_image") else '<div class="muted">no crop saved</div>'
        )
        known_badge = '<span class="pill low">known</span>' if face["is_known"] else ""
        display_name = face["label"] or face["person_name"]
        cards.append(f"""
          <div class="card">
            <div style="font-weight:700; font-size:16px; margin-bottom:8px;">
              <a href="{URL_ROOT}/entity/face/{esc(face["id"])}">{esc(display_name)}</a> {known_badge}
            </div>
            {image}
            <div class="muted" style="margin-top:10px;">
              {esc(face["detection_count"])} sightings · {esc(face["embedding_count"])} embeddings<br/>
              first {esc(format_timestamp(face["first_seen_ts"]))}<br/>
              last {esc(format_timestamp(face["last_seen_ts"]))}
            </div>
            <div style="margin-top:8px;">threat {esc(face["threat_score"])}/100</div>
            {_latest_media_block(latest_media.get(face["id"]))}
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
                "SELECT id, ts, clip_id, source_file, camera, frame_number, det_confidence, "
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
        alerts = _alerts_by_clip(connection)
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
          <td>{'<img class="evidence" loading="lazy" src="' + URL_ROOT + '/media/' + esc(row["image"]) + '" style="max-height:52px;" />' if row["image"] else ""}</td>
          <td>{_media_links(_sighting_media(row, entity_type + "s", alerts)) or '<span class="muted">none saved</span>'}</td>
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
          <tr><th>when</th><th>camera</th><th>clip</th><th>location</th><th>evidence</th><th>media</th>{split_header}</tr>
          {detection_rows}
        </table>
      </div>

      <script>
        const ENTITY_TABLE = "{esc(entity_type)}s";
        const ENTITY_ID = "{esc(entity_id)}";
        let isKnown = {"true" if entity["is_known"] else "false"};

        async function post(payload) {{
          const response = await fetch(`{URL_ROOT}/api/${{ENTITY_TABLE}}/${{ENTITY_ID}}/label`, {{
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
          const response = await fetch(`{URL_ROOT}/api/faces/${{ENTITY_ID}}/merge`, {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ into: target }}),
          }});
          const result = await response.json();
          if (result.ok) location.href = `{URL_ROOT}/entity/face/${{target}}`;
          else alert("Merge failed: " + (result.error || "unknown error"));
        }}
        async function splitDetection(detectionId) {{
          const response = await fetch(`{URL_ROOT}/api/faces/detections/${{detectionId}}/split`, {{
            method: "POST",
          }});
          const result = await response.json();
          if (result.ok) location.href = `{URL_ROOT}/entity/face/${{result.new_face_id}}`;
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

    Leaflet itself is served from the Orin once scripts/fetch_leaflet.sh has
    run (the CDN otherwise). The map TILES still come from OpenStreetMap, so in
    a parked car with no signal the base map stays blank; the tracks and the
    rest of the dashboard are unaffected.
    """
    head_extra = leaflet_head(defer=False)

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
            data = await (await fetch(URL_ROOT + "/api/map")).json();
          } catch (error) {
            summary.innerHTML = '<span class="muted">Could not load map data.</span>';
            return;
          }

          if (typeof L === "undefined") {
            summary.innerHTML = '<span class="muted">Leaflet could not load (no internet). ' +
              'Map data is still available at <a href="' + URL_ROOT + '/api/map">/api/map</a>.</span>';
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
              history.href = URL_ROOT + "/entity/" + encodeURIComponent(kind) + "/" +
                encodeURIComponent(point.entity_id == null ? "" : point.entity_id);
              history.textContent = "history";
              popup.appendChild(history);
              if (point.image) {
                popup.appendChild(document.createElement("br"));
                const evidence = document.createElement("img");
                evidence.className = "evidence";
                evidence.src = URL_ROOT + "/media/" + encodeURIComponent(point.image);
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
        fetch(URL_ROOT + '/api/natix', {cache: 'no-store'}),
        fetch(URL_ROOT + '/api/natix/logs?lines=400', {cache: 'no-store'})
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
    The six-layer signal table: service, device, port, NMEA, fix, motion.

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
        # All six layers unknown rather than absent: an empty table reads as
        # "everything is fine and quiet", which is the opposite of the truth.
        reason = str(gps_state.get("error") or "no heartbeat has ever been written")
        return [{"name": name, "value": unknown, "age": reason}
                for name in ("service", "device", "port", "NMEA", "fix", "motion")]

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
        # The layer above the fix, and the only one that is about a SEQUENCE of
        # them. A single fix cannot distinguish a car at 2mph from a stationary
        # receiver's noise - both report a speed - so this row shows the
        # quantity that actually decides it and the threshold it is measured
        # against. Reading "driving: no · 13.9m of 50m" answers "why is there
        # no drive in the log" without anyone opening the database.
        _gps_motion_layer(heartbeat.get("motion"), unknown),
    ]


def _gps_motion_layer(motion: Any, unknown: str) -> dict[str, str]:
    """
    Render the motion gate's own view of itself.

    Split out because it is the one layer whose absence is expected: a
    heartbeat written by a gps.py older than the gate has no `motion` key, and
    an older page reading a newer heartbeat simply ignores it. Neither should
    raise, and neither should silently render as "not moving".
    """
    if not isinstance(motion, dict):
        return {
            "name": "motion",
            "value": unknown,
            "age": "this heartbeat predates the motion gate",
        }

    driving = motion.get("driving")
    displacement = motion.get("displacement_m")
    enter = motion.get("enter_m")
    samples = motion.get("samples")
    ratio = motion.get("path_ratio")

    measured = (f"{displacement}m of {enter}m needed"
                if displacement is not None and enter is not None else unknown)
    # The ratio is diagnostic only - it is shown because a large one is the
    # signature of a stationary receiver wandering, which is exactly the
    # question someone staring at an empty drive log is asking.
    wander = f" · wander {ratio}x" if ratio else ""

    return {
        "name": "motion",
        "value": f"driving: {'yes' if driving else 'no'} · {measured}{wander}",
        "age": f"over the last {samples} fixes" if samples else "no fixes in the window",
    }


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
      fetch(URL_ROOT + "/api/gps", {cache: "no-store", signal: controller.signal}),
      fetch(URL_ROOT + "/api/gps/logs?lines=400", {cache: "no-store", signal: controller.signal})
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
def leaflet_head(defer: bool = True) -> str:
    """
    Load Leaflet locally if it is actually vendored, and from the CDN if not.

    scripts/fetch_leaflet.sh puts an integrity-checked copy in
    MEDIA_DIR/leaflet; until it has run, the CDN keeps the maps working. The
    source is stamped into the page so a failure message can name what was
    tried instead of blaming the network for a missing directory.

    defer=False is for pages whose own inline script needs `L` as soon as it
    runs (the Timeline).
    """
    vendored_script = MEDIA_DIR / "leaflet" / "leaflet.js"
    vendored_style = MEDIA_DIR / "leaflet" / "leaflet.css"
    if vendored_script.exists() and vendored_style.exists():
        source = "vendored under /media/leaflet"
        style_href = f"{URL_ROOT}/media/leaflet/leaflet.css"
        script_src = f"{URL_ROOT}/media/leaflet/leaflet.js"
    else:
        source = "the unpkg CDN"
        style_href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        script_src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"

    return f"""
<link rel="stylesheet" href="{style_href}" />
<script>window.__leafletSource = "{esc(source)}";</script>
<script {"defer " if defer else ""}src="{script_src}"
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
            <a href="{URL_ROOT}/gps">Receiver detail →</a>
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
    fetch(URL_ROOT + "/api/findmy", {cache: "no-store", signal: controller.signal})
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
                       LOCATION_PAGE_HEAD + leaflet_head())
    # See the note in gps_view: a handler that returns a finished Response gets
    # it passed straight through, so the headers have to be set on it here.
    _apply_location_headers(page)
    return page


# Building a PlateReader loads the YOLO plate model - 3.5 s on the Orin -
# and the Settings page only wants its one-line status. Cached per process,
# refreshed after STAGE_STATUS_TTL_SECONDS, and warmed in the background at
# startup so even the first visit does not pay for it.
STAGE_STATUS_TTL_SECONDS = 600
_stage_status_cache: dict[str, tuple[float, str]] = {}


def _load_stage_status(stage: str) -> str:
    if stage == "alpr":
        from src.alpr import PlateReader
        return PlateReader().status_text()
    from src.faces import FaceEngine
    return FaceEngine().status_text()


def cached_stage_status(stage: str) -> str:
    cached = _stage_status_cache.get(stage)
    if cached and time.time() - cached[0] < STAGE_STATUS_TTL_SECONDS:
        return cached[1]
    text = _load_stage_status(stage)
    _stage_status_cache[stage] = (time.time(), text)
    return text


def _warm_stage_status() -> None:
    for stage in ("alpr", "faces"):
        try:
            cached_stage_status(stage)
        except Exception as error:                       # noqa: BLE001
            print(f"[⚙️ ui] could not pre-load {stage} status: {error}", flush=True)


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
        return cached_stage_status("alpr")

    def face_status():
        return cached_stage_status("faces")

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
          <a href="{URL_ROOT}/api/findings">/api/findings</a> ·
          <a href="{URL_ROOT}/api/plates">/api/plates</a> ·
          <a href="{URL_ROOT}/api/faces">/api/faces</a> ·
          <a href="{URL_ROOT}/api/polls">/api/polls</a> ·
          <a href="{URL_ROOT}/api/drives">/api/drives</a> ·
          <a href="{URL_ROOT}/api/map">/api/map</a> ·
          <a href="{URL_ROOT}/alerts">/alerts</a>
        </div>
      </div>
    """
    return render_page("Settings", body, "/settings")
