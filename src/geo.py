"""
geo.py

Geospatial helpers and the reverse-geocode cache.

Scout stored a `geocodes` collection full of Nominatim responses and pointed
each poll at one via `geocodeID`. We keep that idea but add the thing Scout's
version was missing: a cache key. Scout geocoded per-poll, which at a 1-poll-
per-few-seconds rate is thousands of Nominatim hits per drive - enough to get
your IP banned from OSM's free endpoint. We round coordinates to ~11m and reuse
the answer, which cuts request volume by orders of magnitude on a normal drive.

Nothing here requires the network. Without connectivity `reverse_geocode`
returns None and the rest of the pipeline carries on with raw lat/lon.
"""
import json
import math
import os
import threading
import time
import urllib.parse
import urllib.request

from typing import Any

from src.db import now_ts, upsert

# OSM asks that every client identify itself. Being a good citizen also means
# we honour their 1 request/second rate limit (enforced below).
NOMINATIM_URL = os.environ.get(
    "NOMINATIM_URL", "https://nominatim.openstreetmap.org/reverse"
)
NOMINATIM_USER_AGENT = os.environ.get(
    "NOMINATIM_USER_AGENT", "tesla-vision-scout/1.0 (personal counter-surveillance)"
)
NOMINATIM_MIN_INTERVAL_SECONDS = 1.1

# Rounding to 4 decimal places is roughly 11 metres at mid latitudes - fine
# enough to distinguish two ends of a block, coarse enough to cache well.
GEOCODE_PRECISION = 4

EARTH_RADIUS_MILES = 3958.7613

_last_nominatim_request_at = 0.0

# The rate limiter above is a plain global, and a plain global is not a rate
# limiter under concurrency: uvicorn runs request handlers on a threadpool, so
# two handlers geocoding at once both read the same stale timestamp, both sleep
# until the same instant, and both fire - one request per second becomes N.
# The failure is invisible from here (Nominatim answers both, then bans the IP
# hours later) so the lock is the only thing making the interval mean anything.
# It is per-process: poller, processor and web each run their own limiter, and
# closing THAT gap needs a shared token outside this module.
_NOMINATIM_LOCK = threading.Lock()


def haversine_miles(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """
    Great-circle distance between two points, in miles.

    Used everywhere in the correlation engine: "how far apart were these two
    sightings of the same plate?" is the question that separates a neighbour
    from a tail.
    """
    lat_a_rad, lat_b_rad = math.radians(lat_a), math.radians(lat_b)
    delta_lat = math.radians(lat_b - lat_a)
    delta_lon = math.radians(lon_b - lon_a)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a_rad) * math.cos(lat_b_rad) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(a)))


def bearing_degrees(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """
    Compass bearing from point A to point B, 0-360 degrees.

    Lets us compare where a detection sits relative to the direction the car is
    travelling, which is how we tell "behind me" from "beside me".
    """
    lat_a_rad, lat_b_rad = math.radians(lat_a), math.radians(lat_b)
    delta_lon = math.radians(lon_b - lon_a)

    x = math.sin(delta_lon) * math.cos(lat_b_rad)
    y = math.cos(lat_a_rad) * math.sin(lat_b_rad) - math.sin(lat_a_rad) * math.cos(lat_b_rad) * math.cos(delta_lon)

    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def cache_key_for(lat: float, lon: float) -> str:
    """Rounded coordinate string used as the geocode cache's unique key."""
    return f"{round(lat, GEOCODE_PRECISION)},{round(lon, GEOCODE_PRECISION)}"


def _fetch_from_nominatim(lat: float, lon: float, timeout: float = 8.0) -> dict[str, Any] | None:
    """
    Ask Nominatim what address is at these coordinates.

    Returns the parsed JSON, or None on any failure (offline, rate limited,
    malformed). Callers must treat None as "no address available", never as an
    error worth stopping for - we spend most of our life without connectivity.
    """
    global _last_nominatim_request_at

    query = urllib.parse.urlencode({
        "lat": f"{lat:.6f}",
        "lon": f"{lon:.6f}",
        "format": "jsonv2",
        "zoom": 18,
        "addressdetails": 1,
    })
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{query}",
        headers={"User-Agent": NOMINATIM_USER_AGENT},
    )

    # The lock spans the request itself, not just the arithmetic, because the
    # timestamp is stamped on COMPLETION: releasing it before the call would let
    # a second thread read a timestamp belonging to a request still in flight,
    # decide the interval had elapsed, and fire alongside the first. Serialising
    # costs nothing we had - a 1 req/sec budget has no concurrency to lose - but
    # it does mean a caller can wait out the 8s timeout below behind another
    # thread, which is why no request handler may call this with allow_network.
    with _NOMINATIM_LOCK:
        # Respect the 1 req/sec policy.
        elapsed = time.time() - _last_nominatim_request_at
        if elapsed < NOMINATIM_MIN_INTERVAL_SECONDS:
            time.sleep(NOMINATIM_MIN_INTERVAL_SECONDS - elapsed)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        finally:
            # Stamped even on failure: a request that timed out still consumed
            # a slot at OSM's end, and a failing endpoint is exactly when a
            # caller retries hardest.
            _last_nominatim_request_at = time.time()

    return payload if isinstance(payload, dict) else None


def reverse_geocode(connection, lat: float | None, lon: float | None, allow_network: bool = True) -> dict[str, Any] | None:
    """
    Resolve coordinates to a street address, using the SQLite cache first.

    Args:
        connection: open sqlite3 connection
        lat, lon: coordinates, may be None when we had no GPS fix
        allow_network: set False to stay purely offline (cache-only lookups)

    Returns the geocode row as a dict, or None if unknown.
    """
    if lat is None or lon is None:
        return None

    key = cache_key_for(lat, lon)

    cached = connection.execute(
        "SELECT * FROM geocodes WHERE cache_key=?", (key,)
    ).fetchone()
    if cached:
        return dict(cached)

    if not allow_network:
        return None

    payload = _fetch_from_nominatim(lat, lon)
    if not payload:
        return None

    address = payload.get("address", {}) or {}
    record = {
        "id": key,
        "cache_key": key,
        "lat": lat,
        "lon": lon,
        "display_name": payload.get("display_name", ""),
        "house_number": address.get("house_number"),
        "road": address.get("road"),
        "suburb": address.get("suburb") or address.get("neighbourhood"),
        # Nominatim is inconsistent about which key holds the settlement name.
        "city": address.get("city") or address.get("town") or address.get("village"),
        "county": address.get("county"),
        "state": address.get("state"),
        "postcode": address.get("postcode"),
        "country": address.get("country"),
        "country_code": address.get("country_code"),
        "fetched_ts": now_ts(),
    }

    upsert(connection, "geocodes", record)
    connection.commit()
    return record


def spatial_spread_miles(points: list[tuple[float, float]]) -> float:
    """
    Largest distance between any two points in a set, in miles.

    This is the "how spread out were these sightings" number. A plate seen five
    times within 200 feet is a parked neighbour; the same plate seen five times
    across nine miles has been following you.

    O(n^2), which is fine: a single entity rarely has more than a few hundred
    detections, and we only score entities that crossed a detection threshold.
    """
    if len(points) < 2:
        return 0.0

    widest = 0.0
    for index, (lat_a, lon_a) in enumerate(points):
        for lat_b, lon_b in points[index + 1:]:
            widest = max(widest, haversine_miles(lat_a, lon_a, lat_b, lon_b))
    return widest


def cluster_locations(points: list[tuple[float, float]], radius_miles: float = 0.15) -> int:
    """
    Count how many distinct places a set of coordinates represents.

    Simple greedy clustering: walk the points, and start a new cluster whenever
    one falls further than `radius_miles` from every existing cluster centre.
    ~0.15 miles (about 800 feet) treats one parking lot as one location.

    Distinct *locations* matters more than raw detection count. Twenty sightings
    in your own driveway is one location and zero threat.
    """
    centres: list[tuple[float, float]] = []

    for lat, lon in points:
        if not any(haversine_miles(lat, lon, c_lat, c_lon) <= radius_miles for c_lat, c_lon in centres):
            centres.append((lat, lon))

    return len(centres)
