"""
poller.py

Vehicle telemetry poller - the port of Scout's `scripts/TeslaJS/poll.js`.

Scout polled the Tesla Owner API through TeslaJS and wrote three collections:
`polls` (a telemetry sample), `drives` (a contiguous run of driving polls) and
`geocodes` (the reverse-geocoded address for a poll). We produce the same three
tables, with three corrections:

    1. The Owner API Scout used (owner-api.teslamotors.com) was retired. This
       talks to the Fleet API, and refreshes its own OAuth token.

    2. Scout polled on a fixed cron. That keeps the car awake, which drains the
       battery of a parked vehicle - the exact complaint its README warns about
       under "don't stress the Tesla servers". We check the cheap vehicle-list
       endpoint first and only request full telemetry when the car is *already*
       online, so a sleeping car is left to sleep.

    3. Scout geocoded every poll. We geocode on movement (see geo.py's cache),
       which is the difference between a few hundred Nominatim calls a day and
       a few tens of thousands.

Why this matters to the rest of the system: every plate and face detection gets
stamped with the location and drive it happened on, and that stamp is what the
correlation engine reasons over. Without it, "seen at 4 distinct locations"
degrades to "seen 4 times", and a neighbour becomes indistinguishable from a
tail. The pipeline runs without credentials - it is just less discerning.

Configuration (all via environment, normally .env):
    TESLA_ACCESS_TOKEN     current OAuth access token
    TESLA_REFRESH_TOKEN    refresh token, used to mint new access tokens
    TESLA_CLIENT_ID        your Fleet API application's client id
    TESLA_CLIENT_SECRET    your Fleet API application's client secret
    TESLA_VIN              which vehicle to poll (optional if you have one car)
    TESLA_API_BASE         regional Fleet API host (default: North America)
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from src.common import env_int, env_float
from src.db import connect, now_ts, upsert
from src.geo import haversine_miles, reverse_geocode

TESLA_API_BASE = os.environ.get(
    "TESLA_API_BASE", "https://fleet-api.prd.na.vn.cloud.tesla.com"
)
TESLA_TOKEN_URL = os.environ.get(
    "TESLA_TOKEN_URL", "https://auth.tesla.com/oauth2/v3/token"
)

# Poll cadence. Driving needs resolution to place detections on a route;
# a parked car needs almost none, and asking costs battery.
DRIVING_POLL_SECONDS = env_int("DRIVING_POLL_SECONDS", 10)
PARKED_POLL_SECONDS = env_int("PARKED_POLL_SECONDS", 300)
ASLEEP_POLL_SECONDS = env_int("ASLEEP_POLL_SECONDS", 900)

# A drive ends after this long without movement.
DRIVE_IDLE_TIMEOUT_SECONDS = env_int("DRIVE_IDLE_TIMEOUT_SECONDS", 300)

# Below this speed we treat the car as stationary regardless of shift state.
#
# Applies to the Fleet API path, where a parked car reports speed = None and
# this threshold only has to clear zero. It is NOT what classifies a GNSS
# receiver - see MotionGate in gps.py, which decides from displacement because
# a stationary receiver reports 0.5-3.5 mph of noise indefinitely and this
# threshold called it driving 96% of the time.
MOVING_SPEED_THRESHOLD_MPH = env_float("MOVING_SPEED_THRESHOLD_MPH", 1.0)

# A closed drive whose furthest point is nearer than this to where it started
# never went anywhere, and is deleted. 250 feet.
DRIVE_MIN_DISPLACEMENT_MILES = env_float("DRIVE_MIN_DISPLACEMENT_MILES", 0.047)

# Every table with a drive_id column. Kept next to the schema's consumers so
# that adding one to db.py without adding it here is a visible omission.
DRIVE_ID_TABLES = ("polls", "clips", "plate_detections", "face_detections")


class TeslaFleetClient:
    """
    Minimal Fleet API client - just enough for telemetry.

    Deliberately built on urllib rather than `requests` so the container image
    stays small and this module has no dependency the rest of the app lacks.
    """

    def __init__(self) -> None:
        self.access_token = os.environ.get("TESLA_ACCESS_TOKEN", "").strip()
        self.refresh_token = os.environ.get("TESLA_REFRESH_TOKEN", "").strip()
        self.client_id = os.environ.get("TESLA_CLIENT_ID", "").strip()
        self.client_secret = os.environ.get("TESLA_CLIENT_SECRET", "").strip()
        self.vin = os.environ.get("TESLA_VIN", "").strip()
        self.vehicle_tag: str | None = self.vin or None

    @property
    def configured(self) -> bool:
        """True when we have enough to try an API call."""
        return bool(self.access_token or (self.refresh_token and self.client_id))

    def status_text(self) -> str:
        if not self.configured:
            return "disabled - no TESLA_ACCESS_TOKEN / TESLA_REFRESH_TOKEN set"
        return f"configured (vehicle={self.vehicle_tag or 'first available'})"

    # -- transport --------------------------------------------------------
    def _request(self, path: str, method: str = "GET", retry_on_auth_failure: bool = True):
        """
        Call the Fleet API, refreshing the access token once on a 401.

        Returns the decoded `response` field, or None on any failure. Failure is
        normal here (no signal in a parking garage), so it is never fatal.
        """
        url = f"{TESLA_API_BASE}{path}"
        request = urllib.request.Request(url, method=method)
        request.add_header("Authorization", f"Bearer {self.access_token}")
        request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("response")
        except urllib.error.HTTPError as http_error:
            if http_error.code == 401 and retry_on_auth_failure and self.refresh_access_token():
                return self._request(path, method, retry_on_auth_failure=False)
            if http_error.code == 429:
                # Tesla is rate limiting us. Back off hard - the README's
                # warning about getting the whole library's key revoked is real.
                print("[🚗 poller] rate limited by Tesla, backing off 60s")
                time.sleep(60)
            return None
        except Exception:
            return None

    def refresh_access_token(self) -> bool:
        """Exchange the refresh token for a new access token."""
        if not (self.refresh_token and self.client_id):
            return False

        form = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": self.refresh_token,
        }
        if self.client_secret:
            form["client_secret"] = self.client_secret

        data = urllib.parse.urlencode(form).encode("utf-8")
        request = urllib.request.Request(TESLA_TOKEN_URL, data=data, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exception_object:
            print(f"[🚗 poller] token refresh failed: {exception_object}")
            return False

        self.access_token = payload.get("access_token", self.access_token)
        # Tesla rotates refresh tokens; keeping the new one is what stops the
        # poller silently dying a few weeks after you set it up.
        self.refresh_token = payload.get("refresh_token", self.refresh_token)
        print("[🚗 poller] access token refreshed")
        return bool(self.access_token)

    # -- endpoints --------------------------------------------------------
    def list_vehicles(self) -> list[dict]:
        return self._request("/api/1/vehicles") or []

    def resolve_vehicle_tag(self) -> str | None:
        """
        Find which vehicle to poll, honouring TESLA_VIN when set.

        Cached on the instance because the vehicle list rarely changes and this
        gets called every poll cycle.
        """
        if self.vehicle_tag:
            return self.vehicle_tag

        for vehicle in self.list_vehicles():
            if not self.vin or vehicle.get("vin") == self.vin:
                self.vehicle_tag = vehicle.get("vin") or str(vehicle.get("id"))
                return self.vehicle_tag
        return None

    def vehicle_state(self) -> str:
        """
        Cheap check of whether the car is awake: 'online', 'asleep', 'offline'.

        This is the call that lets us avoid waking a sleeping car.
        """
        for vehicle in self.list_vehicles():
            if not self.vin or vehicle.get("vin") == self.vin:
                return vehicle.get("state", "unknown")
        return "unknown"

    def vehicle_data(self) -> dict | None:
        """Full telemetry. Only call this when the car is already online."""
        tag = self.resolve_vehicle_tag()
        if not tag:
            return None
        query = urllib.parse.urlencode({"endpoints": "drive_state;vehicle_state;charge_state"})
        return self._request(f"/api/1/vehicles/{tag}/vehicle_data?{query}")


def poll_record_from_vehicle_data(data: dict) -> dict:
    """
    Flatten Tesla's nested telemetry into one `polls` row.

    Mirrors the fields Scout's Poll mongoose model captured, plus odometer.
    `status` uses Scout's single-letter convention: D driving, C charging,
    P parked - the dashboard and the correlation engine both read it.
    """
    drive_state = data.get("drive_state", {}) or {}
    charge_state = data.get("charge_state", {}) or {}
    vehicle_state = data.get("vehicle_state", {}) or {}

    latitude = drive_state.get("latitude")
    longitude = drive_state.get("longitude")
    speed = drive_state.get("speed")            # mph, None when stationary
    shift_state = drive_state.get("shift_state")  # 'D' | 'R' | 'N' | 'P' | None

    is_moving = bool(speed and speed > MOVING_SPEED_THRESHOLD_MPH)
    is_charging = str(charge_state.get("charging_state", "")).lower() == "charging"

    status = "D" if is_moving else ("C" if is_charging else "P")

    # Tesla reports drive_state.timestamp in milliseconds.
    raw_timestamp = drive_state.get("timestamp") or 0
    timestamp = int(raw_timestamp / 1000) if raw_timestamp > 1e11 else (int(raw_timestamp) or now_ts())

    return {
        "id": uuid.uuid4().hex[:12],
        "ts": timestamp,
        "lat": latitude,
        "lon": longitude,
        "heading": drive_state.get("heading"),
        "speed": float(speed) if speed else 0.0,
        "power": drive_state.get("power"),
        "shift_state": shift_state,
        "status": status,
        "loc_available": 1 if (latitude is not None and longitude is not None) else 0,
        "odometer": vehicle_state.get("odometer"),
        "street": None,
        "city": None,
        "drive_id": None,
        "geocode_id": None,
    }


def discard_drive_that_went_nowhere(connection, drive_id: str) -> bool:
    """
    Delete a just-closed drive that never left where it started. True if it did.

    This replaces a guard that read `if seconds_since_start < 60` and was
    unreachable. Reaching it required a gap of at least
    DRIVE_IDLE_TIMEOUT_SECONDS (300s) since the anchoring poll, and since that
    poll belongs to this drive, seconds_since_start was necessarily >= 300 too.
    The `< 60` test could never be true, so the comment's promise - that noise
    drives "can't inflate anyone's distinct drives count" - was never kept.

    Duration was the wrong measure anyway. A drive that lasted ten minutes in a
    parking space is exactly the thing worth discarding, and a legitimate
    30-second hop to the end of the street is not. Displacement says which is
    which: a real journey ends somewhere else, or at least passes somewhere
    else on the way.

    The polls are RELINKED, not deleted, and that ordering is the whole
    subtlety. A poll remains a true record that the car was at that place at
    that time, and location_at() reads polls - not drives - to stamp clips.
    Deleting the drives row while leaving polls.drive_id pointing at it would
    leave every detection stamped with a drive that no longer exists, so
    correlate.py would keep counting it in distinct_drives (it reads the
    detection's own column, never joining back to drives) while the operator's
    view of that journey was gone. That is strictly worse than leaving it.
    """
    points = [
        (row["lat"], row["lon"])
        for row in connection.execute(
            "SELECT lat, lon FROM polls WHERE drive_id=? AND lat IS NOT NULL "
            "ORDER BY ts",
            (drive_id,),
        )
    ]

    if len(points) >= 2:
        start_lat, start_lon = points[0]
        displacement = max(
            haversine_miles(start_lat, start_lon, lat, lon)
            for lat, lon in points[1:]
        )
        if displacement >= DRIVE_MIN_DISPLACEMENT_MILES:
            return False
    elif len(points) == 1:
        # One located poll is not evidence of a journey.
        pass
    else:
        # No located polls at all - nothing could have been observed on it.
        pass

    # Every table that carries drive_id, not only polls. Detections and clips
    # are stamped with it by location_at(), and correlate.py counts
    # distinct_drives from the detection's own column without joining back to
    # drives - so a reference left behind here keeps a deleted phantom in the
    # score forever, invisibly. The literal tuple is the whole allowlist.
    for table in DRIVE_ID_TABLES:
        connection.execute(f"UPDATE {table} SET drive_id=NULL WHERE drive_id=?", (drive_id,))
    connection.execute("DELETE FROM drives WHERE id=?", (drive_id,))
    return True


def attach_drive(connection, poll: dict) -> str | None:
    """
    Assign this poll to a drive, opening or closing one as needed.

    A drive opens on the first moving poll and closes once the car has not
    MOVED for DRIVE_IDLE_TIMEOUT_SECONDS. Grouping by drive is what lets the
    correlation engine say "on four separate journeys" rather than "four
    times", which is a far stronger statement.

    "Not moved for", not "no poll for". Those read alike and are not alike, and
    the difference silently disabled the whole mechanism once a GNSS receiver
    became the source. The old test measured the gap to the last poll of ANY
    status, but a parked poll is still attached to the open drive and so
    becomes that last poll - meaning the gap could only grow while nothing was
    being written at all. It happened to work against the Fleet API, where a
    parked car is polled every PARKED_POLL_SECONDS and the write cadence alone
    produced the gap. Against a receiver reporting continuously it made the
    close branch unreachable: one drive stayed open forever and absorbed every
    poll, which pins distinct_drives at 1 - zeroing the 45-point signal
    correlate.py calls its strongest - and makes detect_active_tail's "within a
    single journey" test vacuous, so three sightings days apart score 75.0 and
    fire an urgent push. Anchoring on the last MOVING poll makes the condition
    mean what its name says.
    """
    open_drive = connection.execute(
        "SELECT * FROM drives WHERE is_open=1 ORDER BY start_ts DESC LIMIT 1"
    ).fetchone()

    is_moving = poll["status"] == "D"

    if open_drive:
        last_moving = connection.execute(
            "SELECT ts FROM polls WHERE drive_id=? AND status='D' "
            "ORDER BY ts DESC LIMIT 1",
            (open_drive["id"],),
        ).fetchone()
        last_moving_ts = last_moving["ts"] if last_moving else open_drive["start_ts"]
        last_poll = connection.execute(
            "SELECT ts, lat, lon FROM polls WHERE drive_id=? ORDER BY ts DESC LIMIT 1",
            (open_drive["id"],),
        ).fetchone()

        if is_moving or (poll["ts"] - last_moving_ts) < DRIVE_IDLE_TIMEOUT_SECONDS:
            # Still the same journey - extend it.
            #
            # Distance accrues only on MOVING polls. This is a path length, so
            # summing consecutive steps is right for a car - an odometer
            # measures the road, not the straight line home. It is wrong for a
            # stationary receiver, whose jitter has no net direction but plenty
            # of length: measured on this hardware, 86.2m of summed steps
            # against 5.5m of actual displacement, which accrued 23.6 phantom
            # miles a day. Gating on is_moving keeps the odometer semantics for
            # the case they suit and contributes zero for the case they do not.
            distance_added = 0.0
            if (is_moving and last_poll
                    and last_poll["lat"] is not None and poll["lat"] is not None):
                distance_added = haversine_miles(
                    last_poll["lat"], last_poll["lon"], poll["lat"], poll["lon"]
                )
            connection.execute(
                "UPDATE drives SET end_ts=?, end_lat=?, end_lon=?, end_heading=?, "
                "distance_miles=distance_miles+?, poll_count=poll_count+1 WHERE id=?",
                (poll["ts"], poll["lat"], poll["lon"], poll["heading"],
                 distance_added, open_drive["id"]),
            )
            return open_drive["id"]

        # Idle too long - close it out.
        connection.execute("UPDATE drives SET is_open=0 WHERE id=?", (open_drive["id"],))
        discard_drive_that_went_nowhere(connection, open_drive["id"])

    if not is_moving:
        return None

    drive_id = uuid.uuid4().hex[:12]
    upsert(connection, "drives", {
        "id": drive_id,
        "start_ts": poll["ts"],
        "start_lat": poll["lat"],
        "start_lon": poll["lon"],
        "start_heading": poll["heading"],
        "end_ts": poll["ts"],
        "end_lat": poll["lat"],
        "end_lon": poll["lon"],
        "end_heading": poll["heading"],
        "distance_miles": 0.0,
        "poll_count": 1,
        "is_open": 1,
    })
    print(f"[🚗 poller] new drive {drive_id} started")
    return drive_id


def location_at(connection, timestamp: int, tolerance_seconds: int = 900) -> dict | None:
    """
    Where was the car at this moment?

    Used by processor.py to stamp every detection with a position and a drive.
    We take the nearest poll within `tolerance_seconds` - 15 minutes by default,
    generous because a parked car's position is stable for hours and a clip's
    timestamp only needs to land on the right stop.

    Returns {'lat', 'lon', 'drive_id'} or None.
    """
    row = connection.execute(
        """
        SELECT lat, lon, drive_id, ABS(ts - ?) AS distance_seconds
        FROM polls
        WHERE lat IS NOT NULL AND ABS(ts - ?) <= ?
        ORDER BY distance_seconds ASC
        LIMIT 1
        """,
        (timestamp, timestamp, tolerance_seconds),
    ).fetchone()

    return dict(row) if row else None


def main():
    """Poll loop. Exits immediately with a clear message if unconfigured."""
    connection = connect()
    client = TeslaFleetClient()

    print(f"[🚗 poller] {client.status_text()}")
    if not client.configured:
        print("[🚗 poller] nothing to do - set TESLA_ACCESS_TOKEN and TESLA_REFRESH_TOKEN in .env")
        print("[🚗 poller] the detection pipeline runs fine without this, just without GPS context")
        # Idle rather than exit. Under `restart: unless-stopped` an exit here
        # becomes a restart loop, which shows up in `docker compose ps` as a
        # flapping container and looks like a crash. Sitting quietly is the
        # honest representation of "configured to do nothing".
        while True:
            time.sleep(3600)
            print("[🚗 poller] still idle - no Tesla credentials configured")

    while True:
        sleep_seconds = PARKED_POLL_SECONDS

        try:
            state = client.vehicle_state()

            if state != "online":
                # Let a sleeping car sleep. This is the single most important
                # behaviour difference from Scout's fixed cron.
                print(f"[🚗 poller] vehicle is {state}, not waking it")
                sleep_seconds = ASLEEP_POLL_SECONDS
            else:
                data = client.vehicle_data()
                if data:
                    poll = poll_record_from_vehicle_data(data)
                    poll["drive_id"] = attach_drive(connection, poll)

                    geocode = reverse_geocode(connection, poll["lat"], poll["lon"])
                    if geocode:
                        poll["geocode_id"] = geocode["id"]
                        poll["street"] = geocode.get("road")
                        poll["city"] = geocode.get("city")

                    upsert(connection, "polls", poll)
                    connection.commit()

                    where = poll["street"] or f"{poll['lat']},{poll['lon']}"
                    print(f"[🚗 poller] {poll['status']} {poll['speed']:.0f}mph @ {where}")

                    sleep_seconds = (
                        DRIVING_POLL_SECONDS if poll["status"] == "D" else PARKED_POLL_SECONDS
                    )

        except Exception as exception_object:
            print(f"[🚗 poller] ⧱❗️ ERROR: {exception_object}")

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
