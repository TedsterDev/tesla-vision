#!/bin/bash
# fetch_leaflet.sh - put Leaflet 1.9.4 under media/leaflet so every map page
# (Timeline, Find My, GPS) loads it from the Orin instead of the unpkg CDN.
#
#     ./scripts/fetch_leaflet.sh
#
# Why: from the CDN the Timeline took ~3.9 s to load the first time, and in a
# car with no signal the map could not load at all. The pages already prefer
# a local copy when one exists (ui_app.leaflet_head); nothing created it.
#
# Runs the download inside the web container, which can write the root-owned
# media directory, and refuses files whose hashes do not match the integrity
# values Leaflet publishes for 1.9.4.
set -euo pipefail

cd "$(dirname "$0")/.."
docker compose exec -T web python3 - <<'PY'
import base64, hashlib, pathlib, urllib.request

BASE = "https://unpkg.com/leaflet@1.9.4/dist/"
# Leaflet's published SRI values for 1.9.4 (leafletjs.com/download.html).
EXPECTED = {
    "leaflet.js": "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=",
    "leaflet.css": "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=",
}
IMAGES = ["layers.png", "layers-2x.png", "marker-icon.png", "marker-icon-2x.png", "marker-shadow.png"]

from src.common import MEDIA_DIR
target = MEDIA_DIR / "leaflet"
(target / "images").mkdir(parents=True, exist_ok=True)

def fetch(name: str) -> bytes:
    with urllib.request.urlopen(BASE + name, timeout=30) as response:
        return response.read()

for name, sri in EXPECTED.items():
    data = fetch(name)
    got = "sha256-" + base64.b64encode(hashlib.sha256(data).digest()).decode()
    if got != sri:
        raise SystemExit(f"{name}: integrity mismatch ({got}); nothing written")
    (target / name).write_bytes(data)
    print(f"  ok  {name} ({len(data) // 1024} KB, integrity verified)")
for name in IMAGES:
    data = fetch("images/" + name)
    (target / "images" / name).write_bytes(data)
    print(f"  ok  images/{name}")
print(f"Leaflet is now served from {target}")
PY
