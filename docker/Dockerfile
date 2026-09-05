# dockerfile

FROM python:3.10-slim

# ffmpeg  - GIF generation in gif_worker.py
# libgl1, libglib2.0-0 - OpenCV runtime dependencies
# tesseract-ocr - the character recognition half of the ALPR stage (src/alpr.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libgl1 libglib2.0-0 \
	tesseract-ocr tesseract-ocr-eng \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Installed as its own layer, deliberately NOT added to requirements.txt.
#
# That file pins torch and the CUDA runtime, so any edit to it invalidates the
# layer above and makes Docker re-download several hundred MB of nvidia_* wheels
# - about half an hour on an Orin Nano. A small pure-Python dependency added
# late costs seconds instead, and the expensive layer stays cached.
#
# PyJWT verifies Cloudflare Access identity tokens (src/cfaccess.py). Access
# consumes the Authorization header, so HTTP Basic cannot work behind it; the
# signed JWT it forwards has to be checked properly rather than trusted.
RUN pip install --no-cache-dir "PyJWT[crypto]>=2.8"

COPY src/ /app/src

# Tests and the backfill utility ship in the image so they can be run against
# the exact dependency set the pipeline uses:
#   docker compose exec processor python -u /app/tests/test_alpr.py
#   docker compose exec processor python -u /app/scripts/backfill.py
COPY tests/ /app/tests
COPY scripts/ /app/scripts

ENV PYTHONUNBUFFERED=1
