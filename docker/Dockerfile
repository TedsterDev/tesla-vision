# dockerfile

FROM python:3.10-slim

# ffmpeg  - GIF generation in gif_worker.py
# libgl1, libglib2.0-0 - OpenCV runtime dependencies
# tesseract-ocr - the character recognition half of the ALPR stage (src/alpr.py)
# libopenblas0, libgomp1 - linked by the JetPack PyTorch wheel below
# libjpeg62-turbo, libpng16-16 - torchvision's image extension
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libgl1 libglib2.0-0 \
	tesseract-ocr tesseract-ocr-eng \
	libopenblas0 libgomp1 libjpeg62-turbo libpng16-16 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch built for JetPack 6 / CUDA 12.6, from NVIDIA's Jetson AI Lab index,
# installed BEFORE requirements.txt so ultralytics finds it already satisfied.
# PyPI's own aarch64 torch is built for CUDA 13 on server ARM and never sees
# the Orin's GPU; on the CPU, YOLO took ~2 s a frame against ~35 ms here.
#
# torch 2.8 because 2.9+ links libcudss, which JetPack 6 does not ship.
# The wheel brings no CUDA or cuDNN of its own - JetPack 6's container runtime
# no longer injects them either - so docker-compose.yaml mounts the host's,
# and docker/jetson-env.sh puts them on the library path.
#
# docker/constraints.txt pins these (and numpy<2, which the wheel was built
# against) for every later pip install, so nothing quietly swaps them back.
COPY docker/constraints.txt /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
RUN pip install --no-cache-dir \
	--index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple \
	torch torchvision

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

# OpenCV 5 removed cv2.CascadeClassifier and the bundled haarcascades, which
# both Haar fallbacks (faces.py, alpr.py) rely on - an unpinned rebuild on
# 2026-09-08 picked up 5.0.0 and crash-looped the processor. Pinned here, not
# in requirements.txt, for the same cache reason as PyJWT above.
RUN pip install --no-cache-dir "opencv-python<5"

COPY src/ /app/src

# Tests and the backfill utility ship in the image so they can be run against
# the exact dependency set the pipeline uses:
#   docker compose exec processor python -u /app/tests/test_alpr.py
#   docker compose exec processor python -u /app/scripts/backfill.py
COPY tests/ /app/tests
COPY scripts/ /app/scripts

ENV PYTHONUNBUFFERED=1

# Read by the nvidia runtime, which only the services that set
# `runtime: nvidia` in docker-compose.yaml use; harmless everywhere else.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# In the image's environment rather than set by the entrypoint, so that
# `docker compose exec` (which skips the entrypoint) loads torch too.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/host-cudnn

COPY docker/jetson-env.sh /usr/local/bin/jetson-env.sh
ENTRYPOINT ["/usr/local/bin/jetson-env.sh"]
