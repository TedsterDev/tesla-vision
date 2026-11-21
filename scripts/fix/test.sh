#!/usr/bin/env bash
# The security-fix pipeline's test job for tesla-vision (.tedebyte/fix.json
# jobs.test; tedebyte-admin templates/security-fixes/, merge-only mode,
# FLEET-528). Exit 0 = the fix may be merged; it reaches the car only when the
# owner runs jetson-setup's push scripts from the Mac.
#
# The tests are the README's "Tests" section: every tests/test_*.py run as its
# own script (`python -u tests/test_x.py`; each one sets up its own sandbox
# before importing src/, so they never share a process). Any one failing fails
# the job.
#
# The dependencies follow docker/Dockerfile: torch and torchvision first, then
# requirements.txt, then the Dockerfile's own extra pins, all under
# docker/constraints.txt (PIP_CONSTRAINT). Three differences, because this
# runs in a plain Python image rather than the Jetson's:
#   - torch and torchvision come from PyTorch's CPU index at the versions
#     constraints.txt pins; the Jetson AI Lab wheels the Dockerfile uses are
#     built for JetPack's CUDA on aarch64 only. The tests use fakes for YOLO,
#     so no GPU and no model weights are needed.
#   - opencv-python is swapped for opencv-python-headless at the same version:
#     the image has no libGL, and the tests use no OpenCV window.
#   - tesseract is not installed (no root here), so test_alpr.py skips its
#     Tesseract cases, as it does on any host without it.
# Not run: scripts/selfcheck.py and scripts/mutation_test.py, which need the
# model weights scripts/fetch_models.sh stages on the Jetson's data volume.
#
# Where it runs: `bash scripts/fix/test.sh` from the repository root at the
# commit under test, inside fix.json's image (Python 3.10, as the Dockerfile),
# on a GitHub-hosted runner (fix-job.yml): a public repository never runs on
# the Jetson. The checkout has no .git directory. No token and no secret.
#
# Environment: FIX_KIND=test, FIX_SHA, FIX_REPO, FIX_REQUEST.
#
# It refuses to run with xtrace on (bash -x or set -x).

case $- in
  *x*)
    echo "test.sh: refusing to run with xtrace on (bash -x or set -x)." >&2
    exit 1
    ;;
esac

set -euo pipefail

# The tests behave the same wherever the job runs: CI mode, nothing watching.
export CI=1

# pip and Ultralytics keep files under HOME. The job may run as a user with no
# home of its own, so fall back to a directory inside the throwaway checkout.
if [ -z "${HOME:-}" ] || [ ! -w "$HOME" ]; then
  export HOME="$PWD/.fix-home"
  mkdir -p "$HOME"
fi
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export YOLO_OFFLINE=1

venv="$HOME/.fix-venv"
python3 -m venv "$venv"
export PATH="$venv/bin:$PATH"
export PIP_CONSTRAINT="$PWD/docker/constraints.txt"

constraint() { sed -n "s/^$1==\([^[:space:]#]*\).*/\1/p" docker/constraints.txt; }
torch_v=$(constraint torch)
vision_v=$(constraint torchvision)
[ -n "$torch_v" ] && [ -n "$vision_v" ] || { echo "test.sh: docker/constraints.txt pins no torch or torchvision" >&2; exit 1; }

echo "== $(python3 --version)"
echo "== torch $torch_v, torchvision $vision_v (CPU)"
# The wheels alone from PyTorch's index; their own dependencies from PyPI (the
# installed CPU wheels already satisfy the second line's pins).
python3 -m pip install --quiet --no-deps --index-url https://download.pytorch.org/whl/cpu \
  "torch==$torch_v" "torchvision==$vision_v"
python3 -m pip install --quiet "torch==$torch_v" "torchvision==$vision_v"
echo "== pip install -r requirements.txt"
python3 -m pip install --quiet -r requirements.txt
echo "== docker/Dockerfile's extra pins"
python3 -m pip install --quiet "PyJWT[crypto]>=2.8"
python3 -m pip install --quiet "opencv-python<5"
python3 -m pip install --quiet "fast-plate-ocr==1.1.0" "onnxruntime==1.23.2"

cv_v=$(python3 -m pip show opencv-python | sed -n 's/^Version: //p')
echo "== opencv-python $cv_v -> opencv-python-headless $cv_v"
python3 -m pip uninstall --quiet --yes opencv-python
# Forced: another dependency may have installed the headless wheel already,
# and both wheels share the cv2 directory the uninstall just removed.
python3 -m pip install --quiet --force-reinstall --no-deps "opencv-python-headless==$cv_v"
python3 -c "import cv2, numpy, torch, ultralytics, fastapi" # a broken install fails here, not as skips

failed=()
for t in tests/test_*.py; do
  echo "== python -u $t"
  python3 -u "$t" || failed+=("$t")
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "test.sh: failed: ${failed[*]}" >&2
  exit 1
fi
echo "== every test file passed"
