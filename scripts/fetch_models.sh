#!/bin/bash
#
# fetch_models.sh - download the optional model weights the Scout pipeline uses.
#
# Everything here is optional. The pipeline degrades gracefully:
#   no YuNet/SFace  -> face stage disabled (or detection-only)
#   no plate weights -> ALPR falls back to OpenCV's Haar plate cascade
#
# Run once, with network access:
#   ./scripts/fetch_models.sh
#
# Weights land in $MODELS_DIR (default: the data dir the containers mount at
# /data/models), so they survive image rebuilds.

set -u

MODELS_DIR="${MODELS_DIR:-/mnt/jetsondata/tesla-alerts/models}"
mkdir -p "$MODELS_DIR"

echo "Fetching models into $MODELS_DIR"

download() {
  local description="$1"; shift
  local destination="$1"; shift
  # Remaining arguments are candidate URLs, tried in order.

  if [ -s "$destination" ]; then
    echo "  [skip] $description already present ($(du -h "$destination" | cut -f1))"
    return 0
  fi

  for url in "$@"; do
    echo "  [get ] $description"
    echo "         <- $url"
    if curl -fsSL --retry 2 --connect-timeout 15 -o "$destination.part" "$url"; then
      # A GitHub 404 page is a valid HTTP 200 body sometimes; check for size.
      if [ "$(stat -c%s "$destination.part" 2>/dev/null || echo 0)" -gt 100000 ]; then
        mv "$destination.part" "$destination"
        echo "         ok ($(du -h "$destination" | cut -f1))"
        return 0
      fi
    fi
    rm -f "$destination.part"
    echo "         failed, trying next source"
  done

  echo "  [WARN] could not fetch $description - that stage will run degraded"
  return 1
}

# --- General-purpose detector: YOLO26n (~5MB) -------------------------------
# The person/vehicle detector the pipeline runs on every sampled frame.
# Ultralytics will happily download this at first use, but into the container's
# WRITABLE LAYER, which is discarded on every rebuild or --force-recreate. The
# deployment target is a car with no internet, so staging it here (on the data
# volume) is what stops a recreate in the driveway from killing the pipeline.
download "YOLO26n COCO detector" \
  "$MODELS_DIR/yolo26n.pt" \
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt" \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

# --- Face detection: YuNet (~340KB) ----------------------------------------
download "YuNet face detector" \
  "$MODELS_DIR/face_detection_yunet_2023mar.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# --- Face embedding: SFace (~37MB), our FaceNet replacement ----------------
download "SFace face embedder" \
  "$MODELS_DIR/face_recognition_sface_2021dec.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# --- License plate detector (optional) -------------------------------------
# Scout trained YOLOv3 on Google OpenImages plates and hosted the weights on
# Google Drive (see its README step 24). That link is long dead, so we try a
# couple of maintained YOLO plate models instead. If none resolve, alpr.py
# uses OpenCV's built-in Haar plate cascade, which needs no download.
download "YOLO license plate detector" \
  "$MODELS_DIR/plate_detector.pt" \
  "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt" \
  "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"

echo
echo "Done. Present in $MODELS_DIR:"
ls -lh "$MODELS_DIR" 2>/dev/null | tail -n +2 | awk '{printf "  %-52s %s\n", $9, $5}'
