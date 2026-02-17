sudo tee /usr/local/sbin/teslacam-usb.sh > /dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

GADGET_DIR="/sys/kernel/config/usb_gadget/l4t"
LUN_FILE="$GADGET_DIR/functions/mass_storage.0/lun.0/file"
LUN_RO="$GADGET_DIR/functions/mass_storage.0/lun.0/ro"
UDC_FILE="$GADGET_DIR/UDC"

# Change this if your image location changes
BACKING_IMG="/mnt/teslacam/usb_image/teslacam.img"

# Default UDC on Orin Nano (you confirmed this in your output)
UDC_NAME="3550000.usb"

cmd="${1:-status}"

require_paths() {
  for p in "$GADGET_DIR" "$LUN_FILE" "$LUN_RO" "$UDC_FILE"; do
    if [[ ! -e "$p" ]]; then
      echo "Missing expected path: $p"
      echo "Is configfs mounted and the l4t gadget created?"
      exit 1
    fi
  done
}

status() {
  echo "UDC: $(cat "$UDC_FILE" 2>/dev/null || true)"
  echo -n "Backing file: "; cat "$LUN_FILE" 2>/dev/null || true
  echo -n "Read-only (ro): "; cat "$LUN_RO" 2>/dev/null || true
}

down() {
  require_paths
  # Unbind (disconnect from host)
  echo "" > "$UDC_FILE"
  echo "USB gadget unbound (disconnected)."
}

up() {
  require_paths

  if [[ ! -f "$BACKING_IMG" ]]; then
    echo "Backing image not found: $BACKING_IMG"
    exit 2
  fi

  # 1) Unbind first so nothing is "busy"
  echo "" > "$UDC_FILE"

  # 2) Clear file to force LUN close (prevents 'Device or resource busy')
  echo "" > "$LUN_FILE"

  # 3) Make writable
  echo 0 > "$LUN_RO"

  # 4) Point at our TeslaCam image
  echo "$BACKING_IMG" > "$LUN_FILE"

  # 5) Rebind (reconnect)
  echo "$UDC_NAME" > "$UDC_FILE"

  echo "USB gadget rebound with writable TeslaCam image:"
  status
}

case "$cmd" in
  up) up ;;
  down) down ;;
  status) status ;;
  *)
    echo "Usage: $0 {up|down|status}"
    exit 64
    ;;
esac
EOF

sudo chmod +x /usr/local/sbin/teslacam-usb.sh