#!/bin/bash
set -e
G=/sys/kernel/config/usb_gadget/g1
UDC=3550000.usb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$G" ]; then
  echo "g1 does not exist. Start it first:"
  echo "  sudo $SCRIPT_DIR/usb_gadget_start.sh"
  exit 1
fi

echo "" | sudo tee $G/UDC >/dev/null || true
sleep 2
echo "$UDC" | sudo tee $G/UDC >/dev/null

echo "g1 re-enumerated on $UDC"

