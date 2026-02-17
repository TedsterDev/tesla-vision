#!/bin/bash
set -e

UDC=3550000.usb
G1=/sys/kernel/config/usb_gadget/g1
L4T=/sys/kernel/config/usb_gadget/l4t

echo "Switching to DEV mode..."

# Unbind g1 if active
if [ -f "$G1/UDC" ]; then
    echo "" | sudo tee "$G1/UDC" >/dev/null 2>&1 || true
fi

# Bind l4t gadget (serial + network)
CURRENT_L4T="$(sudo cat "$L4T/UDC" 2>/dev/null || true)"
if [ "$CURRENT_L4T" = "$UDC" ]; then
  echo "l4t already bound to $UDC"
else
  echo "$UDC" | sudo tee "$L4T/UDC" >/dev/null
  echo "l4t bound to $UDC"
fi

echo "Dev mode active (serial available)."

