#!/bin/bash
set -e

UDC=3550000.usb
G=/sys/kernel/config/usb_gadget/g1
L4T=/sys/kernel/config/usb_gadget/l4t

BACKING=/dev/sdb1
MOUNTPOINT=/mnt/teslacam

sudo modprobe libcomposite

# Refuse if mounted (prevents corruption)
if findmnt -n "$MOUNTPOINT" >/dev/null 2>&1; then
  echo "ERROR: $MOUNTPOINT is mounted. Unmount first:"
  echo "  sudo umount $MOUNTPOINT"
  exit 1
fi

# If NVIDIA's l4t gadget is bound, unbind it so we can own the UDC
if [ -f "$L4T/UDC" ]; then
  L4T_BOUND=$(sudo cat "$L4T/UDC" || true)
  if [ -n "$L4T_BOUND" ]; then
    echo "Unbinding l4t gadget from $L4T_BOUND"
    echo "" | sudo tee "$L4T/UDC" >/dev/null || true
    sleep 1
  fi
fi

# Stop existing g1 cleanly (your stop script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo "$SCRIPT_DIR/usb_gadget_stop.sh" || true

# Create gadget g1
sudo mkdir -p "$G"
cd "$G"

echo 0x1d6b | sudo tee idVendor  >/dev/null   # Linux Foundation
echo 0x0104 | sudo tee idProduct >/dev/null   # Mass Storage
echo 0x0100 | sudo tee bcdDevice >/dev/null
echo 0x0200 | sudo tee bcdUSB    >/dev/null

sudo mkdir -p strings/0x409
echo "1234567890" | sudo tee strings/0x409/serialnumber   >/dev/null
echo "Tedster"     | sudo tee strings/0x409/manufacturer   >/dev/null
echo "TeslaCam"    | sudo tee strings/0x409/product        >/dev/null

sudo mkdir -p configs/c.1/strings/0x409
echo "Config 1" | sudo tee configs/c.1/strings/0x409/configuration >/dev/null

sudo mkdir -p functions/mass_storage.0
echo 1        | sudo tee functions/mass_storage.0/stall >/dev/null
echo 0        | sudo tee functions/mass_storage.0/lun.0/removable >/dev/null
echo "$BACKING" | sudo tee functions/mass_storage.0/lun.0/file >/dev/null

sudo ln -s functions/mass_storage.0 configs/c.1/mass_storage.0

# Bind to UDC (this is the "make Tesla see it" moment)
echo "$UDC" | sudo tee UDC >/dev/null

echo "g1 bound to $UDC using backing $BACKING"

# Optional: rebind l4t gadget if present
if [ -f /sys/kernel/config/usb_gadget/l4t/UDC ]; then
  echo "3550000.usb" | sudo tee /sys/kernel/config/usb_gadget/l4t/UDC >/dev/null || true
fi
