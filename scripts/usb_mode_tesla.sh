#!/bin/bash
set -e

UDC=3550000.usb
G1=/sys/kernel/config/usb_gadget/g1
L4T=/sys/kernel/config/usb_gadget/l4t
BACKING=/dev/disk/by-label/TESLACAM

sudo modprobe libcomposite

echo "Switching to TESLA mode..."

# Unbind l4t if active
if [ -f "$L4T/UDC" ]; then
    echo "" | sudo tee "$L4T/UDC" >/dev/null 2>&1 || true
fi

sleep 1

# Create g1 if missing
if [ ! -d "$G1" ]; then
    sudo mkdir -p "$G1"
    cd "$G1"

    echo 0x1d6b | sudo tee idVendor >/dev/null
    echo 0x0104 | sudo tee idProduct >/dev/null
    echo 0x0100 | sudo tee bcdDevice >/dev/null
    echo 0x0200 | sudo tee bcdUSB >/dev/null

    sudo mkdir -p strings/0x409
    echo "1234567890" | sudo tee strings/0x409/serialnumber >/dev/null
    echo "Tedster" | sudo tee strings/0x409/manufacturer >/dev/null
    echo "TeslaCam" | sudo tee strings/0x409/product >/dev/null

    sudo mkdir -p configs/c.1/strings/0x409
    echo "Config 1" | sudo tee configs/c.1/strings/0x409/configuration >/dev/null

    sudo mkdir -p functions/mass_storage.0
    echo 1 | sudo tee functions/mass_storage.0/stall >/dev/null
    echo 0 | sudo tee functions/mass_storage.0/lun.0/removable >/dev/null
    echo "$BACKING" | sudo tee functions/mass_storage.0/lun.0/file >/dev/null

    sudo ln -s functions/mass_storage.0 configs/c.1/mass_storage.0
fi

# Ensure function is linked into config (idempotent)
if [ ! -e "$G1/configs/c.1/mass_storage.0" ]; then
  sudo ln -s "$G1/functions/mass_storage.0" "$G1/configs/c.1/mass_storage.0"
fi

# Always ensure backing file is correct (device names can change)
echo "" | sudo tee "$G1/UDC" >/dev/null 2>&1 || true
echo "$BACKING" | sudo tee "$G1/functions/mass_storage.0/lun.0/file" >/dev/null

# Bind g1
echo "$UDC" | sudo tee "$G1/UDC" >/dev/null
echo "g1 bound to $UDC"


echo "Tesla mode active."

