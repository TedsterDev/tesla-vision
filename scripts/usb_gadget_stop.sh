#!/bin/bash
set -e

G=/sys/kernel/config/usb_gadget/g1

if [ ! -d "$G" ]; then
  echo "No gadget g1 exists."
  exit 0
fi

# 1) Unbind from UDC (detach from host)
echo "" | sudo tee $G/UDC >/dev/null || true
sleep 1

# 2) Remove function links from the config
if [ -L "$G/configs/c.1/mass_storage.0" ]; then
  sudo rm -f $G/configs/c.1/mass_storage.0
fi

# 3) Remove function directory
if [ -d "$G/functions/mass_storage.0" ]; then
  sudo rmdir $G/functions/mass_storage.0 2>/dev/null || true
fi

# 4) Remove config strings and config dir
sudo rmdir $G/configs/c.1/strings/0x409 2>/dev/null || true
sudo rmdir $G/configs/c.1 2>/dev/null || true

# 5) Remove device strings
sudo rmdir $G/strings/0x409 2>/dev/null || true

# 6) Finally remove the gadget itself
sudo rmdir $G 2>/dev/null || true

echo "Gadget stopped and removed."

