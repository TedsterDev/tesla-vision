#!/bin/bash
echo "g1 UDC:  $(sudo cat /sys/kernel/config/usb_gadget/g1/UDC 2>/dev/null)"
echo "l4t UDC: $(sudo cat /sys/kernel/config/usb_gadget/l4t/UDC 2>/dev/null)"
echo -n "g1 backing: "
sudo cat /sys/kernel/config/usb_gadget/g1/functions/mass_storage.0/lun.0/file 2>/dev/null || echo "(none)"
echo -n "TESLACAM label -> "
ls -l /dev/disk/by-label 2>/dev/null | grep -i TESLACAM || echo "(missing)"
