#!/usr/bin/env bash
set -euo pipefail

CONF="${CONF:-$HOME/projects/tesla-alerts/config/wifi_priority.conf}"
IFACE="${IFACE:-wlan0}"

log() { echo "[wifi_autoconnect] $*"; }

# Ensure Wi-Fi radio is on
nmcli radio wifi on >/dev/null 2>&1 || true

# If interface doesn't exist, exit quietly
if ! nmcli -t -f DEVICE device status | cut -d: -f1 | grep -qx "$IFACE"; then
  log "Interface $IFACE not found. (Set IFACE=... if needed)"
  exit 0
fi

# If already connected to wifi, do nothing
STATE="$(nmcli -t -f DEVICE,TYPE,STATE device status | awk -F: -v d="$IFACE" '$1==d {print $3}')"
if [[ "$STATE" == "connected" ]]; then
  exit 0
fi

# Get visible SSIDs (dedup, remove empty)
mapfile -t SSIDS < <(nmcli -t -f SSID dev wifi list ifname "$IFACE" | sed '/^$/d' | awk '!seen[$0]++')

if (( ${#SSIDS[@]} == 0 )); then
  log "No Wi-Fi networks visible right now."
  exit 0
fi

# Helper: connect to SSID (creates/updates a connection)
connect_ssid() {
  local ssid="$1"
  local pass="$2"

  # Connection name we will manage
  local con="wifi:${ssid}"

  # If connection exists, just bring it up
  if nmcli -t -f NAME connection show | grep -qx "$con"; then
    log "Bringing up saved connection: $con"
    nmcli connection up "$con" >/dev/null
    return 0
  fi

  # Otherwise create it
  log "Creating and connecting: $ssid"
  nmcli dev wifi connect "$ssid" password "$pass" ifname "$IFACE" name "$con" >/dev/null

  # Make it autoconnect-capable if later seen again
  nmcli connection modify "$con" connection.autoconnect yes >/dev/null
  # Give it a default priority; our script is the real priority engine
  nmcli connection modify "$con" connection.autoconnect-priority 10 >/dev/null
}

# Read config in priority order
while IFS='|' read -r kind key pass; do
  # skip comments/blank
  [[ -z "${kind:-}" ]] && continue
  [[ "${kind:0:1}" == "#" ]] && continue

  if [[ "$kind" == "EXACT" ]]; then
    for s in "${SSIDS[@]}"; do
      if [[ "$s" == "$key" ]]; then
        connect_ssid "$s" "$pass"
        exit 0
      fi
    done
  elif [[ "$kind" == "REGEX" ]]; then
    for s in "${SSIDS[@]}"; do
      if echo "$s" | grep -Eq "$key"; then
        connect_ssid "$s" "$pass"
        exit 0
      fi
    done
  fi
done < "$CONF"

log "No configured preferred networks matched."
exit 0
