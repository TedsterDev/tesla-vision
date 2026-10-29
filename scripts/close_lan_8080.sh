#!/bin/bash
# close_lan_8080.sh - stop the dashboard answering on the local Wi-Fi while
# keeping it on the tailnet and behind the Cloudflare tunnel.
#
#     ./scripts/close_lan_8080.sh          # close it
#     ./scripts/close_lan_8080.sh --undo   # back to listening everywhere
#
# Run as yourself; it asks for sudo only for `tailscale serve`.
#
# How: the web container is bound to 127.0.0.1 (WEB_BIND in .env, read by
# docker-compose.yaml). cloudflared already connects to 127.0.0.1:8080.
# `tailscale serve` then forwards http://100.98.227.42:8080 to it - more
# robust than binding Docker to the tailnet address, which fails whenever
# Docker starts before tailscaled. Only the :8080 entry is touched; the
# existing https :8446 serve for the status page is left alone.
set -euo pipefail
cd "$(dirname "$0")/.."

TAILNET_IP=$(tailscale ip -4 2>/dev/null | head -1)
LAN_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[\d.]+' || true)
die() { echo "close_lan_8080: $*" >&2; exit 1; }
[[ $EUID -ne 0 ]] || die "run as yourself, not with sudo"
[[ -n "$TAILNET_IP" ]] || die "tailscale is not up"

set_bind() {
  python3 - "$1" <<'PY'
import pathlib, sys
env = pathlib.Path(".env")
lines = [l for l in env.read_text().splitlines() if not l.startswith("WEB_BIND=")]
if sys.argv[1]:
    lines.append(f"WEB_BIND={sys.argv[1]}")
env.write_text("\n".join(lines) + "\n")
PY
}
code() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$1" || true; }

if [[ "${1:-}" == "--undo" ]]; then
  sudo tailscale serve --http=8080 off || true
  set_bind ""
  docker compose up -d web >/dev/null
  echo "Back to listening on every interface (LAN included)."
  exit 0
fi

echo "This binds the dashboard to localhost, then forwards http://$TAILNET_IP:8080 to it with tailscale serve."
read -r -p "Proceed? [y/N] " answer
[[ "$answer" == [yY]* ]] || { echo "Nothing changed."; exit 0; }

set_bind 127.0.0.1
docker compose up -d web >/dev/null
sleep 4
sudo tailscale serve --bg --http=8080 http://127.0.0.1:8080
sleep 2

echo
echo "=== checks (401 = reachable, asking for the password; 000 = unreachable) ==="
local_code=$(code http://127.0.0.1:8080/tesla/plates)
tail_code=$(code "http://$TAILNET_IP:8080/tesla/plates")
lan_code=$(code "http://${LAN_IP:-0.0.0.0}:8080/tesla/plates")
echo "  localhost (Cloudflare tunnel path): $local_code   want 401"
echo "  tailnet  $TAILNET_IP:8080:        $tail_code   want 401"
echo "  LAN      ${LAN_IP:-?}:8080:          $lan_code   want 000"
if [[ "$local_code" == 401 && "$tail_code" == 401 && "$lan_code" == 000 ]]; then
  echo "Done: closed to the Wi-Fi, open on the tailnet and through Cloudflare."
else
  echo "Something is off. To restore the previous setup: ./scripts/close_lan_8080.sh --undo" >&2
  exit 1
fi
