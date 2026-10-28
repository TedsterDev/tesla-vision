#!/bin/bash
# expose_tesla_tunnel.sh - publish the whole tesla-alerts dashboard at
# https://azula.tedebyte.dev/tesla/ through the existing cloudflared tunnel.
#
# Run it as YOURSELF, not with sudo in front: it shows the change, asks, and
# only then calls sudo for the two steps that need root (installing the config
# and restarting cloudflared).
#
#     ./scripts/expose_tesla_tunnel.sh                 # /tesla/*, minus /findmy
#     ./scripts/expose_tesla_tunnel.sh --with-findmy   # ...and the live position
#     ./scripts/expose_tesla_tunnel.sh --remove        # take it back out
#
# What it adds, inserted BEFORE the catch-all that sends azula.tedebyte.dev to
# :7878 (cloudflared matches in file order; appended last it is unreachable):
#
#     - hostname: azula.tedebyte.dev
#       path: '^/tesla/(api/)?findmy(/|$)'
#       service: http_status:404
#     - hostname: azula.tedebyte.dev
#       path: '^/tesla(/|$)'
#       service: http://127.0.0.1:8080
#
# The app needs URL_ROOT=/tesla (set in docker-compose.yaml) so its links carry
# the prefix; it strips the prefix itself before routing.
#
# (/|$) is load-bearing: RE2 anchors nothing at the end, so a bare ^/tesla
# would also claim /teslafoo from the spv-server.
#
# /findmy is the car's LIVE position, and expose_natix_tunnel.sh kept it off
# the public hostname on purpose; the first rule keeps it that way unless you
# pass --with-findmy. It is a convenience, not the security boundary: the app
# still gates the location routes on its own (see LOCATION_ROUTES in
# src/ui_app.py). On the tailnet it stays at http://100.98.227.42:8080/findmy
#
# The existing /natix and /api/natix rule is left alone.
set -euo pipefail

CONFIG=/etc/cloudflared/config.yml
BACKUP_DIR=/etc/cloudflared/backups
MARKER="# tesla-alerts: full dashboard under /tesla"
HOSTNAME_FQDN=azula.tedebyte.dev

die() { echo "expose_tesla_tunnel: $*" >&2; exit 1; }

MODE=apply
WITH_FINDMY=0
for argument in "$@"; do
  case "$argument" in
    --remove)       MODE=remove ;;
    --with-findmy)  WITH_FINDMY=1 ;;
    *) die "unknown argument: $argument (expected --remove or --with-findmy)" ;;
  esac
done

[[ $EUID -ne 0 ]] || die "run this as yourself, without sudo; it asks for sudo when it needs it"
[[ -r "$CONFIG" ]] || die "$CONFIG not found or not readable"
command -v cloudflared >/dev/null || die "cloudflared is not installed"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
NEW="$WORK/config.yml"

# Writes $NEW. Exit 0 = changed, 3 = nothing to do, 4 = unexpected shape.
set +e
python3 - "$CONFIG" "$NEW" "$MODE" "$MARKER" "$WITH_FINDMY" "$HOSTNAME_FQDN" <<'PYEOF'
import sys, pathlib

config_path, new_path, mode, marker, with_findmy, host = sys.argv[1:7]
lines = pathlib.Path(config_path).read_text().splitlines()

rule = [f"  {marker}"]
if with_findmy != "1":
    rule += [
        f"  - hostname: {host}",
        "    path: '^/tesla/(api/)?findmy(/|$)'",
        "    service: http_status:404",
    ]
rule += [
    f"  - hostname: {host}",
    "    path: '^/tesla(/|$)'",
    "    service: http://127.0.0.1:8080",
]


def block_extent(start):
    """Our block runs from the marker to the first line that is neither one of
    our rules' lines nor another '- hostname:' entry that mentions /tesla."""
    end = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if stripped.startswith("- hostname:"):
            # Ours only if its path line names /tesla.
            following = lines[end + 1].strip() if end + 1 < len(lines) else ""
            if "^/tesla" not in following:
                break
        elif not (stripped.startswith("path:") or stripped.startswith("service:")):
            break
        end += 1
    return end


existing = next((i for i, line in enumerate(lines) if marker in line), None)

if mode == "remove":
    if existing is None:
        print("the /tesla rule is not present; nothing to remove")
        sys.exit(3)
    del lines[existing:block_extent(existing)]
elif existing is not None:
    end = block_extent(existing)
    if lines[existing:end] == rule:
        print("the /tesla rule is already present and up to date; nothing to do")
        sys.exit(3)
    lines[existing:end] = rule
else:
    try:
        service_index = next(
            i for i, line in enumerate(lines)
            if "127.0.0.1:7878" in line and line.strip().startswith("service:")
        )
    except StopIteration:
        print("could not find the :7878 catch-all rule to insert before", file=sys.stderr)
        sys.exit(4)
    insert_at = service_index
    while insert_at > 0 and not lines[insert_at].strip().startswith("- hostname:"):
        insert_at -= 1
    lines[insert_at:insert_at] = rule

pathlib.Path(new_path).write_text("\n".join(lines) + "\n")
PYEOF
status=$?
set -e
case $status in
  0) ;;
  3) exit 0 ;;
  4) die "$CONFIG is not the shape this script parses; edit it by hand" ;;
  *) die "config edit failed (python exited $status)" ;;
esac

echo "=== proposed change to $CONFIG ==="
diff -u "$CONFIG" "$NEW" || true
echo

echo "=== validating ==="
cloudflared --config "$NEW" tunnel ingress validate \
  || die "the new config does not validate; nothing was changed"
echo

read -r -p "Install this and restart cloudflared? [y/N] " answer
[[ "$answer" == [yY]* ]] || { echo "Nothing changed."; exit 0; }

sudo -v
sudo mkdir -p "$BACKUP_DIR"
backup="$BACKUP_DIR/config.yml.$(date +%Y%m%d-%H%M%S)"
sudo cp -a "$CONFIG" "$backup"
sudo install -m "$(stat -c %a "$CONFIG")" -o "$(stat -c %u "$CONFIG")" -g "$(stat -c %g "$CONFIG")" \
  "$NEW" "$CONFIG"
echo "backed up the old config to $backup"

echo
echo "=== restarting cloudflared ==="
sudo systemctl restart cloudflared
sleep 4
systemctl is-active cloudflared || die "cloudflared is not running; restore with: sudo cp $backup $CONFIG"

echo
echo "=== checking the routes ==="
# Asked of cloudflared's own matcher, not curl: Access answers 302 for every
# path on the hostname, routed or not, so curl cannot tell them apart.
fails=0
expect() {
  local request_path="$1" want="$2" answer
  answer="$(cloudflared --config "$CONFIG" tunnel ingress rule "https://$HOSTNAME_FQDN$request_path" 2>&1 || true)"
  if grep -q -- "$want" <<<"$answer"; then
    echo "  ok    $request_path -> $want"
  else
    echo "  FAIL  $request_path should go to $want, got:"
    sed 's/^/          /' <<<"$answer"
    fails=$((fails + 1))
  fi
}

if [[ "$MODE" == remove ]]; then
  expect /tesla/ "127.0.0.1:7878"
else
  for request_path in /tesla /tesla/ /tesla/findings /tesla/timeline /tesla/api/map /tesla/media/x.jpg; do
    expect "$request_path" "127.0.0.1:8080"
  done
  if [[ $WITH_FINDMY -eq 1 ]]; then
    expect /tesla/findmy "127.0.0.1:8080"
  else
    expect /tesla/findmy "http_status:404"
    expect /tesla/api/findmy "http_status:404"
  fi
  # Neighbours that must not move.
  expect /teslafoo "127.0.0.1:7878"
  expect / "127.0.0.1:9100"
  expect /natix "127.0.0.1:8080"
fi

[[ $fails -eq 0 ]] || die "$fails route check(s) failed; the old config is at $backup"

if [[ "$MODE" == remove ]]; then
  echo
  echo "Removed. The dashboard is still on the tailnet at http://100.98.227.42:8080/"
else
  echo
  echo "Done. Open https://$HOSTNAME_FQDN/tesla/"
  echo "Cloudflare Access asks first, then the dashboard's own login."
fi
