#!/bin/bash
# expose_natix_tunnel.sh - publish the tesla-alerts diagnostic pages at
# https://azula.tedebyte.dev through the existing cloudflared tunnel.
#
# What it adds
# ------------
# One ingress rule, inserted BEFORE the catch-all that sends azula.tedebyte.dev
# to the spv-server on :7878:
#
#     - hostname: azula.tedebyte.dev
#       path: '^(/api)?/(natix|gps)(/|$)'
#       service: http://127.0.0.1:8080
#
# cloudflared matches ingress rules in order, so placement matters: appended at
# the end it would never be reached, because the :7878 rule matches every path.
#
# Reading that regex, piece by piece, because every piece is load-bearing:
#
#   ^          Go's RE2 anchors ^ at the start of the path. NOTHING anchors the
#              end, which is the whole reason (/|$) below exists.
#   (/api)?    One optional group covers both the HTML page and its JSON
#              sibling, so six paths stay on one line instead of an alternation
#              per route.
#   (natix|gps)  A closed allowlist. Everything else on this dashboard -
#              /timeline, /plates, /faces, /settings, /media, /api/map,
#              /api/polls - stays off the public tunnel.
#   (/|$)      THE ONE THAT MATTERS. It forces the name to end at a segment
#              boundary. The rule this script used to install was `^/natix`,
#              which is a PREFIX match, not an exact one: /natixfoo routes to
#              this dashboard today. Harmless while it is only a 404 from the
#              wrong backend, but /gps without the terminator is a standing
#              claim over every future spv-server path starting with those
#              three letters - a silent routing hijack the day either app grows
#              one. Do not drop it.
#
# Single-quoted because the value contains |, $ and parens. Unquoted happens to
# parse in this YAML today; quote it anyway.
#
# --with-findmy
# -------------
# Adds `findmy` to that allowlist, which publishes /findmy and /api/findmy: the
# car's LIVE position - where it is right now, continuously pollable - to
# anything that gets past Cloudflare Access. That is a different category of
# data from the dashcam pages; it is a home address, a work schedule, and a
# real-time at-home/not-at-home signal.
#
# You almost certainly do not need the flag. The tailnet already reaches the
# page at http://100.98.227.42:8080/findmy from the phone, so the public route
# buys exactly one thing: reading the car's position from a device that is NOT
# on the tailnet. Before using it: make DASHBOARD_PASS 20+ random characters
# (it is 4 digits, brute-forced in ~31 seconds), set CF_ACCESS_ALLOWED_EMAILS,
# and close the empty-DASHBOARD_PASS fail-open in src/ui_app.py.
#
# Auth: azula.tedebyte.dev already sits behind Cloudflare Access, and the
# dashboard adds HTTP Basic on top (DASHBOARD_USER/DASHBOARD_PASS in .env), so
# this is two independent layers rather than a new hole.
#
# Usage:
#     sudo ./scripts/expose_natix_tunnel.sh              # /natix + /gps, reload
#     sudo ./scripts/expose_natix_tunnel.sh --with-findmy  # ...and /findmy
#     sudo ./scripts/expose_natix_tunnel.sh --dry-run    # show the diff only
#     sudo ./scripts/expose_natix_tunnel.sh --remove     # take it back out
#
# Re-running with a different regex UPDATES the rule in place. The previous
# version of this script refused on the marker rather than on the content, so
# editing the rule and re-running printed "already present; nothing to do" and
# changed nothing - the single most likely way to believe you have exposed a
# path that you have not.
set -euo pipefail

CONFIG=/etc/cloudflared/config.yml
BACKUP_DIR=/etc/cloudflared/backups
MARKER="# tesla-alerts: NATIX mirror status + logs"
HOSTNAME_FQDN=azula.tedebyte.dev

die() { echo "expose_natix_tunnel: $*" >&2; exit 1; }

MODE=apply
WITH_FINDMY=0
for argument in "$@"; do
  case "$argument" in
    --dry-run|--remove) MODE="$argument" ;;
    --with-findmy)      WITH_FINDMY=1 ;;
    apply)              MODE=apply ;;
    *) die "unknown argument: $argument (expected --dry-run, --remove, --with-findmy)" ;;
  esac
done

if [[ $WITH_FINDMY -eq 1 ]]; then
  RULE_PATH='^(/api)?/(natix|gps|findmy)(/|$)'
else
  RULE_PATH='^(/api)?/(natix|gps)(/|$)'
fi

[[ $EUID -eq 0 ]] || die "must run as root (sudo $0 $*)"
[[ -f "$CONFIG" ]] || die "$CONFIG not found"

if [[ $WITH_FINDMY -eq 1 && "$MODE" != "--remove" ]]; then
  echo "NOTE: --with-findmy publishes the car's live position at"
  echo "      https://$HOSTNAME_FQDN/findmy behind Cloudflare Access only."
  echo "      Without it, reach the page on the tailnet: http://100.98.227.42:8080/findmy"
  echo
fi

# The Python below never touches $CONFIG. It writes $CONFIG.new and reports what
# it did; the swap happens here, after validation.
#
# Exit codes are part of the contract with the shell: 0 = wrote .new, go on;
# 3 = nothing to do, stop cleanly; 4 = the config is not the shape we parse.
set +e
python3 - "$CONFIG" "$MODE" "$MARKER" "$RULE_PATH" <<'PYEOF'
import sys, pathlib

config_path, mode, marker, rule_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lines = pathlib.Path(config_path).read_text().splitlines()

RULE = [
    f"  {marker}",
    "  - hostname: azula.tedebyte.dev",
    f"    path: '{rule_path}'",
    "    service: http://127.0.0.1:8080",
]


def entry_extent(lines, hostname_index):
    """
    Index one past the last line of the ingress entry starting at hostname_index.

    Measured, never assumed. The old script deleted a fixed `start + len(RULE)`
    slice, so the day the rule grew or lost a line it silently ate the top of
    whatever rule followed it - and the result still validates, because a
    truncated neighbour is usually still legal YAML.
    """
    end = hostname_index + 1
    while end < len(lines):
        line = lines[end]
        stripped = line.strip()
        if not stripped:
            break                       # blank line ends the entry
        if not line[:1].isspace():
            break                       # a top-level key: new section entirely
        if stripped.startswith("- "):
            break                       # the next list item
        if stripped.startswith("#"):
            break                       # a comment introduces what follows it
        end += 1
    return end


def find_existing_block(lines):
    """
    Locate our rule, by marker if it is there and by shape if it is not.

    The shape fallback exists because a hand-edited config.yml (which is how
    the contract says to apply the new regex) has the rule but not the marker.
    Matching on the marker alone would then insert a SECOND azula rule, and
    since cloudflared matches in file order the older, narrower one would win
    while the diff looked perfect.
    """
    for index, line in enumerate(lines):
        if marker in line:
            start = index
            if index + 1 < len(lines) and lines[index + 1].strip().startswith("- "):
                return start, entry_extent(lines, index + 1), "marker"
            return start, index + 1, "marker"

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("path:"):
            continue
        if not any(name in stripped for name in ("natix", "gps", "findmy")):
            continue
        hostname_index = index
        while hostname_index > 0 and not lines[hostname_index].strip().startswith("- hostname:"):
            hostname_index -= 1
        end = entry_extent(lines, hostname_index)
        if any("127.0.0.1:8080" in lines[i] for i in range(hostname_index, end)):
            return hostname_index, end, "unmarked"
    return None


existing = find_existing_block(lines)

if mode == "--remove":
    if not existing:
        print("rule is not present; nothing to remove")
        sys.exit(3)
    start, end, how = existing
    if how == "unmarked":
        print(f"removing an unmarked rule at line {start + 1} (no {marker!r} above it)")
    del lines[start:end]
    pathlib.Path(config_path + ".new").write_text("\n".join(lines) + "\n")
    print(f"rule removed ({end - start} lines from line {start + 1})")
    sys.exit(0)

if existing:
    start, end, how = existing
    if lines[start:end] == RULE:
        print("rule is already present and already matches; nothing to do")
        sys.exit(3)
    # Content differs, so this is an UPDATE, not a no-op. Replacing in place
    # keeps the rule's position relative to the :7878 catch-all, which is the
    # only thing making it reachable at all.
    if how == "unmarked":
        print(f"adopting an unmarked rule at line {start + 1} (no {marker!r} above it)")
    print(f"rule differs; replacing {end - start} lines in place at line {start + 1}")
    for line in lines[start:end]:
        print(f"    was: {line}")
    lines[start:end] = RULE
    pathlib.Path(config_path + ".new").write_text("\n".join(lines) + "\n")
    sys.exit(0)

# Insert before the rule that catches every remaining path for this hostname.
# Find the ':7878' service line, then walk back to its own '- hostname:' line.
try:
    service_index = next(
        i for i, line in enumerate(lines)
        if "127.0.0.1:7878" in line and line.strip().startswith("service:")
    )
except StopIteration:
    print("ERROR: could not find the :7878 catch-all rule to insert before.",
          file=sys.stderr)
    print("The config does not have the shape this script expects; add the rule",
          file=sys.stderr)
    print("by hand instead of letting a script guess.", file=sys.stderr)
    sys.exit(4)

insert_at = service_index
while insert_at > 0 and not lines[insert_at].strip().startswith("- hostname:"):
    insert_at -= 1

lines[insert_at:insert_at] = RULE
pathlib.Path(config_path + ".new").write_text("\n".join(lines) + "\n")
print(f"rule will be inserted at line {insert_at + 1}")
PYEOF
status=$?
set -e

# Written as a case, not as `[[ $status -eq 3 ]] && exit 0`. Under `set -e` that
# one-liner killed the script whenever the test was FALSE, so the "already
# present" and "removed" paths exited non-zero and everything after them was
# dead code that had never once run.
case $status in
  0) ;;
  3) exit 0 ;;
  4) die "the config is not the shape this script parses; edit $CONFIG by hand" ;;
  *) die "config edit failed (python exited $status)" ;;
esac

echo
echo "=== diff ==="
diff -u "$CONFIG" "$CONFIG.new" || true

if [[ "$MODE" == "--dry-run" ]]; then
  rm -f "$CONFIG.new"
  echo
  echo "Dry run - nothing changed."
  exit 0
fi

# Validate BEFORE replacing the live config. A tunnel that will not start takes
# down every other hostname on it, not just this one.
echo
echo "=== validating ==="
if ! cloudflared --config "$CONFIG.new" tunnel ingress validate; then
  rm -f "$CONFIG.new"
  die "the new config does not validate; the live config was NOT touched"
fi

mkdir -p "$BACKUP_DIR"
backup="$BACKUP_DIR/config.yml.$(date +%Y%m%d-%H%M%S)"
cp -a "$CONFIG" "$backup"
mv "$CONFIG.new" "$CONFIG"
echo "backed up to $backup"

echo
echo "=== reloading cloudflared ==="
systemctl restart cloudflared
sleep 4
systemctl --no-pager --lines=5 status cloudflared || true

echo
echo "=== checking the routes ==="
# NOT with curl. Cloudflare Access is scoped to the HOSTNAME, so every path on
# azula.tedebyte.dev - including ones no rule has ever routed - answers 302 to
# the Access login. The old check printed "302 is correct" for paths that did
# not exist, which is a green tick for the exact failure it was there to catch.
# cloudflared's own matcher answers the real question: which service wins.
route_for() {
  cloudflared --config "$CONFIG" tunnel ingress rule "https://$HOSTNAME_FQDN$1" 2>&1
}

expect_route() {
  local request_path="$1" want="$2" answer
  answer="$(route_for "$request_path" || true)"
  if [[ "$want" == "dashboard" ]]; then
    if grep -q '127\.0\.0\.1:8080' <<<"$answer"; then
      echo "  ok    $request_path -> 127.0.0.1:8080"
    else
      echo "  FAIL  $request_path is NOT routed to the dashboard:"
      sed 's/^/          /' <<<"$answer"
    fi
  else
    if grep -q '127\.0\.0\.1:8080' <<<"$answer"; then
      echo "  FAIL  $request_path reaches the dashboard and must not:"
      sed 's/^/          /' <<<"$answer"
    else
      echo "  ok    $request_path stays off the dashboard"
    fi
  fi
}

if [[ "$MODE" == "--remove" ]]; then
  for request_path in /natix /api/natix /gps /api/gps /findmy /api/findmy; do
    expect_route "$request_path" "elsewhere"
  done
else
  for request_path in /natix /api/natix /gps /api/gps; do
    expect_route "$request_path" "dashboard"
  done
  if [[ $WITH_FINDMY -eq 1 ]]; then
    expect_route /findmy dashboard
    expect_route /api/findmy dashboard
  else
    expect_route /findmy elsewhere
  fi
  # The segment terminator, checked rather than trusted. If (/|$) is ever lost
  # from the regex these three start answering 8080 and nothing else notices.
  for request_path in /natixfoo /gpsanything /findmyphone; do
    expect_route "$request_path" "elsewhere"
  done
  # Paths that must stay private no matter what the allowlist grows to.
  for request_path in /timeline /settings /api/polls; do
    expect_route "$request_path" "elsewhere"
  done
fi

if [[ "$MODE" == "--remove" ]]; then
  echo
  echo "Removed. The pages are still on the tailnet at http://100.98.227.42:8080/"
  exit 0
fi

cat <<'SUMMARY'

Done. Open https://azula.tedebyte.dev/natix and https://azula.tedebyte.dev/gps

You will pass two auth layers: Cloudflare Access first, then the dashboard's
HTTP Basic prompt (DASHBOARD_USER / DASHBOARD_PASS from .env).

  /natix   clips mirrored, free space, the attached device and why it was or
           was not accepted, and the tail of the host natix-mirror log
  /gps     receiver health: fix status, satellites, DOP, and whether the
           service is writing at all

/findmy - the car's position - is deliberately NOT published unless you pass
--with-findmy. On the tailnet it is at http://100.98.227.42:8080/findmy

To undo:  sudo ./scripts/expose_natix_tunnel.sh --remove
SUMMARY
