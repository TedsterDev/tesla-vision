#!/bin/bash
# setup_ntfy.sh - bring up the self-hosted ntfy server and wire the
# processor to it. Safe to re-run: it resets the two accounts and the token.
#
#     ./scripts/setup_ntfy.sh
#
# Creates, on a server that is deny-all by default:
#   processor  write-only on the topic, via a token in .env (NTFY_TOKEN)
#   phone      read-only on the topic, for the ntfy app
# The phone's password goes to ~/.config/tesla-alerts/ntfy-phone.txt (mode
# 600) rather than to the terminal, so it does not end up in scrollback.
set -euo pipefail
cd "$(dirname "$0")/.."

TOPIC=tesla-alerts
CREDENTIALS="$HOME/.config/tesla-alerts/ntfy-phone.txt"
NTFY="docker compose exec -T ntfy ntfy"

docker compose up -d ntfy >/dev/null
for _ in $(seq 1 20); do
  curl -sf -o /dev/null http://127.0.0.1:8090/v1/health && break
  sleep 1
done
curl -sf http://127.0.0.1:8090/v1/health >/dev/null || { echo "ntfy did not come up" >&2; exit 1; }

random() { python3 -c "import secrets; print(secrets.token_urlsafe(24))"; }

upsert_user() {   # user role password
  if $NTFY user list 2>&1 | grep -q "^user $1 "; then
    docker compose exec -T -e NTFY_PASSWORD="$3" ntfy ntfy user change-pass "$1" >/dev/null
  else
    docker compose exec -T -e NTFY_PASSWORD="$3" ntfy ntfy user add --role="$2" "$1" >/dev/null
  fi
}

PHONE_PASSWORD=$(random)
upsert_user phone user "$PHONE_PASSWORD"
upsert_user processor user "$(random)"
$NTFY access phone "$TOPIC" read-only >/dev/null
$NTFY access processor "$TOPIC" write-only >/dev/null
for old in $($NTFY token list processor 2>&1 | grep -o 'tk_[A-Za-z0-9]*'); do
  $NTFY token remove processor "$old" >/dev/null
done
TOKEN=$($NTFY token add processor 2>&1 | grep -o 'tk_[A-Za-z0-9]*' | head -1)
[[ -n "$TOKEN" ]] || { echo "could not create a token" >&2; exit 1; }

# The processor reaches ntfy by its compose service name.
python3 - "$TOKEN" "$TOPIC" <<'PY'
import pathlib, sys
token, topic = sys.argv[1], sys.argv[2]
env = pathlib.Path(".env")
wanted = {"NTFY_SERVER": "http://ntfy", "NTFY_TOPIC": topic, "NTFY_TOKEN": token}
lines = env.read_text().splitlines() if env.exists() else []
kept = [line for line in lines if line.split("=", 1)[0] not in wanted]
kept += [f"{key}={value}" for key, value in wanted.items()]
env.write_text("\n".join(kept) + "\n")
PY

mkdir -p "$(dirname "$CREDENTIALS")"
umask 077
cat > "$CREDENTIALS" <<CREDS
ntfy app setup (Tesla Vision alerts)
  Server:   http://100.98.227.42:8090   (over Tailscale)
  Topic:    $TOPIC
  Username: phone
  Password: $PHONE_PASSWORD
In the app: Settings > Users > add the server, username and password;
then Subscribe to topic > use another server > the server above.
CREDS
chmod 600 "$CREDENTIALS"

docker compose up -d processor >/dev/null
echo "ntfy is up at http://100.98.227.42:8090 (deny-all; topic '$TOPIC')."
echo "Processor publishes with a write-only token (saved in .env) and was restarted."
echo "Phone login saved to $CREDENTIALS"
