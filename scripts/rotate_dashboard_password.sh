#!/bin/bash
# rotate_dashboard_password.sh - replace DASHBOARD_PASS with a long random
# password and restart the dashboard. Run it as yourself:
#
#     ./scripts/rotate_dashboard_password.sh
#
# The new password is written to ~/.config/tesla-alerts/dashboard-login.txt
# (mode 600), not to the terminal. The browser will ask for it once, after
# Cloudflare Access; save it in your password manager.
set -euo pipefail
cd "$(dirname "$0")/.."
CREDENTIALS="$HOME/.config/tesla-alerts/dashboard-login.txt"

[[ -f .env ]] || { echo "no .env here" >&2; exit 1; }
read -r -p "Replace the dashboard password with a new random one? [y/N] " answer
[[ "$answer" == [yY]* ]] || { echo "Nothing changed."; exit 0; }

python3 - "$CREDENTIALS" <<'PY'
import os, pathlib, secrets, sys
password = secrets.token_urlsafe(24)
env = pathlib.Path(".env")
lines = env.read_text().splitlines()
user = next((l.split("=", 1)[1] for l in lines if l.startswith("DASHBOARD_USER=")), "") or "tedster"
lines = [l for l in lines if not l.startswith(("DASHBOARD_PASS=", "DASHBOARD_USER="))]
lines += [f"DASHBOARD_USER={user}", f"DASHBOARD_PASS={password}"]
env.write_text("\n".join(lines) + "\n")
creds = pathlib.Path(sys.argv[1])
creds.parent.mkdir(parents=True, exist_ok=True)
os.umask(0o077)
creds.write_text(f"Tesla Vision dashboard login (asked for after Cloudflare Access)\n"
                 f"  Username: {user}\n  Password: {password}\n")
creds.chmod(0o600)
PY

docker compose up -d web >/dev/null
sleep 5
user=$(grep '^DASHBOARD_USER=' .env | cut -d= -f2-)
pass=$(grep '^DASHBOARD_PASS=' .env | cut -d= -f2-)
with=$(curl -s -o /dev/null -w '%{http_code}' -u "$user:$pass" http://127.0.0.1:8080/tesla/plates)
without=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/tesla/plates)
unset pass
echo "with the new password: HTTP $with (want 200) · without: HTTP $without (want 401)"
[[ "$with" == 200 && "$without" == 401 ]] || { echo "check failed - see 'docker compose logs web'" >&2; exit 1; }
echo "Done. Login saved to $CREDENTIALS"
