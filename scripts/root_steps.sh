#!/bin/bash
# root_steps.sh - run the pending privileged work without pasting sudo lines.
#
#     ./scripts/root_steps.sh
#
# Run it as yourself, not with sudo: it asks for the password once, shows the
# dry-run plan from apply_audit_fixes.sh, and applies it only if you say yes.
# That script is idempotent, so running this again later is harmless.

# `sh root_steps.sh` runs dash, which has no pipefail; hop to bash instead.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

[[ $EUID -ne 0 ]] || { echo "run this as yourself, not with sudo - it asks for the password" >&2; exit 1; }

echo "Privileged steps for tesla-alerts need your sudo password."
sudo -v

echo
sudo ./scripts/apply_audit_fixes.sh

echo
read -r -p "Apply the plan above? [y/N] " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
  sudo ./scripts/apply_audit_fixes.sh --execute
else
  echo "Nothing applied."
fi
