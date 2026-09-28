"""
cfaccess.py

Verify Cloudflare Access identity tokens.

Why this exists
---------------
The dashboard sits behind two things at once: Cloudflare Access on
azula.tedebyte.dev, and its own HTTP Basic Auth. Those two cannot be stacked,
because **Cloudflare Access consumes the `Authorization` header**. Observed
directly, in the app's own logs:

    401 no-authorization path=/natix via_cloudflare=True
        access_email=you@example.com access_jwt=True

Access had authenticated the user and forwarded its identity headers, but the
`Authorization` header the browser sent was gone. The origin therefore returned
401, the browser re-prompted, the user re-entered the same correct password,
and round it went. No password could ever have worked.

So for requests arriving through Access we authenticate the way Access intends:
by verifying the signed JWT it puts in `Cf-Access-Jwt-Assertion`. Basic Auth
stays as the path for direct access over the tailnet, where nothing strips the
header.

Why the signature check is not optional
---------------------------------------
It would be far less code to trust the `Cf-Access-Authenticated-User-Email`
header. That would also be a hole: the dashboard's port is published on
0.0.0.0, so anything on the LAN or the tailnet could set that header itself and
walk straight in. A header is a claim; only the signature makes it evidence.

Three things are checked, and all three matter:

  * **signature** - RS256 against the team's published keys, so the token
    really came from Cloudflare
  * **aud** - matches THIS application's audience tag, so a token minted for
    some other app in the same Cloudflare team is not accepted here
  * **exp / nbf** - so a captured token stops working

Fails closed: if `CF_ACCESS_AUD` is unset, or the keys cannot be fetched, no
token is accepted and the request falls through to Basic Auth.
"""
import json
import os
import threading
import time
import urllib.request

from typing import Any

# The Cloudflare Zero Trust team domain, e.g. "tedebyte.cloudflareaccess.com".
CF_ACCESS_TEAM_DOMAIN = os.environ.get("CF_ACCESS_TEAM_DOMAIN", "").strip()

# The Access application's AUD tag. Zero Trust -> Access -> Applications ->
# (your app) -> Overview -> "Application Audience (AUD) Tag".
# Without this we accept nothing: an unpinned audience means any token from any
# app in the team would be honoured here.
CF_ACCESS_AUD = os.environ.get("CF_ACCESS_AUD", "").strip()

# Optional extra restriction. Empty means "anyone Access already let through",
# which is usually right - the Access policy is where identity belongs.
CF_ACCESS_ALLOWED_EMAILS = [
    address.strip().lower()
    for address in os.environ.get("CF_ACCESS_ALLOWED_EMAILS", "").split(",")
    if address.strip()
]

_JWKS_TTL_SECONDS = 3600
_jwks_cache: dict[str, Any] = {"fetched_at": 0.0, "keys": {}}
_jwks_lock = threading.Lock()


def is_configured() -> bool:
    return bool(CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD)


def certs_url() -> str:
    return f"https://{CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs"


def issuer() -> str:
    return f"https://{CF_ACCESS_TEAM_DOMAIN}"


def _load_keys(force: bool = False) -> dict[str, Any]:
    """
    Fetch and cache the team's public keys, keyed by `kid`.

    Cached for an hour, and refetched immediately on an unknown kid so a key
    rotation does not lock everyone out for up to an hour. Network failure
    returns whatever is cached (possibly nothing), because failing closed here
    means falling back to Basic Auth, not letting anyone in.
    """
    import jwt  # imported lazily so the module is importable without PyJWT

    now = time.time()
    with _jwks_lock:
        fresh = (now - _jwks_cache["fetched_at"]) < _JWKS_TTL_SECONDS
        if _jwks_cache["keys"] and fresh and not force:
            return _jwks_cache["keys"]

        try:
            request = urllib.request.Request(
                certs_url(), headers={"User-Agent": "tesla-alerts/natix"}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                document = json.load(response)
        except Exception as error:                      # noqa: BLE001
            print(f"[🔐 cf] could not fetch Access keys: "
                  f"{type(error).__name__}: {error}", flush=True)
            return _jwks_cache["keys"]

        keys: dict[str, Any] = {}
        for entry in document.get("keys", []):
            kid = entry.get("kid")
            if not kid:
                continue
            try:
                keys[kid] = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(entry))
            except Exception:                           # noqa: BLE001
                continue

        if keys:
            _jwks_cache["keys"] = keys
            _jwks_cache["fetched_at"] = now
            print(f"[🔐 cf] loaded {len(keys)} Access signing keys", flush=True)
        return _jwks_cache["keys"]


def verify(token: str) -> dict[str, Any] | None:
    """
    Return the token's claims if it is a valid Access token for THIS app.

    None on any failure, and the reason is logged rather than raised - the
    caller's job is simply to fall back to Basic Auth.
    """
    if not token or not is_configured():
        return None

    try:
        import jwt
    except ImportError:
        print("[🔐 cf] PyJWT is not installed; Access tokens cannot be verified",
              flush=True)
        return None

    try:
        header = jwt.get_unverified_header(token)
    except Exception as error:                          # noqa: BLE001
        print(f"[🔐 cf] malformed Access token: {error}", flush=True)
        return None

    kid = header.get("kid")
    keys = _load_keys()
    key = keys.get(kid)
    if key is None:
        # Unknown kid - most likely a rotation since our last fetch.
        keys = _load_keys(force=True)
        key = keys.get(kid)
    if key is None:
        print(f"[🔐 cf] no signing key for kid={kid}", flush=True)
        return None

    try:
        claims = jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            audience=CF_ACCESS_AUD,
            issuer=issuer(),
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
    except Exception as error:                          # noqa: BLE001
        # Includes expiry, bad audience, wrong issuer and bad signature. The
        # class name alone is enough to debug without logging the token.
        print(f"[🔐 cf] Access token rejected: {type(error).__name__}: {error}",
              flush=True)
        return None

    email = (claims.get("email") or "").lower()
    if CF_ACCESS_ALLOWED_EMAILS and email not in CF_ACCESS_ALLOWED_EMAILS:
        print(f"[🔐 cf] {email or '(no email)'} is not in CF_ACCESS_ALLOWED_EMAILS",
              flush=True)
        return None

    return claims
