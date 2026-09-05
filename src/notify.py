"""
notify.py

Push notifications for high-severity findings.

Scout's headline claim was telling you you're being followed "in real-time",
and its README notes that a Raspberry Pi build gives that up. On the Orin Nano
we can keep it - but only if something actually pushes to your phone, which
Scout never shipped. This is that piece.

Three transports, all optional, all configured by environment variable:

    ntfy      NTFY_TOPIC        - simplest, no account, self-hostable
    webhook   ALERT_WEBHOOK_URL - generic JSON POST (Slack, Discord, n8n, ...)
    pushover  PUSHOVER_TOKEN + PUSHOVER_USER

With none configured, `notify` is a no-op that logs. Nothing in the pipeline
depends on delivery succeeding - a car in a tunnel must not stall the detector.
"""
import json
import os
import urllib.parse
import urllib.request

from src.db import now_ts

NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
PUSHOVER_TOKEN = os.environ.get("PUSHOVER_TOKEN", "").strip()
PUSHOVER_USER = os.environ.get("PUSHOVER_USER", "").strip()

# Only findings at or above this severity are pushed. "medium" would page you
# for a plate seen on two drives, which after a week of commuting is everyone
# who shares your route.
NOTIFY_MIN_SEVERITY = os.environ.get("NOTIFY_MIN_SEVERITY", "high").strip().lower()

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


def transports_configured() -> list[str]:
    """Which delivery methods are set up. Used by the dashboard's status view."""
    configured = []
    if NTFY_TOPIC:
        configured.append("ntfy")
    if ALERT_WEBHOOK_URL:
        configured.append("webhook")
    if PUSHOVER_TOKEN and PUSHOVER_USER:
        configured.append("pushover")
    return configured


def status_text() -> str:
    transports = transports_configured()
    if not transports:
        return "disabled - set NTFY_TOPIC, ALERT_WEBHOOK_URL or PUSHOVER_* to enable"
    return f"ready ({', '.join(transports)}, min severity={NOTIFY_MIN_SEVERITY})"


def _post(url: str, data: bytes, headers: dict[str, str], timeout: float = 10.0) -> bool:
    """Fire-and-forget POST. Returns success, never raises."""
    request = urllib.request.Request(url, data=data, method="POST")
    for header_name, header_value in headers.items():
        request.add_header(header_name, header_value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _send_ntfy(title: str, message: str, severity: str) -> bool:
    # ntfy takes the body as the message and metadata as headers.
    priority = "urgent" if severity == "high" else "default"
    return _post(
        f"{NTFY_SERVER}/{NTFY_TOPIC}",
        message.encode("utf-8"),
        {
            "Title": title,
            "Priority": priority,
            "Tags": "rotating_light" if severity == "high" else "eyes",
        },
    )


def _send_webhook(title: str, message: str, severity: str, payload: dict) -> bool:
    body = json.dumps({
        "title": title,
        # `text` is what Slack and Discord both render, so one payload covers
        # the common cases without per-service formatting.
        "text": f"{title}\n{message}",
        "message": message,
        "severity": severity,
        "timestamp": now_ts(),
        "detail": payload,
    }).encode("utf-8")
    return _post(ALERT_WEBHOOK_URL, body, {"Content-Type": "application/json"})


def _send_pushover(title: str, message: str, severity: str) -> bool:
    form = urllib.parse.urlencode({
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "title": title,
        "message": message,
        # Priority 1 bypasses quiet hours; being followed is worth waking up for.
        "priority": 1 if severity == "high" else 0,
    }).encode("utf-8")
    return _post(
        "https://api.pushover.net/1/messages.json",
        form,
        {"Content-Type": "application/x-www-form-urlencoded"},
    )


def notify(title: str, message: str, severity: str = "high", payload: dict | None = None) -> list[str]:
    """
    Push a message through every configured transport.

    Returns the list of transports that accepted it, so the caller can decide
    whether to mark the finding notified. We only mark on at least one success:
    a finding you were never actually told about must stay pending.
    """
    if SEVERITY_RANK.get(severity, 0) < SEVERITY_RANK.get(NOTIFY_MIN_SEVERITY, 2):
        return []

    delivered: list[str] = []

    if NTFY_TOPIC and _send_ntfy(title, message, severity):
        delivered.append("ntfy")
    if ALERT_WEBHOOK_URL and _send_webhook(title, message, severity, payload or {}):
        delivered.append("webhook")
    if PUSHOVER_TOKEN and PUSHOVER_USER and _send_pushover(title, message, severity):
        delivered.append("pushover")

    if not transports_configured():
        print(f"[🔔 notify] (no transport configured) {severity.upper()}: {title} - {message}")

    return delivered


def notify_correlation(connection, result) -> bool:
    """
    Announce a newly-raised surveillance finding and record that we did.

    Takes a CorrelationResult from correlate.py. The message leads with the
    *reasons*, not the score, because "seen on 4 separate drives across 3 days"
    is actionable and "score 87" is not.
    """
    entity_kind = "Vehicle" if result.entity_type == "plate" else "Person"
    title = f"⚠️ {entity_kind} possibly following you: {result.entity_label}"

    lines = [f"Threat score {result.score}/100 ({result.severity})", ""]
    lines.extend(f"• {reason}" for reason in result.reasons)
    if result.max_separation_miles > 0:
        lines.append(f"• Sightings up to {result.max_separation_miles:.1f} miles apart")
    message = "\n".join(lines)

    delivered = notify(title, message, result.severity, payload={
        "entity_type": result.entity_type,
        "entity_id": result.entity_id,
        "entity_label": result.entity_label,
        "score": result.score,
        "reasons": result.reasons,
        "distinct_drives": result.distinct_drives,
        "distinct_days": result.distinct_days,
        "distinct_locations": result.distinct_locations,
    })

    if delivered:
        connection.execute(
            "UPDATE correlations SET notified=1 WHERE entity_type=? AND entity_id=?",
            (result.entity_type, result.entity_id),
        )
        connection.commit()
        print(f"[🔔 notify] sent via {', '.join(delivered)}: {result.entity_label}")
        return True

    return False
