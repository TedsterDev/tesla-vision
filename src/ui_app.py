"""
ui_app.py

Minimal web dashboard:
- "/" shows newest alerts (or "No alerts yet")
- "/alerts" returns JSON list
- "/media/..." serves JPG/GIF from /data/media
- "/healthz" returns {"ok": true}

Run (in container):
  uvicorn src.ui_app:app --host 0.0.0.0 --port 8080
"""
import json

from pathlib import Path

from typing import Any

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from src.common import (
    MEDIA_DIR,
    ALERTS_DIR,
    ensure_dirs
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On Start-Up
    ensure_dirs()
    yield
    # Shutdown (nothing to do yet)

app = FastAPI(title="Tesla Alerts Dashboard")

# Serve files in /data/media at /media/<filename>
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

def _load_alert(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _list_alerts() -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for alert_path in ALERTS_DIR.glob("*.json"):
        try:
            alerts.append(_load_alert(alert_path))
        except Exception:
            continue
    alerts.sort(key=lambda alert: alert.get("timestamp", 0), reverse=True)
    return alerts

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/alerts")
def alerts_json():
    return _list_alerts()

@app.get("/", response_class=HTMLResponse)
def index():
    alerts = _list_alerts()

    rows = []

    for alert in alerts[:200]:
        alert_id = alert.get("id", "")
        timestamp = alert.get("timestamp", "")
        source_file = alert.get("source_file", "")
        score = alert.get("score", "")
        status = alert.get("status", "")

        jpeg = alert.get("jpeg", "")
        gif = alert.get("gif", "")

        jpeg_tag = f'<img src="/media/{jpeg}" style="max-width:420px; border-radius:12px;" />' if jpeg else ""
        gif_tag = f'<img src="/media/{gif}" style="max-width:420px; border-radius:12px;" />' if gif else ""

        rows.append(f"""
          <div style="display:flex; gap:16px; padding:16px; border:1px solid #333; border-radius:16px; margin:12px 0;">
            <div style="min-width:380px;">
              <div style="font-size:18px; font-weight:700;">Alert {alert_id}</div>
              <div style="opacity:0.85;">ts: {timestamp}</div>
              <div style="opacity:0.85;">file: {source_file}</div>
              <div style="opacity:0.85;">score: {score}</div>
              <div style="opacity:0.85;">status: {status}</div>
              <div style="margin-top:10px;">
                <a href="/alerts" style="color:#7aa2ff;">View JSON list</a>
              </div>
            </div>
            <div style="display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap;">
              <div>{jpeg_tag}</div>
              <div>{gif_tag}</div>
            </div>
          </div>
        """)

    body = "\n".join(rows) if rows else "<p>No alerts yet.</p>"

    html = f"""
    <html>
      <head>
        <title>Tesla Alerts</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="background:#0b0b0f; color:#eee; font-family: ui-sans-serif, system-ui; padding:20px;">
        <h1 style="margin:0 0 12px 0;">Tesla Alerts</h1>
        <div style="opacity:0.8; margin-bottom:18px;">
          Dashboard is live. Alerts will appear here once the car feed is connected.
        </div>
        {body}
      </body>
    </html>
    """
    return HTMLResponse(html)

