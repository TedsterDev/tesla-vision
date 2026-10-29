"""
test_notify.py

Push notifications reach the server in a form it accepts - including the
titles, which carry emoji.

Run:  python3 tests/test_notify.py
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

received: list[dict] = []


class Capture(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        received.append({"path": self.path, "headers": dict(self.headers),
                         "body": self.rfile.read(length)})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Capture)
threading.Thread(target=server.serve_forever, daemon=True).start()
os.environ["NTFY_SERVER"] = f"http://127.0.0.1:{server.server_port}"
os.environ["NTFY_TOPIC"] = "tesla-alerts"
os.environ["NTFY_TOKEN"] = "tk_test"

from src import notify  # noqa: E402


def test_an_emoji_title_is_delivered_not_silently_dropped():
    received.clear()
    delivered = notify.notify("⚠️ Vehicle possibly following you: 7ABC123", "• seen on 3 drives", "high")
    assert delivered == ["ntfy"], delivered
    body = json.loads(received[0]["body"])
    assert body["title"].startswith("⚠️") and body["topic"] == "tesla-alerts", body
    assert body["priority"] == 5, body


def test_the_publish_token_is_sent():
    received.clear()
    notify.notify("plain", "message", "high")
    assert received[0]["headers"].get("Authorization") == "Bearer tk_test", received[0]["headers"]


def test_below_the_minimum_severity_nothing_is_sent():
    received.clear()
    assert notify.notify("quiet", "message", "low") == []
    assert not received


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {test.__name__}: {error}")
        except Exception as error:                        # noqa: BLE001
            failures += 1
            print(f"  ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    server.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
