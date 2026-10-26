"""Lightweight Local Web Dashboard Server for AetherFlow.

Serves the modern 4-tab Linear/Raycast UI on localhost and exposes REST endpoints
for real-time telemetry streaming and interactive workspace adaptation control.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("aetherflow.web")

WEB_DIR = Path(__file__).parent / "web"


class DashboardState:
    """Thread-safe state holder for web telemetry and controls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.session_time: str = "00:00:00"
        self.state: str = "FLOW"
        self.state_title: str = "Deep Flow"
        self.state_desc: str = "Sustained attention, calm motor rhythm."
        self.productivity_score: float = 88.0
        self.wpm: float = 37.0
        self.kpm: float = 185.0
        self.bpm: float = 14.0
        self.ear: float = 0.29
        self.perclos: float = 0.032
        self.gaze: float = 0.92
        self.kelvin: int = 5000
        self.density: str = "STANDARD"
        self.dnd: str = "PRIORITY_ONLY"
        self.context: str = "code.exe — VS Code (Development)"
        self.probabilities: dict[str, float] = {
            "flow": 0.72,
            "focus": 0.18,
            "overload": 0.04,
            "fatigue": 0.03,
            "idle": 0.03,
        }
        # Control signals
        self.override_state: Optional[str] = None
        self.override_kelvin: Optional[int] = None
        self.override_density: Optional[str] = None
        self.override_cooldown: Optional[int] = None
        self.minimize_requested: bool = False

    def update_telemetry(self, data: dict[str, Any]) -> None:
        with self._lock:
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_time": self.session_time,
                "state": self.state,
                "state_title": self.state_title,
                "state_desc": self.state_desc,
                "productivity_score": self.productivity_score,
                "wpm": self.wpm,
                "kpm": self.kpm,
                "bpm": self.bpm,
                "ear": self.ear,
                "perclos": self.perclos,
                "gaze": self.gaze,
                "kelvin": self.kelvin,
                "density": self.density,
                "dnd": self.dnd,
                "context": self.context,
                "probabilities": self.probabilities,
            }


# Global dashboard state singleton
dashboard_state = DashboardState()


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for static assets and REST API endpoints."""

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            index_path = WEB_DIR / "index.html"
            if index_path.exists():
                content = index_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")
        elif self.path == "/api/telemetry":
            payload = json.dumps(dashboard_state.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/override":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len)
            try:
                data = json.loads(post_body.decode("utf-8"))
                with dashboard_state._lock:
                    if "state" in data:
                        dashboard_state.override_state = data["state"]
                    if "kelvin" in data:
                        dashboard_state.override_kelvin = int(data["kelvin"])
                        dashboard_state.kelvin = int(data["kelvin"])
                    if "density" in data:
                        dashboard_state.override_density = data["density"]
                    if "cooldown_sec" in data:
                        dashboard_state.override_cooldown = int(data["cooldown_sec"])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_error(400, f"Invalid JSON payload: {e}")
        elif self.path == "/api/minimize":
            with dashboard_state._lock:
                dashboard_state.minimize_requested = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"minimized"}')
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy access logs in production."""
        pass


class WebDashboardServer:
    """Manages the background threading HTTP server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> str:
        self.server = ThreadingHTTPServer((self.host, self.port), DashboardHTTPHandler)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="WebDashboard")
        self._thread.start()
        url = f"http://{self.host}:{self.port}"
        logger.info(f"AetherFlow Web Dashboard active at: {url}")
        return url

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()


def start_web_server(host: str = "127.0.0.1", port: int = 8765) -> WebDashboardServer:
    srv = WebDashboardServer(host, port)
    srv.start()
    return srv
