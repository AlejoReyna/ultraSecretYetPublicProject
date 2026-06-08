"""Lightweight health check HTTP server (stdlib only)."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.deployment.health_state import HealthState

LOGGER = logging.getLogger(__name__)


def _tail_lines(path: Path, n: int = 50) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def start_health_server(
    state: HealthState,
    *,
    port: int = 8080,
    decision_log_path: str | Path = "decision_log.jsonl",
    dashboard_path: str | Path = "dashboard.html",
) -> ThreadingHTTPServer:
    """Start daemon health server; returns server handle for shutdown."""

    decision_path = Path(decision_log_path)
    dashboard_file = Path(dashboard_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            LOGGER.debug("health %s - %s", self.address_string(), format % args)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, body: str, content_type: str, status: int = 200) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                payload = state.snapshot()
                status = 503 if state.is_stalled() else 200
                if state.is_stalled():
                    payload["status"] = "stalled"
                self._send_json(payload, status=status)
            elif self.path.startswith("/logs"):
                lines = _tail_lines(decision_path, 50)
                self._send_json({"lines": lines})
            elif self.path.startswith("/dashboard"):
                if dashboard_file.exists():
                    self._send_text(dashboard_file.read_text(encoding="utf-8"), "text/html")
                else:
                    self._send_json({"error": "dashboard.html not found"}, status=404)
            else:
                self._send_json({"error": "not found"}, status=404)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="health-server", daemon=True)
    thread.start()
    LOGGER.info("Health server listening on port %s", port)
    return server
