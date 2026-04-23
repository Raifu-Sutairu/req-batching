"""
Minimal inference API for Ashrith's Predictive Dyna-Q model.

Endpoints:
- GET  /health
- POST /infer
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from Ashrith.predictive_dynaq_agent import PredictiveDynaQAgent


class DynaqInferenceService:
    """Loads the selected checkpoint once and serves deterministic actions."""

    def __init__(self, checkpoint_path: str):
        self.agent = PredictiveDynaQAgent(action_dim=2)
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Train or copy the best model first."
            )
        self.agent.load(checkpoint_path)
        self.agent.start_episode()

    def infer(self, state: list[float]) -> dict[str, Any]:
        if len(state) != 6:
            raise ValueError("state must contain exactly 6 normalized features.")
        state_array = np.asarray(state, dtype=np.float32)
        action = int(self.agent.select_action(state_array, explore=False))
        return {
            "action": action,
            "action_name": "SKIP" if action == 1 else "WAIT",
            "model": "predictive_dynaq_best.npy",
        }


def build_handler(service: DynaqInferenceService):
    class RequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "service": "ashrith-dynaq-inference",
                        "checkpoint": service.checkpoint_path,
                    },
                )
                return
            self._send_json(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/infer":
                self._send_json(404, {"error": "not_found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                state = payload.get("state")
                if not isinstance(state, list):
                    raise ValueError("Request body must include: {'state': [6 floats]}")
                result = service.infer(state)
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})

        def log_message(self, format: str, *args):
            # Keep server output clean for reverse-proxy deployment logs.
            return

    return RequestHandler


def run_server(host: str = "0.0.0.0", port: int = 8080):
    checkpoint = os.path.join("Ashrith", "checkpoints", "predictive_dynaq_best.npy")
    service = DynaqInferenceService(checkpoint_path=checkpoint)
    handler = build_handler(service)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dyna-Q API listening on http://{host}:{port}")
    print("Endpoints: GET /health, POST /infer")
    server.serve_forever()


if __name__ == "__main__":
    run_server()