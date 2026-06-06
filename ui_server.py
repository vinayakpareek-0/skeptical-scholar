import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from src.generation.run_generation import run_generation


PROJECT_ROOT = Path(__file__).parent.resolve()
WEB_ROOT = PROJECT_ROOT / "web"


class SkepticalScholarHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/query":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw_body) if raw_body else {}
            query = (payload.get("query") or "").strip()
            if not query:
                self._send_json(400, {"error": "Query is required"})
                return

            result = run_generation(query)
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def main():
    load_dotenv()
    host = os.getenv("SKEPTICAL_UI_HOST", "127.0.0.1")
    port = int(os.getenv("SKEPTICAL_UI_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), SkepticalScholarHandler)
    print(f"Skeptical Scholar UI: http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
