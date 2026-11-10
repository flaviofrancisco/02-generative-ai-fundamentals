"""Placeholder web server shown on the Streamlit port while the vector database is built.

docker-entrypoint.sh starts this before the embedding pipeline and stops it right
before Streamlit takes over the port. The page polls Streamlit's health endpoint and
reloads itself as soon as the real app is up.

Usage: python loading_server.py <port> <pipeline-log-file>
"""

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
LOG_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/embedding.log")

TOTAL_RE = re.compile(r"Total text files to process: (\d+)")
ADDED_RE = re.compile(r"Added (\d+) documents from file: (.+)")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NASA Mission Chat - starting</title>
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0; min-height: 100vh; display: grid; place-items: center;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: #0e1117; color: #fafafa;
  }
  main { text-align: center; padding: 24px; max-width: 520px; }
  .spinner {
    width: 56px; height: 56px; margin: 0 auto 24px;
    border: 5px solid rgba(255, 255, 255, 0.15); border-top-color: #ff4b4b;
    border-radius: 50%; animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  h1 { font-size: 1.35rem; margin: 0 0 8px; font-weight: 600; }
  p { margin: 6px 0; color: #a3a8b8; line-height: 1.5; }
  #detail { font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  @media (prefers-reduced-motion: reduce) { .spinner { animation-duration: 3s; } }
</style>
</head>
<body>
<main>
  <div class="spinner" role="status" aria-label="Loading"></div>
  <h1>Building the vector database&hellip;</h1>
  <p id="detail">Preparing documents</p>
  <p>The chat opens automatically when this finishes.</p>
</main>
<script>
  const detail = document.getElementById("detail");

  async function tick() {
    try {
      // Streamlit answers "ok" here; this placeholder answers 503.
      const health = await fetch("/_stcore/health", { cache: "no-store" });
      if (health.ok) { location.reload(); return; }

      const res = await fetch("/__loading_status", { cache: "no-store" });
      const s = await res.json();
      if (s.file) {
        // Files embedded on an earlier run are skipped, so no "N of total" here.
        detail.textContent = "Embedding " + s.file + " - " + s.chunks + " chunks added this run";
      } else if (s.total) {
        detail.textContent = "Checking " + s.total + " files for new content";
      }
    } catch (e) {
      // Port is changing hands between this page and Streamlit; keep trying.
      detail.textContent = "Starting the chat app";
    }
    setTimeout(tick, 2000);
  }
  tick();
</script>
</body>
</html>
""".encode("utf-8")


def read_status() -> dict:
    """Summarise pipeline progress from its log file."""
    status = {"total": None, "file": None, "files_started": 0, "chunks": 0}
    try:
        text = LOG_FILE.read_text(errors="replace")
    except OSError:
        return status

    seen_files = []
    for line in text.splitlines():
        total = TOTAL_RE.search(line)
        if total:
            status["total"] = int(total.group(1))
            continue
        added = ADDED_RE.search(line)
        if added:
            name = added.group(2).strip()
            if name not in seen_files:
                seen_files.append(name)
            status["chunks"] += int(added.group(1))
            status["file"] = name
    status["files_started"] = len(seen_files)
    return status


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        path = self.path.split("?", 1)[0]
        if path == "/__loading_status":
            self._send(200, json.dumps(read_status()).encode(), "application/json")
        elif path.startswith("/_stcore/"):
            # Keep the Docker healthcheck failing until the real app is up.
            self._send(503, b"starting", "text/plain")
        else:
            self._send(200, PAGE, "text/html; charset=utf-8")

    def log_message(self, *args) -> None:  # keep container logs clean
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
