"""A small web app (standard library only): a chat page and a JSON API.

  GET  /             the chat page
  GET  /api/health   {"status": "ok", ...}
  POST /api/ask      {"question": "...", "retriever": "hybrid", "k": 5} -> answer, sources, usage
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from . import app
from .retrievers import dense_available

STATIC = Path(__file__).parent / "static"
ALLOWED_RETRIEVERS = {"bm25", "tfidf", "dense", "hybrid", "hybrid-lexical", "none"}
MAX_QUESTION_CHARS = 1000


def retriever_options() -> list[str]:
    base = ["hybrid", "dense", "bm25", "tfidf", "hybrid-lexical"] if dense_available() else ["hybrid-lexical", "bm25", "tfidf"]
    return base + [b + "+rerank" for b in base[:1]] + ["none"]


class Handler(BaseHTTPRequestHandler):
    server_version = "pydocs-rag"

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict):
        self._send(status, json.dumps(data).encode(), "application/json")

    def log_message(self, fmt, *args):  # quieter logs
        if os.environ.get("RAG_DEBUG"):
            super().log_message(fmt, *args)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/health":
            self._json(200, {"status": "ok", "chunks": len(app.chunks()), "retrievers": retriever_options(),
                             "default": app.default_retriever(), "llm": "claude" if os.environ.get("ANTHROPIC_API_KEY") else "offline"})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/ask":
            return self._json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 10_000:
                return self._json(413, {"error": "Request too large"})
            body = json.loads(self.rfile.read(length) or b"{}")
            question = str(body.get("question", "")).strip()[:MAX_QUESTION_CHARS]
            retriever = str(body.get("retriever", "auto"))
            k = max(1, min(10, int(body.get("k", 5))))
        except (ValueError, TypeError):
            return self._json(400, {"error": "Invalid JSON body"})
        if not question:
            return self._json(400, {"error": "Please enter a question"})
        base = retriever.replace("+rerank", "")
        if retriever != "auto" and base not in ALLOWED_RETRIEVERS:
            return self._json(400, {"error": f"Unknown retriever '{retriever}'"})
        if base in ("dense", "hybrid") and not dense_available():
            return self._json(400, {"error": "Dense retrieval needs fastembed: pip install -e '.[dense]'"})
        try:
            answer = app.get_pipeline(retriever, k).answer(question)
        except Exception as e:  # noqa: BLE001 - report errors to the page instead of crashing
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        self._json(200, answer.to_dict())


def serve(host: str = "127.0.0.1", port: int = 8000):
    app.chunks()  # load the index before the first request
    httpd = ThreadingHTTPServer((host, port), Handler)
    mode = "Claude" if os.environ.get("ANTHROPIC_API_KEY") else "offline mode (set ANTHROPIC_API_KEY for real answers)"
    print(f"pydocs-rag running at http://{host}:{port}  [{mode}]")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
