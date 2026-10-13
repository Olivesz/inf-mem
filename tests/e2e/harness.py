"""E2E helpers: a real backend process, a fake Groq HTTP server behind it,
and the paths the browser tests need."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
# INF_MEM_EXTENSION_DIR runs the suite against another copy, e.g. the unzipped
# release asset (make release-check).
EXTENSION = Path(os.environ.get("INF_MEM_EXTENSION_DIR", ROOT / "extension")).resolve()
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


class FakeGroq:
    """An OpenAI-compatible HTTP server that records chat requests and
    replies with an empty memory list."""

    def __init__(self):
        self.chats: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body: dict):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"object": "list", "data": []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.chats.append(body)
                self._send({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": "[]"}}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/openai/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def user_messages(self) -> list[str]:
        return [c["messages"][1]["content"] for c in self.chats]


def start_backend(home: Path, groq_url: str | None):
    env = {**os.environ, "HOME": str(home)}
    env.pop("GROQ_API_KEY", None)
    if groq_url:
        env.update(GROQ_API_KEY="test-key", GROQ_BASE_URL=groq_url)
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(ROOT / "backend"),
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env, cwd=home)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            get_json(f"{url}/health")
            return proc, url
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("backend did not start")


def seed(home: Path, items: list, vectors) -> None:
    """Write memories straight into the store the backend will open."""
    from memory_store import MemoryStore
    from models import Memory, MemoryType

    store = MemoryStore(data_dir=home / ".inf_mem")
    mems = [Memory(id=str(uuid4()), text=t, type=MemoryType(k), importance=i) for t, k, i in items]
    store.add_many(mems, vectors)
