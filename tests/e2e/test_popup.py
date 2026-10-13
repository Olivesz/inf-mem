"""E2E: the real popup in Chromium, with the unpacked extension loaded,
talking to a real backend process over HTTP.

Run with ``pytest -m e2e``. Needs Playwright's Chromium:
``PLAYWRIGHT_BROWSERS_PATH=0 python -m playwright install chromium``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extension"

pytestmark = pytest.mark.e2e

SEED = [
    ("User plays the cello in a community orchestra", "FACT", 6),
    ("User is writing a compiler in Rust", "PROJECT", 8),
    ("User prefers concise answers without bullet points", "PREFERENCE", 7),
    ("User spent a week debugging SQLite locking errors", "EPISODE", 4),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


@pytest.fixture
def backend(tmp_path):
    """A real uvicorn process with a store seeded directly on disk."""
    from memory_store import MemoryStore
    from models import Memory, MemoryType

    store = MemoryStore(data_dir=tmp_path / ".inf_mem")
    rng = np.random.default_rng(0)
    mems = [Memory(id=str(uuid4()), text=t, type=MemoryType(k), importance=i) for t, k, i in SEED]
    store.add_many(mems, rng.standard_normal((len(mems), 384)).astype(np.float32))

    port = _free_port()
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("GROQ_API_KEY", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(ROOT / "backend"),
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env, cwd=tmp_path)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                _get(f"{url}/health")
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("backend did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def popup(backend, tmp_path):
    """The extension popup page, pointed at the test backend."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(tmp_path / "profile"), channel="chromium", headless=True,
            args=[f"--disable-extensions-except={EXTENSION}", f"--load-extension={EXTENSION}"])
        try:
            worker = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
            ext_id = worker.url.split("/")[2]
            worker.evaluate(f"chrome.storage.sync.set({{backendUrl: '{backend}', userId: 'default'}})")
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"chrome-extension://{ext_id}/popup/popup.html")
            yield page, backend, errors
        finally:
            ctx.close()


def expect(locator):
    # Extension pages forbid eval, so wait with locator assertions, not wait_for_function.
    from playwright.sync_api import expect as _expect
    return _expect(locator)


def _cards(page):
    return page.locator(".inf-mem-memory-card")


def test_popup_lists_backend_memories(popup):
    page, backend, errors = popup
    expect(page.locator("#memoryCount")).to_have_text("4")
    assert _cards(page).count() == 4 == _get(f"{backend}/memories")["count"]
    shown = set(page.locator(".inf-mem-memory-text").all_inner_texts())
    assert shown == {t for t, _, _ in SEED}
    assert page.locator("#lastUpdated").inner_text() == "just now"
    assert page.locator("#statusText").inner_text() == "Connected"
    assert errors == []


def test_popup_delete_removes_from_backend(popup):
    page, backend, _ = popup
    expect(page.locator("#memoryCount")).to_have_text("4")
    doomed = page.locator(".inf-mem-delete-btn").first.get_attribute("data-id")
    page.locator(".inf-mem-delete-btn").first.click()
    expect(page.locator("#memoryCount")).to_have_text("3")

    remaining = _get(f"{backend}/memories")
    assert remaining["count"] == 3 == _cards(page).count()
    assert doomed not in {m["id"] for m in remaining["memories"]}


def test_popup_clear_all_empties_backend(popup):
    page, backend, _ = popup
    expect(page.locator("#memoryCount")).to_have_text("4")
    page.once("dialog", lambda d: d.accept())
    page.locator("#clearAllBtn").click()
    expect(page.locator("#memoryCount")).to_have_text("0")

    assert _get(f"{backend}/memories")["count"] == 0
    assert page.locator("#footerMessage").inner_text() == "Cleared 4 memories"
    assert "No memories yet" in page.locator("#memoriesList").inner_text()
