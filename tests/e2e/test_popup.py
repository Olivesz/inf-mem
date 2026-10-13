"""E2E: the real popup in Chromium, with the unpacked extension loaded,
talking to a real backend process over HTTP.

Run with ``pytest -m e2e``. Needs Playwright's Chromium:
``PLAYWRIGHT_BROWSERS_PATH=0 python -m playwright install chromium``.
"""

from __future__ import annotations

import numpy as np
import pytest
from harness import get_json, seed, start_backend

pytestmark = pytest.mark.e2e

SEED = [
    ("User plays the cello in a community orchestra", "FACT", 6),
    ("User is writing a compiler in Rust", "PROJECT", 8),
    ("User prefers concise answers without bullet points", "PREFERENCE", 7),
    ("User spent a week debugging SQLite locking errors", "EPISODE", 4),
]


@pytest.fixture
def backend(tmp_path):
    """A real uvicorn process with a store seeded directly on disk."""
    rng = np.random.default_rng(0)
    seed(tmp_path, SEED, rng.standard_normal((len(SEED), 384)).astype(np.float32))
    proc, url = start_backend(tmp_path, groq_url=None)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def popup(backend, launch_extension):
    """The extension popup page, pointed at the test backend."""
    ctx, ext_id = launch_extension(backend)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"chrome-extension://{ext_id}/popup/popup.html")
    return page, backend, errors


def expect(locator):
    # Extension pages forbid eval, so wait with locator assertions, not wait_for_function.
    from playwright.sync_api import expect as _expect
    return _expect(locator)


def _cards(page):
    return page.locator(".inf-mem-memory-card")


def test_popup_lists_backend_memories(popup):
    page, backend, errors = popup
    expect(page.locator("#memoryCount")).to_have_text("4")
    assert _cards(page).count() == 4 == get_json(f"{backend}/memories")["count"]
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

    remaining = get_json(f"{backend}/memories")
    assert remaining["count"] == 3 == _cards(page).count()
    assert doomed not in {m["id"] for m in remaining["memories"]}


def test_popup_clear_all_empties_backend(popup):
    page, backend, _ = popup
    expect(page.locator("#memoryCount")).to_have_text("4")
    page.once("dialog", lambda d: d.accept())
    page.locator("#clearAllBtn").click()
    expect(page.locator("#memoryCount")).to_have_text("0")

    assert get_json(f"{backend}/memories")["count"] == 0
    assert page.locator("#footerMessage").inner_text() == "Cleared 4 memories"
    assert "No memories yet" in page.locator("#memoriesList").inner_text()
