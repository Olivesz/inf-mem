"""
Regenerate docs/popup.png: the real popup, in Chromium with the unpacked
extension, against a real backend whose store holds the demo memories below.
No personal data: the memories are an invented demo user.

    PLAYWRIGHT_BROWSERS_PATH=0 python docs/make_popup_screenshot.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "tests" / "e2e")]

from harness import EXTENSION, seed, start_backend
from playwright.sync_api import sync_playwright

DEMO = [
    ("User is writing a compiler in Rust", "PROJECT", 8),
    ("User prefers concise answers without bullet points", "PREFERENCE", 7),
    ("User plays the cello in a community orchestra", "FACT", 6),
    ("User spent a week debugging SQLite locking errors", "EPISODE", 4),
    ("User is training for a half marathon in April", "PROJECT", 5),
]
WIDTH = 400


def main() -> None:
    home = Path(tempfile.mkdtemp())
    seed(home, DEMO, np.random.default_rng(0).standard_normal((len(DEMO), 384)).astype(np.float32))
    proc, url = start_backend(home, groq_url=None)
    try:
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                str(home / "profile"), channel="chromium", headless=True, device_scale_factor=2,
                viewport={"width": WIDTH, "height": 600},
                args=[f"--disable-extensions-except={EXTENSION}", f"--load-extension={EXTENSION}"])
            worker = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
            worker.evaluate(f"chrome.storage.sync.set({{backendUrl: '{url}'}})")
            page = ctx.new_page()
            page.goto(f"chrome-extension://{worker.url.split('/')[2]}/popup/popup.html")
            page.locator("#memoryCount").filter(has_text=str(len(DEMO))).wait_for()
            page.locator("#statusText").filter(has_text="Connected").wait_for()
            assert page.evaluate("innerWidth") == WIDTH, "viewport was not applied"
            page.locator(".inf-mem-popup").screenshot(path=str(ROOT / "docs" / "popup.png"))
            ctx.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
