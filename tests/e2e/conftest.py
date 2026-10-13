"""E2E fixture: Chromium with the unpacked extension loaded."""

from __future__ import annotations

import pytest
from harness import EXTENSION


@pytest.fixture
def launch_extension(tmp_path):
    """Returns open(backend_url) -> (context, extension_id)."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    contexts = []

    def open_(backend_url: str):
        ctx = pw.chromium.launch_persistent_context(
            str(tmp_path / f"profile{len(contexts)}"), channel="chromium", headless=True,
            args=[f"--disable-extensions-except={EXTENSION}", f"--load-extension={EXTENSION}"])
        contexts.append(ctx)
        worker = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
        worker.evaluate(f"chrome.storage.sync.set({{backendUrl: '{backend_url}', userId: 'default'}})")
        return ctx, worker.url.split("/")[2]

    yield open_
    for ctx in contexts:
        ctx.close()
    pw.stop()
