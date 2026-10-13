"""E2E: Chromium loads the extension with no manifest warnings or errors.

Chromium only records manifest problems in developer mode, so the test turns
it on and reloads the extension. A copy with a deliberately unknown manifest
key runs first, to prove the check can see a warning at all.
"""

from __future__ import annotations

import json
import shutil

import pytest
from harness import EXTENSION

pytestmark = pytest.mark.e2e


def problems(ext, profile) -> dict:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(profile), channel="chromium", headless=True,
            args=[f"--disable-extensions-except={ext}", f"--load-extension={ext}"])
        try:
            worker = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
            ext_id = worker.url.split("/")[2]
            page = ctx.new_page()
            page.goto("chrome://extensions/")
            page.evaluate("() => new Promise(r => chrome.developerPrivate.updateProfileConfiguration("
                          "{inDeveloperMode: true}, r))")
            page.evaluate("id => new Promise(r => chrome.developerPrivate.reload(id, {failQuietly: true}, "
                          "() => r()))", ext_id)
            page.wait_for_timeout(1500)
            info = page.evaluate("id => new Promise(r => chrome.developerPrivate.getExtensionInfo(id, r))", ext_id)
        finally:
            ctx.close()
    return {k: info[k] for k in ("installWarnings", "manifestErrors", "runtimeErrors")}


def test_check_detects_a_planted_manifest_problem(tmp_path):
    bad = tmp_path / "ext"
    shutil.copytree(EXTENSION, bad)
    manifest = json.loads((bad / "manifest.json").read_text())
    manifest["not_a_manifest_key"] = 1
    (bad / "manifest.json").write_text(json.dumps(manifest))
    found = problems(bad, tmp_path / "p1")
    assert any("not_a_manifest_key" in e["message"] for e in found["manifestErrors"])


def test_extension_loads_without_warnings(tmp_path):
    assert problems(EXTENSION, tmp_path / "p2") == {
        "installWarnings": [], "manifestErrors": [], "runtimeErrors": []}
