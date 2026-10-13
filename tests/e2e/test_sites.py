"""E2E: the content scripts on the ChatGPT and Claude fixtures.

The fixture pages are served at https://chatgpt.com/ and https://claude.ai/,
so the manifest's content-script matches apply as they do on the real sites.
The backend is a real process with the real embedding model; Groq is a local
fake HTTP server that records every extraction request.
"""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pytest
from harness import FIXTURES, FakeGroq, seed, start_backend

pytestmark = pytest.mark.e2e

URL = {"chatgpt": "https://chatgpt.com/", "chatgpt-live": "https://chatgpt.com/", "claude": "https://claude.ai/"}
SITES = ["chatgpt", "chatgpt-live", "claude"]
COMPOSER = '[data-fixture="composer"]'
HEADER = "[Memory Context — inf-mem]"

MEMORIES = [
    ("User is writing a compiler in Rust", "PROJECT", 8),
    ("User plays the cello in a community orchestra", "FACT", 6),
]
WITH_MEMORY = "Why does my lexer in Rust panic on unicode identifiers?"
NO_MEMORY = "What is the capital of Australia?"


@pytest.fixture
def stack(tmp_path):
    import retriever

    groq = FakeGroq()
    seed(tmp_path, MEMORIES, np.stack([retriever.encode(t) for t, _, _ in MEMORIES]))
    proc, url = start_backend(tmp_path, groq_url=groq.url)
    # Load the embedding model before the browser starts timing anything.
    warm = urllib.request.Request(f"{url}/retrieve", data=json.dumps({"prompt": "warm up"}).encode(),
                                  headers={"Content-Type": "application/json"})
    urllib.request.urlopen(warm, timeout=120).read()
    yield url, groq
    proc.terminate()
    proc.wait(timeout=10)
    groq.server.shutdown()


@pytest.fixture
def open_site(stack, launch_extension):
    """open_site(site, query='') -> page on the fixture with the extension active."""
    backend, groq = stack

    def open_(site: str, query: str = ""):
        ctx, _ = launch_extension(backend)

        def serve(route):
            name = "fixture.js" if route.request.url.endswith("/fixture.js") else f"{site}.html"
            kind = "application/javascript" if name.endswith(".js") else "text/html"
            route.fulfill(status=200, content_type=kind, body=(FIXTURES / name).read_text())

        ctx.route(f"{URL[site]}**", serve)
        page = ctx.new_page()
        page.goto(URL[site] + query)
        page.wait_for_selector(COMPOSER)
        return page

    return open_, groq


def submits(page) -> list:
    return page.evaluate("window.__submits")


def type_and_enter(page, text: str) -> None:
    page.locator(COMPOSER).click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def wait_until(cond, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


def wait_reply_done(page) -> str:
    """Text of the newest reply once streaming has finished."""
    wait_until(lambda: page.evaluate(
        "!document.querySelector('[data-testid=stop-button], [data-is-streaming=true],"
        " [aria-label=\"Stop generating\"]')"
        " && document.querySelectorAll('[data-message-author-role=assistant], [data-is-streaming]').length > 0"))
    return page.evaluate(
        "(() => { const r = document.querySelectorAll("
        "'[data-message-author-role=assistant], [data-is-streaming]'); return r[r.length-1].innerText.trim(); })()")


@pytest.mark.parametrize("site", SITES)
def test_send_with_memory_submits_exactly_once(open_site, site):
    open_, _ = open_site
    page = open_(site)
    type_and_enter(page, WITH_MEMORY)

    panel = page.locator(".inf-mem-panel")
    panel.wait_for(timeout=20000)
    assert "User is writing a compiler in Rust" in panel.inner_text()
    assert submits(page) == []  # held until the user chooses

    page.get_by_text("Send with memory").click()
    wait_until(lambda: len(submits(page)) == 1)
    wait_reply_done(page)
    time.sleep(1.0)

    sent = submits(page)
    assert len(sent) == 1
    assert sent[0].startswith(HEADER)
    assert sent[0].endswith(WITH_MEMORY)
    assert page.locator(".inf-mem-panel").count() == 0


@pytest.mark.parametrize("site", SITES)
def test_extracts_once_per_streamed_reply_with_full_text(open_site, site):
    open_, groq = open_site
    page = open_(site)
    type_and_enter(page, NO_MEMORY)  # nothing passes the gate, so it sends straight away

    wait_until(lambda: len(submits(page)) == 1)
    reply = wait_reply_done(page)
    wait_until(lambda: len(groq.chats) == 1)
    time.sleep(2.0)  # any duplicate extraction would land here

    assert len(groq.chats) == 1
    sent = groq.user_messages()[0]
    assert sent == f"{NO_MEMORY}\n\nAI Response:\n{reply}"
    assert len(reply.split()) == 40


@pytest.mark.parametrize("site", SITES)
def test_second_prompt_pairs_with_second_reply(open_site, site):
    open_, groq = open_site
    page = open_(site)
    type_and_enter(page, NO_MEMORY)
    first = wait_reply_done(page)
    wait_until(lambda: len(groq.chats) == 1)

    second_prompt = "Give me a recipe for banana bread"
    type_and_enter(page, second_prompt)
    wait_until(lambda: len(submits(page)) == 2)
    time.sleep(0.2)
    second = wait_reply_done(page)
    wait_until(lambda: len(groq.chats) == 2)

    assert first != second
    assert groq.user_messages()[1] == f"{second_prompt}\n\nAI Response:\n{second}"


def test_claude_extracts_the_new_reply_not_the_first(open_site):
    open_, groq = open_site
    page = open_("claude", "?history=3")
    type_and_enter(page, NO_MEMORY)
    wait_until(lambda: len(submits(page)) == 1)
    reply = wait_reply_done(page)
    wait_until(lambda: len(groq.chats) == 1)

    assert reply.startswith("r4w0")
    assert groq.user_messages()[0].endswith(reply)


@pytest.mark.parametrize("site", SITES)
def test_interception_survives_composer_swap(open_site, site):
    open_, _ = open_site
    page = open_(site)
    page.evaluate("window.__swapComposer()")
    type_and_enter(page, WITH_MEMORY)
    page.locator(".inf-mem-panel").wait_for(timeout=20000)
    assert submits(page) == []
