"""Shared test doubles: a fake Groq client and a deterministic fake encoder."""

from __future__ import annotations

import hashlib
import json

import numpy as np


class FakeGroq:
    """Mimics the parts of the OpenAI client the backend uses.

    ``replies`` are returned in order by chat.completions.create; ``calls``
    records each request's kwargs.
    """

    def __init__(self, *replies: str, raise_error: Exception | None = None):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.raise_error = raise_error
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer.raise_error:
                    raise outer.raise_error
                content = outer.replies.pop(0) if outer.replies else "[]"
                msg = type("Msg", (), {"content": content})
                choice = type("Choice", (), {"message": msg})
                return type("Resp", (), {"choices": [choice]})

        class _Models:
            def list(self):
                if outer.raise_error:
                    raise outer.raise_error
                return []

        self.chat = type("Chat", (), {"completions": _Completions()})()
        self.models = _Models()


class FakeEncoder:
    """Deterministic text -> unit vector.

    Texts listed in ``aliases`` map to the vector of another text, which lets a
    test say "these two strings are the same fact" without a real model.
    Every call is counted so tests can assert how often encoding happens.
    """

    def __init__(self, aliases: dict[str, str] | None = None):
        self.aliases = aliases or {}
        self.calls: list[str] = []

    def __call__(self, text: str) -> np.ndarray:
        self.calls.append(text)
        # Queries carry the bge instruction; strip it so a query and a memory
        # with the same text share a vector, as they nearly do for the real model.
        text = text.removeprefix("Represent this sentence for searching relevant passages: ")
        key = self.aliases.get(text, text)
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)


class GroqHTTP:
    """A real OpenAI client whose HTTP transport is mocked.

    Unlike FakeGroq, every request goes through the openai SDK: request
    building, response parsing and its error types. Queue chat replies in
    ``replies``; ``status`` makes every call fail with that HTTP status.
    ``requests`` records (method, path, json body) for each call.
    """

    def __init__(self, *replies: str, status: int = 200):
        import httpx
        from openai import OpenAI

        self.replies = list(replies)
        self.status = status
        self.requests: list[tuple] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else None
            self.requests.append((request.method, request.url.path, body))
            if self.status != 200:
                return httpx.Response(self.status, json={"error": {"message": "mocked failure"}})
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"object": "list", "data": [
                    {"id": "llama-3.1-8b-instant", "object": "model", "created": 0, "owned_by": "groq"}]})
            content = self.replies.pop(0) if self.replies else "[]"
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        self.client = OpenAI(api_key="test-key", base_url="https://api.groq.com/openai/v1",
                             http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                             max_retries=0)
