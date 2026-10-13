"""Shared test doubles: a fake Groq client and a deterministic fake encoder."""

from __future__ import annotations

import hashlib

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
        key = self.aliases.get(text, text)
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)
