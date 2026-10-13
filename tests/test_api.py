"""API tests: every endpoint through FastAPI's TestClient.

The store lives in a tmp HOME, Groq is a fake client and the encoder is a
deterministic fake, so these run offline. They import main.py for real, which
needs the runtime requirements installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import config
import extractor
import main
import retriever
from fakes import FakeEncoder, GroqHTTP

EXT_ORIGIN = "chrome-extension://" + "abcdefghijklmnopabcdefghijklmnop"


@pytest.fixture
def encoder(monkeypatch):
    enc = FakeEncoder()
    monkeypatch.setattr(retriever, "encode", enc)
    return enc


@pytest.fixture
def app_client(tmp_path, monkeypatch, encoder):
    """A TestClient with an empty tmp store and no Groq key configured."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "missing.yaml")
    main._reachability.update(checked_at=0.0, ok=False, error=None)
    for k in extractor.stats:
        extractor.stats[k] = 0
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def with_groq(monkeypatch):
    """Install a real OpenAI client with a mocked HTTP transport.

    Returns the GroqHTTP so tests can queue replies and inspect requests.
    """
    fake = GroqHTTP()
    monkeypatch.setattr(main, "groq_client", fake.client)
    return fake


REPLY = json.dumps([
    {"text": "User has a dog named Biscuit", "type": "FACT", "keywords": ["dog"], "importance": 6},
    {"text": "User prefers concise answers", "type": "PREFERENCE", "keywords": [], "importance": 7},
])


# --------------------------------------------------------------------------- #
# CORS: only the extension may call the backend
# --------------------------------------------------------------------------- #

def test_foreign_origin_gets_no_cors_grant(app_client):
    r = app_client.get("/memories", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in r.headers


def test_foreign_origin_preflight_delete_rejected(app_client):
    r = app_client.options("/memories/x", headers={
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "DELETE",
    })
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


def test_extension_origin_allowed_without_credentials(app_client):
    r = app_client.get("/memories", headers={"Origin": EXT_ORIGIN})
    assert r.headers.get("access-control-allow-origin") == EXT_ORIGIN
    assert "access-control-allow-credentials" not in r.headers

    pre = app_client.options("/memories/x", headers={
        "Origin": EXT_ORIGIN,
        "Access-Control-Request-Method": "DELETE",
    })
    assert pre.status_code == 200


# --------------------------------------------------------------------------- #
# /health tells the truth about Groq
# --------------------------------------------------------------------------- #

def test_health_placeholder_key_is_not_configured(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f'GROQ_API_KEY: "{config.PLACEHOLDER_KEY}"\n')
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert config.load_groq_api_key(cfg) is None


def test_health_without_key(app_client):
    body = app_client.get("/health").json()
    assert body["groq_configured"] is False
    assert body["groq_reachable"] is False
    assert body["memory_count"] == 0


def test_health_reachable_reflects_real_call(app_client, monkeypatch):
    ok = GroqHTTP()
    monkeypatch.setattr(main, "groq_client", ok.client)
    assert app_client.get("/health").json()["groq_reachable"] is True
    assert ok.requests == [("GET", "/openai/v1/models", None)]

    main._reachability.update(checked_at=0.0)
    monkeypatch.setattr(main, "groq_client", GroqHTTP(status=401).client)
    body = app_client.get("/health").json()
    assert body["groq_configured"] is True
    assert body["groq_reachable"] is False
    assert body["groq_error"] == "AuthenticationError"


def test_extract_sends_the_extraction_request(app_client, with_groq):
    with_groq.replies.append("[]")
    app_client.post("/extract", json={"prompt": "I play cello", "response": "Nice"})
    method, path, body = with_groq.requests[0]
    assert (method, path) == ("POST", "/openai/v1/chat/completions")
    assert body["model"] == "llama-3.1-8b-instant"
    assert body["messages"][0]["role"] == "system"
    assert "I play cello" in body["messages"][1]["content"]


def test_extract_groq_error_is_502(app_client, monkeypatch):
    monkeypatch.setattr(main, "groq_client", GroqHTTP(status=500).client)
    r = app_client.post("/extract", json={"prompt": "x"})
    assert r.status_code == 502


def test_compress_groq_error_returns_prompt_unchanged(app_client, monkeypatch):
    monkeypatch.setattr(main, "groq_client", GroqHTTP(status=429).client)
    body = app_client.post("/compress", json={"prompt": "keep me"}).json()
    assert body["compressed"] == "keep me" and body["ratio"] == 1.0


# --------------------------------------------------------------------------- #
# /extract, /retrieve, /memories, /compress
# --------------------------------------------------------------------------- #

def test_extract_without_key_is_503(app_client):
    r = app_client.post("/extract", json={"prompt": "hi"})
    assert r.status_code == 503


def test_compress_without_key_is_503(app_client):
    assert app_client.post("/compress", json={"prompt": "hi"}).status_code == 503


def test_extract_stores_and_lists(app_client, with_groq):
    with_groq.replies.append(REPLY)
    r = app_client.post("/extract", json={"prompt": "I have a dog", "response": "Nice"})
    assert r.status_code == 200 and r.json()["count"] == 2
    listed = app_client.get("/memories").json()
    assert {m["text"] for m in listed["memories"]} == {
        "User has a dog named Biscuit", "User prefers concise answers"}


def test_extract_unparseable_reply_is_502_and_counted(app_client, with_groq):
    with_groq.replies.append("Sure! Here are some memories: none really.")
    r = app_client.post("/extract", json={"prompt": "x"})
    assert r.status_code == 502
    assert app_client.get("/health").json()["extraction"]["parse_failures"] == 1


def test_extension_payload_user_id_is_honoured(app_client, with_groq):
    """The exact body background.js builds: camelCase userId."""
    with_groq.replies.append(REPLY)
    app_client.post("/extract", json={"prompt": "p", "response": "r", "userId": "alice"})
    assert app_client.get("/memories", params={"user_id": "alice"}).json()["count"] == 2
    assert app_client.get("/memories").json()["count"] == 0

    r = app_client.post("/retrieve", json={"prompt": "User has a dog named Biscuit",
                                           "userId": "alice", "threshold": 0.9})
    assert [m["text"] for m in r.json()["memories"]] == ["User has a dog named Biscuit"]


def test_retrieve_scoped_to_user(app_client, with_groq):
    with_groq.replies.append(REPLY)
    app_client.post("/extract", json={"prompt": "p", "user_id": "alice"})
    r = app_client.post("/retrieve", json={"prompt": "User has a dog named Biscuit",
                                           "user_id": "bob", "threshold": 0.0})
    assert r.json()["count"] == 0


def test_delete_one_and_missing(app_client, with_groq):
    with_groq.replies.append(REPLY)
    app_client.post("/extract", json={"prompt": "p"})
    mid = app_client.get("/memories").json()["memories"][0]["id"]
    assert app_client.delete(f"/memories/{mid}").json() == {"success": True}
    assert app_client.get("/memories").json()["count"] == 1
    assert app_client.delete(f"/memories/{mid}").status_code == 404


def test_clear_all_is_user_scoped(app_client, with_groq):
    with_groq.replies += [REPLY, REPLY]
    app_client.post("/extract", json={"prompt": "p", "user_id": "alice"})
    app_client.post("/extract", json={"prompt": "p", "user_id": "bob"})
    assert app_client.delete("/memories", params={"user_id": "alice"}).json() == {"deleted": 2}
    assert app_client.get("/memories", params={"user_id": "alice"}).json()["count"] == 0
    assert app_client.get("/memories", params={"user_id": "bob"}).json()["count"] == 2
    assert app_client.get("/health").json()["memory_count"] == 2


def test_compress_with_key(app_client, with_groq):
    with_groq.replies.append("short")
    body = app_client.post("/compress", json={"prompt": "please make this much shorter"}).json()
    assert body["compressed"] == "short"
