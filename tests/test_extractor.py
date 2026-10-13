"""Extractor tests: reply parsing, dedupe, update, and encode call counts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import extractor
import retriever
from fakes import FakeEncoder, FakeGroq
from memory_store import MemoryStore


@pytest.fixture
def store(tmp_path, monkeypatch) -> MemoryStore:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return MemoryStore()


def _item(text, importance=5, mtype="FACT"):
    return {"text": text, "type": mtype, "keywords": [], "importance": importance}


# --------------------------------------------------------------------------- #
# parse_reply
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("reply", [
    '[{"text": "a", "type": "FACT"}]',
    '```json\n[{"text": "a", "type": "FACT"}]\n```',
    '```\n[{"text": "a", "type": "FACT"}]\n```',
    'Here is the JSON:\n[{"text": "a", "type": "FACT"}]',
    '{"text": "a", "type": "FACT"}',
])
def test_parse_reply_tolerates_model_wrapping(reply):
    assert extractor.parse_reply(reply) == [{"text": "a", "type": "FACT"}]


def test_parse_reply_empty_list():
    assert extractor.parse_reply("[]") == []


@pytest.mark.parametrize("reply", ["no json here", "[not json]", '"a string"'])
def test_parse_reply_rejects_garbage(reply):
    with pytest.raises(extractor.ExtractionParseError):
        extractor.parse_reply(reply)


def test_fenced_reply_end_to_end_stores_memories(store, monkeypatch):
    monkeypatch.setattr(retriever, "encode", FakeEncoder())
    reply = "```json\n" + json.dumps([_item("User plays cello")]) + "\n```"
    out = extractor.extract("I play cello", None, store, FakeGroq(reply))
    assert [m.text for m in out] == ["User plays cello"]


def test_injected_client_receives_prompt_and_response(store, monkeypatch):
    monkeypatch.setattr(retriever, "encode", FakeEncoder())
    groq = FakeGroq("[]")
    extractor.extract("my prompt", "the reply", store, groq)
    user_msg = groq.calls[0]["messages"][1]["content"]
    assert "my prompt" in user_msg and "the reply" in user_msg


# --------------------------------------------------------------------------- #
# apply: ADD / UPDATE / skip
# --------------------------------------------------------------------------- #

def test_paraphrase_updates_instead_of_adding(store, monkeypatch):
    enc = FakeEncoder(aliases={"User owns a dog called Biscuit": "User has a dog named Biscuit"})
    monkeypatch.setattr(retriever, "encode", enc)
    extractor.apply([_item("User has a dog named Biscuit", 5)], store)
    out = extractor.apply([_item("User owns a dog called Biscuit", 8)], store)

    assert len(store.get_all("default")) == 1
    # The returned object is the stored, updated memory, not the stale one.
    assert out[0].text == "User owns a dog called Biscuit" and out[0].importance == 8


def test_weaker_duplicate_is_skipped(store, monkeypatch):
    enc = FakeEncoder(aliases={"dog Biscuit": "User has a dog named Biscuit"})
    monkeypatch.setattr(retriever, "encode", enc)
    extractor.apply([_item("User has a dog named Biscuit", 8)], store)
    assert extractor.apply([_item("dog Biscuit", 3)], store) == []
    assert store.get_all("default")[0].text == "User has a dog named Biscuit"


def test_invalid_items_are_counted_not_stored(store, monkeypatch):
    monkeypatch.setattr(retriever, "encode", FakeEncoder())
    before = extractor.stats["invalid_items"]
    extractor.apply([{"text": "x", "type": "NOT_A_TYPE"}, {"nope": 1}], store)
    assert extractor.stats["invalid_items"] == before + 2
    assert store.get_all("default") == []


def test_encode_called_once_per_candidate_not_per_stored_memory(store, monkeypatch):
    enc = FakeEncoder()
    monkeypatch.setattr(retriever, "encode", enc)
    extractor.apply([_item(f"stored fact {i}") for i in range(20)], store)
    enc.calls.clear()

    extractor.apply([_item("new fact one"), _item("new fact two")], store)
    assert len(enc.calls) == 2
