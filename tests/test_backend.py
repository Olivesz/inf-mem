"""Unit tests for the inf-mem backend: models, store, retriever, compressor.

No network and no embedding-model download: the LLM client is faked and the
embedding function is monkeypatched. The MemoryStore is redirected to a tmp
directory so nothing touches ~/.inf_mem.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from openai import OpenAIError
from pydantic import ValidationError

import compressor
import retriever
from memory_store import MemoryStore
from models import Memory, MemoryType, RetrieveRequest

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def store(tmp_path, monkeypatch) -> MemoryStore:
    """A MemoryStore rooted in a temp dir (never touches the real home dir)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return MemoryStore()


def _mem(text: str, importance: int = 5, mtype: MemoryType = MemoryType.FACT) -> Memory:
    return Memory(id=str(uuid4()), user_id="u1", text=text, type=mtype,
                  keywords=["k"], importance=importance)


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(384).astype(np.float32)


# --------------------------------------------------------------------------- #
# models.py — pydantic validation
# --------------------------------------------------------------------------- #

def test_memory_importance_bounds_enforced():
    with pytest.raises(ValidationError):
        _mem("x", importance=0)
    with pytest.raises(ValidationError):
        _mem("x", importance=11)
    ok = _mem("x", importance=10)
    assert ok.importance == 10


def test_retrieve_request_threshold_bounds():
    with pytest.raises(ValidationError):
        RetrieveRequest(prompt="q", threshold=1.5)
    r = RetrieveRequest(prompt="q")
    assert r.limit == 5 and r.threshold is None


def test_memory_type_enum_roundtrip():
    assert MemoryType("PREFERENCE") is MemoryType.PREFERENCE


# --------------------------------------------------------------------------- #
# memory_store.py — persistence round-trips
# --------------------------------------------------------------------------- #

def test_add_and_get_by_id(store):
    m = _mem("Oliver likes Rust")
    store.add_memory(m, _vec(1))
    fetched = store.get_by_id(m.id)
    assert fetched is not None
    assert fetched.text == "Oliver likes Rust"
    assert fetched.type is MemoryType.FACT


def test_add_memory_rejects_bad_embedding_shape(store):
    with pytest.raises(ValueError):
        store.add_memory(_mem("bad"), np.zeros((10,), dtype=np.float32))


def test_get_all_orders_and_scopes_by_user(store):
    store.add_memory(_mem("a"), _vec(1))
    store.add_memory(_mem("b"), _vec(2))
    all_u1 = store.get_all("u1")
    assert len(all_u1) == 2
    assert store.get_all("someone_else") == []


def test_update_memory_changes_text_and_embedding(store):
    m = _mem("old")
    store.add_memory(m, _vec(1))
    store.update_memory(m.id, "new text", ["kw"], 9, _vec(2))
    updated = store.get_by_id(m.id)
    assert updated.text == "new text"
    assert updated.importance == 9


def test_delete_memory_removes_row_and_vector(store):
    m = _mem("temp")
    store.add_memory(m, _vec(1))
    store.delete_memory(m.id)
    assert store.get_by_id(m.id) is None
    assert store.get_embeddings().shape[0] == 0


def test_increment_access(store):
    m = _mem("counted")
    store.add_memory(m, _vec(1))
    store.increment_access(m.id)
    assert store.get_by_id(m.id).access_count == 1


def test_missing_memory_operations_raise(store):
    with pytest.raises(ValueError):
        store.delete_memory("nope")
    with pytest.raises(ValueError):
        store.increment_access("nope")


def test_store_persists_across_instances(store, tmp_path, monkeypatch):
    m = _mem("durable")
    store.add_memory(m, _vec(1))
    # New instance pointed at the same tmp home should reload the data.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    store2 = MemoryStore()
    assert store2.get_by_id(m.id) is not None
    assert store2.get_embeddings().shape == (1, 384)


# --------------------------------------------------------------------------- #
# retriever.py — similarity + retrieval
# --------------------------------------------------------------------------- #

def test_cosine_similarity_cases():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert retriever.cosine_similarity(a, a) == pytest.approx(1.0)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert retriever.cosine_similarity(a, b) == pytest.approx(0.0)
    assert retriever.cosine_similarity(a, -a) == pytest.approx(-1.0)
    zero = np.zeros(3, dtype=np.float32)
    assert retriever.cosine_similarity(a, zero) == 0.0


def test_retrieve_returns_relevant_memory(store, monkeypatch):
    target = _vec(42)
    other = _vec(7)
    m_relevant = _mem("the relevant one", importance=8)
    m_other = _mem("unrelated", importance=5)
    store.add_memory(m_relevant, target)
    store.add_memory(m_other, other)

    # Query encodes to exactly the relevant memory's vector -> similarity 1.0.
    monkeypatch.setattr(retriever, "encode", lambda text: target)
    results = retriever.retrieve("q", store, user_id="u1", limit=5, threshold=0.5)
    assert results and results[0].id == m_relevant.id


def test_retrieve_respects_limit(store, monkeypatch):
    vecs = [_vec(i) for i in range(5)]
    for i, v in enumerate(vecs):
        store.add_memory(_mem(f"m{i}"), v)
    monkeypatch.setattr(retriever, "encode", lambda text: vecs[0])
    # Low threshold so several match; limit caps the count.
    results = retriever.retrieve("q", store, user_id="u1", limit=2, threshold=-1.0)
    assert len(results) == 2


def test_retrieve_empty_store_returns_empty(store, monkeypatch):
    monkeypatch.setattr(retriever, "encode", lambda text: _vec(1))
    assert retriever.retrieve("q", store, user_id="u1") == []


# --------------------------------------------------------------------------- #
# compressor.py — token estimate + LLM call with a fake client
# --------------------------------------------------------------------------- #

def test_estimate_tokens_is_positive_and_monotonic():
    assert compressor._estimate_tokens("") >= 1
    assert compressor._estimate_tokens("one two three four") > compressor._estimate_tokens("one")


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        msg = type("Msg", (), {"content": self._content})
        choice = type("Choice", (), {"message": msg})
        return type("Resp", (), {"choices": [choice]})


class _FakeGroq:
    def __init__(self, content=None, raise_error=False):
        self.raise_error = raise_error
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()
        if raise_error:
            def _boom(**kwargs):
                raise OpenAIError("api down")
            self.chat.completions.create = _boom


def test_compress_happy_path():
    client = _FakeGroq(content="short version")
    out = compressor.compress("please make this a lot shorter thanks", client)
    assert out["compressed"] == "short version"
    assert out["original_tokens"] >= 1
    assert out["ratio"] > 0


def test_compress_falls_back_on_api_error():
    client = _FakeGroq(raise_error=True)
    prompt = "unchanged prompt"
    out = compressor.compress(prompt, client)
    assert out["compressed"] == prompt
    assert out["ratio"] == 1.0


def test_add_many_matches_add_memory(store):
    mems = [_mem(f"bulk {i}") for i in range(3)]
    store.add_many(mems, np.stack([_vec(i) for i in range(3)]))
    assert [m.id for m in store.get_all("u1")] and len(store.get_all("u1")) == 3
    ids, matrix = store.get_user_vectors("u1")
    assert set(ids) == {m.id for m in mems} and matrix.shape == (3, 384)
    with pytest.raises(ValueError):
        store.add_many(mems, np.zeros((2, 384), dtype=np.float32))


def test_writes_never_rewrite_a_whole_file(store, monkeypatch):
    """Each insert used to np.save the whole matrix; now writes touch only their rows."""
    def boom(*a, **k):
        raise AssertionError("np.save called on a write path")
    monkeypatch.setattr(np, "save", boom)
    m = _mem("one")
    store.add_memory(m, _vec(1))
    store.add_many([_mem("two"), _mem("three")], np.stack([_vec(2), _vec(3)]))
    store.update_memory(m.id, "one!", [], 6, _vec(4))
    store.delete_memory(m.id)
    store.delete_all("u1")


def test_rows_stay_aligned_after_delete_update_and_reopen(store, tmp_path):
    mems = [_mem(f"m{i}") for i in range(5)]
    for i, m in enumerate(mems):
        store.add_memory(m, _vec(i))
    store.delete_memory(mems[1].id)
    store.update_memory(mems[3].id, "m3 new", [], 7, _vec(33))

    reopened = MemoryStore(data_dir=store.data_dir)
    want = {mems[0].id: _vec(0), mems[2].id: _vec(2), mems[3].id: _vec(33), mems[4].id: _vec(4)}
    ids, matrix = reopened.get_user_vectors("u1")
    assert set(ids) == set(want)
    for mid, row in zip(ids, matrix, strict=True):
        assert np.allclose(row, want[mid])


def test_old_npy_store_is_migrated(tmp_path):
    """A store written by the .npy version opens with its vectors intact."""
    import sqlite3

    data = tmp_path / ".inf_mem"
    old = MemoryStore(data_dir=data)
    mems = [_mem(f"old {i}") for i in range(3)]
    old.add_many(mems, np.stack([_vec(i) for i in range(3)]))
    # Recreate the old layout: vectors only in .npy files, none in the table.
    conn = sqlite3.connect(data / "memories.db")
    conn.execute("UPDATE memories SET embedding = NULL")
    conn.commit()
    conn.close()
    np.save(data / "embeddings.npy", np.stack([_vec(i) for i in range(3)]))
    np.save(data / "memory_ids.npy", np.array([m.id for m in mems], dtype=str))

    migrated = MemoryStore(data_dir=data)
    ids, matrix = migrated.get_user_vectors("u1")
    assert ids == [m.id for m in mems]
    assert np.allclose(matrix, np.stack([_vec(i) for i in range(3)]))
    assert not (data / "embeddings.npy").exists()
    assert (data / "embeddings.npy.migrated").exists()
    assert MemoryStore(data_dir=data).get_embeddings().shape == (3, 384)
