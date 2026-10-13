"""Tier 3: the calibrated gates with the real embedding model.

These load BAAI/bge-small-en-v1.5 through fastembed (about 130 MB, cached
after the first run). Run them with ``pytest -m model``.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import extractor
import retriever
from memory_store import MemoryStore
from models import Memory, MemoryType

RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"

PROBE_MEMORIES = [
    "User plays the cello in a community orchestra",
    "User is writing a compiler in Rust",
    "User prefers concise answers without bullet points",
    "User has a dog named Biscuit",
    "User spent a week debugging SQLite locking errors",
]


def test_retrieve_threshold_matches_calibration():
    """No model needed: the constant in code must be the calibrated one."""
    data = json.loads((RESULTS / "locomo-retrieval.json").read_text())
    cal = data["calibration_split"]
    assert retriever.RETRIEVE_THRESHOLD == cal["chosen_threshold"]
    variant = {"bge_instruction": retriever.QUERY_INSTRUCTION != "", "plain": retriever.QUERY_INSTRUCTION == ""}
    assert variant[cal["chosen_variant"]]
    assert data["evaluation_split"]["code_threshold"] == cal["chosen_threshold"]


@pytest.fixture(scope="module")
def probe_store(tmp_path_factory):
    store = MemoryStore(data_dir=tmp_path_factory.mktemp("probe"))
    for text in PROBE_MEMORIES:
        mem = Memory(id=str(uuid4()), user_id="u", text=text, type=MemoryType.FACT, importance=5)
        store.add_memory(mem, retriever.encode(text))
    return store


@pytest.mark.model
@pytest.mark.parametrize("prompt", [
    "What is the capital of Australia?",
    "Give me a recipe for banana bread",
])
def test_unrelated_prompt_retrieves_nothing(probe_store, prompt):
    assert retriever.retrieve(prompt, probe_store, user_id="u") == []


@pytest.mark.model
def test_related_prompt_retrieves_its_memory(probe_store):
    got = retriever.retrieve("Why does my lexer in Rust panic on unicode identifiers?",
                             probe_store, user_id="u")
    assert [m.text for m in got][:1] == ["User is writing a compiler in Rust"]


@pytest.mark.model
def test_dog_and_cat_named_biscuit_stay_two_memories(tmp_path):
    store = MemoryStore(data_dir=tmp_path)
    extractor.apply([{"text": "User has a dog named Biscuit", "type": "FACT", "importance": 5}], store)
    extractor.apply([{"text": "User has a cat named Biscuit", "type": "FACT", "importance": 5}], store)
    assert len(store.get_all("default")) == 2


@pytest.mark.model
def test_near_verbatim_restatement_still_merges(tmp_path):
    store = MemoryStore(data_dir=tmp_path)
    extractor.apply([{"text": "User has a dog named Biscuit", "type": "FACT", "importance": 5}], store)
    extractor.apply([{"text": "The user has a dog named Biscuit", "type": "FACT", "importance": 6}], store)
    assert len(store.get_all("default")) == 1


@pytest.mark.model
def test_real_vectors_are_unit_length():
    v = retriever.encode("hello")
    assert v.shape == (384,) and np.isclose(np.linalg.norm(v), 1.0, atol=1e-3)


def test_dedupe_threshold_matches_calibration():
    """No model needed: the constant in code must be the calibrated one."""
    data = json.loads((RESULTS / "dedupe-calibration.json").read_text())
    chosen = data["calibration_split"]["chosen_threshold"]
    assert extractor.DEDUPE_THRESHOLD == chosen
    assert data["evaluation_split"]["code_threshold"] == chosen
