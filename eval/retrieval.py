"""
Retrieval eval and gate calibration on LoCoMo, with the real embedding model.

Each LoCoMo conversation becomes one inf-mem store, built with the real
MemoryStore and queried through the real retriever.retrieve(). Two kinds of
memory are tested:

- observations: LoCoMo's fact statements ("Caroline attended an LGBTQ
  support group..."), the same shape as inf-mem's extracted memories;
- turns: the raw dialogue lines.

A memory is relevant to a question if it is (or cites) one of the question's
evidence turns. No LLM is called anywhere in this file.

Gate calibration. The retrieval gate should pass memories for questions about
this user and pass nothing for questions about someone else. For every
question we score:
  on-topic:  its best evidence memory, in its own conversation's store;
  off-topic: its best memory in the next conversation's store (a different
             pair of people), where nothing is relevant.
The threshold is the value maximising (on-topic pass rate - off-topic pass
rate), swept in steps of 0.005. See eval/calibration.md.

Usage:
    python eval/retrieval.py            # write eval/results/locomo-retrieval.json
    python eval/retrieval.py --check    # recompute and compare to the committed file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE))

import compare
import locomo
import retriever
from memory_store import MemoryStore
from models import Memory, MemoryType

MODEL_ID = "BAAI/bge-small-en-v1.5"
CACHE = HERE / ".cache" / "embeddings.npz"
RESULTS = HERE / "results" / "locomo-retrieval.json"
K = 5
OLD_THRESHOLD = 0.35  # the hand-picked gate this calibration replaced
SWEEP = np.round(np.arange(0.30, 0.95 + 1e-9, 0.005), 3)


def _h(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def embed_all(texts: list) -> dict:
    """Embed texts with the real model, reusing a local on-disk cache."""
    cache = {}
    if CACHE.exists():
        with np.load(CACHE) as z:
            cache = {k: z[k] for k in z.files}
    todo = sorted({t for t in texts if _h(t) not in cache})
    if todo:
        model = retriever._get_model()
        for t, v in zip(todo, model.embed(todo, batch_size=64), strict=True):
            cache[_h(t)] = np.asarray(v, dtype=np.float32)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **cache)
    return {t: cache[_h(t)] for t in texts}


def build_store(memories: list, vec: dict, root: Path) -> tuple:
    """Load memories into a real MemoryStore; return it and id -> memory map."""
    store = MemoryStore(data_dir=root)
    by_id = {}
    for i, m in enumerate(memories):
        mem = Memory(id=f"m{i:05d}", user_id="eval", text=m.text, type=MemoryType.FACT,
                     keywords=[], importance=5)
        store.add_memory(mem, vec[m.text])
        by_id[mem.id] = m
    return store, by_id


def gate_rates(convs: list, corpus: str, vec: dict) -> dict:
    """Per-question best on-topic and off-topic cosines."""
    on, off = [], []
    for i, conv in enumerate(convs):
        own = getattr(conv, corpus)
        other = getattr(convs[(i + 1) % len(convs)], corpus)
        own_m = np.stack([vec[m.text] for m in own])
        other_m = np.stack([vec[m.text] for m in other])
        for q in conv.questions:
            rel = [bool(q.evidence & m.evidence) for m in own]
            qv = vec[retriever.QUERY_INSTRUCTION + q.text]
            s_own = retriever.cosine_scores(qv, own_m)
            rel = np.array(rel)
            on.append(float(s_own[rel].max()) if rel.any() else -1.0)
            off.append(float(retriever.cosine_scores(qv, other_m).max()))
    return {"on": np.array(on), "off": np.array(off)}


def sweep(rates: dict) -> list:
    rows = []
    for t in SWEEP:
        tpr = float((rates["on"] >= t).mean())
        fpr = float((rates["off"] >= t).mean())
        rows.append({"threshold": float(t), "on_topic_pass": round(tpr, 4),
                     "off_topic_pass": round(fpr, 4), "j": round(tpr - fpr, 4)})
    return rows


def recall_at_k(convs: list, corpus: str, vec: dict, threshold: float) -> dict:
    """Run every question through retriever.retrieve on the real store."""
    per_cat: dict = {}
    recalls, hits, empties = [], [], 0
    # retrieve() calls encode_query -> encode(instruction + prompt); serve
    # those vectors from the cache instead of re-running the model.
    original_encode = retriever.encode
    retriever.encode = lambda text: vec[text]
    try:
        for conv in convs:
            with tempfile.TemporaryDirectory() as d:
                store, by_id = build_store(getattr(conv, corpus), vec, Path(d))
                for q in conv.questions:
                    got = retriever.retrieve(q.text, store, user_id="eval", limit=K, threshold=threshold)
                    found = set().union(*(by_id[m.id].evidence for m in got)) if got else set()
                    r = len(q.evidence & found) / len(q.evidence)
                    recalls.append(r)
                    hits.append(float(r > 0))
                    empties += not got
                    per_cat.setdefault(locomo.ANSWERABLE[q.category], []).append(r)
    finally:
        retriever.encode = original_encode
    return {
        "recall_at_5": round(float(np.mean(recalls)), 4),
        "hit_at_5": round(float(np.mean(hits)), 4),
        "questions_with_no_memory_injected": empties,
        "recall_at_5_by_category": {k: round(float(np.mean(v)), 4) for k, v in sorted(per_cat.items())},
    }


def run() -> dict:
    path = locomo.fetch()
    convs = locomo.load(path)
    texts = [m.text for c in convs for m in c.turns + c.observations] + \
            [retriever.QUERY_INSTRUCTION + q.text for c in convs for q in c.questions]
    vec = embed_all(texts)

    current = retriever.RETRIEVE_THRESHOLD
    out = {
        "benchmark": "LoCoMo (locomo10.json)",
        "dataset_sha256": locomo.SHA256,
        "dataset_commit": locomo.COMMIT,
        "embedding_model": MODEL_ID,
        "k": K,
        "query_instruction": retriever.QUERY_INSTRUCTION,
        "questions": sum(len(c.questions) for c in convs),
        "unresolved_evidence_ids_dropped": sum(c.unresolved_evidence for c in convs),
        "retrieve_threshold_in_code": current,
        "corpora": {},
    }
    for corpus in ("observations", "turns"):
        rates = gate_rates(convs, corpus, vec)
        rows = sweep(rates)
        best = max(rows, key=lambda r: (r["j"], -r["threshold"]))
        at_code = next(r for r in rows if abs(r["threshold"] - current) < 1e-9)
        out["corpora"][corpus] = {
            "memories": sum(len(getattr(c, corpus)) for c in convs),
            # Upper bound on recall: questions with at least one relevant memory.
            "questions_with_a_relevant_memory": int((rates["on"] > -1).sum()),
            "gate": {"best": best, "at_code_threshold": at_code, "sweep": rows},
            "no_gate": recall_at_k(convs, corpus, vec, threshold=-1.0),
            "with_code_threshold": recall_at_k(convs, corpus, vec, threshold=current),
            "with_old_threshold_0_35": recall_at_k(convs, corpus, vec, threshold=OLD_THRESHOLD),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the committed results")
    args = ap.parse_args()
    result = run()
    if args.check:
        return compare.check(RESULTS, result)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(text)
    for corpus, r in result["corpora"].items():
        print(corpus, json.dumps({k: r[k] for k in ("no_gate", "with_code_threshold")}),
              "best gate:", r["gate"]["best"], "at code:", r["gate"]["at_code_threshold"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
