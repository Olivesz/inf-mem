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

Split. The first five conversations (locomo.CALIBRATION) choose the gate;
the last five (locomo.EVALUATION) are only measured, with whatever the code
ships. No number reported for the evaluation split fed into a choice.

Gate calibration. The retrieval gate should pass memories for questions about
this user and pass nothing for questions about someone else. For every
question we score:
  on-topic:  its best evidence memory, in its own conversation's store;
  off-topic: its best memory in the next conversation's store (a different
             pair of people, same split), where nothing is relevant.
The threshold maximises (on-topic pass rate - off-topic pass rate), swept in
steps of 0.005, on the observations store. The same rule decides whether
queries get bge's retrieval instruction. See eval/calibration.md.

Usage:
    python eval/retrieval.py            # write eval/results/locomo-retrieval.json
    python eval/retrieval.py --check    # recompute and compare to the committed file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "
QUERY_VARIANTS = {"bge_instruction": BGE_INSTRUCTION, "plain": ""}


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


def gate_rates(convs: list, corpus: str, vec: dict, prefix: str) -> dict:
    """Per-question best on-topic and off-topic cosines (-1 if no relevant memory)."""
    on, off = [], []
    for i, conv in enumerate(convs):
        own = getattr(conv, corpus)
        other = getattr(convs[(i + 1) % len(convs)], corpus)
        own_m = np.stack([vec[m.text] for m in own])
        other_m = np.stack([vec[m.text] for m in other])
        for q in conv.questions:
            qv = vec[prefix + q.text]
            rel = np.array([bool(q.evidence & m.evidence) for m in own])
            s_own = retriever.cosine_scores(qv, own_m)
            on.append(float(s_own[rel].max()) if rel.any() else -1.0)
            off.append(float(retriever.cosine_scores(qv, other_m).max()))
    return {"on": np.array(on), "off": np.array(off)}


def at(rates: dict, t: float) -> dict:
    tpr = float((rates["on"] >= t).mean())
    fpr = float((rates["off"] >= t).mean())
    return {"threshold": float(t), "on_topic_pass": round(tpr, 4),
            "off_topic_pass": round(fpr, 4), "j": round(tpr - fpr, 4)}


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


def calibrate(cal: list, vec: dict) -> dict:
    """Choose query variant and threshold on the calibration split only."""
    out = {}
    for name, prefix in QUERY_VARIANTS.items():
        rates = gate_rates(cal, "observations", vec, prefix)
        sweep = [at(rates, t) for t in SWEEP]
        out[name] = {"best": max(sweep, key=lambda r: (r["j"], -r["threshold"])), "sweep": sweep}
    # Ties go to the simpler plain query.
    chosen = max(QUERY_VARIANTS, key=lambda n: (out[n]["best"]["j"], n == "plain"))
    return {"variants": out, "chosen_variant": chosen,
            "chosen_threshold": out[chosen]["best"]["threshold"]}


def evaluate(ev: list, vec: dict) -> dict:
    """Measure what the code ships, on the evaluation split."""
    prefix, current = retriever.QUERY_INSTRUCTION, retriever.RETRIEVE_THRESHOLD
    out = {}
    for corpus in ("observations", "turns"):
        rates = gate_rates(ev, corpus, vec, prefix)
        out[corpus] = {
            "memories": sum(len(getattr(c, corpus)) for c in ev),
            # Ceiling on on-topic pass: questions with at least one relevant memory.
            "questions_with_a_relevant_memory": int((rates["on"] > -1).sum()),
            "gate_at_code_threshold": at(rates, current),
            "gate_at_old_threshold": at(rates, OLD_THRESHOLD),
            "no_gate": recall_at_k(ev, corpus, vec, threshold=-1.0),
            "with_old_threshold_0_35": recall_at_k(ev, corpus, vec, threshold=OLD_THRESHOLD),
            "with_code_threshold": recall_at_k(ev, corpus, vec, threshold=current),
        }
    return out


def run() -> dict:
    convs = locomo.load(locomo.fetch())
    cal, ev = convs[locomo.CALIBRATION], convs[locomo.EVALUATION]
    texts = [m.text for c in convs for m in c.turns + c.observations]
    texts += [p + q.text for c in convs for q in c.questions for p in QUERY_VARIANTS.values()]
    vec = embed_all(texts)
    return {
        "benchmark": "LoCoMo (locomo10.json)",
        "dataset_sha256": locomo.SHA256,
        "dataset_commit": locomo.COMMIT,
        "embedding_model": MODEL_ID,
        "k": K,
        "unresolved_evidence_ids_dropped": sum(c.unresolved_evidence for c in convs),
        "calibration_split": {"conversations": [c.sample_id for c in cal],
                              "questions": sum(len(c.questions) for c in cal),
                              **calibrate(cal, vec)},
        "evaluation_split": {"conversations": [c.sample_id for c in ev],
                             "questions": sum(len(c.questions) for c in ev),
                             "code_query_instruction": retriever.QUERY_INSTRUCTION,
                             "code_threshold": retriever.RETRIEVE_THRESHOLD,
                             **evaluate(ev, vec)},
    }


def check(result: dict) -> int:
    """
    Compare a fresh run with the committed results.

    The embedding model is quantized ONNX, whose output differs slightly
    between CPU architectures. On the calibration half the best and second-best
    thresholds are one question apart (0.61 vs 0.605 here), so on another CPU
    the argmax can move one 0.005 step. The committed choice is a decision, not
    a measurement: this check requires that it is still optimal on this
    machine (its J within compare.TOL of the best J), and that every other
    number matches within compare.TOL.
    """
    committed = json.loads(RESULTS.read_text())
    cal = result["calibration_split"]
    variant, chosen = committed["calibration_split"]["chosen_variant"], committed["calibration_split"]["chosen_threshold"]
    problems = []
    best_here = {n: v["best"]["j"] for n, v in cal["variants"].items()}
    if max(best_here, key=best_here.get) != variant and best_here[variant] < max(best_here.values()) - compare.TOL:
        problems.append(f"query variant {variant} is no longer best here: {best_here}")
    sweep = cal["variants"][variant]["sweep"]
    at_chosen = next(r for r in sweep if abs(r["threshold"] - chosen) < 1e-9)
    print(f"shipped {chosen}: J {at_chosen['j']} here; best here {cal['variants'][variant]['best']}")
    if at_chosen["j"] < best_here[variant] - compare.TOL:
        problems.append(f"shipped threshold {chosen} has J {at_chosen['j']}, best here is {best_here[variant]}")

    def strip(d: dict) -> dict:
        d = json.loads(json.dumps(d))
        d["calibration_split"].pop("chosen_threshold")
        for v in d["calibration_split"]["variants"].values():
            v.pop("best")
        return d

    problems += compare.diffs(strip(committed), strip(result))
    out = Path(os.environ.get("RECOMPUTED_OUT", "")) if os.environ.get("RECOMPUTED_OUT") else None
    if out:
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if problems:
        print(f"MISMATCH with {RESULTS.name}:")
        print("\n".join(problems[:20]))
        return 1
    print(f"match: {RESULTS.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the committed results")
    args = ap.parse_args()
    result = run()
    if args.check:
        return check(result)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    c = result["calibration_split"]
    print("calibration chose:", c["chosen_variant"], c["chosen_threshold"],
          {n: v["best"] for n, v in c["variants"].items()})
    for corpus in ("observations", "turns"):
        e = result["evaluation_split"][corpus]
        print("eval", corpus, e["questions_with_a_relevant_memory"], e["gate_at_code_threshold"],
              "old:", e["gate_at_old_threshold"])
        for k in ("no_gate", "with_old_threshold_0_35", "with_code_threshold"):
            print("   ", k, e[k]["recall_at_5"], e[k]["hit_at_5"], e[k]["questions_with_no_memory_injected"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
