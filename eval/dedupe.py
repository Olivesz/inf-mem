"""
Dedupe gate calibration on hand-labelled LoCoMo observation pairs.

LoCoMo's observations are fact statements extracted per session, so the same
fact often recurs across sessions in different words, next to different facts
that share most of their words. Pairs of observations from one conversation
with cosine >= 0.86 were sampled (all 119 pairs >= 0.92, plus 60 of the pairs
in [0.86, 0.92) with numpy seed 0) and labelled by hand in
eval/dedupe_pairs.jsonl. That file holds observation keys, not text (LoCoMo is
CC BY-NC and not redistributed here), plus a hash of each pair's text so a
label can never silently attach to different sentences. The labelling rule is
in eval/calibration.md.

The lower stratum is sampled, so each of its pairs stands for
(pairs in stratum / pairs sampled) pairs; precision and recall are weighted
accordingly. Recall is over same-fact pairs with cosine >= 0.86 only.

Merging two different facts overwrites one of them, which loses data, while
a missed merge only costs a duplicate. So the threshold is the lowest value
whose weighted merge precision is at least MIN_PRECISION.

Usage:
    python eval/dedupe.py            # write eval/results/dedupe-calibration.json
    python eval/dedupe.py --check    # recompute and compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE))

import compare
import extractor
import locomo
import retrieval

PAIRS = HERE / "dedupe_pairs.jsonl"
RESULTS = HERE / "results" / "dedupe-calibration.json"
SWEEP = np.round(np.arange(0.86, 0.995 + 1e-9, 0.005), 3)
MIN_PRECISION = 0.90


def population(vec: dict, convs: list) -> dict:
    """How many within-conversation observation pairs fall in each stratum."""
    counts = {"ge_0.92": 0, "0.86_0.92": 0}
    for c in convs:
        m = np.stack([vec[o.text] for o in c.observations])
        s = (m @ m.T)[np.triu_indices(len(m), 1)]
        counts["ge_0.92"] += int((s >= 0.92).sum())
        counts["0.86_0.92"] += int(((s >= 0.86) & (s < 0.92)).sum())
    return counts


def metrics(cos: np.ndarray, same: np.ndarray, w: np.ndarray, t: float) -> dict:
    merged = cos >= t
    tp = float((w * (merged & same)).sum())
    fp = float((w * (merged & ~same)).sum())
    return {"threshold": float(t),
            "precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
            "recall": round(tp / float((w * same).sum()), 4),
            "merged_pairs_weighted": round(tp + fp, 2)}


def split_arrays(pairs: list, convs: list, vec: dict) -> dict:
    """Cosines, labels and stratum weights for the pairs from these conversations."""
    ids = {c.sample_id for c in convs}
    mine = [p for p in pairs if p["conversation"] in ids]
    pop = population(vec, convs)
    sampled = {k: sum(p["stratum"] == k for p in mine) for k in pop}
    weight = {k: pop[k] / sampled[k] for k in pop}
    return {
        "cos": np.array([float(vec[p["a"]] @ vec[p["b"]]) for p in mine]),
        "same": np.array([p["same_fact"] for p in mine]),
        "w": np.array([weight[p["stratum"]] for p in mine]),
        "summary": {"conversations": sorted(ids), "labelled_pairs": len(mine),
                    "labelled_same_fact": sum(p["same_fact"] for p in mine),
                    "population_pairs": pop, "stratum_weight": {k: round(v, 4) for k, v in weight.items()}},
    }


def load_pairs(convs: list) -> list:
    """Labelled pairs with their observation text filled in and hash-checked."""
    pairs = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]
    text = {(c.sample_id, o.key): o.text for c in convs for o in c.observations}
    for p in pairs:
        p["a"], p["b"] = text[(p["conversation"], p["a"])], text[(p["conversation"], p["b"])]
        digest = hashlib.sha1(f"{p['a']}\n{p['b']}".encode()).hexdigest()[:12]
        if digest != p["text_sha1"]:
            raise RuntimeError(f"label text changed for {p['conversation']} {p['text_sha1']}")
    return pairs


def run() -> dict:
    convs = locomo.load(locomo.fetch())
    pairs = load_pairs(convs)
    vec = retrieval.embed_all([o.text for c in convs for o in c.observations])

    cal = split_arrays(pairs, convs[locomo.CALIBRATION], vec)
    ev = split_arrays(pairs, convs[locomo.EVALUATION], vec)
    sweep = [metrics(cal["cos"], cal["same"], cal["w"], t) for t in SWEEP]
    best = next(r for r in sweep if r["precision"] >= MIN_PRECISION)
    code = extractor.DEDUPE_THRESHOLD
    return {
        "benchmark": "LoCoMo observations (locomo10.json)",
        "dataset_sha256": locomo.SHA256,
        "embedding_model": retrieval.MODEL_ID,
        "criterion": f"lowest threshold with weighted merge precision >= {MIN_PRECISION} on the calibration split",
        "calibration_split": {**cal["summary"], "chosen_threshold": best["threshold"],
                              "at_chosen": best, "sweep": sweep},
        "evaluation_split": {**ev["summary"], "code_threshold": code,
                             "at_code_threshold": metrics(ev["cos"], ev["same"], ev["w"], code),
                             "at_old_threshold_0_85": metrics(ev["cos"], ev["same"], ev["w"], 0.85)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    result = run()
    if args.check:
        return compare.check(RESULTS, result)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    RESULTS.write_text(text)
    c, e = result["calibration_split"], result["evaluation_split"]
    print("calibration chose", c["at_chosen"], "n", c["labelled_pairs"])
    print("evaluation at code", e["at_code_threshold"], "at 0.85", e["at_old_threshold_0_85"], "n", e["labelled_pairs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
