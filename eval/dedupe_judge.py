"""
Measure the LLM merge judge on the hand-labelled LoCoMo pairs.

Each labelled pair (a, b) is put to extractor.judge() as "new memory a,
stored memory b". The judge is right when it says "same" exactly for the
pairs labelled same_fact. Precision and recall use the same stratum weights
as eval/dedupe.py.

Decision rule, fixed before any run: turn USE_MERGE_JUDGE on only if the
judge's weighted merge precision on the calibration half is at least 0.90,
the bar the cosine gate failed. The evaluation half is reported either way.

Every reply is cached in eval/cache/dedupe-judge.jsonl under a hash of
(model, prompt), so --offline re-scores the committed replies without a key.

    GROQ_API_KEY=... python eval/dedupe_judge.py    # call Groq for missing pairs, then score
    python eval/dedupe_judge.py --offline           # score from the cache only
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

import config
import dedupe
import extractor
import locomo
import retrieval

CACHE = HERE / "cache" / "dedupe-judge.jsonl"
RESULTS = HERE / "results" / "dedupe-judge.json"


class Recorder:
    """Wraps a client (or none) so judge() calls are served from or added to the cache."""

    def __init__(self, client, cache: dict):
        self.client, self.cache, self.missing = client, cache, 0
        outer = self

        class _Completions:
            def create(self, **kw):
                key = hashlib.sha256(json.dumps([kw["model"], kw["messages"]], sort_keys=True).encode()).hexdigest()
                if key not in outer.cache:
                    if outer.client is None:
                        outer.missing += 1
                        content = ""
                    else:
                        content = outer.client.chat.completions.create(**kw).choices[0].message.content or ""
                        outer.cache[key] = content
                        with CACHE.open("a") as f:
                            f.write(json.dumps({"key": key, "model": kw["model"], "reply": content}) + "\n")
                msg = type("M", (), {"content": outer.cache.get(key, "")})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})

        self.chat = type("Chat", (), {"completions": _Completions()})()


def score(pairs: list, convs: list, vec: dict, rec: Recorder) -> dict:
    ids = {c.sample_id for c in convs}
    mine = [p for p in pairs if p["conversation"] in ids]
    pop = dedupe.population(vec, convs)
    sampled = {k: sum(p["stratum"] == k for p in mine) for k in pop}
    w = np.array([pop[p["stratum"]] / sampled[p["stratum"]] for p in mine])
    same = np.array([p["same_fact"] for p in mine])
    said = np.array([extractor.judge(rec, p["a"], [p["b"]]) == 0 for p in mine])
    tp, fp = float((w * (said & same)).sum()), float((w * (said & ~same)).sum())
    return {"labelled_pairs": len(mine),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / float((w * same).sum()), 4),
            "unweighted_correct": int((said == same).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    client = None
    if not args.offline:
        key = config.load_groq_api_key()
        if key is None:
            print("No Groq key configured (GROQ_API_KEY or config.yaml). Use --offline to score the cache.")
            return 2
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=config.GROQ_BASE_URL)

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            row = json.loads(line)
            cache[row["key"]] = row["reply"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    rec = Recorder(client, cache)

    convs = locomo.load(locomo.fetch())
    pairs = dedupe.load_pairs(convs)
    vec = retrieval.embed_all([o.text for c in convs for o in c.observations])
    cal = score(pairs, convs[locomo.CALIBRATION], vec, rec)
    ev = score(pairs, convs[locomo.EVALUATION], vec, rec)
    if rec.missing:
        print(f"{rec.missing} pairs have no cached reply; run with a key first.")
        return 2

    result = {
        "model": config.GROQ_MODEL,
        "rule": "enable the judge only if calibration precision >= 0.90",
        "calibration_split": cal,
        "evaluation_split": ev,
        "enable": cal["precision"] is not None and cal["precision"] >= 0.90,
        "judge_failures": extractor.stats["judge_failures"],
    }
    RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
