"""
Live check of extraction against the real Groq model, on 20 fixed LoCoMo turns.

The turns are the first 20 in conv-44 (evaluation half) that a LoCoMo
observation cites. Each turn is sent as the prompt and the next turn as the
reply, the way the extension sends a prompt and its answer. The run fails if
any reply cannot be parsed or names an invalid memory type. It also reports,
per turn, the best cosine between any extracted memory and the turn's
LoCoMo observations: how close extraction lands to the benchmark's own facts.

    GROQ_API_KEY=... python eval/live_extract.py   # writes eval/results/live-extract.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE))

import config
import extractor
import locomo
import retriever
from models import ExtractedMemory, MemoryType

CONVERSATION = "conv-44"
N = 20
RESULTS = HERE / "results" / "live-extract.json"


def main() -> int:
    key = config.load_groq_api_key()
    if key is None:
        print("::error::No Groq key: set the GROQ_API_KEY Actions secret (or config.yaml locally).")
        return 2
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=config.GROQ_BASE_URL)

    conv = next(c for c in locomo.load(locomo.fetch()) if c.sample_id == CONVERSATION)
    cited = {}
    for o in conv.observations:
        for d in o.evidence:
            cited.setdefault(d, []).append(o.text)
    picks = [i for i, t in enumerate(conv.turns[:-1]) if t.key in cited][:N]

    rows, failures = [], []
    for i in picks:
        turn, reply = conv.turns[i], conv.turns[i + 1]
        raw = extractor.call_model(client, turn.text, reply.text)
        try:
            items = [ExtractedMemory(**x) for x in extractor.parse_reply(raw)]
            for m in items:
                MemoryType(m.type.upper())
        except (ValueError, TypeError) as e:
            failures.append({"turn": turn.key, "error": str(e)[:200]})
            continue
        best = None
        if items:
            mem = np.stack([retriever.encode(m.text) for m in items])
            best = max(float(retriever.cosine_scores(retriever.encode(o), mem).max()) for o in cited[turn.key])
        rows.append({"turn": turn.key, "memories": len(items), "best_cosine_to_observation": best})
        time.sleep(0.5)  # stay well inside the free tier's rate limit

    scored = [r["best_cosine_to_observation"] for r in rows if r["best_cosine_to_observation"] is not None]
    result = {
        "model": config.GROQ_MODEL,
        "date": time.strftime("%Y-%m-%d"),
        "turns": len(picks),
        "parse_or_type_failures": failures,
        "turns_with_no_memory": sum(r["memories"] == 0 for r in rows),
        "mean_best_cosine_to_observation": round(float(np.mean(scored)), 4) if scored else None,
        "per_turn": rows,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_turn"}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
