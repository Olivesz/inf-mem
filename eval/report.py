"""
Render the README results block from the committed result files.

    python eval/report.py           # rewrite the block in README.md
    python eval/report.py --check   # exit 1 if README.md is out of date

tests/test_readme.py runs the check, so a number in the README cannot drift
from the file it came from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def render() -> str:
    r = _load("eval/results/locomo-retrieval.json")
    d = _load("eval/results/dedupe-calibration.json")
    ev = r["evaluation_split"]
    obs = ev["observations"]
    gate, old = obs["gate_at_code_threshold"], obs["gate_at_old_threshold"]
    g, n, o = obs["with_code_threshold"], obs["no_gate"], obs["with_old_threshold_0_35"]
    dc, de = d["calibration_split"], d["evaluation_split"]
    bench = sorted((ROOT / "bench").glob("results-*.json"))

    lines = [
        START,
        (f"**Retrieval** on [LoCoMo](https://github.com/snap-research/locomo), real embedding model, no LLM. "
         f"The gate was chosen on {r['calibration_split']['questions']} questions from five conversations; "
         f"these numbers are from the other five ({ev['questions']} questions, "
         f"{obs['memories']:,} fact-style memories), which played no part in the choice. "
         "Method and caveats: [eval/calibration.md](eval/calibration.md)."),
        "",
        "| Retrieval gate | Recall@5 | Off-topic prompts that still inject memories |",
        "|---|---|---|",
        f"| none | {n['recall_at_5']:.3f} | all of them (no gate) |",
        f"| 0.35 (old, hand-picked) | {o['recall_at_5']:.3f} | {old['off_topic_pass']:.0%} |",
        f"| **{gate['threshold']} (calibrated)** | **{g['recall_at_5']:.3f}** | **{gate['off_topic_pass']:.1%}** |",
        "",
        (f"**Dedupe**: on {dc['labelled_pairs'] + de['labelled_pairs']} hand-labelled pairs, no cosine threshold that merged anything "
         "was at least 90% correct on the calibration pairs, so only near-verbatim restatements merge "
         f"(cosine ≥ {de['code_threshold']}). On the evaluation pairs, {1 - de['at_old_threshold_0_85']['precision']:.0%} "
         f"of the merges the old {de['at_old_threshold_0_85']['threshold']} gate made joined two different facts."),
        "",
        "**Latency** through the full HTTP stack, real model and store, Groq stubbed out:",
        "",
        "| Machine | Memories | /retrieve p50 / p95 | /extract p50 / p95 |",
        "|---|---|---|---|",
    ]
    for path in bench:
        b = json.loads(path.read_text())
        for row in b["results"]:
            lines.append(
                f"| {b['machine']['cpu']} | {row['memories']:,} | "
                f"{row['retrieve']['p50_ms']:.0f} / {row['retrieve']['p95_ms']:.0f} ms | "
                f"{row['extract']['p50_ms']:.0f} / {row['extract']['p95_ms']:.0f} ms |")
    lines.append(END)
    return "\n".join(lines)


def current() -> str:
    text = README.read_text()
    return text[text.index(START): text.index(END) + len(END)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        ok = current() == render()
        print("README results block is " + ("current" if ok else "OUT OF DATE: run python eval/report.py"))
        return 0 if ok else 1
    text = README.read_text()
    README.write_text(text.replace(current(), render()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
