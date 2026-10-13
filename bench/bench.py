"""
Latency of POST /retrieve and POST /extract at 100, 1k and 10k memories.

Requests go through FastAPI's TestClient against the real app, the real
MemoryStore on disk and the real embedding model. Each store is filled with
LoCoMo observations and turns (repeated to reach 10k), embedded beforehand.

/extract normally waits on Groq. Here the Groq client is a stub that returns
a fixed reply naming two new facts, so the timing covers what inf-mem itself
does: parse, embed each candidate, compare with stored vectors, write.

Usage:
    python bench/bench.py                      # all sizes, writes bench/results-<machine>.json
    python bench/bench.py --sizes 1000 --out x.json
    python bench/bench.py --sizes 1000 --compare bench/results-ci.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "eval"), str(ROOT / "tests")]

from fastapi.testclient import TestClient

import locomo
import main
import retrieval
from fakes import FakeGroq
from models import Memory, MemoryType

SIZES = (100, 1000, 10000)
RETRIEVE_N = 200
EXTRACT_N = 50
WARMUP = 10
# --compare fails if a p95 exceeds this multiple of the baseline's.
MAX_SLOWDOWN = 2.0
# --scaling fails if /extract p50 at 10k memories exceeds this multiple of its
# p50 at 1k. A ratio, so it holds on any machine.
MAX_EXTRACT_SCALING = 2.0


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        text = Path("/proc/cpuinfo").read_text()
        return re.search(r"model name\s*:\s*(.+)", text).group(1).strip()
    except Exception:  # noqa: BLE001 - descriptive only
        return platform.processor() or "unknown"


def machine() -> dict:
    return {"cpu": cpu_name(), "os": f"{platform.system()} {platform.release()}",
            "arch": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "load_avg_1m_at_start": round(os.getloadavg()[0], 2)}


def pct(xs: list) -> dict:
    a = np.array(xs) * 1000
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
            "n": len(xs)}


def corpus() -> tuple:
    convs = locomo.load(locomo.fetch())
    texts = [m.text for c in convs for m in c.observations + c.turns]
    queries = [q.text for c in convs for q in c.questions]
    vec = retrieval.embed_all(texts)
    return texts, vec, queries


def fill(store, texts: list, vec: dict, n: int) -> None:
    picked = [texts[i % len(texts)] for i in range(n)]
    mems = [Memory(id=f"b{i:06d}", text=t, type=MemoryType.FACT, importance=1 + i % 10)
            for i, t in enumerate(picked)]
    store.add_many(mems, np.stack([vec[t] for t in picked]))


def run_size(n: int, texts: list, vec: dict, queries: list, profile: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as home:
        old_home = Path.home
        Path.home = staticmethod(lambda: Path(home))
        try:
            with TestClient(main.app) as client:
                fill(main.store, texts, vec, n)
                assert len(main.store.memory_ids) == n

                times = []
                for i, q in enumerate(queries[: WARMUP + RETRIEVE_N]):
                    t0 = time.perf_counter()
                    r = client.post("/retrieve", json={"prompt": q})
                    dt = time.perf_counter() - t0
                    assert r.status_code == 200
                    if i >= WARMUP:
                        times.append(dt)
                retrieve = pct(times)

                times = []
                for i in range(WARMUP + EXTRACT_N):
                    reply = json.dumps([
                        {"text": f"User finished reading book number {i} in the series",
                         "type": "EPISODE", "importance": 4},
                        {"text": f"User wants to visit city number {i} next year",
                         "type": "PROJECT", "importance": 5},
                    ])
                    main.groq_client = FakeGroq(reply)
                    t0 = time.perf_counter()
                    r = client.post("/extract", json={"prompt": f"turn {i}", "response": "ok"})
                    dt = time.perf_counter() - t0
                    assert r.status_code == 200, r.text
                    if i >= WARMUP:
                        times.append(dt)
                extract = pct(times)
                if profile:
                    # A separate, untimed pass: the profiler slows calls down,
                    # so it must never run while latencies are being recorded.
                    import cProfile
                    import io
                    import pstats
                    prof = cProfile.Profile()
                    for i in range(EXTRACT_N):
                        main.groq_client = FakeGroq(json.dumps([
                            {"text": f"User profiled fact number {i}", "type": "FACT", "importance": 4}]))
                        prof.enable()
                        client.post("/extract", json={"prompt": f"profile {i}", "response": "ok"})
                        prof.disable()
                    out = io.StringIO()
                    pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(15)
                    print(f"--- /extract profile at {n} memories (by own time) ---")
                    print(out.getvalue())
        finally:
            Path.home = old_home
    return {"memories": n, "retrieve": retrieve, "extract": extract}


def compare(result: dict, baseline_path: Path) -> int:
    base = {r["memories"]: r for r in json.loads(baseline_path.read_text())["results"]}
    bad = []
    for r in result["results"]:
        b = base.get(r["memories"])
        if b is None:
            bad.append(f"no baseline at {r['memories']} memories")
            continue
        for ep in ("retrieve", "extract"):
            now, then = r[ep]["p95_ms"], b[ep]["p95_ms"]
            status = "ok" if now <= MAX_SLOWDOWN * then else "SLOW"
            print(f"{ep} @ {r['memories']}: p95 {now} ms vs baseline {then} ms -> {status}")
            if status == "SLOW":
                bad.append(f"{ep} p95 {now} ms > {MAX_SLOWDOWN}x baseline {then} ms")
    if bad:
        print("FAIL:", "; ".join(bad))
        return 1
    return 0


def scaling(result: dict) -> int:
    by = {r["memories"]: r for r in result["results"]}
    if not {1000, 10000} <= by.keys():
        print("FAIL: --scaling needs results at 1000 and 10000 memories")
        return 1
    small, big = by[1000]["extract"]["p50_ms"], by[10000]["extract"]["p50_ms"]
    ratio = big / small
    status = "ok" if ratio <= MAX_EXTRACT_SCALING else "FAIL"
    print(f"extract p50: {small} ms at 1k, {big} ms at 10k, ratio {ratio:.2f} "
          f"(limit {MAX_EXTRACT_SCALING}) -> {status}")
    return 0 if status == "ok" else 1


def main_() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--scaling", action="store_true", help="fail if /extract slows with store size")
    ap.add_argument("--profile", action="store_true", help="print a /extract profile at the largest size")
    args = ap.parse_args()

    texts, vec, queries = corpus()
    result = {
        "machine": machine(),
        "date": time.strftime("%Y-%m-%d"),
        "method": f"TestClient, real model and store; {RETRIEVE_N} /retrieve and {EXTRACT_N} /extract "
                  f"per size after {WARMUP} warm-up calls; Groq stubbed with a fixed 2-fact reply",
        "results": [],
    }
    for n in args.sizes:
        r = run_size(n, texts, vec, queries, profile=args.profile and n == max(args.sizes))
        print(json.dumps(r))
        result["results"].append(r)

    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    status = 0
    if args.compare:
        status |= compare(result, args.compare)
    if args.scaling:
        status |= scaling(result)
    return status


if __name__ == "__main__":
    raise SystemExit(main_())
