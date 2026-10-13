"""
Compare recomputed eval results with the committed file.

Embeddings come from a quantized ONNX model whose output differs slightly
between CPU architectures, so a question whose score sits on a threshold can
fall on either side. Rates are compared within TOL, which allows three such
questions in a 760-question half (3/760 = 0.0039; CI's x86 runner flipped two
against the ARM results). TOL stays below the 0.005 sweep step, so a threshold
that moves still fails.
"""

from __future__ import annotations

import json
from pathlib import Path

TOL = 0.004


def diffs(a, b, path="") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing" for k in a.keys() ^ b.keys()]
        for k in a.keys() & b.keys():
            out += diffs(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in diffs(x, y, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int | float):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    return [] if abs(a - b) <= max(TOL, TOL * abs(a)) else [f"{path}: {a} != {b}"]


def check(committed: Path, recomputed: dict) -> int:
    problems = diffs(json.loads(committed.read_text()), recomputed)
    if problems:
        print(f"MISMATCH with {committed.name}:")
        print("\n".join(problems[:20]))
        return 1
    print(f"match: {committed.name}")
    return 0
