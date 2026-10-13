"""
Compare recomputed eval results with the committed file.

Embeddings come from ONNX Runtime, whose float output can differ in the last
bits across CPUs, so numbers are compared within TOL rather than byte for
byte. TOL is below the 0.005 sweep step, so a calibrated threshold that moves
still fails the check.
"""

from __future__ import annotations

import json
from pathlib import Path

TOL = 0.002


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
