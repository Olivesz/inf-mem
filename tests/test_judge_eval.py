"""The judge eval's scorer, checked with judges whose answers are known.

Needs the LoCoMo download and embeddings, so it runs in the model tier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

pytestmark = pytest.mark.model


class _Scripted:
    """A client whose judge answers come from a function of the prompt."""

    def __init__(self, answer):
        msg = lambda content: type("M", (), {"content": content})
        create = lambda **kw: type("R", (), {"choices": [
            type("C", (), {"message": msg(answer(kw["messages"][1]["content"]))})]})
        self.chat = type("Chat", (), {"completions": type("X", (), {"create": staticmethod(create)})()})()


@pytest.fixture(scope="module")
def data():
    import dedupe
    import locomo
    import retrieval

    convs = locomo.load(locomo.fetch())
    pairs = dedupe.load_pairs(convs)
    vec = retrieval.embed_all([o.text for c in convs for o in c.observations])
    return convs, pairs, vec


def test_oracle_judge_scores_perfectly(data):
    import dedupe_judge

    import locomo

    convs, pairs, vec = data
    truth = {(p["a"], p["b"]): p["same_fact"] for p in pairs}

    def oracle(prompt):
        new = prompt.split("New memory: ")[1].split("\n")[0]
        stored = prompt.split("1. ")[1]
        return '{"same_as": 1}' if truth[(new, stored)] else '{"same_as": null}'

    r = dedupe_judge.score(pairs, convs[locomo.EVALUATION], vec, _Scripted(oracle))
    assert (r["precision"], r["recall"], r["unweighted_correct"]) == (1.0, 1.0, r["labelled_pairs"])


def test_always_same_judge_scores_the_base_rate(data):
    import dedupe_judge
    import numpy as np

    import dedupe
    import locomo

    convs, pairs, vec = data
    ev = convs[locomo.EVALUATION]
    r = dedupe_judge.score(pairs, ev, vec, _Scripted(lambda _: '{"same_as": 1}'))
    arrays = dedupe.split_arrays(pairs, ev, vec)
    base = float((arrays["w"] * arrays["same"]).sum() / arrays["w"].sum())
    assert r["recall"] == 1.0 and r["precision"] == pytest.approx(base, abs=1e-4)
    assert not np.isnan(base)
