# Calibrating the two gates

inf-mem makes two decisions by thresholding a cosine similarity between
`BAAI/bge-small-en-v1.5` embeddings:

| Gate | Where | Question it answers | Was | Now |
|---|---|---|---|---|
| Retrieval | `retriever.RETRIEVE_THRESHOLD` | Is this stored memory relevant to the prompt? | 0.35 | **0.61** |
| Dedupe | `extractor.DEDUPE_THRESHOLD` | Is this new memory the same fact as a stored one? | 0.85 | **0.945** |

The old values were picked by hand and never measured. Both numbers below come
from committed result files, and `make calibrate` regenerates them. Tests
assert that the constants in code equal the calibrated values.

## Data

[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) has 10 long
conversations between two people, 1,986 questions about them, and for each
question the dialogue turns that hold the answer. It also lists **observations**
per session: one-sentence facts such as "Caroline attended an LGBTQ support
group recently", each citing its source turn. Those have the same shape as the
memories inf-mem's extractor writes, so they stand in for memories here.

The file is CC BY-NC 4.0 and is not committed. `eval/locomo.py` downloads it
from a pinned commit and checks its SHA-256.

The 446 adversarial questions (category 5, answer not in the conversation) are
left out. Of the remaining 1,540, 4 have evidence IDs that match no turn and
are dropped, which leaves **1,536**.

## Retrieval gate

**Criterion.** The gate should let memories through for questions about this
user and block them for questions about someone else. For every question:

- *on-topic*: does its best evidence memory, in its own conversation's store,
  pass the gate?
- *off-topic*: does any memory in the next conversation's store (two other
  people, so nothing is relevant) pass the gate?

The threshold maximises on-topic pass rate minus off-topic pass rate (Youden's
J), swept from 0.30 to 0.95 in steps of 0.005, on the observations store.

**Query instruction.** bge models are trained to embed queries with the prefix
`Represent this sentence for searching relevant passages: ` and documents
without it. The retriever now does this. On the observations store it raised
J from 0.707 to 0.727 and recall@5 from 0.552 to 0.562.

**Result** (`results/locomo-retrieval.json`, observations store, 2,541 memories):

| Gate | On-topic pass | Off-topic pass | Recall@5 | Hit@5 |
|---|---|---|---|---|
| none | 0.854 | 1.000 | 0.562 | 0.640 |
| 0.35 (old) | 0.854 | **1.000** | 0.562 | 0.640 |
| **0.61** | 0.759 | **0.033** | 0.556 | 0.633 |

At the old 0.35, every question asked against a stranger's store still got
memories injected. At 0.61, 3.3% do, and recall@5 drops by 0.006.

*On-topic pass* is out of all 1,536 questions. Only 1,312 (0.854) have an
observation citing their evidence, so 0.854 is its ceiling, and the old gate
reached it. Recall@5 counts the evidence turns covered by the 5 retrieved
memories, averaged over all 1,536 questions.

By question type at 0.61: single-hop 0.599, temporal 0.712, multi-hop 0.348,
open-domain 0.262. Multi-hop answers span several turns, and one prompt
embedding rarely lands near all of them.

Raw dialogue turns as memories (5,882) do worse: recall@5 0.457 at 0.61, and
their best J is at 0.615 with 9.1% off-topic. That is one reason to store
extracted facts rather than transcripts.

## Dedupe gate

**Labels.** All pairs of observations within one conversation were scored.
1,239 pairs have cosine ≥ 0.86. All 120 pairs ≥ 0.92 were labelled, plus 60 of
the 1,119 pairs in [0.86, 0.92) drawn with numpy seed 0. Each sampled pair
stands for 18.65 pairs, and precision and recall are weighted to match. The
labels are in `dedupe_pairs.jsonl` (128 same, 52 different).

The labelling rule: **same** if both state the same fact about the same subject,
so a reader would not want both stored. Wording and minor elaboration don't
matter. **Different** if the subject or direction differs ("Sam thanks Evan" /
"Evan thanks Sam"), if they describe different events ("attended a festival" /
"enjoys festivals"), or if one adds a distinct fact ("...and received a medal").

**Criterion.** Merging two different facts overwrites one of them, so data is
lost. A missed merge only stores a duplicate. The threshold is therefore the
lowest value whose weighted merge precision is at least 0.90.

**Result** (`results/dedupe-calibration.json`):

| Threshold | Merge precision | Same-fact pairs merged |
|---|---|---|
| 0.86 | 0.588 | 1.000 |
| 0.90 | 0.599 | 0.283 |
| 0.92 | 0.783 | 0.129 |
| **0.945** | **0.906** | **0.040** |

**Cosine similarity alone cannot do this job.** Precision stays between 0.55
and 0.80 from 0.86 up to 0.94, because near-identical sentences with the
subject swapped score as high as true paraphrases (0.996 for "Evan plans a
painting session with Sam" vs "Sam plans a painting session with Evan"). At
0.945 merges are safe, but only 4% of duplicates get caught. The old 0.85
merged "has a dog named Biscuit" with "has a cat named Biscuit" (0.890); at
0.945 they stay separate, and a test checks it.

The next step is to let the extraction model make the ADD/UPDATE/skip call
with the nearest stored memories in its prompt, using the cosine gate only to
pick candidates. That needs a Groq key to measure.

## What these numbers are not

- They measure retrieval over LoCoMo's own observations, not over memories
  produced by inf-mem's extractor. The end-to-end number (extract with Groq,
  then retrieve and answer) needs a Groq key and comes next.
- Off-topic questions are drawn from other LoCoMo conversations, which are
  everyday chat like the on-topic ones. Unrelated prompts in real use
  ("capital of Australia") are easier to reject. The tests check those cases
  with the real model.
- Dedupe labels were made by hand. They are committed so the labels can be
  disputed line by line.
