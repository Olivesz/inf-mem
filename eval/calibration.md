# Calibrating the two gates

inf-mem makes two decisions by thresholding a cosine similarity between
`BAAI/bge-small-en-v1.5` embeddings:

| Gate | Where | Question it answers | Was | Now |
|---|---|---|---|---|
| Retrieval | `retriever.RETRIEVE_THRESHOLD` | Is this stored memory relevant to the prompt? | 0.35 | **0.61** |
| Dedupe | `extractor.DEDUPE_THRESHOLD` | Is this new memory the same fact as a stored one? | 0.85 | **0.98** |

The old values were picked by hand and never measured. Every number below is
in a committed result file, `make calibrate` regenerates them, and tests check
that the constants in code equal the calibrated values.

## Data and split

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
are dropped, which leaves 1,536.

**Held-out split.** This was fixed before any result was looked at. The first
five conversations in file order (conv-26 to conv-43, 760 questions) choose
every setting. The last five (conv-44 to conv-50, 776 questions) only measure
what the code ships. Every number reported below as a result comes from the
evaluation half.

## Retrieval gate

**Criterion.** The gate should let memories through for questions about this
user and block them for questions about someone else. For every question:

- *on-topic*: does its best evidence memory, in its own conversation's store,
  pass the gate?
- *off-topic*: does any memory in the next conversation's store (two other
  people, so nothing is relevant) pass the gate?

On the calibration half's observations, the threshold maximises on-topic pass
rate minus off-topic pass rate (Youden's J), swept from 0.30 to 0.95 in steps
of 0.005. The same rule decides whether queries carry bge's retrieval
instruction (`Represent this sentence for searching relevant passages: `).
With the instruction, J was 0.712 at 0.61; without it, 0.688 at 0.63. The
instruction won, and the retriever uses it.

**Across CPUs.** The embedding model is quantized ONNX, and its output
differs slightly between ARM (where the calibration was run) and x86 (CI).
Here J is 0.7118 at 0.61 and 0.7105 at 0.605, one question apart. On CI's x86
runner one question flips the other way, and the argmax lands on 0.605. So
CI does not require the argmax to match. It requires the shipped 0.61 to
stay optimal within 0.002 of J on its own machine, and every other number to
match within 0.002. CI uploads its recomputed file as an artifact.

**Result on the evaluation half** (`results/locomo-retrieval.json`,
observations store, 1,331 memories, 776 questions):

| Gate | On-topic pass | Off-topic pass | Recall@5 | Hit@5 |
|---|---|---|---|---|
| none | 0.878 | 1.000 | 0.575 | 0.664 |
| 0.35 (old) | 0.878 | **1.000** | 0.575 | 0.664 |
| **0.61** | 0.789 | **0.044** | 0.571 | 0.657 |

At the old 0.35, every question asked against a stranger's store still got
memories injected. At 0.61, 4.4% do, and recall@5 drops by 0.004.

*On-topic pass* is out of all 776 questions. Only 681 (0.878) have an
observation citing their evidence, so 0.878 is its ceiling, and the old gate
reached it. Recall@5 counts the evidence turns covered by the 5 retrieved
memories, averaged over all 776 questions.

By question type at 0.61: single-hop 0.623, temporal 0.705, multi-hop 0.373,
open-domain 0.236. Multi-hop answers span several turns, and one prompt
embedding rarely lands near all of them.

Raw dialogue turns as memories (3,122) do worse: recall@5 0.453 at 0.61, with
15.7% off-topic. That is one reason to store extracted facts, not transcripts.

## Dedupe gate

**Labels.** All pairs of observations within one conversation were scored, and
1,239 have cosine ≥ 0.86. All 120 pairs ≥ 0.92 were labelled, plus 60 of the
1,119 pairs in [0.86, 0.92), drawn with numpy seed 0. Within each half, a
sampled pair from the lower band stands for (pairs in band / pairs sampled)
pairs, and precision and recall are weighted to match. Recall covers same-fact
pairs scoring ≥ 0.86 only.

`dedupe_pairs.jsonl` holds observation keys and labels, not text, since LoCoMo
is not redistributed here. It also stores a hash of each pair's text, so a
label cannot silently attach to different sentences.

The labelling rule: **same** if both state the same fact about the same subject,
so a reader would not want both stored. Wording and minor elaboration don't
matter. **Different** if the subject or direction differs ("Sam thanks Evan" /
"Evan thanks Sam"), if they describe different events ("attended a festival" /
"enjoys festivals"), or if one adds a distinct fact ("...and received a medal").

**Criterion.** Merging two different facts overwrites one of them, so data is
lost. A missed merge only stores a duplicate. The threshold is therefore the
lowest value whose weighted merge precision on the calibration half is at
least 0.90.

**Calibration half** (88 labelled pairs):

| Threshold | Merge precision | Same-fact pairs merged |
|---|---|---|
| 0.86 | 0.567 | 1.000 |
| 0.90 | 0.535 | 0.229 |
| 0.92 | 0.737 | 0.127 |
| 0.945 | 0.882 | 0.045 |
| **0.98** | no labelled pair merges | 0 |

No threshold that merged anything reached 0.90, so the rule lands at 0.98.
Only near-verbatim restatements merge there: "User has a dog named Biscuit"
vs "The user has a dog named Biscuit" scores 0.989.

**Evaluation half** (92 labelled pairs): at 0.98, 2 pairs merge, 1 of them
wrongly. The wrong one is "Evan plans a painting session with Sam" vs "Sam
plans a painting session with Evan", at cosine 0.996. Of the merges the old
0.85 made, 39% joined two different facts. That included "has a dog named
Biscuit" and "has a cat named Biscuit" (0.890), which a test now keeps apart.

**Cosine similarity alone cannot make this decision.** A subject swap scores
as high as a true paraphrase. The fix is to let the extraction model decide
ADD/UPDATE/skip with the nearest stored memories in its prompt, using cosine
only to pick candidates. Measuring that needs a Groq key.

## What these numbers are not

- They measure retrieval over LoCoMo's own observations, not over memories
  produced by inf-mem's extractor. The end-to-end number (extract with Groq,
  then retrieve and answer) needs a Groq key and comes next.
- Off-topic questions come from other LoCoMo conversations, which are
  everyday chat like the on-topic ones. Unrelated prompts in real use
  ("capital of Australia") are easier to reject. The tests check those cases
  with the real model.
- Dedupe labels were made by hand. They are committed so they can be
  disputed line by line.
