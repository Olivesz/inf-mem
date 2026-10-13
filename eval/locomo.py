"""
LoCoMo: fetch the pinned benchmark file and turn it into eval records.

LoCoMo (Maharana et al., ACL 2024) is 10 long multi-session conversations
with 1,986 questions, each labelled with the dialogue turns that hold the
answer. It is licensed CC BY-NC 4.0, so this repo does not redistribute it:
`fetch()` downloads it from a pinned commit and checks its SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
URL = f"https://raw.githubusercontent.com/snap-research/locomo/{COMMIT}/data/locomo10.json"
SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DATA_PATH = Path(__file__).resolve().parent / "data" / "locomo10.json"

# Category 5 questions are adversarial: the answer is not in the conversation.
# They are excluded from recall, which needs a real evidence turn.
ANSWERABLE = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


@dataclass
class Memory:
    key: str          # dia_id for a turn, "<dia_id>#<n>" for an observation
    text: str
    evidence: set     # dia_ids this memory is evidence for


@dataclass
class Question:
    text: str
    category: int
    evidence: set


@dataclass
class Conversation:
    sample_id: str
    turns: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    unresolved_evidence: int = 0


def fetch(path: Path = DATA_PATH) -> Path:
    """Download the pinned file if absent; raise if its hash is wrong."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{path} has SHA-256 {digest}, expected {SHA256}")
    return path


def _norm_id(raw: str) -> str:
    """'D30:05' -> 'D30:5'. Returns '' for ids that cannot be read."""
    m = re.fullmatch(r"D(\d+):(\d+)", raw.strip())
    return f"D{int(m.group(1))}:{int(m.group(2))}" if m else ""


def _evidence_ids(raw_list) -> set:
    """Evidence entries are sometimes 'D1:2; D3:4' or 'D1:2 D3:4' in one string."""
    out = set()
    for raw in raw_list:
        for part in re.split(r"[;,\s]+", raw):
            if part:
                out.add(_norm_id(part))
    return out


def load(path: Path = DATA_PATH) -> list:
    """Parse the benchmark into Conversation records."""
    convs = []
    for sample in json.loads(path.read_text()):
        conv = Conversation(sample_id=sample["sample_id"])
        c = sample["conversation"]
        known = set()

        for key in sorted((k for k in c if re.fullmatch(r"session_\d+", k)),
                          key=lambda k: int(k.split("_")[1])):
            for t in c[key]:
                dia = _norm_id(t["dia_id"])
                known.add(dia)
                text = f'{t["speaker"]}: {t["text"]}'
                if t.get("blip_caption"):
                    text += f' [shares a photo: {t["blip_caption"]}]'
                conv.turns.append(Memory(key=dia, text=text, evidence={dia}))

        for skey, by_speaker in sample.get("observation", {}).items():
            for speaker_obs in by_speaker.values():
                for n, (text, dia_raw) in enumerate(speaker_obs):
                    ev = _evidence_ids(dia_raw if isinstance(dia_raw, list) else [dia_raw]) & known
                    conv.observations.append(
                        Memory(key=f"{skey}#{len(conv.observations)}", text=text, evidence=ev))

        for q in sample["qa"]:
            if q["category"] not in ANSWERABLE:
                continue
            ev = _evidence_ids(q.get("evidence", []))
            resolved = ev & known
            conv.unresolved_evidence += len(ev - resolved)
            if resolved:
                conv.questions.append(Question(q["question"], q["category"], resolved))

        convs.append(conv)
    return convs
