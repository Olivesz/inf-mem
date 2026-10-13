"""
Extract memories from a prompt/response pair using Groq's LLM API.

PURPOSE:
Ask llama-3.1-8b-instant (via Groq) for memorable facts, projects,
preferences and episodes in a conversation turn, then deduplicate each
candidate against the user's stored memory vectors.

INPUTS:
User prompt, optional AI response, an OpenAI-compatible Groq client, MemoryStore

OUTPUTS:
List of newly added or updated Memory objects
"""

import json
import re
from typing import Any
from uuid import uuid4

import numpy as np

import retriever
from config import GROQ_MODEL
from memory_store import MemoryStore
from models import ExtractedMemory, Memory, MemoryType

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system for a personal AI assistant.
Given a user's message and optionally an AI's response, extract important facts worth remembering long-term.

Extract ONLY things that are genuinely useful to remember across conversations:
- FACT: Personal facts about the user (name, job, location, background, relationships)
- PROJECT: Active work or projects the user is doing
- PREFERENCE: How the user likes to work, communicate, or be responded to
- EPISODE: Significant events or context worth remembering

Rules:
- Only extract if the information is explicitly stated or very clearly implied
- Skip conversational filler, greetings, and one-off questions
- Each memory should be a single clear statement
- Importance 1-10: 10 = critical long-term fact, 1 = minor detail

Respond ONLY with a JSON array. Example:
[{"text": "User is a CS student at UPenn", "type": "FACT", "keywords": ["student", "UPenn", "CS"], "importance": 8}]

If nothing worth storing, respond with: []
"""

# A candidate whose cosine similarity to a stored memory is at or above this
# value is treated as the same fact. See eval/calibration.md.
DEDUPE_THRESHOLD = 0.98

# Below DEDUPE_THRESHOLD, cosine cannot tell a paraphrase from a different
# fact (eval/calibration.md). With the merge judge on, stored memories at or
# above JUDGE_CANDIDATE_THRESHOLD are shown to the model, which decides. It
# stays off until eval/dedupe_judge.py measures it against the same bar the
# cosine gate had to meet; that needs a Groq key.
USE_MERGE_JUDGE = False
JUDGE_CANDIDATE_THRESHOLD = 0.86
JUDGE_MAX_CANDIDATES = 3

JUDGE_SYSTEM_PROMPT = """You decide whether a new memory about a user repeats one already stored.
Two memories are the same fact if they say the same thing about the same subject, even in different words.
They are different if the subject or direction differs ("Sam thanks Evan" vs "Evan thanks Sam"),
if they describe different events, or if one adds a distinct fact.

Respond ONLY with JSON: {"same_as": <number of the stored memory that is the same fact, or null>}
"""

# Counters surfaced by GET /health so failures are visible, not silent.
stats = {"calls": 0, "parse_failures": 0, "invalid_items": 0, "judge_calls": 0, "judge_failures": 0}

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class ExtractionParseError(ValueError):
    """The model's reply could not be read as a JSON array."""


def parse_reply(text: str) -> list[Any]:
    """
    Parse the model's reply into a list.

    llama-3.1-8b often wraps JSON in ``` fences or adds a sentence before it,
    so strip fences and fall back to the outermost [...] span.
    """
    body = _FENCE.sub("", text.strip()).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("["), body.rfind("]")
        if start == -1 or end <= start:
            raise ExtractionParseError(f"no JSON array in reply: {text[:200]!r}")
        try:
            parsed = json.loads(body[start:end + 1])
        except json.JSONDecodeError as e:
            raise ExtractionParseError(f"bad JSON in reply: {e}") from e

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ExtractionParseError(f"reply is {type(parsed).__name__}, not a list")
    return parsed


def call_model(client, prompt: str, response: str | None) -> str:
    """Send one extraction request and return the raw reply text."""
    combined = prompt if not response else f"{prompt}\n\nAI Response:\n{response}"
    reply = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": combined},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return reply.choices[0].message.content or ""


def judge(client, new_text: str, stored_texts: list[str]) -> int | None:
    """
    Ask the model which stored memory, if any, states the same fact.

    Returns an index into stored_texts, or None. An unreadable reply counts
    as a judge failure and returns None, so the new memory is added rather
    than overwriting one it may not match.
    """
    stats["judge_calls"] += 1
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(stored_texts))
    reply = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"New memory: {new_text}\n\nStored memories:\n{listing}"},
        ],
        temperature=0.0,
        max_tokens=50,
    )
    text = _FENCE.sub("", (reply.choices[0].message.content or "").strip()).strip()
    try:
        same_as = json.loads(text[text.find("{"): text.rfind("}") + 1])["same_as"]
    except (ValueError, KeyError, TypeError):
        stats["judge_failures"] += 1
        return None
    if same_as is None:
        return None
    if isinstance(same_as, int) and not isinstance(same_as, bool) and 1 <= same_as <= len(stored_texts):
        return same_as - 1
    stats["judge_failures"] += 1
    return None


def find_duplicate(store: MemoryStore, user_id: str, embedding: np.ndarray, text: str, client) -> str | None:
    """Return the ID of the stored memory that this one restates, or None."""
    ids, sims, _ = store.search(user_id, embedding)
    if not ids:
        return None
    best = int(np.argmax(sims))
    if sims[best] >= DEDUPE_THRESHOLD:
        return ids[best]
    if not (USE_MERGE_JUDGE and client is not None):
        return None
    order = [i for i in np.argsort(-sims)[:JUDGE_MAX_CANDIDATES] if sims[i] >= JUDGE_CANDIDATE_THRESHOLD]
    if not order:
        return None
    pick = judge(client, text, [store.get_by_id(ids[i]).text for i in order])
    return None if pick is None else ids[order[pick]]


def apply(
    candidates: list[Any],
    store: MemoryStore,
    user_id: str = "default",
    client=None,
) -> list[Memory]:
    """
    Deduplicate parsed candidates against the store and write them.

    For each candidate: embed it once, compare it with the user's stored
    vectors in one matrix product (and, if enabled, ask the merge judge),
    then ADD, UPDATE or skip.
    """
    result: list[Memory] = []
    for item in candidates:
        try:
            extracted = ExtractedMemory(**item)
            mtype = MemoryType(extracted.type.upper())
        except (TypeError, ValueError):
            stats["invalid_items"] += 1
            continue

        embedding = retriever.encode(extracted.text)
        match_id = find_duplicate(store, user_id, embedding, extracted.text, client)

        if match_id is not None:
            existing = store.get_by_id(match_id)
            better = (extracted.importance > existing.importance
                      or len(extracted.text) > len(existing.text))
            if better:
                store.update_memory(
                    memory_id=existing.id,
                    text=extracted.text,
                    keywords=extracted.keywords,
                    importance=extracted.importance,
                    embedding=embedding,
                )
                result.append(store.get_by_id(existing.id))
            continue

        memory = Memory(
            id=str(uuid4()),
            user_id=user_id,
            text=extracted.text,
            type=mtype,
            keywords=extracted.keywords,
            importance=extracted.importance,
        )
        store.add_memory(memory, embedding)
        result.append(memory)

    return result


def extract(
    prompt: str,
    response: str | None,
    store: MemoryStore,
    client,
    user_id: str = "default",
) -> list[Memory]:
    """
    Extract memories from a prompt/response pair and store them.

    Raises:
        ExtractionParseError: the reply was not a JSON array (also counted)
        Exception: whatever the Groq client raises on an API failure
    """
    stats["calls"] += 1
    text = call_model(client, prompt, response)
    try:
        candidates = parse_reply(text)
    except ExtractionParseError:
        stats["parse_failures"] += 1
        raise
    return apply(candidates, store, user_id, client)
