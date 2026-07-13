"""
Extract memories from user prompts using Groq's LLM API.

PURPOSE:
Use a free Groq API (llama-3.1-8b-instant) to intelligently extract
memorable facts, projects, preferences, and episodes from conversations,
then deduplicate against existing memories using embedding similarity.

INPUTS:
User prompt, optional AI response, Groq API key from config, MemoryStore, retriever module

OUTPUTS:
List of newly added or updated Memory objects
"""

import json
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from openai import OpenAI

from models import Memory, MemoryType, ExtractedMemory
from memory_store import MemoryStore


# System prompt for Groq extraction (must be exact as specified)
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


def _load_groq_api_key() -> str:
    """
    Load Groq API key from config.yaml.

    Config file location: project_root/config.yaml (2 levels up from backend/)

    Returns:
        Groq API key string

    Raises:
        FileNotFoundError: If config.yaml not found
        ValueError: If GROQ_API_KEY not in config
    """
    # Navigate 1 level up from backend/ to project root (inf-mem/)
    backend_dir = Path(__file__).parent
    project_root = backend_dir.parent
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            f"Expected at project root with GROQ_API_KEY."
        )

    # Simple YAML parsing (we only need to extract GROQ_API_KEY)
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GROQ_API_KEY"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[1].strip()
                    # Handle quoted values
                    if key.startswith('"') and key.endswith('"'):
                        key = key[1:-1]
                    elif key.startswith("'") and key.endswith("'"):
                        key = key[1:-1]
                    return key

    raise ValueError("GROQ_API_KEY not found in config.yaml")


def extract(
    prompt: str,
    response: Optional[str],
    store: MemoryStore,
    user_id: str = "default"
) -> List[Memory]:
    """
    Extract memories from a prompt/response pair using Groq API.

    Algorithm:
    1. Build combined text from prompt and response
    2. Call Groq API with extraction system prompt
    3. Parse JSON response into ExtractedMemory objects
    4. For each extracted memory:
       a. Generate embedding via retriever.encode()
       b. Check against existing memories for near-duplicates (similarity > 0.85)
       c. If duplicate found and new is higher quality: UPDATE the existing
       d. If duplicate found and new is lower quality: skip (NOOP)
       e. If new: ADD with generated UUID
    5. Return list of Memory objects that were added/updated

    Args:
        prompt: User's prompt text
        response: Optional AI response text
        store: MemoryStore instance to add/update memories
        user_id: User identifier

    Returns:
        List of Memory objects that were added or updated

    Raises:
        ValueError: If Groq API returns invalid JSON
        Exception: If Groq API call fails
    """
    # Import here to avoid circular imports
    import retriever

    # Build combined input text
    combined_text = prompt
    if response:
        combined_text = f"{prompt}\n\nAI Response:\n{response}"

    # Initialize Groq client
    try:
        api_key = _load_groq_api_key()
    except (FileNotFoundError, ValueError) as e:
        # If config not available, log and return empty list
        # In production, you might want to raise instead
        print(f"Warning: Could not load Groq API key: {e}")
        return []

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    # Call Groq API for extraction
    try:
        response_obj = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": EXTRACTION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": combined_text
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent extraction
            max_tokens=1024
        )
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return []

    # Extract response text
    groq_response = response_obj.choices[0].message.content.strip()

    # Parse JSON response
    try:
        extracted_list = json.loads(groq_response)
    except json.JSONDecodeError as e:
        print(f"Error parsing Groq response as JSON: {e}")
        print(f"Response was: {groq_response}")
        return []

    # If empty response, return early
    if not extracted_list:
        return []

    # Process extracted memories
    result = []

    for extracted_dict in extracted_list:
        try:
            extracted = ExtractedMemory(**extracted_dict)
        except Exception as e:
            print(f"Error validating extracted memory: {e}")
            continue

        # Generate embedding for this extracted memory
        try:
            embedding = retriever.encode(extracted.text)
        except Exception as e:
            print(f"Error encoding memory text: {e}")
            continue

        # Check for near-duplicate memories
        existing_memories = store.get_all(user_id)
        duplicate_found = False

        for existing in existing_memories:
            try:
                existing_embedding = retriever.encode(existing.text)
                similarity = retriever.cosine_similarity(embedding, existing_embedding)

                if similarity > 0.85:
                    # Near-duplicate found
                    duplicate_found = True

                    # Decide whether to update or skip
                    # UPDATE if: new importance is higher or text is more complete
                    if extracted.importance > existing.importance or len(extracted.text) > len(existing.text):
                        # Update the existing memory
                        try:
                            store.update_memory(
                                memory_id=existing.id,
                                text=extracted.text,
                                keywords=extracted.keywords,
                                importance=extracted.importance,
                                embedding=embedding
                            )
                            result.append(existing)
                        except ValueError as e:
                            print(f"Error updating memory: {e}")

                    # else: skip (existing is better or equal)
                    break

            except Exception as e:
                # If encoding fails for existing memory, skip duplicate check
                print(f"Error checking duplicate: {e}")
                continue

        # If no duplicate, add new memory
        if not duplicate_found:
            try:
                memory = Memory(
                    id=str(uuid4()),
                    user_id=user_id,
                    text=extracted.text,
                    type=MemoryType(extracted.type),
                    keywords=extracted.keywords,
                    importance=extracted.importance
                )
                store.add_memory(memory, embedding)
                result.append(memory)
            except Exception as e:
                print(f"Error adding memory: {e}")
                continue

    return result
