"""
Embedding-based memory retrieval using sentence-transformers.

PURPOSE:
Encode user prompts and memories into 384-dimensional vectors,
compute similarity scores, and retrieve the most relevant stored memories.

INPUTS:
Prompts (strings), embeddings matrix from MemoryStore, Memory objects

OUTPUTS:
Ranked list of relevant Memory objects, ordered by similarity and importance
"""

from typing import List, Optional

import numpy as np
from fastembed import TextEmbedding

from models import Memory
from memory_store import MemoryStore


# Global model instance — lazy loaded on first call.
# Uses fastembed (ONNX-based, no PyTorch required) with BAAI/bge-small-en-v1.5,
# a 384-dimensional model comparable in quality to all-MiniLM-L6-v2.
_model: Optional[TextEmbedding] = None


def _get_model() -> TextEmbedding:
    """
    Lazy-load the fastembed embedding model on first use.

    fastembed uses ONNX Runtime under the hood — no PyTorch or GPU required.
    The model (~130MB) is downloaded once to ~/.cache/fastembed/ on first call.

    Returns:
        TextEmbedding model (BAAI/bge-small-en-v1.5, 384-dim)
    """
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def encode(text: str) -> np.ndarray:
    """
    Encode text to a 384-dimensional embedding vector.

    Args:
        text: Text to encode (prompt, memory text, etc.)

    Returns:
        Numpy array of shape (384,) with float32 dtype
    """
    model = _get_model()
    # fastembed.embed() returns a generator — take the first (and only) result
    embedding = next(model.embed([text]))
    return np.array(embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Handles edge cases like zero vectors gracefully.

    Args:
        a: Vector of shape (384,)
        b: Vector of shape (384,)

    Returns:
        Similarity score in range [-1, 1], typically [0, 1] for normalized embeddings
    """
    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Compute cosine similarity
    similarity = np.dot(a, b) / (norm_a * norm_b)
    return float(similarity)


def retrieve(
    prompt: str,
    store: MemoryStore,
    user_id: str,
    limit: int = 5,
    threshold: float = 0.35
) -> List[Memory]:
    """
    Retrieve the most relevant memories for a prompt.

    Algorithm:
    1. Encode the prompt to vector
    2. Compute cosine similarity against all stored memories
    3. Filter by threshold to remove low-relevance results
    4. Score each memory by: similarity_score * (1 + importance/10)
       This gives importance a 10-30% boost depending on value
    5. Sort by score descending and return top-k

    Args:
        prompt: Query text to find relevant memories
        store: MemoryStore instance with embeddings
        user_id: User identifier to filter by
        limit: Maximum number of memories to return
        threshold: Minimum similarity score (0-1) to include

    Returns:
        List of Memory objects, ordered by relevance score (highest first)
    """
    # Get all memory IDs for this user
    user_memory_ids = store.get_memory_ids(user_id)

    if not user_memory_ids:
        return []

    # Encode the query prompt
    prompt_embedding = encode(prompt)

    # Get all embeddings from store
    all_embeddings = store.get_embeddings()

    if len(all_embeddings) == 0:
        return []

    # Build mapping of memory IDs to their positions in the embeddings array
    all_ids = store.memory_ids
    user_id_to_idx = {mid: i for i, mid in enumerate(all_ids) if mid in user_memory_ids}

    # Compute similarities for user's memories
    similarities = {}
    for mem_id, idx in user_id_to_idx.items():
        sim = cosine_similarity(prompt_embedding, all_embeddings[idx])
        if sim >= threshold:
            similarities[mem_id] = sim

    if not similarities:
        return []

    # Fetch memory objects and compute final scores
    scored_memories = []
    for mem_id, similarity in similarities.items():
        memory = store.get_by_id(mem_id)
        if memory is not None:
            # Score combines similarity and importance
            # importance ranges 1-10, so multiplier ranges 1.1-2.0
            importance_multiplier = 1.0 + (memory.importance / 10.0)
            final_score = similarity * importance_multiplier
            scored_memories.append((memory, final_score))

    # Sort by score descending
    scored_memories.sort(key=lambda x: x[1], reverse=True)

    # Extract Memory objects and return top-k
    result = [mem for mem, score in scored_memories[:limit]]

    # Increment access counts for returned memories
    for memory in result:
        try:
            store.increment_access(memory.id)
        except ValueError:
            # Memory was deleted between retrieval and access increment; skip
            pass

    return result
