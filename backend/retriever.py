"""
Embedding-based memory retrieval using fastembed (BAAI/bge-small-en-v1.5).

PURPOSE:
Encode user prompts and memories into 384-dimensional vectors,
compute similarity scores, and retrieve the most relevant stored memories.

INPUTS:
Prompts (strings), embeddings matrix from MemoryStore, Memory objects

OUTPUTS:
Ranked list of relevant Memory objects, ordered by similarity and importance
"""


import numpy as np
from fastembed import TextEmbedding

from memory_store import MemoryStore
from models import Memory

# Global model instance — lazy loaded on first call.
# Uses fastembed (ONNX-based, no PyTorch required) with BAAI/bge-small-en-v1.5,
# a 384-dimensional model.
_model: TextEmbedding | None = None

# Memories scoring below this cosine similarity to the prompt are not
# retrieved. See eval/calibration.md.
RETRIEVE_THRESHOLD = 0.35


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


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of one vector against every row of a matrix.

    Args:
        query: Vector of shape (384,)
        matrix: Array of shape (N, 384)

    Returns:
        Array of shape (N,); rows or queries with zero norm score 0
    """
    if len(matrix) == 0:
        return np.zeros(0, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)


def retrieve(
    prompt: str,
    store: MemoryStore,
    user_id: str,
    limit: int = 5,
    threshold: float | None = None
) -> list[Memory]:
    """
    Retrieve the most relevant memories for a prompt.

    Algorithm:
    1. Encode the prompt to a vector
    2. Score it against all of the user's stored vectors in one matrix product
    3. Drop memories below the similarity threshold
    4. Rank the rest by similarity * (1 + importance/10), so importance
       multiplies the score by 1.1 to 2.0
    5. Return the top `limit`

    Args:
        prompt: Query text to find relevant memories
        store: MemoryStore instance with embeddings
        user_id: User identifier to filter by
        limit: Maximum number of memories to return
        threshold: Minimum similarity to include; None means RETRIEVE_THRESHOLD

    Returns:
        List of Memory objects, ordered by relevance score (highest first)
    """
    if threshold is None:
        threshold = RETRIEVE_THRESHOLD
    ids, matrix = store.get_user_vectors(user_id)
    if not ids:
        return []

    sims = cosine_scores(encode(prompt), matrix)
    passing = np.flatnonzero(sims >= threshold)
    if len(passing) == 0:
        return []

    by_id = {m.id: m for m in store.get_many([ids[i] for i in passing])}
    scored = []
    for i in passing:
        memory = by_id.get(ids[i])
        if memory is not None:
            scored.append((memory, float(sims[i]) * (1.0 + memory.importance / 10.0)))
    scored.sort(key=lambda x: x[1], reverse=True)

    result = [mem for mem, _ in scored[:limit]]
    store.increment_access_many([m.id for m in result])
    return result
