"""
FastAPI server for the inf-mem personal LLM memory system.

PURPOSE:
Expose the memory system via REST API with CORS support for browser extensions.
Handle memory extraction, retrieval, compression, and management.

INPUTS:
HTTP requests with JSON payloads from browser extension or clients

OUTPUTS:
JSON responses with memory data, extraction results, compression results
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

import compressor
import extractor
import retriever
from memory_store import MemoryStore
from models import CompressRequest, ExtractRequest, RetrieveRequest


# Global instances
app = FastAPI(
    title="inf-mem API",
    description="Personal LLM memory system",
    version="1.0.0"
)
store: Optional[MemoryStore] = None
groq_client: Optional[OpenAI] = None


# Configure CORS
# Allow localhost and chrome-extension:// origins for browser extension compatibility
app.add_middleware(
    CORSMiddleware,
    # Allow all origins — this backend is local-only (127.0.0.1),
    # so there's no meaningful security boundary to enforce here.
    # Chrome extensions require their origin (chrome-extension://...) to be allowed.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    """
    Initialize application on startup.

    Loads Groq API configuration, initializes MemoryStore, and sets up client.
    """
    global store, groq_client

    try:
        # Initialize memory store
        store = MemoryStore()
        print(f"Initialized MemoryStore at {store.data_dir}")

        # Initialize Groq client
        try:
            api_key = extractor._load_groq_api_key()
            groq_client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            print("Groq client initialized successfully")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not initialize Groq client: {e}")
            groq_client = None

    except Exception as e:
        print(f"Error during startup: {e}")
        raise


@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns basic system status including memory count and Groq connectivity.

    Returns:
        Dictionary with keys:
        - status (str): "ok"
        - memory_count (int): Total memories in storage
        - groq_connected (bool): Whether Groq API is available
    """
    if store is None:
        raise HTTPException(status_code=500, detail="MemoryStore not initialized")

    # Count total memories
    # Get all unique memory IDs across all users
    all_memories = set(store.memory_ids)
    memory_count = len(all_memories)

    # Check if Groq client is available
    groq_connected = groq_client is not None

    return {
        "status": "ok",
        "memory_count": memory_count,
        "groq_connected": groq_connected
    }


@app.post("/retrieve")
async def retrieve_memories(request: RetrieveRequest) -> dict:
    """
    Retrieve relevant memories for a prompt.

    Uses embedding-based similarity search to find the most relevant stored memories.

    Args:
        request: RetrieveRequest with prompt, limit, user_id, threshold

    Returns:
        Dictionary with keys:
        - memories (list): List of Memory objects, sorted by relevance
        - count (int): Number of memories returned
    """
    if store is None:
        raise HTTPException(status_code=500, detail="MemoryStore not initialized")

    try:
        # Load retriever model on first use
        _ = retriever._get_model()

        # Retrieve relevant memories
        memories = retriever.retrieve(
            prompt=request.prompt,
            store=store,
            user_id=request.user_id,
            limit=request.limit,
            threshold=request.threshold
        )

        return {
            "memories": [mem.model_dump() for mem in memories],
            "count": len(memories)
        }

    except Exception as e:
        print(f"Error retrieving memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract_memories(request: ExtractRequest) -> dict:
    """
    Extract memories from a prompt/response pair.

    Uses Groq LLM to identify and extract memorable information.
    Deduplicates against existing memories.

    Args:
        request: ExtractRequest with prompt, response, user_id

    Returns:
        Dictionary with keys:
        - extracted (list): List of Memory objects that were added/updated
        - count (int): Number of memories extracted
    """
    if store is None:
        raise HTTPException(status_code=500, detail="MemoryStore not initialized")

    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq API not configured - cannot extract memories"
        )

    try:
        # Extract memories using Groq
        extracted = extractor.extract(
            prompt=request.prompt,
            response=request.response,
            store=store,
            user_id=request.user_id
        )

        return {
            "extracted": [mem.model_dump() for mem in extracted],
            "count": len(extracted)
        }

    except Exception as e:
        print(f"Error extracting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compress")
async def compress_prompt(request: CompressRequest) -> dict:
    """
    Compress a prompt to be more concise and precise.

    Args:
        request: CompressRequest with prompt text

    Returns:
        Dictionary with keys:
        - compressed (str): Compressed prompt text
        - original_tokens (int): Estimated tokens in original
        - compressed_tokens (int): Estimated tokens in compressed
        - ratio (float): Compression ratio (compressed / original)
    """
    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq API not configured - cannot compress"
        )

    try:
        result = compressor.compress(request.prompt, groq_client)
        return result

    except Exception as e:
        print(f"Error compressing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories")
async def list_memories(user_id: str = "default") -> dict:
    """
    List all memories for a user.

    Args:
        user_id: User identifier (query parameter)

    Returns:
        Dictionary with keys:
        - memories (list): All memories for the user, ordered by creation date
        - count (int): Number of memories
    """
    if store is None:
        raise HTTPException(status_code=500, detail="MemoryStore not initialized")

    try:
        memories = store.get_all(user_id)
        return {
            "memories": [mem.model_dump() for mem in memories],
            "count": len(memories)
        }

    except Exception as e:
        print(f"Error listing memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """
    Delete a memory by ID.

    Args:
        memory_id: ID of memory to delete

    Returns:
        Dictionary with key:
        - success (bool): Whether deletion succeeded
    """
    if store is None:
        raise HTTPException(status_code=500, detail="MemoryStore not initialized")

    try:
        memory = store.get_by_id(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="Memory not found")

        store.delete_memory(memory_id)
        return {"success": True}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Run the server on port 7842
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=7842,
        log_level="info"
    )
