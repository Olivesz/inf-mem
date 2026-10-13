"""
FastAPI server for the inf-mem personal LLM memory system.

PURPOSE:
Expose memory retrieval, extraction, compression and management over REST
to the Chrome extension. Only the extension's origin may call it.

INPUTS:
HTTP requests with JSON payloads from the browser extension

OUTPUTS:
JSON responses with memory data, extraction results, compression results
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError

import compressor
import config
import extractor
import retriever
from memory_store import MemoryStore
from models import CompressRequest, ExtractRequest, RetrieveRequest

store: MemoryStore | None = None
groq_client: OpenAI | None = None

# Chrome extension IDs are 32 characters from a-p. Any web page calling the
# backend arrives with an http(s) origin and gets no CORS grant, so it cannot
# read or delete the memory store.
EXTENSION_ORIGIN = r"^chrome-extension://[a-p]{32}$"

# GET /health checks Groq for real at most this often.
REACHABILITY_TTL_S = 300.0
_reachability = {"checked_at": 0.0, "ok": False, "error": None}


def make_groq_client(api_key: str | None) -> OpenAI | None:
    """Build a Groq client, or None when no usable key is configured."""
    if api_key is None:
        return None
    return OpenAI(api_key=api_key, base_url=config.GROQ_BASE_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, groq_client
    store = MemoryStore()
    print(f"Initialized MemoryStore at {store.data_dir}")
    groq_client = make_groq_client(config.load_groq_api_key())
    if groq_client is None:
        print("Groq key not configured: /extract and /compress will return 503")
    yield


app = FastAPI(
    title="inf-mem API",
    description="Personal LLM memory system",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=EXTENSION_ORIGIN,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


def _require_store() -> MemoryStore:
    if store is None:
        raise HTTPException(status_code=500, detail="MemoryStore not initialized")
    return store


def _require_groq():
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Groq API key not configured")
    return groq_client


def _groq_reachable() -> tuple:
    """Make a real, cheap Groq call (list models), cached for REACHABILITY_TTL_S."""
    if groq_client is None:
        return False, "not configured"
    now = time.monotonic()
    if _reachability["checked_at"] and now - _reachability["checked_at"] < REACHABILITY_TTL_S:
        return _reachability["ok"], _reachability["error"]
    try:
        groq_client.models.list()
        ok, error = True, None
    except OpenAIError as e:
        ok, error = False, type(e).__name__
    _reachability.update(checked_at=now, ok=ok, error=error)
    return ok, error


@app.get("/health")
def health_check() -> dict:
    """
    Report store size, whether a Groq key is configured, whether Groq
    actually answers, and extraction failure counters.
    """
    s = _require_store()
    reachable, error = _groq_reachable()
    return {
        "status": "ok",
        "memory_count": len(s.memory_ids),
        "groq_configured": groq_client is not None,
        "groq_reachable": reachable,
        "groq_error": error,
        "extraction": dict(extractor.stats),
    }


@app.post("/retrieve")
def retrieve_memories(request: RetrieveRequest) -> dict:
    """Return the user's memories relevant to a prompt, best first."""
    s = _require_store()
    memories = retriever.retrieve(
        prompt=request.prompt,
        store=s,
        user_id=request.user_id,
        limit=request.limit,
        threshold=request.threshold,
    )
    return {"memories": [m.model_dump(mode="json") for m in memories], "count": len(memories)}


@app.post("/extract")
def extract_memories(request: ExtractRequest) -> dict:
    """Extract memories from a prompt/response pair and store them."""
    s = _require_store()
    client = _require_groq()
    try:
        extracted = extractor.extract(
            prompt=request.prompt,
            response=request.response,
            store=s,
            client=client,
            user_id=request.user_id,
        )
    except extractor.ExtractionParseError as e:
        raise HTTPException(status_code=502, detail=f"unparseable model reply: {e}")
    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Groq call failed: {type(e).__name__}")
    return {"extracted": [m.model_dump(mode="json") for m in extracted], "count": len(extracted)}


@app.post("/compress")
def compress_prompt(request: CompressRequest) -> dict:
    """Ask Groq to shorten a prompt; returns estimated token counts too."""
    return compressor.compress(request.prompt, _require_groq())


@app.get("/memories")
def list_memories(user_id: str = "default") -> dict:
    """List all of a user's memories, newest first."""
    memories = _require_store().get_all(user_id)
    return {"memories": [m.model_dump(mode="json") for m in memories], "count": len(memories)}


@app.delete("/memories")
def clear_memories(user_id: str = "default") -> dict:
    """Delete every memory belonging to a user."""
    return {"deleted": _require_store().delete_all(user_id)}


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str) -> dict:
    """Delete one memory by ID."""
    s = _require_store()
    if s.get_by_id(memory_id) is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    s.delete_memory(memory_id)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7842, log_level="info")
