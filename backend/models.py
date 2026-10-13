"""
Pydantic models for the inf-mem memory system.

PURPOSE:
Define all data structures used across the API, database, and internal modules.
Ensures type safety and validation for memory objects, API requests, and responses.

INPUTS:
None (defines models only)

OUTPUTS:
Pydantic models: MemoryType, Memory, ExtractRequest, RetrieveRequest, CompressRequest, ExtractedMemory
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import AliasChoices, BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


# The extension sends camelCase ``userId``; accept either spelling.
_USER_ID = AliasChoices("user_id", "userId")


class MemoryType(str, Enum):
    """Enum for the four types of memories the system can extract."""
    FACT = "FACT"
    PROJECT = "PROJECT"
    PREFERENCE = "PREFERENCE"
    EPISODE = "EPISODE"


class Memory(BaseModel):
    """
    Core memory object stored in the system.

    Represents a single memorable piece of information extracted from conversations.
    """
    id: str = Field(..., description="Unique identifier (UUID)")
    user_id: str = Field(default="default", description="User this memory belongs to")
    text: str = Field(..., description="The memory content")
    type: MemoryType = Field(..., description="Category of the memory")
    keywords: list[str] = Field(default_factory=list, description="Tags/keywords for the memory")
    importance: int = Field(default=5, ge=1, le=10, description="Importance score 1-10")
    created_at: datetime = Field(default_factory=_now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=_now, description="Last update timestamp")
    access_count: int = Field(default=0, description="Number of times retrieved")


class ExtractRequest(BaseModel):
    """API request to extract memories from a prompt/response pair."""
    prompt: str = Field(..., description="User prompt text")
    response: str | None = Field(default=None, description="Optional AI response to extract from")
    user_id: str = Field(default="default", validation_alias=_USER_ID, description="User identifier")


class RetrieveRequest(BaseModel):
    """API request to retrieve relevant memories."""
    prompt: str = Field(..., description="Query text to find relevant memories")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum memories to return")
    user_id: str = Field(default="default", validation_alias=_USER_ID, description="User identifier")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="Similarity threshold (0-1); omit for the calibrated default")


class CompressRequest(BaseModel):
    """API request to compress a prompt."""
    prompt: str = Field(..., description="Prompt text to compress")


class ExtractedMemory(BaseModel):
    """Output format from Groq extraction (intermediate format)."""
    text: str = Field(..., description="The memory text")
    type: str = Field(..., description="Memory type as string")
    keywords: list[str] = Field(default_factory=list, description="Keywords/tags")
    importance: int = Field(default=5, ge=1, le=10, description="Importance score")
