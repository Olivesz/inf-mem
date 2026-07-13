"""
SQLite + numpy storage backend for the inf-mem system.

PURPOSE:
Provide persistent storage for memories and their embeddings.
Manages both relational data (SQLite) and high-dimensional vectors (numpy).

INPUTS:
Memory objects, embeddings (numpy arrays of shape (N, 384))

OUTPUTS:
Persisted memories and embeddings; retrieved Memory objects from database
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import numpy as np

from models import Memory, MemoryType


class MemoryStore:
    """
    Manages persistent storage of memories using SQLite + numpy embeddings.

    Maintains two storage backends:
    - SQLite database: structured memory metadata
    - Numpy file: embedding vectors (shape: N x 384 for all-MiniLM-L6-v2)

    The embedding matrix rows correspond to memory IDs in a fixed order.
    """

    def __init__(self) -> None:
        """
        Initialize storage directories and database.

        Creates ~/.inf_mem directory if needed, initializes SQLite schema,
        and loads or creates the embeddings numpy array.
        """
        # Create storage directory
        self.data_dir = Path.home() / ".inf_mem"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "memories.db"
        self.embeddings_path = self.data_dir / "embeddings.npy"
        self.ids_path = self.data_dir / "memory_ids.npy"  # Maintain order of memory IDs

        # Initialize database
        self._init_db()

        # Load or create embeddings array
        # Shape: (N, 384) where N is number of memories and 384 is embedding dimension
        if self.embeddings_path.exists() and self.ids_path.exists():
            self.embeddings = np.load(self.embeddings_path, allow_pickle=False)
            self.memory_ids = np.load(self.ids_path, allow_pickle=False).tolist()
        else:
            # Start with empty array and ID list
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.memory_ids = []

    def _init_db(self) -> None:
        """
        Initialize SQLite database schema if it doesn't exist.

        Creates the memories table with all required columns.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                type TEXT NOT NULL,
                keywords TEXT NOT NULL,
                importance INTEGER NOT NULL DEFAULT 5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        # Create index on user_id for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
        """)

        conn.commit()
        conn.close()

    def add_memory(self, memory: Memory, embedding: np.ndarray) -> None:
        """
        Add a new memory to storage.

        Args:
            memory: Memory object with populated fields
            embedding: 1D numpy array of shape (384,) from sentence-transformers

        Raises:
            ValueError: If embedding shape is incorrect
        """
        if embedding.shape != (384,):
            raise ValueError(f"Embedding must have shape (384,), got {embedding.shape}")

        # Insert into SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO memories
            (id, user_id, text, type, keywords, importance, created_at, updated_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.user_id,
            memory.text,
            memory.type.value,
            json.dumps(memory.keywords),
            memory.importance,
            memory.created_at.isoformat(),
            memory.updated_at.isoformat(),
            memory.access_count
        ))

        conn.commit()
        conn.close()

        # Append to numpy array and ID list
        self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
        self.memory_ids.append(memory.id)

        # Save updated arrays
        self._save_embeddings()

    def update_memory(
        self,
        memory_id: str,
        text: str,
        keywords: List[str],
        importance: int,
        embedding: np.ndarray
    ) -> None:
        """
        Update an existing memory's text, keywords, importance, and embedding.

        Args:
            memory_id: ID of memory to update
            text: New memory text
            keywords: New keywords list
            importance: New importance score
            embedding: New embedding vector

        Raises:
            ValueError: If memory not found or embedding shape incorrect
        """
        if embedding.shape != (384,):
            raise ValueError(f"Embedding must have shape (384,), got {embedding.shape}")

        if memory_id not in self.memory_ids:
            raise ValueError(f"Memory {memory_id} not found")

        # Update SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE memories
            SET text = ?, keywords = ?, importance = ?, updated_at = ?
            WHERE id = ?
        """, (
            text,
            json.dumps(keywords),
            importance,
            datetime.utcnow().isoformat(),
            memory_id
        ))

        conn.commit()
        conn.close()

        # Update numpy array
        idx = self.memory_ids.index(memory_id)
        self.embeddings[idx] = embedding

        # Save updated arrays
        self._save_embeddings()

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from storage.

        Args:
            memory_id: ID of memory to delete

        Raises:
            ValueError: If memory not found
        """
        if memory_id not in self.memory_ids:
            raise ValueError(f"Memory {memory_id} not found")

        # Delete from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

        conn.commit()
        conn.close()

        # Delete from numpy array
        idx = self.memory_ids.index(memory_id)
        self.embeddings = np.delete(self.embeddings, idx, axis=0)
        self.memory_ids.pop(idx)

        # Save updated arrays
        self._save_embeddings()

    def get_all(self, user_id: str) -> List[Memory]:
        """
        Retrieve all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            List of Memory objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, user_id, text, type, keywords, importance, created_at, updated_at, access_count
            FROM memories
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        memories = []
        for row in rows:
            memories.append(self._row_to_memory(row))

        return memories

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a single memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory object if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, user_id, text, type, keywords, importance, created_at, updated_at, access_count
            FROM memories
            WHERE id = ?
        """, (memory_id,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_memory(row)

    def get_embeddings(self) -> np.ndarray:
        """
        Get the full embeddings matrix.

        Returns:
            Numpy array of shape (N, 384) where N is number of memories
        """
        return self.embeddings.copy()

    def get_memory_ids(self, user_id: str) -> List[str]:
        """
        Get ordered list of memory IDs for a user.

        The returned order corresponds to rows in the embeddings matrix
        for memories belonging to this user.

        Args:
            user_id: User identifier

        Returns:
            Ordered list of memory IDs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id FROM memories
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def increment_access(self, memory_id: str) -> None:
        """
        Increment the access count for a memory.

        Args:
            memory_id: Memory identifier

        Raises:
            ValueError: If memory not found
        """
        memory = self.get_by_id(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} not found")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE memories
            SET access_count = access_count + 1
            WHERE id = ?
        """, (memory_id,))

        conn.commit()
        conn.close()

    def _save_embeddings(self) -> None:
        """Save embeddings and ID list to numpy files.

        IDs are stored as a plain unicode string array (not a pickled object
        array) so they can be reloaded with ``allow_pickle=False`` — safer and
        avoids the load-time crash that pickled object arrays caused.
        """
        np.save(self.embeddings_path, self.embeddings, allow_pickle=False)
        np.save(self.ids_path, np.array(self.memory_ids, dtype=str), allow_pickle=False)

    @staticmethod
    def _row_to_memory(row: tuple) -> Memory:
        """
        Convert a SQLite row to a Memory object.

        Args:
            row: Tuple from database query

        Returns:
            Memory object
        """
        (
            memory_id, user_id, text, mem_type, keywords_json,
            importance, created_at, updated_at, access_count
        ) = row

        return Memory(
            id=memory_id,
            user_id=user_id,
            text=text,
            type=MemoryType(mem_type),
            keywords=json.loads(keywords_json),
            importance=importance,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            access_count=access_count
        )
