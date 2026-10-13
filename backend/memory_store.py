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
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from models import Memory, MemoryType


class MemoryStore:
    """
    Manages persistent storage of memories using SQLite + numpy embeddings.

    Maintains three files:
    - memories.db: SQLite, structured memory metadata
    - embeddings.npy: embedding vectors, shape (N, 384) for bge-small-en-v1.5
    - memory_ids.npy: the memory ID of each embedding row, in row order
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """
        Initialize storage directories and database.

        Creates the data directory (default ~/.inf_mem) if needed, initializes
        the SQLite schema, and loads or creates the embeddings array.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".inf_mem"
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

    def add_many(self, memories: list[Memory], embeddings: np.ndarray) -> None:
        """
        Add many memories with one SQLite transaction and one file write.

        Args:
            memories: Memory objects, in the same order as the embedding rows
            embeddings: Array of shape (len(memories), 384)
        """
        if embeddings.shape != (len(memories), 384):
            raise ValueError(f"Embeddings must have shape ({len(memories)}, 384), got {embeddings.shape}")
        conn = sqlite3.connect(self.db_path)
        conn.executemany("""
            INSERT INTO memories
            (id, user_id, text, type, keywords, importance, created_at, updated_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(m.id, m.user_id, m.text, m.type.value, json.dumps(m.keywords), m.importance,
               m.created_at.isoformat(), m.updated_at.isoformat(), m.access_count) for m in memories])
        conn.commit()
        conn.close()
        self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        self.memory_ids.extend(m.id for m in memories)
        self._save_embeddings()

    def update_memory(
        self,
        memory_id: str,
        text: str,
        keywords: list[str],
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
            datetime.now(UTC).isoformat(),
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

    def get_all(self, user_id: str) -> list[Memory]:
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

    def get_by_id(self, memory_id: str) -> Memory | None:
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

    def get_memory_ids(self, user_id: str) -> list[str]:
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

    def get_user_vectors(self, user_id: str) -> tuple[list[str], np.ndarray]:
        """
        Get a user's memory IDs and their embedding rows, in matching order.

        Returns:
            (ids, matrix) where matrix has shape (len(ids), 384)
        """
        wanted = set(self.get_memory_ids(user_id))
        rows = [i for i, mid in enumerate(self.memory_ids) if mid in wanted]
        return [self.memory_ids[i] for i in rows], self.embeddings[rows]

    def get_many(self, memory_ids: list[str]) -> list[Memory]:
        """Fetch several memories in one query (order not guaranteed)."""
        if not memory_ids:
            return []
        conn = sqlite3.connect(self.db_path)
        marks = ",".join("?" * len(memory_ids))
        rows = conn.execute(f"""
            SELECT id, user_id, text, type, keywords, importance, created_at, updated_at, access_count
            FROM memories WHERE id IN ({marks})
        """, memory_ids).fetchall()
        conn.close()
        return [self._row_to_memory(row) for row in rows]

    def increment_access_many(self, memory_ids: list[str]) -> None:
        """Increment the access count of several memories; unknown IDs are ignored."""
        if not memory_ids:
            return
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
            [(mid,) for mid in memory_ids],
        )
        conn.commit()
        conn.close()

    def delete_all(self, user_id: str) -> int:
        """
        Delete every memory belonging to a user.

        Returns:
            Number of memories deleted
        """
        doomed = set(self.get_memory_ids(user_id))
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()

        keep = [i for i, mid in enumerate(self.memory_ids) if mid not in doomed]
        self.embeddings = self.embeddings[keep]
        self.memory_ids = [self.memory_ids[i] for i in keep]
        self._save_embeddings()
        return len(doomed)

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
            created_at=_aware(datetime.fromisoformat(created_at)),
            updated_at=_aware(datetime.fromisoformat(updated_at)),
            access_count=access_count
        )


def _aware(dt: datetime) -> datetime:
    """Rows written before timestamps carried a zone were naive UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
