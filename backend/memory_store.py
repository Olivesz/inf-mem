"""
SQLite storage for memories and their embeddings.

PURPOSE:
Persist memories with their 384-dim embedding vectors in one SQLite table,
and keep the vectors in an in-memory matrix for fast similarity search.

Every write touches only the rows it changes. (An earlier version saved the
whole embedding matrix to a .npy file on every insert, so /extract slowed as
the store grew; bench/bench.py --scaling now fails if that comes back.)

INPUTS:
Memory objects, embeddings (numpy arrays of shape (384,))

OUTPUTS:
Persisted memories; Memory objects and embedding rows on read
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from models import Memory, MemoryType

DIM = 384


class MemoryStore:
    """
    Persistent memory storage: one SQLite file, vectors held in memory.

    memories.db holds each memory's fields and its embedding (float32 BLOB).
    On open, the embeddings are loaded into a matrix whose row i belongs to
    memory_ids[i]. The matrix grows by doubling, so an insert does not copy it.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """
        Open (or create) the store in data_dir, default ~/.inf_mem.

        Stores written by older versions, with vectors in embeddings.npy and
        memory_ids.npy, are migrated into the database once; the .npy files
        are then renamed with a .migrated suffix, not deleted.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".inf_mem"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "memories.db"

        self._init_db()
        self._migrate_npy()
        self._load()

    # ------------------------------------------------------------------ #
    # schema, migration, loading
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
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
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if "embedding" not in columns:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)")
        conn.commit()
        conn.close()

    def _migrate_npy(self) -> None:
        emb_path = self.data_dir / "embeddings.npy"
        ids_path = self.data_dir / "memory_ids.npy"
        if not (emb_path.exists() and ids_path.exists()):
            return
        vectors = np.load(emb_path, allow_pickle=False)
        ids = np.load(ids_path, allow_pickle=False).tolist()
        conn = self._connect()
        conn.executemany(
            "UPDATE memories SET embedding = ? WHERE id = ? AND embedding IS NULL",
            [(_blob(v), mid) for mid, v in zip(ids, vectors, strict=True)],
        )
        conn.commit()
        conn.close()
        emb_path.rename(emb_path.with_name("embeddings.npy.migrated"))
        ids_path.rename(ids_path.with_name("memory_ids.npy.migrated"))

    def _load(self) -> None:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, user_id, embedding, importance FROM memories WHERE embedding IS NOT NULL ORDER BY rowid"
        ).fetchall()
        conn.close()
        self.memory_ids: list[str] = [r[0] for r in rows]
        self._users: list[str] = [r[1] for r in rows]
        cap = max(16, len(rows))
        self._buf = np.zeros((cap, DIM), dtype=np.float32)
        self._imp = np.zeros(cap, dtype=np.float32)
        for i, r in enumerate(rows):
            self._buf[i] = np.frombuffer(r[2], dtype=np.float32)
            self._imp[i] = r[3]
        self._norms = np.linalg.norm(self._buf, axis=1)
        self._row = {mid: i for i, mid in enumerate(self.memory_ids)}
        self._users_np = None

    @property
    def embeddings(self) -> np.ndarray:
        """The live (N, 384) matrix; row i is memory_ids[i]. Do not write to it."""
        return self._buf[: len(self.memory_ids)]

    def _append_rows(self, ids: list[str], users: list[str], vectors: np.ndarray, importance: list[int]) -> None:
        n, k = len(self.memory_ids), len(ids)
        if n + k > len(self._buf):
            cap = max(2 * len(self._buf), n + k)
            self._buf = _grow(self._buf, cap, n)
            self._norms = _grow(self._norms, cap, n)
            self._imp = _grow(self._imp, cap, n)
        self._buf[n: n + k] = vectors
        self._norms[n: n + k] = np.linalg.norm(vectors, axis=1)
        self._imp[n: n + k] = importance
        for offset, (mid, user) in enumerate(zip(ids, users, strict=True)):
            self._row[mid] = n + offset
        self.memory_ids.extend(ids)
        self._users.extend(users)
        self._users_np = None

    def _drop_rows(self, rows: list[int]) -> None:
        keep = np.setdiff1d(np.arange(len(self.memory_ids)), rows)
        n = len(keep)
        self._buf[:n] = self._buf[keep]
        self._norms[:n] = self._norms[keep]
        self._imp[:n] = self._imp[keep]
        self.memory_ids = [self.memory_ids[i] for i in keep]
        self._users = [self._users[i] for i in keep]
        self._row = {mid: i for i, mid in enumerate(self.memory_ids)}
        self._users_np = None

    # ------------------------------------------------------------------ #
    # writes: each one touches only its own rows
    # ------------------------------------------------------------------ #

    def add_memory(self, memory: Memory, embedding: np.ndarray) -> None:
        """
        Add a new memory.

        Raises:
            ValueError: If embedding shape is not (384,)
        """
        if embedding.shape != (DIM,):
            raise ValueError(f"Embedding must have shape ({DIM},), got {embedding.shape}")
        self.add_many([memory], embedding.reshape(1, -1))

    def add_many(self, memories: list[Memory], embeddings: np.ndarray) -> None:
        """
        Add many memories in one transaction.

        Args:
            memories: Memory objects, in the same order as the embedding rows
            embeddings: Array of shape (len(memories), 384)
        """
        if embeddings.shape != (len(memories), DIM):
            raise ValueError(f"Embeddings must have shape ({len(memories)}, {DIM}), got {embeddings.shape}")
        vectors = embeddings.astype(np.float32)
        conn = self._connect()
        conn.executemany("""
            INSERT INTO memories
            (id, user_id, text, type, keywords, importance, created_at, updated_at, access_count, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(m.id, m.user_id, m.text, m.type.value, json.dumps(m.keywords), m.importance,
               m.created_at.isoformat(), m.updated_at.isoformat(), m.access_count, _blob(v))
              for m, v in zip(memories, vectors, strict=True)])
        conn.commit()
        conn.close()
        self._append_rows([m.id for m in memories], [m.user_id for m in memories], vectors,
                          [m.importance for m in memories])

    def update_memory(
        self,
        memory_id: str,
        text: str,
        keywords: list[str],
        importance: int,
        embedding: np.ndarray
    ) -> None:
        """
        Update a memory's text, keywords, importance and embedding.

        Raises:
            ValueError: If memory not found or embedding shape incorrect
        """
        if embedding.shape != (DIM,):
            raise ValueError(f"Embedding must have shape ({DIM},), got {embedding.shape}")
        if memory_id not in self._row:
            raise ValueError(f"Memory {memory_id} not found")
        vector = embedding.astype(np.float32)
        conn = self._connect()
        conn.execute("""
            UPDATE memories
            SET text = ?, keywords = ?, importance = ?, updated_at = ?, embedding = ?
            WHERE id = ?
        """, (text, json.dumps(keywords), importance, datetime.now(UTC).isoformat(), _blob(vector), memory_id))
        conn.commit()
        conn.close()
        row = self._row[memory_id]
        self._buf[row] = vector
        self._norms[row] = np.linalg.norm(vector)
        self._imp[row] = importance

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory.

        Raises:
            ValueError: If memory not found
        """
        if memory_id not in self._row:
            raise ValueError(f"Memory {memory_id} not found")
        conn = self._connect()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        conn.close()
        self._drop_rows([self._row[memory_id]])

    def delete_all(self, user_id: str) -> int:
        """
        Delete every memory belonging to a user.

        Returns:
            Number of memories deleted
        """
        rows = [i for i, u in enumerate(self._users) if u == user_id]
        conn = self._connect()
        conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        self._drop_rows(rows)
        return len(rows)

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #

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
        """A copy of the (N, 384) embedding matrix, rows in memory_ids order."""
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
        rows = np.flatnonzero(self._user_array() == user_id)
        if len(rows) == len(self.memory_ids):
            # One user owns every row (the usual case): a view, no copy.
            return list(self.memory_ids), self.embeddings
        return [self.memory_ids[i] for i in rows], self.embeddings[rows]

    def search(self, user_id: str, query: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Cosine similarity of query against each of a user's memories.

        Row norms are kept up to date on every write, so this is one
        matrix-vector product with no temporary copy of the matrix.

        Returns:
            (ids, cosines, importance), aligned
        """
        rows = np.flatnonzero(self._user_array() == user_id)
        n = len(self.memory_ids)
        sel = slice(0, n) if len(rows) == n else rows
        ids = list(self.memory_ids) if len(rows) == n else [self.memory_ids[i] for i in rows]
        qn = float(np.linalg.norm(query))
        dots = self._buf[sel] @ query
        denom = self._norms[sel] * qn
        sims = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
        return ids, sims, self._imp[sel].copy()

    def _user_array(self) -> np.ndarray:
        if self._users_np is None or len(self._users_np) != len(self._users):
            self._users_np = np.array(self._users, dtype=object)
        return self._users_np

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


def _grow(arr: np.ndarray, cap: int, n: int) -> np.ndarray:
    out = np.zeros((cap, *arr.shape[1:]), dtype=arr.dtype)
    out[:n] = arr[:n]
    return out


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _aware(dt: datetime) -> datetime:
    """Rows written before timestamps carried a zone were naive UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
