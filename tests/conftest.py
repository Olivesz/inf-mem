"""Test configuration for inf-mem.

The backend modules import each other flatly (``from models import Memory``),
so we add ``backend/`` to the import path here. We also stub out ``fastembed``
when it isn't installed (and always avoid loading the real model) so the suite
runs fast and offline — tests monkeypatch ``retriever.encode`` directly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

try:  # pragma: no cover - environment dependent
    import fastembed  # noqa: F401
except ImportError:  # fastembed not installed -> inject a lightweight stub
    stub = types.ModuleType("fastembed")

    class TextEmbedding:  # minimal placeholder; real embedding is monkeypatched
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            raise RuntimeError("stub fastembed.embed should not be called in tests")

    stub.TextEmbedding = TextEmbedding
    sys.modules["fastembed"] = stub
