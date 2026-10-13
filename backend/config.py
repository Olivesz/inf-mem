"""
Groq API key loading.

The key comes from the GROQ_API_KEY environment variable if set (CI uses an
Actions secret), otherwise from config.yaml at the project root.
"""

import os
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# The value shipped in config.example.yaml. A key equal to this is not a key.
PLACEHOLDER_KEY = "your-groq-api-key-here"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"


def load_groq_api_key(config_path: Path | None = None) -> str | None:
    """
    Return the configured Groq key, or None if there is no usable key.

    A missing file, a missing GROQ_API_KEY line, an empty value and the
    placeholder from config.example.yaml all count as "not configured".
    """
    config_path = config_path or CONFIG_PATH
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key and config_path.exists():
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GROQ_API_KEY"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[1].strip().strip("'\"")
                break

    if not key or key == PLACEHOLDER_KEY:
        return None
    return key
