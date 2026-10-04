"""
config.py
---------
Central place for every setting the application uses.

Keeping all "magic numbers" (model names, chunk sizes, limits) in one file
makes the project easier to tune and easier to explain: if you want to change
how the system behaves, you only need to look here.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Read variables from a local .env file (if it exists) into the environment.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ---- Application identity -------------------------------------------
    app_name: str = "ScholarLens"
    app_tagline: str = "Grounded question answering over your PDF documents"

    # ---- OpenAI models ---------------------------------------------------
    # Embedding model: turns text into vectors (1536 numbers per chunk).
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # Chat model: reads the retrieved context and writes the answer.
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # ---- Chunking --------------------------------------------------------
    # Chunk size is measured in characters (~4 characters ≈ 1 token).
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200

    # ---- Retrieval & generation -----------------------------------------
    default_top_k: int = 4
    max_top_k: int = 10
    default_temperature: float = 0.1

    # ---- PDF quality checks ---------------------------------------------
    # A page with fewer characters than this is treated as "no text".
    min_chars_per_page: int = 20
    # If a whole PDF has fewer characters than this, we warn the user that it
    # is probably a scanned document (images only) and needs OCR.
    min_chars_per_document: int = 200
    max_file_size_mb: int = 50


SETTINGS = Settings()


def get_api_key() -> str | None:
    """Return the OpenAI API key from the environment, or None if missing."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    # Treat the placeholder from .env.example as "missing".
    if not key or key.startswith("sk-your"):
        return None
    return key
