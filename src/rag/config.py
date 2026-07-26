"""Settings. Every tunable lives here; nothing downstream hardcodes a value.

embed_dim in particular is load-bearing: it is baked into the vec0 table's DDL,
so changing the model means rebuilding that table and re-embedding everything.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("rag.db")
    corpus_dir: Path = Path("corpus")

    embed_url: str = "http://127.0.0.1:8092"
    embed_model: str = "Qwen3-Embedding-0.6B"
    embed_dim: int = 1024
    embed_batch: int = 16

    # ~512 tokens at ~4 chars/token. An approximation, deliberately: a real
    # tokenizer would be a dependency and a network hop for a sizing decision
    # that only needs to be roughly right.
    chunk_chars: int = 2048
    chunk_overlap_chars: int = 256

    rrf_k: int = 60
    arm_depth: int = 50

    # Qwen3 embedding is asymmetric: queries carry an instruction, documents
    # never do. Getting this wrong degrades silently.
    query_instruction: str = (
        "Given a search query, retrieve relevant passages from personal notes"
    )

    @classmethod
    def from_env(cls) -> "Settings":
        d = {}
        if v := os.getenv("RAG_DB_PATH"):
            d["db_path"] = Path(v)
        if v := os.getenv("RAG_CORPUS_DIR"):
            d["corpus_dir"] = Path(v)
        if v := os.getenv("RAG_EMBED_URL"):
            d["embed_url"] = v
        if v := os.getenv("RAG_EMBED_MODEL"):
            d["embed_model"] = v
        if v := os.getenv("RAG_EMBED_DIM"):
            d["embed_dim"] = int(v)
        return cls(**d)
