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

    # Measured 2026-07-26 against 28 verified questions over a 156-chunk corpus
    # of real technical notes; see docs/TUNING.md.
    #
    # rrf_k is inert here: at arm_depth 10, k=10 and k=60 score identically
    # (MRR@10 0.529 both). It reshapes the curve among candidates already
    # admitted, so it cannot undo what arm_depth lets in. Left at the value from
    # the original RRF paper.
    rrf_k: int = 60
    # arm_depth is the lever that matters. At 50 each arm contributed 40-odd
    # candidates that BM25's IDF had already ranked as worthless, and RRF gave
    # every one of them a vote. Dropping to 10 dominated 50 on every metric
    # measured — MRR@10, and recall at 1, 3, 5, 10 and 20.
    #
    # Going further to 5 scores higher on MRR but caps the candidate pool at
    # two arms x 5, freezing recall at 0.786 however deep you look. 10 keeps
    # the ceiling open.
    #
    # This is corpus-size dependent: 10 of 156 chunks is 6%, and the right value
    # on a much larger corpus may differ. Re-run the harness rather than
    # assuming this transfers.
    arm_depth: int = 10

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
        # Retrieval constants are overridable so the eval harness can sweep them
        # without editing this file. That is the whole point of the harness.
        if v := os.getenv("RAG_RRF_K"):
            d["rrf_k"] = int(v)
        if v := os.getenv("RAG_ARM_DEPTH"):
            d["arm_depth"] = int(v)
        return cls(**d)
