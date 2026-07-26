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
    # rrf_k is flat for k >= 10: at arm_depth 10, k=10, 60 and 100 all score
    # 0.529. Below that it does matter (k=1 -> 0.512), so "inert" applies to the
    # upper range only. It reshapes the curve among candidates already admitted,
    # so it cannot undo what arm_depth lets in. Left at the RRF paper's value.
    rrf_k: int = 60
    # arm_depth is the lever that matters. At 50 each arm contributed candidates
    # that BM25's IDF had already ranked worthless, and RRF gave every one a
    # vote worth more than half a rank-1 vote. Dropping to 10 beat 50 on every
    # chunk-level metric measured.
    #
    # 5 vs 10 is a judgment call, not a result: 5 scores better at MRR and at
    # hit rate for k <= 5, but two arms x 5 caps the candidate pool at 10, so
    # three questions become unreachable at any depth. 10 keeps that open and
    # degrades better as the corpus grows. If the consumer only ever reads the
    # top 3-5 chunks, 5 is the better setting — RAG_ARM_DEPTH=5, no code change.
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
        # without editing this file. That is the whole point of the harness —
        # which is also why every failure mode here must be loud. A sweep that
        # silently used a different value than it printed would produce numbers
        # nobody could trust.
        d["rrf_k"] = _positive_int_env("RAG_RRF_K", d.get("rrf_k", cls.rrf_k))
        d["arm_depth"] = _positive_int_env(
            "RAG_ARM_DEPTH", d.get("arm_depth", cls.arm_depth))
        return cls(**d)


def _positive_int_env(name: str, default: int) -> int:
    """Read an int env var, failing loudly on anything unusable.

    An unset variable falls through to the default. An empty or non-numeric one
    does NOT: `RAG_ARM_DEPTH=""` silently taking the default is exactly the
    "reported numbers for settings it never used" failure this guards against.

    Values below 1 are rejected because they fail in confusing ways rather than
    obviously: arm_depth 0 makes both arms return nothing and search silently
    yields no results, arm_depth -1 becomes SQLite's "unlimited" in the keyword
    arm while the vector arm raises, and rrf_k 0 divides by zero on the first
    rank-0 chunk.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not an integer. Unset it to use the default "
            f"({default})."
        ) from None
    if value < 1:
        raise ValueError(f"{name}={value} must be >= 1 (default {default}).")
    return value
