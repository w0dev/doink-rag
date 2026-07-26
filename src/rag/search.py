"""Hybrid retrieval.

The two arms fail differently: BM25 misses vocabulary mismatch ("why is it slow"
vs a note titled "the memory-fit problem"), and dense retrieval blurs exact
identifiers like b10075 or --no-repack. RRF is how both get a vote.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from .config import Settings
from .db import pack_vector

_FTS_SAFE = re.compile(r'[^\w\s]')


@dataclass
class Hit:
    chunk_id: int
    score: float
    text: str
    heading_path: str
    path: str
    title: str


def _sanitise(query: str) -> str:
    """FTS5 MATCH takes a query language, not a string. Stripping punctuation
    alone is not enough: AND, OR and NOT are ordinary word characters, so a
    bare "NOT working" still parses as the NOT operator and raises a syntax
    error rather than matching anything. Quoting each token turns it into a
    string literal FTS5 cannot read as an operator. Joining with OR (not a
    bare space) matters just as much: adjacent quoted strings are implicitly
    ANDed by FTS5, which would require a chunk to contain *every* term in the
    query -- for any natural-language, multi-word question that is nearly
    always false, silently zeroing out the keyword arm's contribution while
    RRF still produces plausible-looking fused scores from the vector arm
    alone. OR keeps this arm doing what BM25 ranking already assumes: reward
    partial matches instead of requiring a perfect one. Entirely-punctuation
    input reduces to no tokens and this still returns ""."""
    terms = _FTS_SAFE.sub(" ", query).split()
    return " OR ".join(f'"{t}"' for t in terms)


def keyword_arm(con: sqlite3.Connection, settings: Settings, query: str) -> list[int]:
    terms = _sanitise(query)
    if not terms:
        return []
    rows = con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
        "ORDER BY rank LIMIT ?",
        (terms, settings.arm_depth),
    ).fetchall()
    return [r[0] for r in rows]


def vector_arm(con: sqlite3.Connection, settings: Settings,
               vector: list[float]) -> list[int]:
    blob = pack_vector(vector)
    rows = con.execute(
        "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
        "ORDER BY distance",
        (blob, settings.arm_depth),
    ).fetchall()
    return [r[0] for r in rows]


def fuse(keyword_ids: list[int], vector_ids: list[int],
         settings: Settings) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score = sum over arms of 1/(k + rank).

    Separate from search() so an offline parameter sweep ranks identically to
    the live service. A sweep with its own copy of this arithmetic silently
    measures a different ranker — which is exactly what happened during the
    2026-07-26 tuning run, see docs/TUNING.md.

    Computed in Python rather than SQL because both arms are already
    materialised as small ranked lists.
    """
    scores: dict[int, float] = {}
    for ids in (keyword_ids, vector_ids):
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (settings.rrf_k + rank)

    # Ties are common at small arm_depth. A stable sort would break them by dict
    # insertion order — "whichever arm was iterated first wins" — which is an
    # accident of the loop above, not a policy. chunk_id as secondary key makes
    # the order explicit and reproducible.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def search(con: sqlite3.Connection, settings: Settings, embedder,
           query: str, limit: int = 10) -> list[Hit]:
    keyword_ids = keyword_arm(con, settings, query)
    vector_ids = vector_arm(con, settings, embedder.embed_query(query))

    ordered = fuse(keyword_ids, vector_ids, settings)[:limit]
    if not ordered:
        return []

    placeholders = ",".join("?" * len(ordered))
    rows = {
        r["id"]: r
        for r in con.execute(
            f"SELECT c.id, c.text, c.heading_path, d.path, d.title "
            f"FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE c.id IN ({placeholders})",
            [cid for cid, _ in ordered],
        )
    }
    return [
        Hit(cid, score, rows[cid]["text"], rows[cid]["heading_path"] or "",
            rows[cid]["path"], rows[cid]["title"] or "")
        for cid, score in ordered
        if cid in rows
    ]
