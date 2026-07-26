"""Hybrid retrieval.

The two arms fail differently: BM25 misses vocabulary mismatch ("why is it slow"
vs a note titled "the memory-fit problem"), and dense retrieval blurs exact
identifiers like b10075 or --no-repack. RRF is how both get a vote.
"""
from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass

from .config import Settings

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
    """FTS5 MATCH takes a query language, not a string. User text containing
    quotes, parentheses or bare AND/OR raises rather than returning nothing, so
    reduce it to bare terms."""
    return " ".join(_FTS_SAFE.sub(" ", query).split())


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
    blob = struct.pack(f"{len(vector)}f", *vector)
    rows = con.execute(
        "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
        "ORDER BY distance",
        (blob, settings.arm_depth),
    ).fetchall()
    return [r[0] for r in rows]


def search(con: sqlite3.Connection, settings: Settings, embedder,
           query: str, limit: int = 10) -> list[Hit]:
    keyword_ids = keyword_arm(con, settings, query)
    vector_ids = vector_arm(con, settings, embedder.embed_query(query))

    # Reciprocal Rank Fusion: score = sum over arms of 1/(k + rank). Computed in
    # Python rather than SQL because both arms are already materialised as small
    # ranked lists, and the SQL version is harder to read for no gain.
    scores: dict[int, float] = {}
    for ids in (keyword_ids, vector_ids):
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (settings.rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
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
