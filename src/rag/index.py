"""Ingestion.

A library function, not a CLI command: the CLI calls it today and the planned
`POST /documents` upload endpoint will call the same thing.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .chunk import chunk_text
from .config import Settings
from .extract import SUPPORTED_SUFFIXES, extract


@dataclass
class IndexResult:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def index_corpus(con: sqlite3.Connection, settings: Settings, embedder) -> IndexResult:
    result = IndexResult()
    # con is caller-supplied and may be a long-lived connection the CLI or the
    # upload endpoint keeps reusing. If extract/chunk/embed raises partway
    # through the loop, the transaction must be rolled back here rather than
    # left open: otherwise a later, unrelated commit() on the same connection
    # would silently persist this run's partial inserts.
    try:
        on_disk: dict[str, Path] = {
            str(p.relative_to(settings.corpus_dir)): p
            for p in sorted(settings.corpus_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        }
        known = {
            row["path"]: (row["id"], row["content_hash"])
            for row in con.execute("SELECT id, path, content_hash FROM documents")
        }

        for rel, path in on_disk.items():
            digest = _hash(path)
            if rel in known and known[rel][1] == digest:
                result.unchanged += 1
                continue
            if rel in known:
                # Delete and reinsert rather than diff chunks: content changed,
                # so chunk boundaries may have moved and old chunks are
                # meaningless.
                con.execute("DELETE FROM documents WHERE id=?", (known[rel][0],))
                result.updated += 1
            else:
                result.added += 1
            _index_file(con, settings, embedder, rel, path, digest)

        for rel, (doc_id, _) in known.items():
            if rel not in on_disk:
                con.execute("DELETE FROM documents WHERE id=?", (doc_id,))
                result.removed += 1

        con.commit()
    except Exception:
        # All-or-nothing: a failing run leaves no trace, not a partial
        # document with no chunks that would then block re-indexing that
        # same path on the next run via the path UNIQUE constraint.
        con.rollback()
        raise

    return result


def _index_file(con, settings, embedder, rel: str, path: Path, digest: str) -> None:
    title, text = extract(path)
    chunks = chunk_text(text, settings)
    stat = path.stat()

    cur = con.execute(
        "INSERT INTO documents(path,content_hash,mtime,size,title,indexed_at) "
        "VALUES (?,?,?,?,?,?)",
        (rel, digest, stat.st_mtime, stat.st_size, title,
         datetime.now(timezone.utc).isoformat()),
    )
    doc_id = cur.lastrowid
    if not chunks:
        return

    vectors = embedder.embed_documents([c.text for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        cur = con.execute(
            "INSERT INTO chunks(document_id,ordinal,text,heading_path) "
            "VALUES (?,?,?,?)",
            (doc_id, chunk.ordinal, chunk.text, chunk.heading_path),
        )
        con.execute(
            "INSERT INTO chunks_vec(chunk_id,embedding) VALUES (?,?)",
            (cur.lastrowid, _pack(vector)),
        )
