"""Schema and connections.

One file holds everything: documents, chunks, the FTS5 index and the vectors.
That is the entire reason this design needs no vector database — sqlite-vec is
an extension, not a service.
"""
from __future__ import annotations

import sqlite3
import struct

from .config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id            INTEGER PRIMARY KEY,
  path          TEXT UNIQUE NOT NULL,
  content_hash  TEXT NOT NULL,
  mtime         REAL NOT NULL,
  size          INTEGER NOT NULL,
  title         TEXT,
  indexed_at    TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id           INTEGER PRIMARY KEY,
  document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal      INTEGER NOT NULL,
  text         TEXT NOT NULL,
  heading_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- external-content FTS5: the index stores no copy of the text, so chunks is
-- the single source of truth and the triggers below keep them in step.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, heading_path,
  content='chunks', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, heading_path)
  VALUES (new.id, new.text, new.heading_path);
END;

-- External-content tables need the 'delete' command with the OLD values,
-- otherwise the index keeps a phantom entry that matches queries forever.
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
  VALUES ('delete', old.id, old.text, old.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
  VALUES ('delete', old.id, old.text, old.heading_path);
  INSERT INTO chunks_fts(rowid, text, heading_path)
  VALUES (new.id, new.text, new.heading_path);
END;

-- chunks_vec is a vec0 virtual table, so it is not covered by chunks'
-- ON DELETE CASCADE from documents. Without this trigger a deleted chunk
-- leaves an orphaned vector that keeps surfacing in search results forever.
CREATE TRIGGER IF NOT EXISTS chunks_ad_vec AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_vec WHERE chunk_id = old.id;
END;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def connect(settings: Settings) -> sqlite3.Connection:
    con = sqlite3.connect(settings.db_path)
    con.row_factory = sqlite3.Row
    try:
        con.enable_load_extension(True)
    except AttributeError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "This Python cannot load SQLite extensions, so sqlite-vec is "
            "unavailable. pyenv builds Python without them. Run under uv."
        ) from exc
    import sqlite_vec

    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    return con


def create_schema(con: sqlite3.Connection, settings: Settings) -> None:
    con.executescript(SCHEMA)
    # chunks_vec is CREATE ... IF NOT EXISTS, so once it exists on disk its
    # baked-in dimension is fixed regardless of what settings says now. Check
    # before touching it, or a config drift silently leaves meta and the real
    # table disagreeing until a query fails deep inside vec0 with a confusing
    # dimension error.
    existing_dim = schema_dim(con)
    if existing_dim is not None and existing_dim != settings.embed_dim:
        raise RuntimeError(
            f"Database at {settings.db_path} was built with embed_dim="
            f"{existing_dim}, but config specifies embed_dim="
            f"{settings.embed_dim}. chunks_vec bakes its dimension into its "
            f"DDL and cannot be resized in place. Changing embedding models "
            f"requires dropping chunks_vec and re-embedding the corpus."
        )
    # vec0 bakes the dimension into its DDL, so it cannot be created from a
    # constant. Changing models means dropping this table and re-embedding.
    con.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{settings.embed_dim}])"
    )
    con.execute(
        "INSERT INTO meta(key,value) VALUES('embed_dim',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(settings.embed_dim),),
    )
    con.commit()


def schema_dim(con: sqlite3.Connection) -> int | None:
    row = con.execute("SELECT value FROM meta WHERE key='embed_dim'").fetchone()
    return int(row[0]) if row else None


def pack_vector(vector: list[float]) -> bytes:
    """Pack a vector into the byte layout `chunks_vec`'s FLOAT[N] column
    expects. This is the single place that decision is made: index.py,
    search.py and cli.py's rebuild-vectors all write or read this same blob
    format, and if any of the three disagreed on width or byte order the
    dense arm would silently return wrong neighbours -- no exception, just
    bad results. Explicit little-endian (`<Nf`) rather than native order:
    equivalent on this hardware, but removes the assumption that native order
    is always little-endian.
    """
    return struct.pack(f"<{len(vector)}f", *vector)
