"""Command line entry point.

`rebuild-vectors` is first-class rather than an afterthought: the vec0 table
bakes its dimension into DDL, so changing embedding model requires dropping and
rebuilding it, and that will happen every time a model is evaluated.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import httpx

from .config import Settings
from .db import connect, create_schema, pack_vector, schema_dim
from .embed import Embedder
from .index import index_corpus
from .search import search


def _embedder(settings: Settings):
    return Embedder(settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="sync the index with the corpus directory")
    sub.add_parser("stats", help="show index contents")
    sub.add_parser("rebuild-vectors", help="drop and re-embed every chunk")
    p_search = sub.add_parser("search", help="hybrid search")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--limit", type=int, default=10)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse raises SystemExit(0) for --help/-h and SystemExit(2) for a
        # genuine usage error. exc.code is only ever an int here, but guard
        # against None defensively rather than let `int(None)` raise.
        return exc.code if isinstance(exc.code, int) else 0

    settings = Settings.from_env()
    con = None
    try:
        con = connect(settings)
        create_schema(con, settings)

        if args.command == "index":
            result = index_corpus(con, settings, _embedder(settings))
            print(f"added={result.added} updated={result.updated} "
                  f"removed={result.removed} unchanged={result.unchanged}")
            return 0

        if args.command == "stats":
            docs = con.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
            vectors = con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0]
            print(f"documents={docs} chunks={chunks} vectors={vectors} "
                  f"dim={schema_dim(con)} model={settings.embed_model}")
            if chunks != vectors:
                print("WARNING: chunk and vector counts differ; the dense arm is "
                      "missing entries. Run `rag rebuild-vectors`.", file=sys.stderr)
            return 0

        if args.command == "rebuild-vectors":
            embedder = _embedder(settings)
            con.execute("DELETE FROM chunks_vec")
            rows = con.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
            for start in range(0, len(rows), settings.embed_batch):
                batch = rows[start:start + settings.embed_batch]
                vectors = embedder.embed_documents([r["text"] for r in batch])
                for row, vector in zip(batch, vectors):
                    con.execute(
                        "INSERT INTO chunks_vec(chunk_id,embedding) VALUES (?,?)",
                        (row["id"], struct.pack(f"{len(vector)}f", *vector)),
                    )
            con.commit()
            print(f"re-embedded {len(rows)} chunks")
            return 0

        if args.command == "search":
            for hit in search(con, settings, _embedder(settings), args.query, args.limit):
                location = f"{hit.path}" + (f" :: {hit.heading_path}" if hit.heading_path else "")
                print(f"{hit.score:.4f}  {location}")
                print(f"        {hit.text[:160].strip()}")
            return 0

        return 2
    except (RuntimeError, ValueError, httpx.HTTPError, sqlite3.Error) as exc:
        # These are the operational failures a user can act on: the embedding
        # server unreachable or returning the wrong model (httpx.HTTPError,
        # Embedder's dimension ValueError), an embed_dim mismatch between
        # this database and the current config (create_schema's RuntimeError),
        # or the database itself (sqlite3.Error) -- e.g. "database is locked"
        # from two overlapping `rag index` runs, or a disk-full write. Anything
        # not in this set is a programming error and should surface as a
        # traceback rather than being swallowed here. A mid-batch failure must
        # not persist: nothing above this point calls commit() until the
        # command finishes, so discarding con without committing leaves the
        # old vectors intact.
        if con is not None:
            con.rollback()
        print(f"error: {exc}", file=sys.stderr)
        return 1
