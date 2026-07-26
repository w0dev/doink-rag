"""HTTP surface.

Binds loopback in deployment; nothing off-box reaches it directly. Every planned
consumer — chat context injection, the family agent harness — talks to this same
endpoint, which is why it exists before any of them do.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI

from .config import Settings
from .db import connect, create_schema
from .embed import Embedder
from .search import search


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


def create_app(settings: Settings | None = None, embedder=None) -> FastAPI:
    settings = settings or Settings.from_env()
    embedder = embedder or Embedder(settings)

    # Establish schema once, at startup, not per request. Nothing about the
    # schema or embed_dim changes during the process's lifetime, so redoing it
    # on every call only buys a write transaction (create_schema commits an
    # INSERT into meta) that every /search and /healthz call would then have
    # to queue behind under WAL's single-writer rule. Doing it here instead
    # means a dimension mismatch fails loud at startup — a service that starts
    # and then fails every request is worse than one that refuses to start.
    con = connect(settings)
    try:
        create_schema(con, settings)
    finally:
        con.close()

    app = FastAPI(title="doink-rag")

    def _con():
        # A connection per request: SQLite connections are not thread-safe
        # across FastAPI's threadpool, and opening one is microseconds.
        return connect(settings)

    @app.post("/search")
    def do_search(request: SearchRequest):
        con = _con()
        try:
            hits = search(con, settings, embedder, request.query, request.limit)
            return {"query": request.query, "hits": [vars(h) for h in hits]}
        finally:
            con.close()

    @app.get("/healthz")
    def healthz():
        con = _con()
        try:
            docs = con.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
            vectors = con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0]
            return {
                "documents": docs,
                "chunks": chunks,
                "vectors": vectors,
                "vectors_complete": chunks == vectors,
                "embedder": embedder.health(),
            }
        finally:
            con.close()

    return app


app = create_app  # uvicorn factory target: uvicorn rag.api:app --factory
