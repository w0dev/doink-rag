"""HTTP surface.

Binds loopback in deployment; nothing off-box reaches it directly. Every planned
consumer — chat context injection, the family agent harness — talks to this same
endpoint, which is why it exists before any of them do.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, UploadFile

from .config import Settings
from .db import connect, create_schema
from .embed import Embedder
from .extract import SUPPORTED_SUFFIXES
from .index import index_corpus
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

    @app.get("/documents")
    def list_documents():
        con = _con()
        try:
            rows = con.execute(
                "SELECT d.path, d.title, d.size, d.indexed_at, count(c.id) AS chunks "
                "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
                "GROUP BY d.id ORDER BY d.path"
            ).fetchall()
            return {"documents": [dict(row) for row in rows]}
        finally:
            con.close()

    @app.post("/documents", status_code=201)
    def upload_document(file: UploadFile):
        # The upload lands in corpus_dir and then the normal sync runs, so an
        # uploaded file is indistinguishable from an rsync'd one: same change
        # detection, same all-or-nothing transaction, and corpus-sync.yml will
        # rsync it back out rather than fight over it.
        name = Path(file.filename or "").name  # strip any client-supplied dirs
        if not name:
            raise HTTPException(status_code=422, detail="missing filename")
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=422,
                detail=f"unsupported file type {Path(name).suffix!r}; "
                f"supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
            )
        settings.corpus_dir.mkdir(parents=True, exist_ok=True)
        (settings.corpus_dir / name).write_bytes(file.file.read())

        con = _con()
        try:
            result = index_corpus(con, settings, embedder)
        except Exception:
            # index_corpus rolled the transaction back; remove the file too so
            # a failed upload leaves no trace for the next sync to trip over
            # (a PDF that PyMuPDF cannot parse would otherwise fail every
            # subsequent index run for every caller).
            (settings.corpus_dir / name).unlink(missing_ok=True)
            raise HTTPException(
                status_code=422, detail=f"indexing failed for {name!r}"
            ) from None
        finally:
            con.close()
        return {"path": name, "index": vars(result)}

    @app.delete("/documents/{path:path}")
    def delete_document(path: str):
        target = (settings.corpus_dir / path).resolve()
        # Uploaded files are always flat, but rsync'd corpus files can be
        # nested, so path may contain separators; the resolve-and-contain
        # check is what stops it from escaping corpus_dir.
        if not target.is_relative_to(settings.corpus_dir.resolve()):
            raise HTTPException(status_code=422, detail="path escapes corpus dir")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no such document: {path}")
        target.unlink()

        con = _con()
        try:
            # Removal is just another sync: index_corpus sees the file gone
            # and deletes its rows, same as if rsync had removed it.
            result = index_corpus(con, settings, embedder)
        finally:
            con.close()
        return {"path": path, "index": vars(result)}

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
