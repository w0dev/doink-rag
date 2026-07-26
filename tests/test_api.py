import sqlite3

import pytest
from fastapi.testclient import TestClient

import rag.api as api_module
from rag.api import create_app
from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus


class Fake:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    def health(self):
        return True


def _client(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nthe memory fit problem\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())
    con.close()
    return TestClient(create_app(s, Fake()))


def test_search_returns_hits(tmp_path):
    r = _client(tmp_path).post("/search", json={"query": "memory"})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["path"] == "a.md"
    assert "score" in body["hits"][0]


def test_healthz_reports_index_and_embedder(tmp_path):
    r = _client(tmp_path).get("/healthz")
    assert r.status_code == 200
    assert r.json()["embedder"] is True
    assert r.json()["documents"] == 1


def test_empty_query_is_rejected(tmp_path):
    assert _client(tmp_path).post("/search", json={"query": "  "}).status_code == 422


def test_create_app_rejects_dimension_mismatch(tmp_path):
    """embed_dim is baked into chunks_vec's DDL; a process started against a
    database built for a different dimension must fail at startup rather than
    on every subsequent request."""
    corpus = tmp_path / "c"
    corpus.mkdir()
    s4 = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s4)
    create_schema(con, s4)
    con.close()

    s8 = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=8)
    with pytest.raises(RuntimeError):
        create_app(s8, Fake())


def test_search_request_closes_its_connection(tmp_path, monkeypatch):
    """Regression guard for the per-request connection lifecycle: wrap the
    module's `connect` to observe the connection a real request opens, then
    assert it was closed. A closed sqlite3.Connection raises on any further
    use, which is the observable consequence — no production instrumentation
    needed."""
    client = _client(tmp_path)

    opened = []
    real_connect = api_module.connect

    def tracking_connect(settings):
        con = real_connect(settings)
        opened.append(con)
        return con

    monkeypatch.setattr(api_module, "connect", tracking_connect)

    r = client.post("/search", json={"query": "memory"})
    assert r.status_code == 200
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")
