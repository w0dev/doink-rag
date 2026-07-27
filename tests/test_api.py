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


def _upload(client, name, content=b"# B\n\nfresh note about antennas\n"):
    return client.post("/documents", files={"file": (name, content)})


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


def test_list_documents_reports_paths_and_chunk_counts(tmp_path):
    r = _client(tmp_path).get("/documents")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert [d["path"] for d in docs] == ["a.md"]
    assert docs[0]["chunks"] >= 1


def test_upload_document_is_searchable(tmp_path):
    client = _client(tmp_path)
    r = _upload(client, "b.md")
    assert r.status_code == 201
    assert r.json()["path"] == "b.md"
    assert r.json()["index"]["added"] == 1
    # The upload went through the same index path as a synced file, so it is
    # immediately searchable and visible in the listing.
    assert client.get("/healthz").json()["documents"] == 2
    hits = client.post("/search", json={"query": "antennas"}).json()["hits"]
    assert any(h["path"] == "b.md" for h in hits)


def test_upload_strips_client_supplied_directories(tmp_path):
    client = _client(tmp_path)
    r = _upload(client, "../../evil.md")
    assert r.status_code == 201
    # The file must land flat inside corpus_dir, never above it.
    assert r.json()["path"] == "evil.md"
    assert (tmp_path / "c" / "evil.md").is_file()
    assert not (tmp_path / "evil.md").exists()


def test_upload_rejects_unsupported_type(tmp_path):
    assert _upload(_client(tmp_path), "b.exe").status_code == 422


def test_failed_upload_leaves_no_file_behind(tmp_path):
    """A file that cannot be indexed must not stay in corpus_dir, or every
    subsequent index run -- uploads and rsync syncs alike -- would fail on
    it."""
    client = _client(tmp_path)
    r = _upload(client, "broken.pdf", b"not actually a pdf")
    assert r.status_code == 422
    assert not (tmp_path / "c" / "broken.pdf").exists()
    assert client.get("/healthz").json()["documents"] == 1


def test_delete_document_removes_index_rows(tmp_path):
    client = _client(tmp_path)
    r = client.delete("/documents/a.md")
    assert r.status_code == 200
    assert r.json()["index"]["removed"] == 1
    assert client.get("/healthz").json()["documents"] == 0
    assert not (tmp_path / "c" / "a.md").exists()


def test_delete_unknown_document_is_404(tmp_path):
    assert _client(tmp_path).delete("/documents/nope.md").status_code == 404


def test_delete_cannot_escape_corpus_dir(tmp_path):
    (tmp_path / "outside.md").write_text("do not touch")
    r = _client(tmp_path).delete("/documents/../outside.md")
    assert r.status_code in (404, 422)
    assert (tmp_path / "outside.md").exists()


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
