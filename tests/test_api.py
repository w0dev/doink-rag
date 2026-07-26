from fastapi.testclient import TestClient
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
