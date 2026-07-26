import pytest
from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus


class FakeEmbedder:
    """Returns a deterministic vector per text so tests never touch a network."""
    def __init__(self, dim=4):
        self.dim = dim
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]


@pytest.fixture
def env(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4,
                 chunk_chars=400, chunk_overlap_chars=40)
    con = connect(s)
    create_schema(con, s)
    return con, s, corpus


def test_indexes_a_new_file(env):
    con, s, corpus = env
    (corpus / "a.md").write_text("# A\n\nthe memory fit problem\n")
    r = index_corpus(con, s, FakeEmbedder())
    assert r.added == 1
    assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] >= 1
    assert con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] >= 1


def test_unchanged_file_is_not_re_embedded(env):
    con, s, corpus = env
    (corpus / "a.md").write_text("# A\n\nstable content\n")
    index_corpus(con, s, FakeEmbedder())
    e = FakeEmbedder()
    r = index_corpus(con, s, e)
    assert r.unchanged == 1 and r.added == 0
    assert e.calls == 0, "re-embedding unchanged content wastes the whole budget"


def test_modified_file_is_reindexed(env):
    con, s, corpus = env
    p = corpus / "a.md"
    p.write_text("# A\n\noriginal\n")
    index_corpus(con, s, FakeEmbedder())
    p.write_text("# A\n\ncompletely different words\n")
    r = index_corpus(con, s, FakeEmbedder())
    assert r.updated == 1
    hits = con.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'original'"
    ).fetchone()[0]
    assert hits == 0, "stale chunks must not survive a reindex"


def test_deleted_file_is_removed_from_index(env):
    con, s, corpus = env
    p = corpus / "a.md"
    p.write_text("# A\n\ndoomed content\n")
    index_corpus(con, s, FakeEmbedder())
    p.unlink()
    r = index_corpus(con, s, FakeEmbedder())
    assert r.removed == 1
    assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == 0


def test_unsupported_files_are_skipped(env):
    con, s, corpus = env
    (corpus / "x.docx").write_text("nope")
    (corpus / "a.md").write_text("# A\n\nyes\n")
    r = index_corpus(con, s, FakeEmbedder())
    assert r.added == 1
