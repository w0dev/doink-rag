from pathlib import Path

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


class FailingEmbedder:
    """Succeeds for the first `fail_after` files, then raises -- simulating a
    real embedding server timing out or rate-limiting partway through a
    multi-file corpus."""
    def __init__(self, fail_after=1):
        self.calls = 0
        self.fail_after = fail_after

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("embedding server timeout")
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]


class ShortEmbedder:
    """Returns one fewer vector than it was given chunks for -- simulating a
    provider that silently drops a trailing item (batch truncation, a
    filtered-out empty string, etc). zip(chunks, vectors) stops at the
    shorter iterable, so this pairs up correctly for every item it does
    produce; the mismatch is invisible unless the lengths are checked before
    zipping."""
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts][:-1]


def test_short_vector_list_raises_instead_of_silently_dropping_chunks(env):
    con, s, corpus = env
    # Two paragraphs under one heading with a small chunk budget, forced
    # apart so chunk_text produces more than one chunk for embed_documents to
    # under-supply vectors for.
    s = Settings(db_path=s.db_path, corpus_dir=s.corpus_dir, embed_dim=4,
                 chunk_chars=20, chunk_overlap_chars=2)
    (corpus / "a.md").write_text(
        "# A\n\nfirst paragraph is long enough on its own\n\n"
        "second paragraph is also long enough on its own\n"
    )
    with pytest.raises(ValueError):
        index_corpus(con, s, ShortEmbedder())
    # a partial/mismatched write must not persist -- same all-or-nothing
    # guarantee as any other mid-run failure.
    assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_uppercase_suffixes_are_indexed(env):
    # index.py and extract.py both lowercase the suffix before comparing
    # against SUPPORTED_SUFFIXES; nothing pinned that end to end before this.
    con, s, corpus = env
    (corpus / "A.MD").write_text("# A\n\nmarkdown with an uppercase suffix\n")
    (corpus / "B.TXT").write_text("plain text with an uppercase suffix\n")
    r = index_corpus(con, s, FakeEmbedder())
    assert r.added == 2
    paths = {row["path"] for row in con.execute("SELECT path FROM documents")}
    assert paths == {"A.MD", "B.TXT"}


def test_nested_subdirectory_corpus_stores_relative_path(env):
    # rglob finds files at any depth and relative_to derives the stored path
    # from settings.corpus_dir; nothing pinned that the separators and nesting
    # survive intact before this.
    con, s, corpus = env
    nested = corpus / "sub" / "deep"
    nested.mkdir(parents=True)
    (nested / "note.md").write_text("# Note\n\nnested content\n")
    r = index_corpus(con, s, FakeEmbedder())
    assert r.added == 1
    path = con.execute("SELECT path FROM documents").fetchone()["path"]
    assert path == str(Path("sub") / "deep" / "note.md")


def test_failure_partway_through_leaves_db_unchanged(env):
    con, s, corpus = env
    (corpus / "a.md").write_text("# A\n\nfirst file content\n")
    (corpus / "b.md").write_text("# B\n\nsecond file content\n")

    with pytest.raises(RuntimeError):
        index_corpus(con, s, FailingEmbedder(fail_after=1))

    # a.md would have been inserted successfully before b.md's embed call
    # raised; the whole run must roll back, not leave a.md half-indexed.
    assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == 0
