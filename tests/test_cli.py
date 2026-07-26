import sqlite3
import struct

from rag.cli import main
from rag.config import Settings
from rag.db import connect as db_connect


def test_stats_on_empty_db(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(tmp_path))
    assert main(["stats"]) == 0
    assert "documents" in capsys.readouterr().out


def test_index_reports_counts(tmp_path, capsys, monkeypatch):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nhello world\n")
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("RAG_EMBED_DIM", "4")
    monkeypatch.setattr("rag.cli._embedder", lambda s: _Fake())
    assert main(["index"]) == 0
    assert "added=1" in capsys.readouterr().out


def test_unknown_command_returns_nonzero(capsys):
    assert main(["nope"]) != 0


def test_help_returns_zero(capsys):
    # argparse raises SystemExit(0) for --help; 0 is falsy, so a naive
    # `exc.code or 2` maps success to failure. A caller scripting
    # `rag --help && ...` must not see this as an error.
    assert main(["--help"]) == 0


def test_invalid_argument_returns_nonzero(capsys):
    # `search` requires a positional query; omitting it is a genuine usage
    # error and must still exit non-zero after the --help fix.
    assert main(["search"]) != 0


def test_search_prints_hits_for_matching_query(tmp_path, capsys, monkeypatch):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nthe memory fit problem\n")
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("RAG_EMBED_DIM", "4")
    monkeypatch.setattr("rag.cli._embedder", lambda s: _Fake())

    assert main(["index"]) == 0
    capsys.readouterr()

    assert main(["search", "memory"]) == 0
    assert "a.md" in capsys.readouterr().out


def _vector_for(text: str) -> list[float]:
    """Distinct per input text (length, vowel counts) so a chunk<->vector
    mis-association would change the stored bytes, unlike a constant vector
    that would pass under any shuffling."""
    return [float(len(text)), float(text.count("a")), float(text.count("e")), 1.0]


class _DerivedFake:
    def embed_documents(self, texts):
        return [_vector_for(t) for t in texts]

    def embed_query(self, text):
        return _vector_for(text)


def test_rebuild_vectors_reembeds_and_preserves_association(tmp_path, capsys, monkeypatch):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nalpha alpha alpha\n")
    (corpus / "b.md").write_text("# B\n\nbravissimo eeeeeee\n")
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("RAG_EMBED_DIM", "4")
    monkeypatch.setattr("rag.cli._embedder", lambda s: _DerivedFake())

    assert main(["index"]) == 0
    capsys.readouterr()

    assert main(["rebuild-vectors"]) == 0
    assert "re-embedded 2 chunks" in capsys.readouterr().out

    settings = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = db_connect(settings)
    chunks = con.execute("SELECT id, text FROM chunks").fetchall()
    assert len(chunks) == 2
    assert con.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == 2

    for row in chunks:
        vec_row = con.execute(
            "SELECT embedding FROM chunks_vec WHERE chunk_id = ?", (row["id"],)
        ).fetchone()
        stored = list(struct.unpack("4f", vec_row["embedding"]))
        assert stored == _vector_for(row["text"])


class _FailingEmbedder:
    """Simulates the served model's dimension not matching settings.embed_dim
    -- the exact ValueError Embedder._post raises for a real mismatch."""
    def embed_documents(self, texts):
        raise ValueError("embedding dimension 999 does not match configured 4")

    def embed_query(self, text):
        raise ValueError("embedding dimension 999 does not match configured 4")


def test_failing_embedder_returns_nonzero_with_message(tmp_path, capsys, monkeypatch):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nhello world\n")
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(corpus))
    monkeypatch.setenv("RAG_EMBED_DIM", "4")
    monkeypatch.setattr("rag.cli._embedder", lambda s: _FailingEmbedder())

    assert main(["index"]) != 0
    assert capsys.readouterr().err.strip() != ""


class _Fake:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


def test_sqlite_error_is_reported_cleanly_not_as_a_traceback(tmp_path, capsys, monkeypatch):
    """`except (RuntimeError, ValueError, httpx.HTTPError)` covers embedding
    and dimension failures but not the database itself. Two overlapping `rag
    index` runs (plausible if a corpus-sync timer overruns) raise
    sqlite3.OperationalError: database is locked -- and disk-full does too --
    which must surface as the same clean `error: ...` path as any other
    operational failure, not an uncaught traceback."""
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("RAG_CORPUS_DIR", str(tmp_path))

    def _raise(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("rag.cli.index_corpus", _raise)
    assert main(["index"]) == 1
    assert "database is locked" in capsys.readouterr().err
