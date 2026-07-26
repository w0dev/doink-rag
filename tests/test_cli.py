from rag.cli import main


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


class _Fake:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]
