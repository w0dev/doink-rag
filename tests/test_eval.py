import sys
import pathlib

import pytest

from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "eval"))
from run_eval import evaluate, load_questions, Question  # noqa: E402


class Fake:
    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)

    @staticmethod
    def _v(t):
        return [0.0, 1.0, 0.0, 0.0] if "tls" in t.lower() else [1.0, 0.0, 0.0, 0.0]


def test_evaluate_scores_all_three_arms(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "mem.md").write_text("# Memory\n\nthe repack buffer was swapping\n")
    (corpus / "net.md").write_text("# Net\n\ntls termination in tailscale\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())

    scores = evaluate(con, s, Fake(),
                      [Question("swapping", ["mem.md"])], k=5)
    assert set(scores) == {"keyword", "vector", "fused"}
    assert scores["keyword"].recall == 1.0
    assert 0.0 <= scores["fused"].mrr <= 1.0


def test_score_ids_returns_the_reciprocal_of_the_first_gold_rank():
    """Pins that MRR is rank-sensitive. A scorer that returns 1.0 for any hit
    passes every integration test in this file but is not MRR."""
    from run_eval import _score_ids
    assert _score_ids([9, 8, 7], {9}) == (1.0, 1.0)
    assert _score_ids([9, 8, 7], {8}) == (1.0, 0.5)
    assert _score_ids([9, 8, 7], {7}) == (1.0, 1.0 / 3)
    assert _score_ids([9, 8, 7], {4}) == (0.0, 0.0)
    # first gold wins, not last
    assert _score_ids([9, 8, 7], {8, 7}) == (1.0, 0.5)


def test_gold_chunks_selects_only_chunks_containing_the_phrase(tmp_path):
    """Pins that expect_phrase narrows the gold set. A _gold_chunks that
    returned every chunk of the expected document would pass the integration
    tests while making chunk scoring meaningless."""
    from run_eval import _gold_chunks
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "notes.md").write_text(
        "# Notes\n\n## One\n\nalpha content here\n\n"
        "## Two\n\nbeta DISTINCTIVE content\n\n## Three\n\ngamma content\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4,
                 chunk_chars=60, chunk_overlap_chars=10)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())

    total = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    gold = _gold_chunks(con, ["notes.md"], "DISTINCTIVE")
    assert total > 1, "fixture must produce several chunks or this proves nothing"
    assert 0 < len(gold) < total, "gold must be a strict subset of the document"
    for cid in gold:
        text = con.execute("SELECT text FROM chunks WHERE id=?", (cid,)).fetchone()[0]
        assert "DISTINCTIVE" in text


def test_chunk_metrics_are_not_document_metrics(tmp_path):
    """The decisive test: a case where document scoring says perfect and chunk
    scoring says missed. Any implementation that computes chunk metrics from
    document paths collapses these two and fails here."""
    from run_eval import evaluate, Question
    corpus = tmp_path / "c"
    corpus.mkdir()
    # 'padding' appears in many sections so the keyword arm returns many
    # candidates; the gold phrase sits in the last one, beyond k.
    body = "\n\n".join(
        f"## S{i}\n\npadding padding padding section {i}" for i in range(12))
    body += "\n\n## Target\n\npadding UNIQUEGOLDTOKEN here"
    (corpus / "big.md").write_text("# Big\n\n" + body + "\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4,
                 chunk_chars=80, chunk_overlap_chars=10)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())

    q = Question("padding", ["big.md"], expect_phrase="UNIQUEGOLDTOKEN")
    scores = evaluate(con, s, Fake(), [q], k=3)

    # Only one document exists, so document-level scoring cannot miss.
    assert scores["keyword"].recall == 1.0
    assert scores["keyword"].mrr == 1.0
    # But the gold chunk is not in the top 3, so chunk-level scoring must miss.
    assert scores["keyword"].chunk_recall == 0.0
    assert scores["keyword"].chunk_mrr == 0.0


def test_evaluate_refuses_to_silently_shrink_its_sample(tmp_path):
    """A phrase that matches nothing means the corpus and questions have
    drifted. Dropping the question would report a score over a smaller sample
    than the header claims."""
    from run_eval import evaluate, Question
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nthe repack buffer was swapping\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())

    q = Question("swapping", ["a.md"], expect_phrase="THIS PHRASE IS NOT PRESENT")
    with pytest.raises(ValueError, match="drifted apart"):
        evaluate(con, s, Fake(), [q], k=5)


def test_shipped_questions_file_is_well_formed():
    """The shipped questions are the tuning baseline; a malformed entry would
    silently shrink the eval."""
    import pathlib
    qs = load_questions(pathlib.Path(__file__).parent.parent / "eval" / "questions.yaml")
    assert len(qs) == 28, "the tuning baseline in docs/TUNING.md is 28 questions"
    assert all(q.expect_paths for q in qs)
    assert all(q.expect_phrase for q in qs), "chunk-level ground truth is required"
    assert {q.kind for q in qs} == {"vocab", "exact", "natural"}
