import sys
import pathlib

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


def test_chunk_level_scoring_distinguishes_the_right_passage(tmp_path):
    """Document-level recall on a small corpus is nearly free. This pins that
    expect_phrase actually scores at chunk level, which is where fusion
    dilution shows."""
    corpus = tmp_path / "c"
    corpus.mkdir()
    # One document, several sections: document recall is trivially 1.0, so only
    # chunk-level scoring can tell whether the right section was retrieved.
    (corpus / "notes.md").write_text(
        "# Notes\n\n## Swapping\n\nthe repack buffer was swapping badly\n\n"
        "## Networking\n\ntls termination in tailscale\n\n"
        "## Cooling\n\nthe fan curve is aggressive\n"
    )
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())

    q = Question("swapping", ["notes.md"], expect_phrase="repack")
    scores = evaluate(con, s, Fake(), [q], k=5)

    # Document level cannot discriminate — there is only one document.
    assert scores["keyword"].recall == 1.0
    # Chunk level must actually find the section containing "repack".
    assert scores["keyword"].chunk_mrr > 0.0
    assert scores["keyword"].chunk_recall == 1.0


def test_questions_without_a_phrase_still_score_at_document_level(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nthe repack buffer was swapping\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, Fake())

    scores = evaluate(con, s, Fake(), [Question("swapping", ["a.md"])], k=5)
    assert scores["keyword"].recall == 1.0
    # No expect_phrase supplied, so chunk metrics stay at zero rather than
    # inventing a score.
    assert scores["keyword"].chunk_mrr == 0.0


def test_shipped_questions_file_is_well_formed():
    """The shipped questions are the tuning baseline; a malformed entry would
    silently shrink the eval."""
    import pathlib
    qs = load_questions(pathlib.Path(__file__).parent.parent / "eval" / "questions.yaml")
    assert len(qs) >= 20, "the tuning baseline is 28 questions"
    assert all(q.expect_paths for q in qs)
    assert all(q.expect_phrase for q in qs), "chunk-level ground truth is required"
    assert {q.kind for q in qs} == {"vocab", "exact", "natural"}
