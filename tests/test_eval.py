import sys
import pathlib

from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "eval"))
from run_eval import evaluate, Question  # noqa: E402


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
