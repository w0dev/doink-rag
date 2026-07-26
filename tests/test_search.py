import pytest
from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus
from rag.search import search, keyword_arm


class FakeEmbedder:
    """Vector arm returns [1,0,0,0] for everything except 'tls', which is
    orthogonal — so the two arms disagree and RRF has something to fuse."""
    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(t):
        return [0.0, 1.0, 0.0, 0.0] if "tls" in t.lower() else [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def env(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "mem.md").write_text("# Memory\n\nthe repack buffer was swapping badly\n")
    (corpus / "net.md").write_text("# Network\n\ntls termination happens in tailscale\n")
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=4)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, FakeEmbedder())
    return con, s


def test_keyword_arm_finds_stemmed_match(env):
    con, s = env
    ids = keyword_arm(con, s, "swap")
    assert len(ids) == 1


def test_search_returns_hits_with_provenance(env):
    con, s = env
    hits = search(con, s, FakeEmbedder(), "swapping")
    assert hits
    assert hits[0].path == "mem.md"
    assert hits[0].title == "Memory"
    assert "repack" in hits[0].text


def test_fusion_ranks_documents_matched_by_both_arms_first(env):
    con, s = env
    hits = search(con, s, FakeEmbedder(), "tls")
    assert hits[0].path == "net.md", "tls matches both arms and must rank first"


def test_query_with_no_keyword_match_still_returns_vector_hits(env):
    con, s = env
    hits = search(con, s, FakeEmbedder(), "zzzznotawordanywhere")
    assert hits, "the dense arm must still contribute when BM25 finds nothing"


def test_limit_is_respected(env):
    con, s = env
    assert len(search(con, s, FakeEmbedder(), "the", limit=1)) <= 1


def test_fts_special_characters_do_not_crash(env):
    con, s = env
    # Unescaped, this is an FTS5 syntax error rather than zero results.
    search(con, s, FakeEmbedder(), 'what is "memory fit" AND (swap)')
