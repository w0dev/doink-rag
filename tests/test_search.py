import re

import pytest
from rag.config import Settings
from rag.db import connect, create_schema
from rag.index import index_corpus
from rag.search import search, keyword_arm, vector_arm


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


class MarkerEmbedder:
    """Assigns each chunk a 1-D vector equal to the integer found in its
    "markerN" tag. Query embeds to [0.0], so vector-arm distance is exactly N
    — nearest-first order is 1,2,3,... This lets a test pin dense-arm rank
    independently of keyword content, deliberately opposite the keyword arm's
    order (see env_many), rather than relying on incidental embedding
    behaviour."""
    _MARKER = re.compile(r"marker(\d+)")

    def embed_documents(self, texts):
        return [[float(self._MARKER.search(t).group(1))] for t in texts]

    def embed_query(self, text):
        return [0.0]


@pytest.fixture
def env_many(tmp_path):
    """9 single-chunk documents. Document i repeats "flarple" i times, so
    BM25 ranks them i=9 (most repeats) best .. i=1 (fewest) worst — verified
    empirically, not assumed. MarkerEmbedder makes the dense arm's distance
    equal to i, so its nearest-first order is i=1 best .. i=9 worst: the
    exact reverse of the keyword arm. arm_depth=5, less than the 9-chunk
    corpus, so each arm's top-5 excludes some documents entirely rather than
    merely ranking them low — the distinction Finding 3 and Finding 4 both
    depend on.

    With arm_depth=5: keyword top-5 is i in {9,8,7,6,5} (ranks 1-5); vector
    top-5 is i in {1,2,3,4,5} (ranks 1-5). Document 5 is rank 5 (worst) in
    both arms. Document 9 is rank 1 (best) in the keyword arm only —
    excluded from the vector arm's top-5 entirely.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(1, 10):
        text = f"# Doc{i}\n\n" + ("flarple " * i) + f"marker{i} filler text padding here\n"
        (corpus / f"d{i}.md").write_text(text)
    s = Settings(db_path=tmp_path / "t.db", corpus_dir=corpus, embed_dim=1, arm_depth=5)
    con = connect(s)
    create_schema(con, s)
    index_corpus(con, s, MarkerEmbedder())
    return con, s


def _path_of(con, chunk_id: int) -> str:
    row = con.execute(
        "SELECT d.path FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE c.id = ?",
        (chunk_id,),
    ).fetchone()
    return row["path"]


def test_keyword_arm_finds_stemmed_match(env):
    con, s = env
    ids = keyword_arm(con, s, "swap")
    assert len(ids) == 1


def test_keyword_arm_matches_natural_language_multi_term_query(env):
    """Every other keyword test in this file uses a single term ("swap",
    "flarple", "the", "memory") -- that is precisely the blind spot that let
    an AND-joined query survive eleven prior reviews. Adjacent quoted FTS5
    strings are implicitly ANDed, so a query built by quoting each term and
    joining with a bare space requires the chunk to contain *every* term.
    A real natural-language question almost never does: mem.md contains
    "swapping" but not the literal word "problem", so under AND this query
    matches nothing at all, even though it is obviously about the right
    document. The keyword arm must fall back to OR semantics so any term
    matching is enough -- exactly what BM25 ranking already assumes when it
    scores partial matches."""
    con, s = env
    ids = keyword_arm(con, s, "why is the repack buffer swapping problem")
    assert ids, "a natural multi-word question must match via OR, not require every term"


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
    # An empty result also satisfies "<= 1" — assert the exact count so a
    # broken search() that silently returns nothing cannot pass this test.
    hits = search(con, s, FakeEmbedder(), "the", limit=1)
    assert len(hits) == 1


def test_limit_truncates_a_larger_fused_candidate_set(env_many):
    con, s = env_many
    # 9 chunks are eligible across both arms (see env_many docstring), well
    # over limit=2 — this proves truncation happens on the fused/ranked
    # result, not on some coincidentally-small candidate set.
    hits = search(con, s, MarkerEmbedder(), "flarple", limit=2)
    assert len(hits) == 2


def test_fts_special_characters_do_not_crash(env):
    # Unescaped, quotes/parens are FTS5 syntax errors. Bare AND/OR/NOT are
    # *not* stripped by punctuation removal (they are word characters) and
    # are read as query-language operators — "NOT working" or a trailing
    # "AND" raises rather than returning zero results.
    cases = [
        'what is "memory fit" AND (swap)',
        "NOT working",
        "swap AND",
        "AND",
        "OR OR",
        "a AND OR b",
    ]
    con, s = env
    for query in cases:
        hits = search(con, s, FakeEmbedder(), query)
        assert isinstance(hits, list), f"query {query!r} did not return a list"


def test_worst_in_both_arms_outranks_best_in_one_arm_only(env_many):
    """The design's core claim: a chunk both arms agree on, even weakly,
    should beat a chunk only one arm loves. This needs enough chunks that
    ranks exceed the number of arms (2) and an arm_depth that actually
    excludes some documents — a 2-chunk fixture can't distinguish "ranked
    worst" from "not ranked at all", and can't fail if rrf_k were a literal
    or arm_depth were ignored/swapped for limit (verified: swapping this
    fixture's arm_depth for "no truncation" flips the ordering asserted
    below, so this is not a redundant check on top of the simpler fixture)."""
    con, s = env_many
    hits = search(con, s, MarkerEmbedder(), "flarple", limit=9)
    scores = {h.path: h.score for h in hits}
    assert scores["d5.md"] > scores["d9.md"], "worst-in-both must beat best-keyword-only"
    assert scores["d5.md"] > scores["d1.md"], "worst-in-both must beat best-vector-only"


def test_vector_arm_orders_nearest_first(env_many):
    con, s = env_many
    ids = vector_arm(con, s, [0.0])
    paths = [_path_of(con, cid) for cid in ids]
    assert paths == ["d1.md", "d2.md", "d3.md", "d4.md", "d5.md"]


def test_vector_arm_respects_arm_depth(env_many):
    con, s = env_many
    ids = vector_arm(con, s, [0.0])
    # The corpus has 9 chunks; arm_depth=5 must bound the result, not return
    # every match.
    assert len(ids) == s.arm_depth == 5
