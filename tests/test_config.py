"""Settings.from_env is the entire deployment configuration surface, and the
retrieval constants are the knobs the eval harness sweeps. Both are worth
pinning: a silently-ignored override would make a tuning run measure nothing.
"""
from pathlib import Path

import pytest

from rag.config import Settings


def test_defaults_match_the_measured_values():
    # Changed on evidence, not taste — see docs/TUNING.md. If these move,
    # the measurement should move with them.
    s = Settings()
    assert s.rrf_k == 60
    assert s.arm_depth == 10
    assert s.embed_dim == 1024


def test_retrieval_constants_are_env_overridable(monkeypatch):
    monkeypatch.setenv("RAG_RRF_K", "17")
    monkeypatch.setenv("RAG_ARM_DEPTH", "3")
    s = Settings.from_env()
    assert s.rrf_k == 17
    assert s.arm_depth == 3


def test_env_values_are_coerced_to_int(monkeypatch):
    monkeypatch.setenv("RAG_ARM_DEPTH", "25")
    monkeypatch.setenv("RAG_EMBED_DIM", "768")
    s = Settings.from_env()
    assert s.arm_depth == 25 and isinstance(s.arm_depth, int)
    assert s.embed_dim == 768 and isinstance(s.embed_dim, int)


def test_unset_env_falls_through_to_defaults(monkeypatch):
    for var in ("RAG_RRF_K", "RAG_ARM_DEPTH", "RAG_EMBED_DIM", "RAG_DB_PATH"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert (s.rrf_k, s.arm_depth) == (Settings().rrf_k, Settings().arm_depth)


def test_paths_are_wrapped_as_path_objects(monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", "/tmp/x.db")
    monkeypatch.setenv("RAG_CORPUS_DIR", "/tmp/corpus")
    s = Settings.from_env()
    assert s.db_path == Path("/tmp/x.db")
    assert s.corpus_dir == Path("/tmp/corpus")


def test_a_non_numeric_override_fails_loudly(monkeypatch):
    # Silently falling back to the default would make a sweep report numbers
    # for settings it never actually used.
    monkeypatch.setenv("RAG_ARM_DEPTH", "ten")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_empty_string_override_fails_loudly(monkeypatch):
    """`RAG_ARM_DEPTH=""` silently taking the default is the same class of bug
    as a non-numeric value: the sweep would report a number for a setting it
    never used."""
    monkeypatch.setenv("RAG_ARM_DEPTH", "")
    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_values_below_one_are_rejected(monkeypatch, value):
    """These fail confusingly rather than obviously: arm_depth 0 makes search
    silently return nothing, and -1 becomes SQLite's 'unlimited' in the keyword
    arm while the vector arm raises."""
    monkeypatch.setenv("RAG_ARM_DEPTH", value)
    with pytest.raises(ValueError, match=">= 1"):
        Settings.from_env()


def test_rrf_k_below_one_is_rejected(monkeypatch):
    # rrf_k=0 divides by zero on the first rank-0 chunk.
    monkeypatch.setenv("RAG_RRF_K", "0")
    with pytest.raises(ValueError, match=">= 1"):
        Settings.from_env()
