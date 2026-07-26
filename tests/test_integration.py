"""Deselected by default (addopts = -m 'not integration').

Run against a live embedding server once llama_embed is deployed:
    RAG_EMBED_URL=http://doink-brain:8092 uv run pytest -m integration -v
"""
import pytest
from rag.config import Settings
from rag.embed import Embedder


@pytest.mark.integration
def test_live_embedding_dimension_matches_config():
    settings = Settings.from_env()
    embedder = Embedder(settings)
    assert embedder.health(), f"no embedding server at {settings.embed_url}"
    vector = embedder.embed_query("does the model swap under load")
    assert len(vector) == settings.embed_dim, (
        f"served model returns {len(vector)} dims but config says "
        f"{settings.embed_dim}; chunks_vec was built for the configured value"
    )


@pytest.mark.integration
def test_query_and_document_embeddings_differ():
    """If prefixing is broken, these collapse to the same vector and retrieval
    quietly degrades."""
    settings = Settings.from_env()
    embedder = Embedder(settings)
    text = "tailscale serve terminates TLS"
    assert embedder.embed_query(text) != embedder.embed_documents([text])[0]
