"""Embeddings over HTTP.

Deliberately the only thing that talks to a model, and deliberately not an ML
library: llama-server already runs on this box, so embedding is an HTTP call and
this package needs no torch, no onnx and no transformers.
"""
from __future__ import annotations

import httpx

from .config import Settings


class Embedder:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self._client = client or httpx.Client(
            base_url=settings.embed_url, timeout=120.0
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Documents carry no instruction prefix. Adding one here would make
        document vectors incomparable with query vectors."""
        out: list[list[float]] = []
        for start in range(0, len(texts), self.settings.embed_batch):
            batch = texts[start:start + self.settings.embed_batch]
            out.extend(self._post(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        """Queries MUST carry the instruction. Qwen3 embedding is asymmetric and
        omitting this silently costs retrieval quality."""
        prompt = f"Instruct: {self.settings.query_instruction}\nQuery: {text}"
        return self._post([prompt])[0]

    def health(self) -> bool:
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def _post(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        response = self._client.post(
            "/v1/embeddings",
            json={"model": self.settings.embed_model, "input": inputs},
        )
        response.raise_for_status()
        vectors = [row["embedding"] for row in response.json()["data"]]
        for vector in vectors:
            if len(vector) != self.settings.embed_dim:
                raise ValueError(
                    f"embedding dimension {len(vector)} does not match "
                    f"configured {self.settings.embed_dim}; the served model is "
                    f"not the one this database was built for"
                )
        return vectors
