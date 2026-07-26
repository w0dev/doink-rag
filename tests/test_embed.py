import httpx
import pytest
from rag.config import Settings
from rag.embed import Embedder

S = Settings(embed_dim=4, embed_batch=2)


def _client(capture: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(request.content)
        capture.append(body)
        inputs = body["input"]
        return httpx.Response(200, json={
            "data": [{"embedding": [1.0, 0.0, 0.0, 0.0]} for _ in inputs]
        })
    return httpx.Client(base_url="http://localhost", transport=httpx.MockTransport(handler))


def test_documents_get_no_instruction_prefix():
    seen: list = []
    e = Embedder(S, client=_client(seen))
    e.embed_documents(["plain body text"])
    assert seen[0]["input"] == ["plain body text"]


def test_query_gets_the_instruction_prefix():
    seen: list = []
    e = Embedder(S, client=_client(seen))
    e.embed_query("how does swapping happen")
    sent = seen[0]["input"][0]
    assert sent.startswith("Instruct: ")
    assert "Query: how does swapping happen" in sent


def test_batching_respects_embed_batch():
    seen: list = []
    e = Embedder(S, client=_client(seen))
    e.embed_documents(["a", "b", "c"])
    assert [len(c["input"]) for c in seen] == [2, 1]


def test_dimension_mismatch_is_rejected():
    def handler(request):
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})
    e = Embedder(S, client=httpx.Client(base_url="http://localhost", transport=httpx.MockTransport(handler)))
    with pytest.raises(ValueError, match="dimension"):
        e.embed_documents(["x"])


def test_empty_input_makes_no_request():
    seen: list = []
    e = Embedder(S, client=_client(seen))
    assert e.embed_documents([]) == []
    assert seen == []
