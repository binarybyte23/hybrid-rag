"""Full pipeline over in-memory fakes: wiring, RRF, trim, citations, timings."""
import pytest
from hybrid_rag.config import Settings
from hybrid_rag.pipeline import HybridPipeline
from hybrid_rag.retrieval.base import RetrievalResult as R


class FakeEmb:
    async def embed_query(self, text):
        return [1.0, 0.0]

    async def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeLLM:
    def __init__(self):
        self.seen = []

    async def generate(self, prompt, system=""):
        self.seen.append((prompt, system))
        if "hypothetical" in prompt.lower():
            return "hypothetical vacation passage"
        return "Vacation is 20 days [1] and refunds 30 days [2]."


class FakeDense:
    def __init__(self):
        self.got = None

    async def search(self, embedding, top_k):
        self.got = embedding
        return [R("d1", "vacation 20 days", "handbook.txt", 0.9),
                R("d2", "refund 30 days", "handbook.txt", 0.8)]


class FakeSparse:
    def __init__(self):
        self.got = None

    async def search(self, query, top_k):
        self.got = query
        return [R("d2", "refund 30 days", "handbook.txt", 5.0),
                R("d1", "vacation 20 days", "handbook.txt", 4.0)]


class FakeRerank:
    async def rerank(self, query, docs, top_n):
        return docs[:top_n]


def make(**kw):
    s = Settings(**{**dict(hyde_enabled=True, hyde_include_raw_query=False,
                           top_k_dense=20, top_k_sparse=20, rrf_k=60,
                           rerank_candidates=20, top_n_final=5), **kw})
    return HybridPipeline(s, FakeEmb(), FakeLLM(), FakeDense(), FakeSparse(), FakeRerank())


@pytest.mark.asyncio
async def test_wiring_dense_gets_hyde_sparse_gets_raw():
    pipe = make()
    await pipe.query("How many vacation days?")
    assert pipe.dense.got == [1.0, 0.0]
    assert pipe.sparse.got == "How many vacation days?"
    assert "How many vacation days?" in pipe.llm.seen[0][0]


@pytest.mark.asyncio
async def test_citations_and_timings():
    pipe = make()
    res = await pipe.query("vacation?")
    assert res.answer.sources and res.answer.sources[0]["doc_id"] in ("d1", "d2")
    for k in ("hyde_ms", "retrieve_ms", "rrf_ms", "rerank_ms", "generate_ms", "total_ms"):
        assert k in res.timings_ms


@pytest.mark.asyncio
async def test_top_n_trim():
    pipe = make(top_n_final=1)
    res = await pipe.query("q")
    assert len(res.contexts) == 1
