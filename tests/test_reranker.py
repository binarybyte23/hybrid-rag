"""Reranker ordering/trimming with fakes + one real FlashRank (slow)."""
import pytest
from hybrid_rag.fusion import FusedDoc
from hybrid_rag.reranker import NoopReranker, build_reranker


def docs(*ids):
    return [FusedDoc(doc_id=i, rrf_score=1.0, content=f"content {i}") for i in ids]


class ReverseReranker:
    async def rerank(self, query, docs, top_n):
        return list(reversed(docs))[:top_n]


@pytest.mark.asyncio
async def test_noop_trims():
    out = await NoopReranker().rerank("q", docs("a", "b", "c"), 2)
    assert [d.doc_id for d in out] == ["a", "b"]


@pytest.mark.asyncio
async def test_noop_empty():
    assert await NoopReranker().rerank("q", [], 5) == []


@pytest.mark.asyncio
async def test_custom_order():
    out = await ReverseReranker().rerank("q", docs("a", "b"), 2)
    assert [d.doc_id for d in out] == ["b", "a"]


@pytest.mark.asyncio
async def test_custom_trim():
    out = await ReverseReranker().rerank("q", docs("a", "b", "c"), 1)
    assert [d.doc_id for d in out] == ["c"]


def test_builder_none():
    assert isinstance(build_reranker("none", "x"), NoopReranker)


def test_builder_unknown_falls_back():
    assert isinstance(build_reranker("bogus", "x"), NoopReranker)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_flashrank_real():
    r = build_reranker("flashrank", "ms-marco-MiniLM-L-12-v2")
    out = await r.rerank("vacation days?",
                         docs("a", "b"), 2)
    assert len(out) == 2
