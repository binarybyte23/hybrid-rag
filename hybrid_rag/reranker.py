"""Stage 4 — cross-encoder re-rank on the fused shortlist. Original query in, trimmed top-N out."""
from __future__ import annotations
from hybrid_rag.fusion import FusedDoc


class BaseReranker:
    async def rerank(self, query: str, docs: list[FusedDoc], top_n: int) -> list[FusedDoc]:
        raise NotImplementedError


class NoopReranker(BaseReranker):
    async def rerank(self, query: str, docs: list[FusedDoc], top_n: int) -> list[FusedDoc]:
        return docs[:top_n]


class FlashRankReranker(BaseReranker):
    def __init__(self, model: str = "ms-marco-MiniLM-L-12-v2"):
        self.model = model
        self._r = None

    def _r_or_load(self):
        if self._r is None:
            from flashrank import Ranker  # lazy: optional dep
            self._r = Ranker(model_name=self.model)
        return self._r

    async def rerank(self, query: str, docs: list[FusedDoc], top_n: int) -> list[FusedDoc]:
        import asyncio
        ranker = self._r_or_load()
        pairs = [{"id": d.doc_id, "text": d.content} for d in docs]
        scored = await asyncio.to_thread(ranker.rerank,
                                         [{"query": query, "passage": p["text"]} for p in pairs])
        # flashrank returns in ranked order; map back
        order = sorted(zip(docs, scored), key=lambda t: -t[1]["score"])
        return [d for d, _ in order[:top_n]]


class HFCrossEncoderReranker(BaseReranker):
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = model
        self._m = None

    def _m_or_load(self):
        if self._m is None:
            from sentence_transformers import CrossEncoder  # lazy
            self._m = CrossEncoder(self.model)
        return self._m

    async def rerank(self, query: str, docs: list[FusedDoc], top_n: int) -> list[FusedDoc]:
        import asyncio
        m = self._m_or_load()
        scores = await asyncio.to_thread(m.predict, [(query, d.content) for d in docs])
        order = sorted(zip(docs, scores), key=lambda t: -float(t[1]))
        return [d for d, _ in order[:top_n]]


def build_reranker(provider: str, model: str) -> BaseReranker:
    if provider == "flashrank":
        return FlashRankReranker(model)
    if provider == "hf":
        return HFCrossEncoderReranker(model)
    return NoopReranker()
