"""Orchestration: HyDE → dense+sparse (gather) → RRF → rerank → generate. All timings captured."""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass

from hybrid_rag.config import Settings
from hybrid_rag.embeddings import EmbeddingProvider
from hybrid_rag.fusion import FusedDoc, rrf_fuse
from hybrid_rag.generation import Answer, generate_answer
from hybrid_rag.hyde import HyDEExpander
from hybrid_rag.llm.base import LLMProvider
from hybrid_rag.reranker import BaseReranker
from hybrid_rag.retrieval.base import DenseRetriever, SparseRetriever


@dataclass
class PipelineResult:
    query: str
    answer: Answer
    contexts: list[FusedDoc]
    timings_ms: dict[str, float]


class HybridPipeline:
    def __init__(self, settings: Settings, embedder: EmbeddingProvider,
                 llm: LLMProvider, dense: DenseRetriever,
                 sparse: SparseRetriever, reranker: BaseReranker):
        self.s = settings
        self.embedder = embedder
        self.llm = llm
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.hyde = HyDEExpander(llm, enabled=settings.hyde_enabled)

    async def query(self, query: str) -> PipelineResult:
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        hyp = await self.hyde.expand(query)
        t["hyde_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        emb_task = self.embedder.embed_query(hyp)
        # dense needs embedding; sparse needs raw query — run embedding + sparse concurrently
        sparse_task = self.sparse.search(query, self.s.top_k_sparse)
        q_emb, sparse_hits = await asyncio.gather(emb_task, sparse_task)
        dense_hits = await self.dense.search(q_emb, self.s.top_k_dense)
        # hedge: optional third list — dense over raw query
        lists = {"dense": dense_hits, "sparse": sparse_hits}
        if self.s.hyde_include_raw_query and hyp != query:
            raw_emb = await self.embedder.embed_query(query)
            lists["dense_raw"] = await self.dense.search(raw_emb, self.s.top_k_dense)
        t["retrieve_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        fused = rrf_fuse(lists, k=self.s.rrf_k)[:self.s.rerank_candidates]
        t["rrf_ms"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        # reranker judges the ORIGINAL query (precision), not HyDE expansion
        contexts = await self.reranker.rerank(query, fused, self.s.top_n_final)
        t["rerank_ms"] = (time.perf_counter() - t3) * 1000

        t4 = time.perf_counter()
        answer = await generate_answer(self.llm, query, contexts)
        t["generate_ms"] = (time.perf_counter() - t4) * 1000
        t["total_ms"] = (time.perf_counter() - t0) * 1000
        return PipelineResult(query=query, answer=answer, contexts=contexts, timings_ms=t)
