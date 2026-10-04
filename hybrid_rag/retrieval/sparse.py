"""Postgres full-text keyword search (ts_rank_cd)."""
from __future__ import annotations
from hybrid_rag.db import get_pool
from hybrid_rag.retrieval.base import SparseRetriever, RetrievalResult


class PostgresFTSSparseRetriever(SparseRetriever):
    async def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        pool = await get_pool()
        rows = await pool.fetch(
            """SELECT id, content, source, page,
                      ts_rank_cd(tsv, websearch_to_tsquery('english', $1)) AS rank
               FROM chunks
               WHERE tsv @@ websearch_to_tsquery('english', $1)
               ORDER BY rank DESC LIMIT $2""",
            query, top_k,
        )
        return [RetrievalResult(doc_id=r["id"], content=r["content"],
                                source=r["source"], page=r["page"], score=float(r["rank"]))
                for r in rows]
