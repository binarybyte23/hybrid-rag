"""pgvector dense search via native asyncpg SQL."""
from __future__ import annotations
from hybrid_rag.db import get_pool
from hybrid_rag.retrieval.base import DenseRetriever, RetrievalResult


class PgVectorDenseRetriever(DenseRetriever):
    async def search(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        pool = await get_pool()
        rows = await pool.fetch(
            """SELECT id, content, source, page, embedding <=> $1 AS dist
               FROM chunks ORDER BY embedding <=> $1 LIMIT $2""",
            embedding, top_k,
        )
        return [RetrievalResult(doc_id=r["id"], content=r["content"],
                                source=r["source"], page=r["page"], score=1 - float(r["dist"]))
                for r in rows]
