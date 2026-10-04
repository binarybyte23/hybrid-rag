"""Swappable dense backend. Chunk text stays in Postgres; Qdrant holds vectors only."""
from __future__ import annotations
from hybrid_rag.retrieval.base import DenseRetriever, RetrievalResult


class QdrantDenseRetriever(DenseRetriever):
    def __init__(self, url: str, collection: str = "chunks"):
        self.url = url
        self.collection = collection
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # lazy: optional dep
            self._client = QdrantClient(url=self.url)
        return self._client

    async def search(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        import asyncio
        client = self._client_or_raise()
        hits = await asyncio.to_thread(
            client.search, self.collection, embedding, limit=top_k)
        return [RetrievalResult(doc_id=str(h.id), content=str(h.payload.get("content", "")),
                                source=str(h.payload.get("source", "")),
                                page=h.payload.get("page"), score=float(h.score))
                for h in hits]
