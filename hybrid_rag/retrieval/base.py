from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    content: str
    source: str = ""
    page: int | None = None


@dataclass
class RetrievalResult:
    doc_id: str
    content: str
    source: str
    score: float
    page: int | None = None


class DenseRetriever:
    async def search(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError


class SparseRetriever:
    async def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError
