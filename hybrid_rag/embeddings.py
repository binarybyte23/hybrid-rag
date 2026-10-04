"""Embedding providers: local sentence-transformers or OpenAI-compatible API."""
from __future__ import annotations


class EmbeddingProvider:
    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalEmbedding(EmbeddingProvider):
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.model = model
        self._m = None

    def _m_or_load(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer  # lazy
            self._m = SentenceTransformer(self.model)
        return self._m

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        m = self._m_or_load()
        vecs = await asyncio.to_thread(m.encode, texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class ApiEmbedding(EmbeddingProvider):
    def __init__(self, base_url: str, model: str, api_key: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import urllib.request, json
        out: list[list[float]] = []
        for t in texts:
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps({"model": self.model, "input": t}).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out.append(json.load(r)["data"][0]["embedding"])
        return out


def build_embedder(provider: str, model: str, api_url: str = "", api_key: str = "ollama") -> EmbeddingProvider:
    if provider == "local":
        return LocalEmbedding(model)
    return ApiEmbedding(api_url, model, api_key)
