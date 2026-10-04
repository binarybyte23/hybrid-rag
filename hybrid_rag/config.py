"""Env-driven settings. All tunables live here — see .env.example."""
from __future__ import annotations
import os
from dataclasses import dataclass


def _load_dotenv(path: str = ".env") -> None:
    # ponytail: stdlib .env parser, no python-dotenv dependency
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()
_load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # infra
    database_url: str = _get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
    # llm
    llm_provider: str = _get("LLM_PROVIDER", "mock")
    llm_base_url: str = _get("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model: str = _get("LLM_MODEL", "llama3.1")
    llm_api_key: str = _get("LLM_API_KEY", "ollama")
    # embeddings
    embedding_provider: str = _get("EMBEDDING_PROVIDER", "local")
    embedding_model: str = _get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dim: int = _int("EMBEDDING_DIM", 384)
    embedding_api_url: str = _get("EMBEDDING_API_URL", "http://localhost:11434/api/embeddings")
    # retrieval
    dense_backend: str = _get("DENSE_BACKEND", "pgvector")
    qdrant_url: str = _get("QDRANT_URL", "http://localhost:6333")
    top_k_dense: int = _int("TOP_K_DENSE", 20)
    top_k_sparse: int = _int("TOP_K_SPARSE", 20)
    rrf_k: int = _int("RRF_K", 60)
    rerank_candidates: int = _int("RERANK_CANDIDATES", 20)
    top_n_final: int = _int("TOP_N_FINAL", 5)
    # hyde
    hyde_enabled: bool = _bool("HYDE_ENABLED", True)
    hyde_include_raw_query: bool = _bool("HYDE_INCLUDE_RAW_QUERY", False)
    # reranker
    reranker_provider: str = _get("RERANKER_PROVIDER", "none")
    reranker_model: str = _get("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")


settings = Settings()
