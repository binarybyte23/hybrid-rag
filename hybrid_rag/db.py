"""Postgres pool + schema. pgvector codec registered on connect."""
from __future__ import annotations
import asyncpg
from pgvector.asyncpg import register_vector

from hybrid_rag.config import settings

_pool: asyncpg.Pool | None = None


async def _init_conn(conn: asyncpg.Connection) -> None:
    try:
        await register_vector(conn)
    except ValueError:
        pass  # fresh DB without vector extension yet; init_db creates it first


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.database_url, init=_init_conn)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_db(dim: int | None = None) -> None:
    """Create extension + chunks table (idempotent)."""
    # CREATE EXTENSION via bare connection (no codec init — type may not exist yet)
    bare = await asyncpg.connect(settings.database_url)
    try:
        await bare.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await bare.close()
    pool = await get_pool()
    dim = dim or settings.embedding_dim
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                page INT,
                embedding vector({dim}),
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            )"""
        )
        await conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page INT")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv)"
        )
