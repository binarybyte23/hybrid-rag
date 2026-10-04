"""CLI: init-db | ingest | query | eval."""
from __future__ import annotations
import argparse
import asyncio
from hybrid_rag.config import settings
from hybrid_rag.db import init_db, close_pool
from hybrid_rag.embeddings import build_embedder
from hybrid_rag.ingest import ingest_paths
from hybrid_rag.llm import build_llm
from hybrid_rag.pipeline import HybridPipeline
from hybrid_rag.reranker import build_reranker
from hybrid_rag.retrieval.dense import PgVectorDenseRetriever
from hybrid_rag.retrieval.qdrant_dense import QdrantDenseRetriever
from hybrid_rag.retrieval.sparse import PostgresFTSSparseRetriever


def _build_pipeline() -> HybridPipeline:
    emb = build_embedder(settings.embedding_provider, settings.embedding_model,
                         settings.embedding_api_url, settings.llm_api_key)
    llm = build_llm(settings.llm_provider, settings.llm_base_url,
                    settings.llm_model, settings.llm_api_key)
    dense = (QdrantDenseRetriever(settings.qdrant_url)
             if settings.dense_backend == "qdrant" else PgVectorDenseRetriever())
    return HybridPipeline(settings, emb, llm, dense,
                          PostgresFTSSparseRetriever(),
                          build_reranker(settings.reranker_provider, settings.reranker_model))


async def _query(q: str, verbose: bool):
    pipe = _build_pipeline()
    res = await pipe.query(q)
    print(f"\nAnswer:\n{res.answer.text}\n")
    for s in res.answer.sources:
        loc = s['source'] + (f", p.{s['page']}" if s.get("page") is not None else "")
        print(f"[{s['n']}] {loc} | id={s['doc_id']} | "
              f"rrf={s['rrf_score']:.4f} | ranks={s['ranks']}")
    t = res.timings_ms
    print(f"timings: hyde={t.get('hyde_ms', 0):.1f}ms retrieve={t.get('retrieve_ms', 0):.1f}ms "
          f"rrf={t.get('rrf_ms', 0):.1f}ms rerank={t.get('rerank_ms', 0):.1f}ms "
          f"generate={t.get('generate_ms', 0):.1f}ms total={t.get('total_ms', 0):.1f}ms")
    await close_pool()


async def _eval():
    from hybrid_rag.eval.harness import run_eval
    rows = await run_eval(_build_pipeline())
    for r in rows:
        print(r)
    await close_pool()


def main():
    ap = argparse.ArgumentParser(prog="hybrid_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("paths", nargs="+")
    p_q = sub.add_parser("query")
    p_q.add_argument("question")
    p_q.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("eval")
    a = ap.parse_args()
    if a.cmd == "init-db":
        asyncio.run(init_db())
        print("db ready")
    elif a.cmd == "ingest":
        async def _ing():
            n = await ingest_paths(
                a.paths,
                build_embedder(settings.embedding_provider, settings.embedding_model,
                               settings.embedding_api_url, settings.llm_api_key))
            print(f"ingested {n} chunks")
            await close_pool()
        asyncio.run(_ing())
    elif a.cmd == "query":
        asyncio.run(_query(a.question, a.verbose))
    elif a.cmd == "eval":
        asyncio.run(_eval())


if __name__ == "__main__":
    main()
