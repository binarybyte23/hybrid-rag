# Hybrid RAG — framework-free, production-grade

A retrieval-augmented generation pipeline built **directly on native drivers and
open-source libraries** — `asyncpg` + pgvector, native Postgres full-text search,
hand-written Reciprocal Rank Fusion, and direct cross-encoder inference.
**No LangChain / LlamaIndex in the pipeline.** (Ragas is used only as an
*evaluator*, where its judge needs an LLM client; the pipeline under test stays
framework-free.)

```
                    ┌──────────────────────── STAGE 1: HyDE ─────────────────────────┐
  user query ─────► │ LLM writes a hypothetical answer passage (async, swappable)     │
                    └──────────────┬──────────────────────────────┬──────────────────┘
                        embed(hypothetical doc)          raw query text (keywords)
                                   ▼                               ▼
        STAGE 2 ── dense retrieve (pgvector / Qdrant)   sparse retrieve (tsvector/ts_rank_cd)
         (asyncio.gather: both run concurrently, each returns top-K with scores)
                                   └───────────┬───────────────────┘
                                               ▼
                  STAGE 3 ── Reciprocal Rank Fusion  score(d) = Σ_lists 1/(k + rank)
                                               ▼   fused shortlist (top ~20)
                  STAGE 4 ── Cross-encoder re-rank (FlashRank / HF ms-marco-MiniLM)
                                               ▼   trimmed to top-N (5)
                  STAGE 5 ── Grounded LLM generation, [n] citations, source attribution
                                               ▼
                  STAGE 6 ── Ragas eval: Faithfulness · Answer Relevancy · Ctx P/R
```

## Project layout

```
hybrid_rag/
├── hyde.py                 # Stage 1 — Hypothetical Document Embeddings (async)
├── retrieval/
│   ├── base.py             # Chunk / RetrievalResult + Dense/Sparse ABCs
│   ├── dense.py            # pgvector dense search (native asyncpg SQL)
│   ├── sparse.py           # Postgres tsvector/ts_rank_cd keyword search
│   └── qdrant_dense.py     # swappable dense backend (DENSE_BACKEND=qdrant)
├── fusion.py               # Stage 3 — RRF written from scratch
├── reranker.py             # Stage 4 — FlashRank / HF cross-encoder
├── generation.py           # Stage 5 — grounded prompt + citation parsing
├── pipeline.py             # orchestration (asyncio.gather, timings, fakes-friendly)
├── embeddings.py           # local (sentence-transformers) or API embeddings
├── llm/                    # provider-agnostic LLM interface (base/mock/openai-compatible)
├── ingest.py               # hand-rolled paragraph-aware chunker + bulk insert
├── db.py                   # pool, pgvector codec registration, dynamic schema
├── config.py               # all tunables via env vars / .env
├── cli.py                  # init-db | ingest | query | eval
└── eval/                   # Stage 6 — golden dataset + Ragas harness
tests/                      # unit tests: RRF (9), reranker (7), HyDE (5), pipeline (3)
docker-compose.yml          # Postgres+pgvector (Qdrant behind a profile)
```

## Quickstart

```bash
docker compose up -d postgres                 # add `--profile qdrant` if using Qdrant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                          # defaults: Ollama llama3.1 + local bge-small embeddings

python -m hybrid_rag.cli init-db
python -m hybrid_rag.cli ingest sample_data/
python -m hybrid_rag.cli query "How many vacation days do employees get?" -v
python -m hybrid_rag.cli eval                 # Ragas metrics + eval_results.csv

pytest -q -m "not slow"                       # 25 tests, no infra needed
```

The default LLM endpoint is a local Ollama server (`ollama serve`, `ollama pull llama3.1`),
but any OpenAI-compatible endpoint works by editing `.env` — including OpenAI itself,
vLLM, LM Studio, Groq, Together. Set `LLM_PROVIDER=mock` to run query flow end-to-end
with no LLM at all (handy for CI).

## Architectural decisions

### Why RRF instead of blending raw similarity scores?

Dense and sparse scores are **uncalibrated and incommensurable**:

* Cosine similarity lives in a narrow model-dependent band (typically 0.6–0.9 for a
  given embedding model), and the band shifts with the embedding model, the corpus,
  and HyDE-vs-raw-query inputs.
* `ts_rank_cd` / BM25 is unbounded and drifts with corpus size, document length
  distribution, and query term count.

Linear blending (`α·dense + (1−α)·sparse`) therefore needs per-corpus min-max or
z-score normalisation plus a tuned α — three knobs that silently change ranking
under distribution shift, and normalisation is computed *per result window*, so a
document's blended score changes depending on what else happened to be retrieved.

**RRF uses only rank positions** (`1/(k+rank)`), which is the one signal both
retrievers are actually optimised to produce. Documents strong in *both* lists are
rewarded multiplicatively in count of appearances, a single outlier list cannot
dominate (k=60 damps top ranks), and there is nothing to tune per corpus. The
paper (Cormack et al., SIGIR 2009) shows it matches or beats calibrated score
fusion in practice. `fusion.py` is ~80 lines, depends on nothing, and is verified
against hand-computed expectations in `tests/test_fusion.py`
(e.g. `test_doc_strong_in_both_lists_beats_doc_strong_in_one`).

### Why does cross-encoder reranking happen *after* fusion, not before?

1. **Cost.** A cross-encoder runs one full transformer forward pass per
   (query, document) pair. Dense retrieval scores the whole corpus for ~free via
   HNSW ANN; re-scoring the corpus with a cross-encoder would be 4–6 orders of
   magnitude more compute. It is only tractable on a small shortlist.
2. **Recall funnel.** The correct funnel is: cheap high-recall retrieval
   (top-20×2 lists) → RRF (dedupe + consensus, still high recall) → expensive
   high-precision rerank on the fused top-~20 → **top-5 into the LLM context**.
   Reranking each list *before* fusion spends cross-encoder passes on 40 documents
   (many overlapping), and then rank-fuses already-refined orderings for no benefit.
3. **Cleaner inputs.** The cross-encoder sees the *original user query* — not the
   HyDE expansion — because judgement here is precision, not recall: HyDE's invented
   vocabulary helps recall at retrieval time, but the reranker should judge the
   genuine relevance of a chunk to what the user asked.

### Why HyDE, and why does sparse search still use the raw query?

Dense embedding models map queries and documents into the same vector space, but a
two-line question and a paragraph answer sit in *different regions* of it. HyDE
asks the LLM to hallucinate a plausible answer passage and embeds **that** —
moving the query embedding into "document space" where it neighbours real answer
chunks. The facts in the hallucination are often wrong; the topical and lexical
neighbourhood is right, which is all ANN retrieval needs.

Two failure modes and how the pipeline handles them:
* **HyDE drift** (hallucinated topic pulls retrieval off-course): `HYDE_INCLUDE_RAW_QUERY=true`
  fuses a third ranked list — dense search over the raw-query embedding — so RRF
  arbitrates between them, and `HyDEExpander` falls back to the raw query on any
  LLM error.
* **Keyword corruption**: lexical search *needs the user's literal terms*
  (product names, numbers). Hallucinated HyDE vocabulary actively hurts FTS.
  Sparse retrieval therefore always receives the raw query — this asymmetry
  is deliberate and is why hybrid > dense-only in practice.

### Sparse: `ts_rank_cd` vs true BM25

This project deliberately stays on stock Postgres: a generated
`tsvector` column + GIN index queried with `websearch_to_tsquery` (web-style
quotes/OR/minus operators) and ranked by `ts_rank_cd` (cover-density — rewards
documents where query terms cluster, with document-length normalisation options).
For true BM25 swap `PostgresFTSSparseRetriever` for one backed by ParadeDB's
`pg_search` extension — it satisfies the same `SparseRetriever` interface, so
nothing downstream changes.

### Concurrency model

Everything on the hot path is async: HyDE generates while anything else could
run, then `asyncio.gather` fires dense (pgvector over the pooled asyncpg
connection) and sparse (FTS) in parallel. Blocking model work (local embeddings,
cross-encoder) is pushed off the event loop with `asyncio.to_thread`. Per-stage
timings are returned in every `PipelineResult` (`-v` flag shows them).

### Prompt discipline

Generation uses a strict grounded prompt (system: *answer only from the passages,
cite with [n]*), and `generation.py` parses the `[n]` markers back out into a
structured `sources` list — doc_id, source file, rerank/RRF scores, and provenance
ranks from each retriever — so answers are auditable.

## Configuration

Everything is env-driven (`.env`); see `.env.example` for the annotated full list.
Key knobs:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` / `LLM_BASE_URL` / `LLM_MODEL` | openai-compatible / Ollama / llama3.1 | one interface for HyDE + generation (+ Ragas judge) |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | local / bge-small-en-v1.5 (384d) | local ST model or API embeddings |
| `DENSE_BACKEND` | pgvector | `qdrant` swaps in Qdrant (chunk text still lives in Postgres) |
| `TOP_K_DENSE` / `TOP_K_SPARSE` | 20 / 20 | per-retriever shortlist sizes |
| `RRF_K` | 60 | RRF damping constant (paper default) |
| `RERANK_CANDIDATES` | 20 | fused shortlist handed to the cross-encoder |
| `TOP_N_FINAL` | 5 | contexts that reach the LLM — the cost/latency lever |
| `HYDE_ENABLED` / `HYDE_INCLUDE_RAW_QUERY` | true / false | HyDE on/off; hedge with raw-query dense list |
| `RERANKER_PROVIDER` / `RERANKER_MODEL` | flashrank / ms-marco-MiniLM-L-12-v2 | flashrank · hf · none |

## Evaluation

`python -m hybrid_rag.cli eval` runs the golden set in `eval/dataset.py` through
the **full** pipeline and reports Faithfulness, Answer Relevancy, Context
Precision and Context Recall per question plus aggregates, writing
`eval_results.csv` so runs can be diffed across pipeline changes (e.g. is HyDE
actually helping? flip `HYDE_ENABLED`, re-run, compare). The judge LLM and
embeddings come from the same `.env`, so the eval harness also runs fully offline
against Ollama + a local embedding model.

## Testing

```bash
pytest -q -m "not slow"     # 25 tests, no Postgres / no models / no LLM required
pytest -q                   # additionally runs a real FlashRank rerank (downloads ~50MB ONNX)
```

* `test_fusion.py` — RRF against hand-computed formula values: cross-list
  agreement beats single-list strength, provenance capture, k-sensitivity,
  duplicate handling, deterministic tie-breaking.
* `test_reranker.py` — ordering/trimming logic pure, with a fake cross-encoder,
  plus a real FlashRank integration test.
* `test_hyde.py` — enabled/disabled/concat/fallback paths of HyDE.
* `test_pipeline.py` — full pipeline over in-memory fakes: retriever wiring
  (dense gets HyDE embedding, sparse gets raw query), RRF integration,
  rerank trimming, citation attribution, per-stage timings.

Because every stage is an injected ABC (`LLMProvider`, `EmbeddingProvider`,
`DenseRetriever`, `SparseRetriever`, `BaseReranker`), the pipeline's wiring is
testable with zero infrastructure — which is the point of building it on plain
drivers rather than a framework.

Swapping in your own backend = implement the 2-method ABC, register it in the
factory (`build_llm` / `build_embedder` / `build_dense` / `build_reranker`), set
an env var. Nothing else in the pipeline changes.
