# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Start infrastructure (Postgres + Redis)
docker-compose up -d

# Run database migrations
alembic upgrade head

# Start API server
uvicorn app.main:app --reload

# Start Celery worker (document ingestion)
celery -A app.celery_app worker --loglevel=info

# Generate a new migration
alembic revision --autogenerate -m "description"
```

Environment: copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, and `CORS_ALLOWED_URL` (required). The DB default is `localhost:5433` (note: not the standard 5432 — docker-compose maps `pgvector/pgvector:pg17` to 5433).

## Architecture

Self-RAG is a FastAPI app wrapping a **LangGraph state machine** (`app/rag/graph.py`) that implements two nested self-correction loops before streaming a final answer. Document ingestion runs as a separate **Celery** background task. A single PostgreSQL instance (with pgvector) serves as the relational store, the vector database, and the LangGraph checkpoint backend.

### Request Flow

```
Client HTTP → FastAPI
  → APITraceMiddleware → AuthMiddleware
  → ThreadController.ask_a_query
      → asyncio.Queue (stream_queue injected into LangGraph config)
      → RAGGraph.ainvoke()   ← runs in background asyncio.Task
      → SafeStreamingResponse (consumes queue as SSE/chunks)
```

### LangGraph Graph (`app/bot/nodes/`)

The graph is built in `app/rag/graph.py`. Each node is a separate file. The `RAGState` (`app/bot/state.py`) extends LangGraph's `MessagesState`.

**Retrieval path:**
```
START → upsert_thread → should_retrieve
  ├─(need_retrieval=False)→ generate_direct → END
  └─(need_retrieval=True)→ context_retriever → context_relevance_checker
       ├─(no relevant docs)→ no_answer_found → END
       └─(relevant docs)→ generate_from_context → answer_relevance_checker
            ├─(FULLY_SUPPORTED)→ check_answer_usefulness
            └─(PARTIALLY/NOT + ans_iteration<3)→ rewrite_answer ↺ answer_relevance_checker
                                                      ↓ (exhausted)
                                             check_answer_usefulness
                                               ├─(useful)→ stream_answer → END
                                               ├─(not useful + rewrite_tries<3)→ rewrite_question ↺ context_retriever
                                               └─(exhausted)→ no_answer_found → END
```

**Two self-correction loops:**
- **Answer-rewrite loop**: `generate_from_context → answer_relevance_checker → rewrite_answer`, bounded by `ans_iteration` (max 3, `_MAX_ANS_ITERATION`)
- **Question-rewrite loop**: `check_answer_usefulness → rewrite_question → context_retriever`, bounded by `rewrite_tries` (max 3, `_MAX_QUE_ITERATION`)

**Streaming design**: The answer is held in state throughout the correction loops. `generate_direct` streams tokens live; `stream_answer` pushes the fully assembled answer at the end of the RAG path. Both send `None` as a sentinel to signal stream end. **Do not stream mid-loop** — this is intentional.

### Key Modules

| Path | Purpose |
|---|---|
| `app/bot/state.py` | `RAGState` — all graph state fields |
| `app/bot/llm.py` | Shared `ChatGoogleGenerativeAI` instance + `StrOutputParser` |
| `app/rag/graph.py` | `RAGGraph` — builds and compiles the StateGraph |
| `app/rag/retriever.py` | pgvector similarity search (`Retriever.get()`) |
| `app/rag/ingestor/` | `BaseIngestor` + `PdfIngestor`: load → chunk → embed → store |
| `app/api/controller/thread.py` | Runs the graph, manages `asyncio.Queue` for streaming |
| `app/core/config.py` | `Settings` (Pydantic settings, all env vars) |
| `app/db/client.py` | SQLAlchemy async engine + session factory |
| `app/worker/tasks.py` | `ingest_document` Celery task |

### Document Ingestion

`POST /api/v1/documents/upload` → validates PDF (magic bytes) → saves to `app/uploads/` → inserts `Document` row (status=PROCESSING) → enqueues Celery task → `PdfIngestor.ainvoke()`:
- `PyPDFLoader` → `RecursiveCharacterTextSplitter` (600 chars / 150 overlap)
- `GoogleGenerativeAIEmbeddings` in batches of 10, with 5 s sleep between batches and tenacity retry on HTTP 429
- Stores `chunks` rows with `Vector(768)` embeddings; HNSW index via Alembic DDL migration

### Known Issues

- **`rewrite_question_router`** (`app/bot/nodes/rewrite_question.py:13–15`): condition is inverted — returns `no_answer_found` when tries are low and `rewrite_question` when tries are high. Fix by swapping the two return values.
- **`generate_direct`** does not push to `stream_queue`; the API controller handles this path separately.
- Dead streaming code block in `generate_from_context` (lines ~33–55) can be removed.

### LangGraph Checkpointing

`AsyncPostgresSaver` stores the full graph state per `thread_id` in `langgraph_*` tables, enabling multi-turn conversations that resume exactly where they left off. The checkpointer is initialized in `app/main.py` lifespan and passed to `RAGGraph.init()`.

## Infrastructure

| Service | Image | Port |
|---|---|---|
| PostgreSQL + pgvector | `pgvector/pgvector:pg17` | 5432 (container) → 5432 (host) |
| Redis | `redis:alpine` | 6379 |
| FastAPI | uvicorn | 8000 |
| Celery worker | `celery_aio_pool` (AsyncIO pool) | — |

The Celery worker uses `celery_aio_pool` so async coroutines work natively inside tasks.