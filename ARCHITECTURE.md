# Architecture

## Overview

Self-RAG is a production-grade Retrieval-Augmented Generation API that wraps a LangGraph state machine. Instead of a naive single-pass retrieve-then-answer loop, it runs two nested self-correction loops: one that rewrites the *answer* if it is not grounded in context, and one that rewrites the *question* if the answer is not useful. Streaming is served over HTTP via FastAPI. Document ingestion is handled asynchronously by a Celery worker.

---

## System Components

```
┌─────────────────────────────────────────────────────────┐
│                        Client                           │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP (SSE / plain text stream)
┌───────────────────────▼─────────────────────────────────┐
│                    FastAPI App                          │
│  Middleware: APITraceMiddleware → AuthMiddleware        │
│  Routers:  /api/v1/threads/**   /api/v1/documents/**    │
│  Static:   SPA at /chat/**                              │
└──────┬──────────────────────────────────┬───────────────┘
       │ ainvoke (LangGraph)               │ .delay (Celery)
┌──────▼───────────────┐        ┌──────────▼──────────────┐
│    RAGGraph           │        │    Celery Worker        │
│  (LangGraph compiled) │        │  ingest_document task   │
└──────┬───────────────┘        └──────────┬──────────────┘
       │                                    │
┌──────▼───────────────────────────────────▼──────────────┐
│             PostgreSQL + pgvector                       │
│  Tables: users, threads, messages, documents, chunks   │
│  LangGraph checkpoint tables (langgraph_*)             │
└─────────────────────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │  Redis  │
                    │ (Celery │
                    │ broker) │
                    └─────────┘
```

---

## Directory Layout

```
app/
├── main.py                  # FastAPI app, lifespan setup
├── celery_app.py            # Celery instance (Redis broker/backend)
│
├── api/
│   ├── routers/             # HTTP route definitions
│   │   ├── thread.py        # GET /threads, GET /threads/{id}, POST /threads/{id}/query
│   │   ├── document.py      # POST /documents/upload, GET, DELETE
│   ├── controller/          # Business logic called by routers
│   │   ├── thread.py        # Runs the RAG graph, manages asyncio.Queue for streaming
│   │   └── document.py      # Validates PDF, saves file, enqueues Celery task
│   └── models/              # Pydantic request/response schemas
│
├── bot/
│   ├── state.py             # RAGState (extends MessagesState)
│   └── nodes/               # One module per LangGraph node (see Graph section)
│
├── rag/
│   ├── graph.py             # Builds and compiles the StateGraph
│   ├── retriever.py         # Embedding-based vector search (pgvector)
│   └── ingestor/
│       ├── abstract.py      # BaseIngestor: load → chunk → embed → store
│       └── pdf_ingestor.py  # Concrete loader using PyPDFLoader
│
├── db/
│   ├── client.py            # SQLAlchemy async engine + session factory
│   ├── migrate.py           # Runs Alembic migrations on startup
│   ├── models/              # ORM models: User, Thread, Message, Document, Chunk
│   └── services/            # Thin async CRUD wrappers per model
│
├── worker/
│   └── tasks.py             # ingest_document Celery task
│
├── middlewares/
│   ├── api_trace.py         # Request ID injection, latency logging
│   └── auth.py              # AuthMiddleware (handles Bearer token/Internal token)
│
└── core/
    ├── config.py            # Pydantic Settings (env vars)
    ├── logging.py           # Structlog / contextvars setup
    └── custom_exceptions.py
```

---

## LangGraph: Self-RAG Graph

### State (`app/bot/state.py`)

| Field | Type | Purpose |
|---|---|---|
| `question` | `str` | Original user question |
| `retrieval_query` | `str` | Rewritten query used for vector search (may differ from `question`) |
| `need_retrieval` | `bool` | Gate: route to retriever or generate direct |
| `docs` | `List[Document]` | Raw retrieval results |
| `relevant_docs` | `List[Document]` | Docs that passed the relevance checker |
| `context_str` | `str` | Concatenated text of `relevant_docs` |
| `answer` | `str` | Current candidate answer |
| `answer_relevance` | `Literal[FULLY_SUPPORTED \| PARTIALLY_SUPPORTED \| NOT_SUPPORTED]` | Grounding verdict |
| `ans_iteration` | `int` | Answer-rewrite loop counter (max 3) |
| `is_ans_useful` | `bool` | Usefulness verdict |
| `reason` | `str` | LLM explanation for usefulness verdict |
| `rewrite_tries` | `int` | Question-rewrite loop counter (max 3) |
| `messages` | `List[BaseMessage]` | LangGraph chat history (inherited from MessagesState) |

### Node Reference

| Node | File | Role |
|---|---|---|
| `upsert_thread` | `thread_handler.py` | Creates thread record in DB if new; loads chat history into state |
| `should_retrieve` | `should_retrieve.py` | LLM decides `need_retrieval` (bool) |
| `generate_direct` | `generate_direct.py` | Answers from model knowledge; streams tokens to `stream_queue`; writes DB message |
| `context_retriever` | `context_retriever.py` | Vector search via `Retriever.get()` using `retrieval_query` (falls back to `question`) |
| `context_relevance_checker` | `context_relevance_checker.py` | Scores each doc in parallel; populates `relevant_docs` and `context_str` |
| `no_answer_found` | `no_answer_found.py` | Terminal: writes "no answer" message to DB and signals stream end |
| `generate_from_context` | `generate_from_context.py` | Generates `answer` from `context_str`; sets `ans_iteration = 0` |
| `answer_relevance_checker` | `answer_relevance_checker.py` | Grades `answer` as FULLY / PARTIALLY / NOT supported by context |
| `rewrite_answer` | `rewrite_answer.py` | Revises answer to use only direct quotes from the context; increments `ans_iteration` |
| `check_answer_usefulness` | `answer_usefulness.py` | LLM checks if answer actually addresses the question |
| `rewrite_question` | `rewrite_question.py` | Rewrites `retrieval_query` for better vector recall; increments `rewrite_tries`; clears `docs`/`relevant_docs`/`context_str` |
| `stream_answer` | `stream_answer.py` | Pushes final `answer` to `stream_queue`; writes DB message |

### Graph Edges

```
START
  └─► upsert_thread
        └─► should_retrieve
              ├─(need_retrieval=False)─► generate_direct ──► END
              └─(need_retrieval=True)──► context_retriever
                                              └─► context_relevance_checker
                                                    ├─(no relevant docs)──► no_answer_found ──► END
                                                    └─(relevant docs found)─► generate_from_context
                                                                                    └─► answer_relevance_checker
                                                                                          ├─(FULLY_SUPPORTED)──────────────────────────► check_answer_usefulness
                                                                                          ├─(PARTIALLY/NOT + tries < 3)─► rewrite_answer ─► answer_relevance_checker (loop)
                                                                                          └─(tries exhausted)─────────────────────────► check_answer_usefulness
                                                                                                                                              ├─(useful)──────────────────────────► stream_answer ──► END (implicit)
                                                                                                                                              ├─(not useful + rewrite_tries < 3)─► rewrite_question ─► context_retriever (loop)
                                                                                                                                              └─(rewrite_tries exhausted)─────────► no_answer_found ──► END
```

### Two Self-Correction Loops

**Answer-rewrite loop** (`generate_from_context` → `answer_relevance_checker` → `rewrite_answer`)
- Iterates until the answer is `FULLY_SUPPORTED` or `ans_iteration` reaches 3.
- `rewrite_answer` forces the model to produce only direct quotes from the context, stripping interpretive language.

**Question-rewrite loop** (`check_answer_usefulness` → `rewrite_question` → `context_retriever`)
- Triggers when the answer is not useful (even if grounded) and `rewrite_tries < 3`.
- `rewrite_question` generates a new retrieval-optimised query, resets document state, and re-enters the retrieval branch.

### Streaming

Streaming is delivered via an `asyncio.Queue` injected into `config["configurable"]["stream_queue"]`.
- `generate_direct` streams tokens one-by-one as the model produces them.
- `stream_answer` pushes the fully-assembled string at the end of the RAG path, then sends `None` as a sentinel.
- The HTTP layer (`ThreadController.ask_a_query`) consumes the queue via an async generator and returns a `SafeStreamingResponse`.

---

## Document Ingestion Pipeline

```
POST /api/v1/documents/upload
  │
  ├─ Validate PDF (magic bytes + extension)
  ├─ Save file to disk (app/uploads/)
  ├─ Insert Document row (status=PROCESSING)
  └─ celery_app.task.delay(ingest_document)
          │
          └─ PdfIngestor.ainvoke()
               ├─ PyPDFLoader.aload()
               ├─ RecursiveCharacterTextSplitter (600 chars / 150 overlap)
               ├─ GoogleGenerativeAIEmbeddings (batched, 10/batch, 5 s sleep between batches)
               │   └─ tenacity retry on HTTP 429 (exponential back-off, max 5 attempts)
               ├─ ChunkService.create_many() → chunks table (Vector(768))
               └─ DocumentService.update(status=COMPLETED | FAILED)
```

---

## Database Schema

| Table | Key Columns | Notes |
|---|---|---|
| `users` | `id`, `email`, … | Auth identity |
| `threads` | `id`, `user_id`, `title` | One conversation per thread |
| `messages` | `id`, `thread_id`, `role`, `content`, `latency_ms`, `token_count` | Persisted chat history |
| `documents` | `id`, `user_id`, `filename`, `file_path`, `status`, `error` | Upload tracking |
| `chunks` | `id`, `document_id`, `content`, `embedding` (Vector 768), `chunk_index`, `metadata_` | HNSW index on embedding |
| `langgraph_*` | (managed by LangGraph) | Conversation checkpoints; enables multi-turn resumption |

`chunks.embedding` uses an HNSW index created via raw DDL in an Alembic migration for approximate-nearest-neighbour search via pgvector.

---

## Infrastructure

| Component | Image / Tech | Port | Purpose |
|---|---|---|---|
| FastAPI (uvicorn) | Python 3.12 | 8000 | API + SPA serving |
| PostgreSQL | `pgvector/pgvector:pg17` | 5433 | Primary store + vector index + LangGraph checkpoints |
| Redis | `redis:alpine` | 6379 | Celery broker and result backend |
| Celery Worker | `celery_aio_pool` (AsyncIO pool) | — | Background document ingestion |

---

## Key Design Decisions

- **Single DB for everything**: PostgreSQL serves as the relational store, the vector database (pgvector), and the LangGraph checkpoint backend. No separate vector DB.
- **Streaming via asyncio.Queue**: The graph runs in a background `asyncio.Task`; the HTTP response reads from the queue concurrently. This avoids blocking the event loop during long LLM calls.
- **LangGraph checkpointer**: `AsyncPostgresSaver` stores the full graph state per `thread_id`, enabling multi-turn conversations that resume exactly where they left off.
- **Celery + AsyncIO pool**: The ingestion worker uses `celery_aio_pool` so async coroutines (embedding API calls, DB writes) work natively inside Celery tasks.
- **Dual loop max = 3**: Both `_MAX_ANS_ITERATION` and `_MAX_QUE_ITERATION` are hardcoded to 3, bounding worst-case LLM calls per query to ~20.
- **Security Scopes**: Access control enforced via AuthMiddleware (Bearer token).
