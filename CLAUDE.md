# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Start infrastructure (Postgres + localstack for S3/SQS)
docker-compose up -d

# Run database migrations
alembic upgrade head

# Start API server
uvicorn app.main:app --reload

# Start SQS consumer (document ingestion)
python -m app.worker.sqs_consumer

# Generate a new migration
alembic revision --autogenerate -m "description"
```

Environment: copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, and `CORS_ALLOWED_URL` (required). The DB default is `localhost:5433` (note: not the standard 5432 — docker-compose maps `pgvector/pgvector:pg17` to 5433). AWS config (`AWS_ENDPOINT_URL`, `S3_BUCKET_NAME`, `SQS_QUEUE_NAME`) defaults to localstack at `http://localhost:4566` for local dev.

## Architecture

Self-RAG is a FastAPI app wrapping a **LangGraph state machine** (`app/rag/graph.py`) that implements two nested self-correction loops before streaming a final answer. Document ingestion is decoupled via **S3 + SQS**: uploads go to S3 and a message is enqueued to SQS, consumed by a standalone async poller (`app/worker/sqs_consumer.py`) that runs the ingestion pipeline. Locally, S3/SQS are provided by **localstack**. A single PostgreSQL instance (with pgvector) serves as the relational store, the vector database, and the LangGraph checkpoint backend.

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
| `app/rag/ingestor/` | `BaseIngestor` + `PdfIngestor`: load from S3 → chunk → embed → store |
| `app/api/controller/thread.py` | Runs the graph, manages `asyncio.Queue` for streaming |
| `app/core/config.py` | `Settings` (Pydantic settings, all env vars) |
| `app/db/client.py` | SQLAlchemy async engine + session factory |
| `app/core/aws.py` | Shared `aioboto3` session + S3/SQS client factories (localstack endpoint) |
| `app/core/s3.py` | `upload_bytes` / `download_bytes` / `delete_object` helpers |
| `app/worker/sqs_queue.py` | `send_ingest_message` (producer) + `receive_messages` / `delete_message` (consumer) |
| `app/worker/sqs_consumer.py` | Standalone long-polling SQS consumer that runs `PdfIngestor.ainvoke()` |

### Document Ingestion

`POST /api/v1/documents/upload` → validates PDF (magic bytes) → uploads bytes to S3 (`{user_id}/{filename}` key) → inserts `Document` row (status=PROCESSING, `file_path` = S3 key) → sends an SQS message (`document_id`, `user_id`, `filename`, `s3_key`) → `app/worker/sqs_consumer.py` long-polls the queue and, per message, runs `PdfIngestor.ainvoke()`:
- Downloads the object from S3 into a temp file → `PyPDFLoader` → `RecursiveCharacterTextSplitter` (600 chars / 150 overlap)
- `GoogleGenerativeAIEmbeddings` in batches of 10, with 5 s sleep between batches and tenacity retry on HTTP 429
- Stores `chunks` rows with `Vector(768)` embeddings; HNSW index via Alembic DDL migration
- On success, deletes the SQS message. On failure, the message is left in place — it becomes visible again after the visibility timeout and SQS's redrive policy moves it to the DLQ (`{SQS_QUEUE_NAME}-dlq`) after 5 receives.

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
| localstack (S3 + SQS) | `localstack/localstack` | 4566 |
| FastAPI | uvicorn | 8000 |
| SQS consumer | `python -m app.worker.sqs_consumer` | — |

`scripts/localstack-init.sh` runs on localstack startup (mounted into `/etc/localstack/init/ready.d/`) and provisions the S3 bucket and SQS queue + DLQ with a redrive policy (max 5 receives).