from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LOG_LEVEL: str = "DEBUG"
    ENV: str = "local"
    APP_NAME: str = "Self Rag API"
    APP_VERSION: str = "0.1.0"

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5433
    DB_NAME: str = "self_rag"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_ECHO: bool = False

    # AWS (S3 + SQS, backed by localstack for local dev)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = "test"
    AWS_SECRET_ACCESS_KEY: str = "test"
    AWS_ENDPOINT_URL: str | None = "http://localhost:4566"  # localstack; unset in real AWS

    S3_BUCKET_NAME: str = "self-rag-documents"

    SQS_QUEUE_NAME: str = "self-rag-ingest"
    SQS_POLL_WAIT_SECONDS: int = 20  # long polling
    SQS_VISIBILITY_TIMEOUT: int = 300

    EMBEDDING_DIM: int = 768  # Change to match your model (768 gemini, 1536 openai-small, 3072 openai-large)
    EMBEDDING_BATCH_SIZE: int = 10
    DB_INSERT_BATCH_SIZE: int = 100
    DB_INSERT_MAX_RETRIES: int = 3
    MAX_CHAT_HISTORY: int = 6

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 100
    SEMANTIC_CHUNKING: bool = False  # use SemanticChunker instead of RecursiveCharacterTextSplitter
    SEMANTIC_CHUNKING_BREAKPOINT_TYPE: str = "percentile"  # percentile | standard_deviation | interquartile

    CORS_ALLOWED_URL: str | None = None

    # Internal service-to-service auth
    INTERNAL_TOKEN: str | None = None

    # Graceful shutdown: max seconds to wait for in-flight RAG queries to finish
    GRACEFUL_SHUTDOWN_TIMEOUT: float = 30.0

    # retrieval strategy
    RETRIEVER_TOP_K: int = 3
    RETRIEVER_HYBRID: bool = True   # enable dense+BM25 RRF fusion
    RETRIEVER_RERANK: bool = False  # enable cross-encoder reranking (requires sentence-transformers)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # context eval
    CONTEXT_EVAL_HIGHER_THR: float = 0.7
    CONTEXT_EVAL_LOWER_THR: float = 0.3

    # llm
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str | None = None
    GEMINI_EMBEDDING_MODEL: str | None = None

    def validate_env_variables(self) -> None:
        required = {
            "CORS_ALLOWED_URL": self.CORS_ALLOWED_URL,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def db_url_psycopg(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
if settings.ENV not in ("test", "unittest"):
    settings.validate_env_variables()
