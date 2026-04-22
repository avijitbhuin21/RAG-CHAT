from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path so Settings() works regardless of where the process is launched
# from (e.g. `py backend/app.py` run from inside the backend/ directory).
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), extra="ignore", case_sensitive=True
    )

    APP_NAME: str = "1staid4sme-Agent"
    FRONTEND_URL: str = "http://localhost:3000"
    BACKEND_URL: str = "http://localhost:8000"
    CORS_ORIGINS: str = "http://localhost:3000"
    # True in prod (cross-subdomain HTTPS): cookies need SameSite=None + Secure.
    # False for local http://localhost dev.
    CROSS_SITE_COOKIES: bool = False

    ADMIN_USERNAME: str
    ADMIN_PASSWORD: str

    GOOGLE_OAUTH_CLIENT_ID: str
    GOOGLE_OAUTH_CLIENT_SECRET: str
    GOOGLE_OAUTH_REDIRECT_URI: str

    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 43200

    BIFROST_BASE_URL: str
    BIFROST_API_KEY: str
    BIFROST_LLM_MODEL: str = "claude-sonnet-4-6"
    BIFROST_EMBEDDING_MODEL: str = "gemini-embedding-001"
    BIFROST_ANTHROPIC_BETA: str = "context-1m-2025-08-07"
    # Anthropic-native endpoint on Bifrost. Docs: point the Anthropic SDK
    # at `{host}/anthropic` (SDK appends `/v1/messages` internally). Leave
    # blank to auto-derive from BIFROST_BASE_URL by stripping the OpenAI-
    # compat `/v1` suffix.
    BIFROST_ANTHROPIC_BASE_URL: str = ""

    EMBEDDING_DIM: int = 1536
    THINKING_BUDGET_TOKENS: int = 5000
    # Sonnet 4.6's standard output cap is 64k tokens. Anthropic's API requires
    # max_tokens to be set explicitly; when we don't pass it, Bifrost's
    # OpenAI→Anthropic translator picks an unsafe small default (~8) that
    # silently truncates long answers.
    LLM_MAX_OUTPUT_TOKENS: int = 64000
    CHAT_HISTORY_MAX_MESSAGES: int = 40
    CHAT_RATE_LIMIT_PER_HOUR: int = 100

    DATABASE_URL: str

    QDRANT_URL: str
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "documents"

    S3_ENDPOINT: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_SECURE: bool = True
    S3_REGION: str = "auto"
    S3_BUCKET: str
    S3_PREFIX_ORIGINALS: str = "originals/"
    S3_PREFIX_PAGE_RENDERS: str = "page-renders/"

    # Verified ceilings from scripts/test_bifrost_embeddings.py at 75% safety margin.
    EMBED_MAX_BATCH_ITEMS: int = 75
    EMBED_MAX_BATCH_TOKENS: int = 22500
    EMBED_MAX_TOKENS_PER_TEXT: int = 4500

    # Keep low on Railway — Docling is ~1–2 GB RSS per concurrent file even
    # with the singleton. 2 is safe on 8 GB; 3 on 16 GB.
    INGEST_CONCURRENT_FILES: int = 2

    # Fast path: if pymupdf pulls at least this many chars per page from a
    # PDF, treat it as a native-text PDF. Scanned PDFs fall below this
    # threshold; with no OCR fallback they produce zero extractable text
    # and the ingest will fail cleanly with "document produced zero
    # chunks" rather than crashing the container.
    PYMUPDF_MIN_CHARS_PER_PAGE: int = 50

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def bifrost_anthropic_base_url(self) -> str:
        """Resolve the Anthropic-native base URL on Bifrost.

        Bifrost segregates provider paths: OpenAI-compat lives under
        `/openai/v1/...` and Anthropic-native under `/anthropic/v1/...`.
        We derive the Anthropic base by stripping any `/openai` / `/v1`
        tail from BIFROST_BASE_URL and appending `/anthropic`. The
        Anthropic SDK then appends `/v1/messages` itself.

        Set BIFROST_ANTHROPIC_BASE_URL explicitly to override.
        """
        if self.BIFROST_ANTHROPIC_BASE_URL:
            return self.BIFROST_ANTHROPIC_BASE_URL.rstrip("/")
        base = self.BIFROST_BASE_URL.rstrip("/")
        for suffix in ("/v1", "/openai", "/openai/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        # Second pass in case /openai/v1 was trimmed to /openai.
        if base.endswith("/openai"):
            base = base[: -len("/openai")]
        return f"{base}/anthropic"


settings = Settings()
