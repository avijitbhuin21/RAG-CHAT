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

    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_LLM_MODEL: str = "z-ai/glm-5.3-flash"
    OPENROUTER_EMBEDDING_MODEL: str = "google/gemini-embedding-001"
    # OpenRouter `reasoning.effort`. GLM 5.3 Flash emits zero reasoning tokens
    # below `max`; keep `max` if the thinking panel should show anything.
    LLM_REASONING_EFFORT: str = "max"

    EMBEDDING_DIM: int = 1536
    LLM_MAX_OUTPUT_TOKENS: int = 32000
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

    # Verified ceilings from scripts/test_openrouter_embeddings.py at 75% safety margin.
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


settings = Settings()
