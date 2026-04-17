"""Application entry point.

Run directly:
    python backend/app.py

Or, for auto-reload during development:
    python -m uvicorn backend.app:app --reload --port 8000
"""
import logging
import logging.config
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Force line-buffered stdout/stderr. Without this, when this module is imported
# inside uvicorn's reload-subprocess (Windows spawn in particular), the child's
# stderr is a pipe to the parent and can sit block-buffered -- so background
# task log lines don't appear in real time during long operations like Docling
# parsing. line_buffering=True makes each '\n' flush immediately.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

# Configure logging BEFORE any other imports, so any log calls emitted during
# module-load (e.g. during `startup.run_startup` or settings validation) are
# captured. We use dictConfig (not basicConfig) so this is deterministic even
# if uvicorn has already installed its own handlers in this process.
_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "default",
            "level": "INFO",
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        # Our application namespaces — be explicit so they propagate to root
        # and can be tuned independently if they ever get too chatty.
        "task": {"level": "INFO"},
        "api": {"level": "INFO"},
        "startup": {"level": "INFO"},
        # Silence noisy third-party INFO logs by default.
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
    },
}
logging.config.dictConfig(_LOG_CONFIG)

# Make `python backend/app.py` work by putting the project root on sys.path,
# so the absolute `from backend.*` imports below resolve.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from starlette.middleware.sessions import SessionMiddleware  # noqa: E402

from backend.api import admin_files, auth, chat  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.startup import run_startup  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_startup()
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.JWT_SECRET,
    session_cookie="oauth_state",
    same_site="lax",
    https_only=False,
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(admin_files.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import os

    import uvicorn

    # Uvicorn's reload spawns a subprocess that re-imports "backend.app:app";
    # that subprocess's CWD must be the project root (not backend/) so Python
    # can find the `backend` package.
    os.chdir(_ROOT)
    # log_config=None: don't let uvicorn re-install its own root handlers on
    # top of ours. Our dictConfig already covers everything we care about.
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,
    )
