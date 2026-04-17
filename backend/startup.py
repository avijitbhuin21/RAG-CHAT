import asyncio
import logging

from sqlalchemy import text

from . import models  # noqa: F401  — register tables on Base.metadata
from .db import Base, engine
from .services import bifrost, qdrant, s3

log = logging.getLogger("startup")


async def _check_postgres() -> None:
    def go():
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    await asyncio.to_thread(go)
    log.info("postgres OK")


async def _check_qdrant() -> None:
    info = await qdrant.health()
    await qdrant.ensure_collection()
    log.info(f"qdrant OK -> {info}")


async def _check_s3() -> None:
    await asyncio.to_thread(s3.head_bucket)
    log.info("s3 OK")


async def _check_bifrost() -> None:
    await bifrost.health()
    log.info("bifrost OK")


def _init_schema() -> None:
    Base.metadata.create_all(engine)
    log.info("schema create_all OK")


async def run_startup() -> None:
    log.info("running startup checks...")
    await asyncio.to_thread(_init_schema)
    await asyncio.gather(_check_postgres(), _check_qdrant(), _check_s3(), _check_bifrost())
    log.info("all startup checks passed")
