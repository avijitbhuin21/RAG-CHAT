import re

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


def _normalize_db_url(url: str) -> str:
    """Point any Postgres URL variant (postgres://, postgresql+psycopg://, ...) at the installed psycopg2 driver."""
    return re.sub(r"^postgres(?:ql)?(?:\+[a-z0-9_]+)?://", "postgresql+psycopg2://", url, count=1)


engine = create_engine(
    _normalize_db_url(settings.DATABASE_URL),
    pool_pre_ping=False,
    pool_recycle=280,
    pool_size=5,
    max_overflow=10,
    future=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
