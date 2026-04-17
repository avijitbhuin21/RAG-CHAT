"""Reset ALL state: Postgres schema + Qdrant collection.

DESTRUCTIVE — wipes users, chats, messages, files, chunks, and all vector
points. Use when model definitions change or to wipe a dev environment.
(We don't use Alembic by design; schema changes are handled by resetting.)

Run:
  python scripts/reset_db.py --yes
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from backend import models  # noqa: F401 — register tables on Base.metadata
from backend.config import settings
from backend.db import Base, engine
from backend.services import qdrant as qdrant_svc


async def _reset_qdrant() -> None:
    client = qdrant_svc.client()
    name = settings.QDRANT_COLLECTION
    cols = await client.get_collections()
    if any(c.name == name for c in cols.collections):
        print(f"Deleting Qdrant collection '{name}'…")
        await client.delete_collection(collection_name=name)
    print(f"Recreating Qdrant collection '{name}'…")
    await qdrant_svc.ensure_collection()


def main() -> None:
    if "--yes" not in sys.argv:
        print("Refusing to run without --yes (this wipes Postgres + Qdrant).")
        sys.exit(1)
    print("Dropping Postgres public schema (CASCADE)…")
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    print("Creating all Postgres tables…")
    Base.metadata.create_all(engine)
    asyncio.run(_reset_qdrant())
    print("Done. (S3 objects, if any, are left behind — delete manually if needed.)")


if __name__ == "__main__":
    main()
