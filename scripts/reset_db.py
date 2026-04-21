"""Reset ALL state: Postgres schema + Qdrant collection + S3 bucket.

DESTRUCTIVE — wipes users, chats, messages, files, chunks, all vector
points, and every object in the S3 bucket. Use when model definitions
change or to wipe a dev environment.
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
from backend.services import s3 as s3_svc


async def _reset_qdrant() -> None:
    client = qdrant_svc.client()
    name = settings.QDRANT_COLLECTION
    cols = await client.get_collections()
    if any(c.name == name for c in cols.collections):
        print(f"Deleting Qdrant collection '{name}'…")
        await client.delete_collection(collection_name=name)
    print(f"Recreating Qdrant collection '{name}'…")
    await qdrant_svc.ensure_collection()


def _reset_s3() -> None:
    client = s3_svc.client()
    bucket = settings.S3_BUCKET
    print(f"Clearing S3 bucket '{bucket}'…")
    paginator = client.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=bucket):
        contents = page.get("Contents") or []
        if not contents:
            continue
        # delete_objects caps at 1000 keys per request; pages are already ≤1000.
        resp = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": obj["Key"]} for obj in contents], "Quiet": True},
        )
        for err in resp.get("Errors") or []:
            print(f"  failed to delete {err.get('Key')}: {err.get('Message')}")
        total += len(contents)
    print(f"Deleted {total} object(s) from '{bucket}'.")


def main() -> None:
    if "--yes" not in sys.argv:
        print("Refusing to run without --yes (this wipes Postgres + Qdrant + S3).")
        sys.exit(1)
    print("Dropping Postgres public schema (CASCADE)…")
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    print("Creating all Postgres tables…")
    Base.metadata.create_all(engine)
    asyncio.run(_reset_qdrant())
    _reset_s3()
    print("Done.")


if __name__ == "__main__":
    main()
