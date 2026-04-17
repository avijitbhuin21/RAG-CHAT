"""Wipe the knowledge base: Qdrant collection + S3 originals + Postgres
files/chunks rows. Chats and users are left alone by default.

Run: python -m backend.scripts.clean_db --yes
     python -m backend.scripts.clean_db --yes --include-chats
     python -m backend.scripts.clean_db            (dry run — no flag = no action)

Bulk state clear — exactly what you want after a dev reset left drift between
Qdrant, S3, and Postgres. After running you can re-upload files from scratch.
"""
from __future__ import annotations

import argparse
import asyncio

from botocore.exceptions import ClientError
from qdrant_client.models import Distance, VectorParams

from backend.config import settings
from backend.db import SessionLocal
from backend.models import Chat, Chunk, FileRecord, Message
from backend.services import qdrant as qdrant_svc
from backend.services import s3 as s3_svc


def _summary() -> dict:
    db = SessionLocal()
    try:
        return {
            "files": db.query(FileRecord).count(),
            "chunks": db.query(Chunk).count(),
            "chats": db.query(Chat).count(),
            "messages": db.query(Message).count(),
        }
    finally:
        db.close()


def _list_s3_keys() -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs = {
            "Bucket": settings.S3_BUCKET,
            "Prefix": settings.S3_PREFIX_ORIGINALS,
        }
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3_svc.client().list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            keys.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def _delete_s3_batch(keys: list[str]) -> None:
    # delete_objects accepts up to 1000 keys per call.
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        s3_svc.client().delete_objects(
            Bucket=settings.S3_BUCKET,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )


async def _reset_qdrant() -> int:
    """Drop and recreate the collection — faster and more thorough than
    deleting every point with a filter, and guarantees no stale schema."""
    client = qdrant_svc.client()
    name = settings.QDRANT_COLLECTION
    # Count before we drop, purely for reporting.
    try:
        info = await client.get_collection(name)
        count = info.points_count or 0
    except Exception:
        count = 0
    try:
        await client.delete_collection(name)
    except Exception:
        pass
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=settings.EMBEDDING_DIM, distance=Distance.COSINE
        ),
    )
    return count


def _wipe_postgres(include_chats: bool) -> None:
    db = SessionLocal()
    try:
        # Chunks and files: delete chunks first (FK to files).
        db.query(Chunk).delete(synchronize_session=False)
        db.query(FileRecord).delete(synchronize_session=False)
        if include_chats:
            db.query(Message).delete(synchronize_session=False)
            db.query(Chat).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform the wipe. Without this flag the script only reports.",
    )
    parser.add_argument(
        "--include-chats",
        action="store_true",
        help="Also drop all chats and messages. Users are always preserved.",
    )
    args = parser.parse_args()

    before = _summary()
    s3_keys = _list_s3_keys()
    try:
        q_info = await qdrant_svc.client().get_collection(settings.QDRANT_COLLECTION)
        q_points = q_info.points_count or 0
    except Exception:
        q_points = 0

    print("=" * 72)
    print(f"Bucket:        {settings.S3_BUCKET}")
    print(f"Collection:    {settings.QDRANT_COLLECTION}")
    print("=" * 72)
    print("Current state:")
    print(f"  Postgres files rows    : {before['files']}")
    print(f"  Postgres chunk rows    : {before['chunks']}")
    print(f"  Postgres chat rows     : {before['chats']}")
    print(f"  Postgres message rows  : {before['messages']}")
    print(f"  S3 objects ({settings.S3_PREFIX_ORIGINALS}): {len(s3_keys)}")
    print(f"  Qdrant points          : {q_points}")
    print()
    print("Plan:")
    print("  - drop and recreate Qdrant collection")
    print("  - delete all S3 objects under the originals prefix")
    print("  - delete all rows in files + chunks")
    if args.include_chats:
        print("  - delete all rows in chats + messages  [--include-chats]")
    else:
        print("  - KEEP chats + messages (pass --include-chats to drop these too)")
    print("  - KEEP users (always preserved)")
    print()

    if not args.yes:
        print("Dry run. Re-run with --yes to actually perform the wipe.")
        return

    print("Proceeding...")
    print()

    print("[qdrant] resetting collection...")
    q_deleted = await _reset_qdrant()
    print(f"  dropped ~{q_deleted} points, recreated empty collection")

    print("[s3] deleting objects...")
    try:
        _delete_s3_batch(s3_keys)
        print(f"  deleted {len(s3_keys)} objects")
    except ClientError as e:
        print(f"  WARNING: S3 delete failed: {e}")

    print("[postgres] deleting rows...")
    _wipe_postgres(args.include_chats)
    after = _summary()
    print(f"  files: {before['files']} -> {after['files']}")
    print(f"  chunks: {before['chunks']} -> {after['chunks']}")
    if args.include_chats:
        print(f"  chats: {before['chats']} -> {after['chats']}")
        print(f"  messages: {before['messages']} -> {after['messages']}")

    print()
    print("Done. Knowledge base is empty — re-upload files to rebuild.")


if __name__ == "__main__":
    asyncio.run(main())
