"""Read-only audit of knowledge-base state across Qdrant, Postgres, and S3.

Run: python -m backend.scripts.audit_kb

Detects three kinds of drift:
  1. Qdrant chunks whose file_id has no matching row in files (orphan chunks) —
     this is what causes "file not found in database" on citation click.
  2. files rows that have no chunks in Qdrant (orphan rows / failed ingest).
  3. files rows whose S3 object is missing (deleted out of band).
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

from backend.config import settings
from backend.db import SessionLocal
from backend.models import Chunk, FileRecord
from backend.services import qdrant as qdrant_svc
from backend.services import s3 as s3_svc


async def _scan_qdrant() -> dict[str, int]:
    """Return {file_id: point_count} for every point currently in the collection."""
    client = qdrant_svc.client()
    counts: dict[str, int] = defaultdict(int)
    offset = None
    scanned = 0
    while True:
        points, offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=512,
            with_payload=["file_id"],
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            fid = (p.payload or {}).get("file_id")
            if fid:
                counts[str(fid)] += 1
        scanned += len(points)
        if offset is None or not points:
            break
    print(f"[qdrant] scanned {scanned} points across {len(counts)} distinct file_ids")
    return dict(counts)


def _scan_postgres() -> dict[str, dict]:
    """Return {file_id: {filename, s3_key, status, chunk_count}} for every files row."""
    db = SessionLocal()
    try:
        rows = db.query(FileRecord).all()
        chunk_counts_raw = (
            db.query(Chunk.file_id, Chunk.id).all()
        )
        chunk_counts: dict[str, int] = defaultdict(int)
        for fid, _ in chunk_counts_raw:
            chunk_counts[str(fid)] += 1
        out: dict[str, dict] = {}
        for r in rows:
            out[str(r.id)] = {
                "filename": r.filename,
                "s3_key": r.s3_key,
                "status": r.status,
                "chunk_count_pg": chunk_counts.get(str(r.id), 0),
            }
        print(f"[postgres] {len(rows)} files rows, {sum(chunk_counts.values())} chunk rows")
        return out
    finally:
        db.close()


def _list_s3_originals() -> set[str]:
    keys: set[str] = set()
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
            keys.add(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    print(f"[s3] {len(keys)} objects under {settings.S3_PREFIX_ORIGINALS!r}")
    return keys


async def main() -> None:
    print("=" * 72)
    print(f"Bucket: {settings.S3_BUCKET}  Collection: {settings.QDRANT_COLLECTION}")
    print("=" * 72)

    q_counts = await _scan_qdrant()
    pg = _scan_postgres()
    s3_keys = _list_s3_originals()
    referenced_s3_keys = {row["s3_key"] for row in pg.values()}

    q_ids = set(q_counts)
    pg_ids = set(pg)

    orphan_chunks = sorted(q_ids - pg_ids)
    orphan_rows = sorted(pg_ids - q_ids)
    present_in_both = q_ids & pg_ids
    orphan_s3 = sorted(s3_keys - referenced_s3_keys)
    missing_s3 = sorted(
        fid for fid, row in pg.items() if row["s3_key"] not in s3_keys
    )

    print()
    print(f"Qdrant file_ids      : {len(q_ids)}")
    print(f"Postgres file rows   : {len(pg_ids)}")
    print(f"Intersection (good)  : {len(present_in_both)}")
    print()

    print(f"--- Orphan Qdrant chunks (file_id has no files row) : {len(orphan_chunks)}")
    print("    These cause 'file not found in database' on citation click.")
    for fid in orphan_chunks[:50]:
        print(f"    {fid}  ({q_counts[fid]} points)")
    if len(orphan_chunks) > 50:
        print(f"    ... and {len(orphan_chunks) - 50} more")
    print()

    print(f"--- files rows with no Qdrant chunks              : {len(orphan_rows)}")
    print("    Likely failed or in-progress ingests.")
    for fid in orphan_rows[:50]:
        r = pg[fid]
        print(f"    {fid}  status={r['status']:<18} chunks_pg={r['chunk_count_pg']}  {r['filename']}")
    if len(orphan_rows) > 50:
        print(f"    ... and {len(orphan_rows) - 50} more")
    print()

    print(f"--- S3 objects not referenced by any files row    : {len(orphan_s3)}")
    for k in orphan_s3[:30]:
        print(f"    {k}")
    if len(orphan_s3) > 30:
        print(f"    ... and {len(orphan_s3) - 30} more")
    print()

    print(f"--- files rows whose S3 object is missing         : {len(missing_s3)}")
    for fid in missing_s3[:30]:
        r = pg[fid]
        print(f"    {fid}  s3_key={r['s3_key']}  {r['filename']}")
    print()

    verdict = "clean" if not (orphan_chunks or orphan_rows or orphan_s3 or missing_s3) else "DRIFT DETECTED"
    print(f"Result: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
