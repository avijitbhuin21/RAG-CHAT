import asyncio
import logging
import os
import tempfile
import time
from typing import Any
from uuid import UUID, uuid4

from qdrant_client.models import PointStruct

from ..config import settings
from ..db import SessionLocal
from ..models import Chunk, FileRecord
from ..progress_broker import broker
from . import bifrost, qdrant, s3

log = logging.getLogger("task.ingest")


def _prefix(task: str, file_id: UUID, filename: str | None = None) -> str:
    """Every log line for a background task starts with one of these so you
    can grep a single task end-to-end."""
    tag = str(file_id)[:8]
    if filename:
        return f"[{task} {tag} {filename!r}]"
    return f"[{task} {tag}]"

_semaphore: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.INGEST_CONCURRENT_FILES)
    return _semaphore


# ---------- sync DB helpers (called via to_thread) ----------
def _load_file_row(file_id: UUID) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = db.get(FileRecord, file_id)
        if row is None:
            raise RuntimeError(f"file {file_id} not found")
        return {"s3_key": row.s3_key, "filename": row.filename}
    finally:
        db.close()


def _update_status(file_id: UUID, **kwargs) -> None:
    db = SessionLocal()
    try:
        row = db.get(FileRecord, file_id)
        if row is None:
            return
        for k, v in kwargs.items():
            setattr(row, k, v)
        db.commit()
    finally:
        db.close()


def _insert_chunk_rows(file_id: UUID, records: list[dict]) -> None:
    db = SessionLocal()
    try:
        db.bulk_insert_mappings(
            Chunk,
            [
                {
                    "id": uuid4(),
                    "file_id": file_id,
                    "qdrant_point_id": r["point_id"],
                    "page": r.get("page"),
                    "chunk_text": r["text"],
                    "element_type": None,
                }
                for r in records
            ],
        )
        db.commit()
    finally:
        db.close()


# ---------- parsing + chunking ----------
def _download(s3_key: str, prefix: str) -> bytes:
    t0 = time.perf_counter()
    log.info("%s downloading from S3 key=%s", prefix, s3_key)
    resp = s3.client().get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
    data = resp["Body"].read()
    log.info(
        "%s downloaded %d bytes in %.2fs",
        prefix,
        len(data),
        time.perf_counter() - t0,
    )
    return data


def _parse_with_docling(data: bytes, filename: str, prefix: str) -> str:
    """Blocking -- call via asyncio.to_thread. Returns markdown."""
    suffix = os.path.splitext(filename)[1] or ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(data)
        tmp_path = tf.name
    try:
        log.info(
            "%s parsing with Docling (first run may download models, this can take minutes)...",
            prefix,
        )
        t0 = time.perf_counter()
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(tmp_path)
        md = result.document.export_to_markdown()
        log.info(
            "%s Docling parse done in %.2fs (%d chars of markdown)",
            prefix,
            time.perf_counter() - t0,
            len(md),
        )
        return md
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _chunk_markdown(text: str, target_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    """Paragraph-aware sliding window. Target ~500 tokens (≈2000 chars)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= target_chars:
            buf = buf + "\n\n" + p
            continue
        chunks.append(buf)
        tail = buf[-overlap_chars:] if overlap_chars else ""
        buf = (tail + "\n\n" + p) if tail else p
    if buf:
        chunks.append(buf)
    return chunks


# ---------- orchestrator ----------
async def run_ingest(file_id: UUID) -> None:
    """Entry point scheduled via BackgroundTasks. Guarded by a global
    semaphore so at most INGEST_CONCURRENT_FILES run at the same time."""
    try:
        meta = await asyncio.to_thread(_load_file_row, file_id)
    except Exception:
        log.exception("[ingest %s] could not load file row -- aborting", str(file_id)[:8])
        return

    filename = meta["filename"]
    s3_key = meta["s3_key"]
    prefix = _prefix("ingest", file_id, filename)

    log.info("%s ====== starting ingest task ======", prefix)
    log.info("%s file_id=%s s3_key=%s", prefix, file_id, s3_key)

    sem = _sem()
    if sem.locked():
        log.info(
            "%s all %d concurrency slots busy, waiting for a slot...",
            prefix,
            settings.INGEST_CONCURRENT_FILES,
        )
    async with sem:
        log.info("%s acquired concurrency slot", prefix)
        task_t0 = time.perf_counter()
        try:
            await _do_ingest(file_id, filename, s3_key, prefix)
            log.info(
                "%s ====== done in %.2fs -- status=ready ======",
                prefix,
                time.perf_counter() - task_t0,
            )
        except Exception as e:  # noqa: BLE001 -- top-level task guard
            log.exception(
                "%s FAILED after %.2fs -- %s",
                prefix,
                time.perf_counter() - task_t0,
                e,
            )
            await asyncio.to_thread(
                _update_status, file_id, status="failed", error_message=str(e)[:500]
            )
            broker.publish(str(file_id), {"status": "failed", "error": str(e)[:200]})


async def _do_ingest(file_id: UUID, filename: str, s3_key: str, prefix: str) -> None:
    # 1. parse
    await asyncio.to_thread(
        _update_status, file_id, status="parsing", stage_current=0, stage_total=0
    )
    broker.publish(str(file_id), {"status": "parsing", "filename": filename})

    data = await asyncio.to_thread(_download, s3_key, prefix)
    markdown = await asyncio.to_thread(_parse_with_docling, data, filename, prefix)

    # 2. chunk
    log.info("%s chunking markdown (target ~500 tokens per chunk)...", prefix)
    t_chunk = time.perf_counter()
    await asyncio.to_thread(_update_status, file_id, status="chunking")
    broker.publish(str(file_id), {"status": "chunking"})
    chunks = _chunk_markdown(markdown)
    total = len(chunks)
    log.info(
        "%s chunked into %d pieces in %.2fs",
        prefix,
        total,
        time.perf_counter() - t_chunk,
    )
    if total == 0:
        raise RuntimeError("document produced zero chunks")
    await asyncio.to_thread(_update_status, file_id, stage_current=0, stage_total=total)
    broker.publish(
        str(file_id), {"status": "chunking", "stage_current": 0, "stage_total": total}
    )

    # 3. embed + upsert in batches
    await asyncio.to_thread(_update_status, file_id, status="embedding")
    broker.publish(
        str(file_id), {"status": "embedding", "stage_current": 0, "stage_total": total}
    )
    batch_size = settings.EMBED_MAX_BATCH_ITEMS
    batch_total = (total + batch_size - 1) // batch_size
    chunk_records: list[dict] = []

    for batch_idx, start in enumerate(range(0, total, batch_size), 1):
        batch = chunks[start : start + batch_size]
        log.info(
            "%s embedding batch %d/%d (%d items)...",
            prefix,
            batch_idx,
            batch_total,
            len(batch),
        )
        t_embed = time.perf_counter()
        vectors = await bifrost.embed_texts(batch)
        log.info(
            "%s  embed done in %.2fs -- upserting to Qdrant",
            prefix,
            time.perf_counter() - t_embed,
        )
        points = []
        for j, (text, vec) in enumerate(zip(batch, vectors)):
            point_id = str(uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "file_id": str(file_id),
                        "filename": filename,
                        "chunk_index": start + j,
                        "chunk_text": text,
                    },
                )
            )
            chunk_records.append({"point_id": point_id, "text": text})
        t_upsert = time.perf_counter()
        await qdrant.client().upsert(
            collection_name=settings.QDRANT_COLLECTION, points=points
        )
        log.info(
            "%s  upsert done in %.2fs (%d points)",
            prefix,
            time.perf_counter() - t_upsert,
            len(points),
        )
        done = min(start + batch_size, total)
        await asyncio.to_thread(_update_status, file_id, stage_current=done)
        broker.publish(
            str(file_id),
            {"status": "embedding", "stage_current": done, "stage_total": total},
        )

    # 4. persist chunk rows (for traceability; Qdrant is the source of truth)
    log.info("%s persisting %d chunk rows to Postgres", prefix, len(chunk_records))
    await asyncio.to_thread(_insert_chunk_rows, file_id, chunk_records)

    # 5. done
    await asyncio.to_thread(_update_status, file_id, status="ready")
    broker.publish(
        str(file_id), {"status": "ready", "stage_current": total, "stage_total": total}
    )


# ---------- delete orchestrator ----------
def _fetch_s3_key(file_id: UUID) -> str | None:
    db = SessionLocal()
    try:
        row = db.get(FileRecord, file_id)
        return row.s3_key if row else None
    finally:
        db.close()


def _drop_row(file_id: UUID) -> None:
    db = SessionLocal()
    try:
        row = db.get(FileRecord, file_id)
        if row is None:
            return
        db.delete(row)
        db.commit()
    finally:
        db.close()


async def run_delete(file_id: UUID) -> None:
    """Background delete pipeline: Qdrant points → S3 object → DB row.
    Ordering matters -- we need the s3_key from the row, so DB goes last.
    On failure, records which stage blew up so the UI can surface it."""
    try:
        filename = (await asyncio.to_thread(_load_file_row, file_id))["filename"]
    except Exception:
        filename = None  # row may be missing on retry

    prefix = _prefix("delete", file_id, filename)
    log.info("%s ====== starting delete task ======", prefix)
    task_t0 = time.perf_counter()

    stage = "start"
    try:
        stage = "qdrant"
        log.info("%s stage=qdrant -- removing all points for file_id", prefix)
        t = time.perf_counter()
        await qdrant.delete_points_for_file(file_id)
        log.info("%s  qdrant done in %.2fs", prefix, time.perf_counter() - t)

        stage = "s3"
        s3_key = await asyncio.to_thread(_fetch_s3_key, file_id)
        if s3_key:
            log.info("%s stage=s3 -- removing object key=%s", prefix, s3_key)
            t = time.perf_counter()
            await asyncio.to_thread(s3.delete_object, s3_key)
            log.info("%s  s3 done in %.2fs", prefix, time.perf_counter() - t)
        else:
            log.info("%s stage=s3 -- no s3_key to delete (already gone)", prefix)

        stage = "db"
        log.info("%s stage=db -- dropping row (cascades to chunks)", prefix)
        t = time.perf_counter()
        await asyncio.to_thread(_drop_row, file_id)
        log.info("%s  db done in %.2fs", prefix, time.perf_counter() - t)

        broker.publish(str(file_id), {"status": "deleted"})
        log.info(
            "%s ====== done in %.2fs -- status=deleted ======",
            prefix,
            time.perf_counter() - task_t0,
        )
    except Exception as e:  # noqa: BLE001
        err = f"{stage}: {str(e)[:300]}"
        log.exception("%s FAILED at stage=%s after %.2fs", prefix, stage, time.perf_counter() - task_t0)
        await asyncio.to_thread(
            _update_status, file_id, status="delete_failed", error_message=err
        )
        broker.publish(str(file_id), {"status": "delete_failed", "error": err})
