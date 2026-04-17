import asyncio
import hashlib
import json
import logging
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Chunk, FileRecord
from ..progress_broker import broker
from ..schemas import FileOut
from ..security import require_admin
from ..services import qdrant as qdrant_svc
from ..services import s3 as s3_svc

router = APIRouter(prefix="/admin/files", tags=["admin-files"])
log = logging.getLogger("api.admin_files")


# ---------- helpers ----------
def _s3_key_for(content_hash: str, filename: str) -> str:
    safe_name = filename.replace("/", "_").replace("\\", "_")[:200]
    return f"{settings.S3_PREFIX_ORIGINALS}{content_hash[:16]}-{safe_name}"


async def _put_s3(key: str, body: bytes, mime: str) -> None:
    def go():
        s3_svc.client().put_object(
            Bucket=settings.S3_BUCKET, Key=key, Body=body, ContentType=mime
        )
    await asyncio.to_thread(go)


async def _delete_s3(key: str) -> None:
    def go():
        s3_svc.client().delete_object(Bucket=settings.S3_BUCKET, Key=key)
    await asyncio.to_thread(go)


def _schedule_ingest(tasks: BackgroundTasks, file_id: UUID) -> None:
    from ..services.ingest import run_ingest  # lazy to avoid Docling import on app boot
    tasks.add_task(run_ingest, file_id)


# ---------- routes ----------
@router.get("", response_model=list[FileOut])
def list_files(_: dict = Depends(require_admin), db: Session = Depends(get_db)):
    # Hide rows mid-delete; they reappear automatically if the delete fails
    # (status flips to delete_failed, which is visible).
    rows = (
        db.query(FileRecord)
        .filter(FileRecord.status != "deleting")
        .order_by(FileRecord.created_at.desc())
        .all()
    )
    return rows


@router.post("")
async def upload_files(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    _: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Multipart upload, one or many files. For each:
      - hash match  -> 'exact_duplicate'
      - name match  -> 'name_conflict' (frontend prompts replace)
      - otherwise   -> upload to S3, insert row, schedule ingest.
    """
    results = []
    for f in files:
        body = await f.read()
        if not body:
            results.append({"filename": f.filename, "status": "empty"})
            continue
        h = hashlib.sha256(body).hexdigest()

        exact = db.query(FileRecord).filter_by(content_hash=h).one_or_none()
        if exact:
            results.append(
                {
                    "filename": f.filename,
                    "status": "exact_duplicate",
                    "existing_file_id": str(exact.id),
                    "existing_filename": exact.filename,
                }
            )
            continue

        name_match = db.query(FileRecord).filter_by(filename=f.filename).one_or_none()
        if name_match:
            results.append(
                {
                    "filename": f.filename,
                    "status": "name_conflict",
                    "existing_file_id": str(name_match.id),
                    "existing_status": name_match.status,
                }
            )
            continue

        key = _s3_key_for(h, f.filename)
        mime = f.content_type or "application/octet-stream"
        await _put_s3(key, body, mime)

        row = FileRecord(
            id=uuid4(),
            filename=f.filename,
            content_hash=h,
            s3_key=key,
            size_bytes=len(body),
            mime_type=mime,
            status="pending_ingest",
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        _schedule_ingest(background, row.id)
        log.info(
            "[upload] queued ingest task id=%s filename=%r size=%d bytes",
            str(row.id)[:8],
            row.filename,
            len(body),
        )
        broker.publish(str(row.id), {"status": "pending_ingest", "filename": row.filename})
        results.append({"filename": f.filename, "status": "queued", "file_id": str(row.id)})

    return {"results": results}


@router.post("/{file_id}/replace")
async def replace_file(
    file_id: UUID,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    _: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    row = db.query(FileRecord).filter_by(id=file_id).one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="file not found")

    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="empty file")
    new_hash = hashlib.sha256(body).hexdigest()

    if new_hash == row.content_hash:
        raise HTTPException(status_code=409, detail="new file is identical to existing")

    hash_clash = (
        db.query(FileRecord)
        .filter(FileRecord.content_hash == new_hash, FileRecord.id != file_id)
        .one_or_none()
    )
    if hash_clash:
        raise HTTPException(
            status_code=409,
            detail=f"content already exists under file id {hash_clash.id}",
        )

    # wipe old vectors + chunks + S3 object
    await qdrant_svc.delete_points_for_file(file_id)
    db.query(Chunk).filter_by(file_id=file_id).delete(synchronize_session=False)
    await _delete_s3(row.s3_key)

    new_key = _s3_key_for(new_hash, file.filename or row.filename)
    new_mime = file.content_type or row.mime_type
    await _put_s3(new_key, body, new_mime)

    row.filename = file.filename or row.filename
    row.content_hash = new_hash
    row.s3_key = new_key
    row.size_bytes = len(body)
    row.mime_type = new_mime
    row.status = "pending_ingest"
    row.stage_current = 0
    row.stage_total = 0
    row.error_message = None
    db.commit()
    db.refresh(row)

    _schedule_ingest(background, row.id)
    broker.publish(str(row.id), {"status": "pending_ingest", "filename": row.filename})
    return {"file_id": str(row.id), "status": "queued"}


@router.delete("/{file_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_file(
    file_id: UUID,
    background: BackgroundTasks,
    _: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Optimistic delete — flip status to 'deleting' and return immediately.
    The real work (Qdrant points + S3 object + DB row) happens in a background
    task. On failure, status flips to 'delete_failed' with an error message
    indicating which stage blew up; the file reappears on the next list call."""
    row = db.query(FileRecord).filter_by(id=file_id).one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="file not found")
    if row.status == "deleting":
        return {"file_id": str(file_id), "status": "deleting"}

    row.status = "deleting"
    row.error_message = None
    db.commit()

    from ..services.ingest import run_delete  # lazy (keeps Docling off the boot path)
    background.add_task(run_delete, file_id)
    log.info(
        "[delete] queued delete task id=%s filename=%r",
        str(file_id)[:8],
        row.filename,
    )

    broker.publish(str(file_id), {"status": "deleting", "filename": row.filename})
    return {"file_id": str(file_id), "status": "deleting"}


@router.get("/events")
async def files_events(_: dict = Depends(require_admin)):
    async def stream():
        async for event in broker.subscribe_all():
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/health")
def admin_files_health() -> dict:
    return {"status": "ok"}
