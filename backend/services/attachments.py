"""Chat attachments: validation, text extraction and S3 storage.

Objects live under chat-attachments/{user_id}/{attachment_id}/ so ownership is
enforced by the key itself: a user can only ever resolve ids inside their own
prefix.
"""
import base64
import json
import logging
import re
from uuid import UUID, uuid4

from ..config import settings
from . import s3

log = logging.getLogger("attachments")

PREFIX = "chat-attachments"

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_DOC_BYTES = 10 * 1024 * 1024
MAX_IMAGES_PER_MESSAGE = 4
MAX_DOCS_PER_MESSAGE = 3
MAX_DOC_CHARS = 150_000

IMAGE_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}

TEXT_EXTS = {
    "txt", "md", "markdown", "csv", "tsv", "json", "xml", "html", "htm",
    "log", "yaml", "yml", "ini", "rtf", "sql",
}
DOC_EXTS = {"pdf", "docx", "xlsx"} | TEXT_EXTS


class AttachmentError(ValueError):
    """Raised for user-facing validation failures (bad type, too large, unreadable)."""


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _safe_name(filename: str) -> str:
    """Strip directories and unsafe characters from an uploaded filename."""
    base = re.split(r"[\\/]", filename or "")[-1].strip() or "file"
    base = re.sub(r"[^\w.\- ()]+", "_", base)
    return base[:180]


def classify(filename: str) -> str | None:
    """Return 'image', 'document' or None (unsupported) from the filename extension."""
    ext = _ext(filename)
    if ext in IMAGE_MIME:
        return "image"
    if ext in DOC_EXTS:
        return "document"
    return None


def _key(user_id: UUID, att_id: str, name: str) -> str:
    return f"{PREFIX}/{user_id}/{att_id}/{name}"


def _extract_pdf(data: bytes) -> str:
    """Extract text from every page of a PDF with pymupdf."""
    import fitz

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # noqa: BLE001
        raise AttachmentError("This PDF could not be opened.") from e
    try:
        parts = []
        for i, page in enumerate(doc, 1):
            txt = (page.get_text("text") or "").strip()
            if txt:
                parts.append(f"--- Page {i} ---\n{txt}")
        return "\n\n".join(parts)
    finally:
        doc.close()


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from a supported document, raising AttachmentError if none is found."""
    from .ingest import _extract_docx_markdown, _extract_xlsx_markdown

    ext = _ext(filename)
    tag = f"[attach {filename}]"
    try:
        if ext == "pdf":
            text = _extract_pdf(data)
        elif ext == "docx":
            text = _extract_docx_markdown(data, tag)
        elif ext == "xlsx":
            text = _extract_xlsx_markdown(data, tag)
        else:
            text = data.decode("utf-8", errors="replace")
    except AttachmentError:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("%s extraction failed", tag)
        raise AttachmentError("Couldn't read text from this file.") from e
    text = text.strip()
    if not text:
        raise AttachmentError(
            "No readable text found in this file (scanned PDFs and image-only documents aren't supported)."
        )
    return text


def store(user_id: UUID, filename: str, data: bytes) -> dict:
    """Validate, extract and persist one upload; returns its public metadata."""
    name = _safe_name(filename)
    kind = classify(name)
    if kind is None:
        raise AttachmentError(
            "Unsupported file type. Attach images (PNG, JPG, WEBP, GIF) or documents "
            "(PDF, DOCX, XLSX, TXT, MD, CSV, JSON…)."
        )
    size = len(data)
    if size == 0:
        raise AttachmentError("This file is empty.")
    limit = MAX_IMAGE_BYTES if kind == "image" else MAX_DOC_BYTES
    if size > limit:
        raise AttachmentError(f"File is too large (max {limit // (1024 * 1024)} MB for {kind}s).")

    att_id = uuid4().hex
    ext = _ext(name)
    mime = IMAGE_MIME.get(ext) if kind == "image" else _doc_mime(ext)
    meta = {
        "id": att_id,
        "kind": kind,
        "filename": name,
        "mime": mime,
        "size": size,
    }
    client = s3.client()
    if kind == "document":
        text = extract_text(name, data)
        truncated = len(text) > MAX_DOC_CHARS
        text = text[:MAX_DOC_CHARS]
        meta["chars"] = len(text)
        meta["truncated"] = truncated
        client.put_object(
            Bucket=settings.S3_BUCKET,
            Key=_key(user_id, att_id, "text.txt"),
            Body=text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
    client.put_object(
        Bucket=settings.S3_BUCKET,
        Key=_key(user_id, att_id, "original"),
        Body=data,
        ContentType=mime,
    )
    client.put_object(
        Bucket=settings.S3_BUCKET,
        Key=_key(user_id, att_id, "meta.json"),
        Body=json.dumps(meta).encode("utf-8"),
        ContentType="application/json",
    )
    log.info("stored %s attachment %s (%s, %d bytes) for %s", kind, att_id, name, size, user_id)
    return meta


def _doc_mime(ext: str) -> str:
    return {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "json": "application/json",
        "html": "text/html",
        "htm": "text/html",
        "md": "text/markdown",
        "markdown": "text/markdown",
    }.get(ext, "text/plain")


def _valid_id(att_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", att_id or ""))


def _get(user_id: UUID, att_id: str, name: str) -> bytes:
    obj = s3.client().get_object(Bucket=settings.S3_BUCKET, Key=_key(user_id, att_id, name))
    return obj["Body"].read()


def load_meta(user_id: UUID, att_id: str) -> dict:
    """Read an attachment's metadata from the caller's own prefix; KeyError if missing."""
    if not _valid_id(att_id):
        raise KeyError(att_id)
    try:
        return json.loads(_get(user_id, att_id, "meta.json"))
    except Exception as e:  # noqa: BLE001
        raise KeyError(att_id) from e


def load_original(user_id: UUID, att_id: str) -> tuple[bytes, dict]:
    """Return the original bytes and metadata for one of the caller's attachments."""
    meta = load_meta(user_id, att_id)
    return _get(user_id, att_id, "original"), meta


def load_text(user_id: UUID, att_id: str) -> str:
    """Return the extracted text stored for a document attachment."""
    return _get(user_id, att_id, "text.txt").decode("utf-8", errors="replace")


def image_data_url(user_id: UUID, meta: dict) -> str:
    """Encode an image attachment as a base64 data URL for multimodal model input."""
    data = _get(user_id, meta["id"], "original")
    return f"data:{meta['mime']};base64,{base64.b64encode(data).decode('ascii')}"


def validate_counts(metas: list[dict]) -> None:
    """Enforce per-message image/document count limits."""
    images = sum(1 for m in metas if m.get("kind") == "image")
    docs = sum(1 for m in metas if m.get("kind") == "document")
    if images > MAX_IMAGES_PER_MESSAGE:
        raise AttachmentError(f"You can attach at most {MAX_IMAGES_PER_MESSAGE} images per message.")
    if docs > MAX_DOCS_PER_MESSAGE:
        raise AttachmentError(f"You can attach at most {MAX_DOCS_PER_MESSAGE} documents per message.")
