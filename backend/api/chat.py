import asyncio
import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..db import SessionLocal, get_db
from ..models import Chat, Message
from ..schemas import ChatOut, MessageOut, SendMessageRequest
from ..security import require_user, user_id_from_claims
from ..services.chat_service import stream_chat

router = APIRouter(prefix="/chat", tags=["chat"])
log = logging.getLogger("api.chat")

# Event types where we don't want to log every delta (would be very noisy).
_QUIET_EVENT_TYPES = {"thinking_delta", "content_delta"}

# Module-load banner so we can confirm uvicorn reload picked up the new file.
log.info("api.chat router module loaded")
print(">>> api/chat.py module loaded", flush=True)


@router.get("/chats", response_model=list[ChatOut])
def list_chats(claims: dict = Depends(require_user), db: Session = Depends(get_db)):
    uid = user_id_from_claims(claims)
    return (
        db.query(Chat)
        .filter(Chat.user_id == uid)
        .order_by(Chat.updated_at.desc())
        .all()
    )


@router.post("/chats", response_model=ChatOut)
def create_chat(claims: dict = Depends(require_user), db: Session = Depends(get_db)):
    uid = user_id_from_claims(claims)
    chat = Chat(user_id=uid, title="New chat")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: UUID, claims: dict = Depends(require_user), db: Session = Depends(get_db)
):
    uid = user_id_from_claims(claims)
    chat = db.query(Chat).filter_by(id=chat_id, user_id=uid).one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="chat not found")
    db.delete(chat)
    db.commit()


@router.get("/chats/{chat_id}/messages", response_model=list[MessageOut])
def list_messages(
    chat_id: UUID, claims: dict = Depends(require_user), db: Session = Depends(get_db)
):
    uid = user_id_from_claims(claims)
    chat = db.query(Chat).filter_by(id=chat_id, user_id=uid).one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="chat not found")
    return (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
        .all()
    )


@router.post("/chats/{chat_id}/messages")
async def send_message(
    chat_id: UUID,
    body: SendMessageRequest,
    claims: dict = Depends(require_user),
):
    uid = user_id_from_claims(claims)
    user_email = claims.get("email")
    tag = str(chat_id)[:8]

    # print() in addition to log so we see the entry even if logging is misbehaving.
    print(
        f"\n>>> [api {tag} {user_email or '?'}] POST /messages "
        f"({len(body.content)} chars): {body.content[:200]!r}",
        flush=True,
    )
    log.info(
        "[api %s %s] POST /messages | content=%r (%d chars)",
        tag,
        user_email or "?",
        body.content[:200],
        len(body.content),
    )

    def _authz() -> bool:
        db = SessionLocal()
        try:
            return (
                db.query(Chat).filter_by(id=chat_id, user_id=uid).one_or_none()
                is not None
            )
        finally:
            db.close()

    if not await asyncio.to_thread(_authz):
        raise HTTPException(status_code=404, detail="chat not found")

    async def stream():
        thinking_chars = 0
        content_chars = 0
        try:
            async for event in stream_chat(chat_id, body.content, user_email):
                etype = event.get("type")
                if etype == "thinking_delta":
                    thinking_chars += len(event.get("content", ""))
                elif etype == "content_delta":
                    content_chars += len(event.get("content", ""))
                if etype not in _QUIET_EVENT_TYPES:
                    log.info(
                        "[api %s] SSE -> %s %s",
                        tag,
                        etype,
                        {k: v for k, v in event.items() if k != "type"},
                    )
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            # StreamingResponse swallows exceptions silently after headers are
            # sent; log + surface to client as a final SSE event so we can see
            # what went wrong on both sides.
            log.exception("[api %s] stream generator crashed", tag)
            yield f"data: {json.dumps({'type': 'error', 'message': 'stream crashed (see server logs)'})}\n\n"
        finally:
            log.info(
                "[api %s] SSE stream closed | thinking=%d chars, content=%d chars",
                tag,
                thinking_chars,
                content_chars,
            )

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/health")
def chat_health() -> dict:
    return {"status": "ok"}
