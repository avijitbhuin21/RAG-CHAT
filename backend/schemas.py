from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


# ---------- auth ----------
class AdminLoginRequest(BaseModel):
    username: str
    password: str


class UserSessionInfo(BaseModel):
    user_id: UUID
    email: str
    name: str | None = None
    picture_url: str | None = None


class AdminSessionInfo(BaseModel):
    type: Literal["admin"] = "admin"


class SessionResponse(BaseModel):
    # Both can be populated in the same browser when a user is also signed in
    # as admin — each lives in its own cookie (user_session / admin_session).
    user: UserSessionInfo | None = None
    admin: AdminSessionInfo | None = None


# ---------- admin files ----------
class FileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    filename: str
    content_hash: str
    size_bytes: int
    mime_type: str
    status: str
    stage_current: int
    stage_total: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class UploadResultItem(BaseModel):
    filename: str
    status: Literal["queued", "duplicate"]
    file_id: UUID | None = None
    existing_file_id: UUID | None = None
    existing_status: str | None = None


# ---------- chat ----------
class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    chat_id: UUID
    role: str
    content: str
    thinking: str | None
    citations: list | None
    tool_calls: list | None = None
    created_at: datetime


class SendMessageRequest(BaseModel):
    content: str
