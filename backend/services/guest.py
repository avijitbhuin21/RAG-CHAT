from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from ..config import settings
from ..models import GuestUsage


def _window_start() -> datetime:
    """Return the start of the rolling guest quota window."""
    return datetime.now(timezone.utc) - timedelta(hours=settings.GUEST_WINDOW_HOURS)


def usage(db: Session, ip: str) -> dict:
    """Summarise how many guest messages this IP has used in the current window."""
    since = _window_start()
    used, oldest = (
        db.query(func.count(GuestUsage.id), func.min(GuestUsage.created_at))
        .filter(GuestUsage.ip == ip, GuestUsage.created_at >= since)
        .one()
    )
    limit = settings.GUEST_MESSAGE_LIMIT
    resets_at = (
        oldest + timedelta(hours=settings.GUEST_WINDOW_HOURS) if used and oldest else None
    )
    return {
        "used": int(used),
        "limit": limit,
        "remaining": max(0, limit - int(used)),
        "resets_at": resets_at,
    }


def consume(db: Session, ip: str, guest_id: UUID) -> tuple[bool, dict]:
    """Atomically record one guest message for this IP if quota remains."""
    db.execute(text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": f"guest:{ip}"})
    current = usage(db, ip)
    if current["remaining"] <= 0:
        db.rollback()
        return False, current
    db.add(GuestUsage(ip=ip, guest_id=guest_id))
    db.commit()
    return True, usage(db, ip)
