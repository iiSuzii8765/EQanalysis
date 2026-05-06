import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import AnalysisSession


def create_session(db: Session, *, context: str, video_path: str) -> AnalysisSession:
    row = AnalysisSession(context=context, status="queued", video_path=video_path)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_session(db: Session, session_id: uuid.UUID) -> AnalysisSession | None:
    return db.get(AnalysisSession, session_id)


def set_status(db: Session, session_id: uuid.UUID, status: str, error_message: str | None = None) -> AnalysisSession:
    row = db.get(AnalysisSession, session_id)
    if row is None:
        raise ValueError(f"Session not found: {session_id}")
    row.status = status
    row.error_message = error_message
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return row


def set_result(db: Session, session_id: uuid.UUID, result: dict) -> AnalysisSession:
    row = db.get(AnalysisSession, session_id)
    if row is None:
        raise ValueError(f"Session not found: {session_id}")
    row.status = "completed"
    row.result_json = result
    row.error_message = None
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return row
