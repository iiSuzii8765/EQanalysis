import uuid

from app.celery_app import celery_app
from app.db import SessionLocal
from app.pipeline import run_week3_pipeline
from app.services import set_result, set_status


@celery_app.task(name="app.tasks.process_analysis_session")
def process_analysis_session(session_id: str, context: str, video_path: str) -> dict:
    db = SessionLocal()
    session_uuid = uuid.UUID(session_id)
    try:
        set_status(db, session_uuid, "processing")
        result = run_week3_pipeline(video_path=video_path, context=context, session_id=session_id)
        set_result(db, session_uuid, result)
        return {"session_id": session_id, "status": "completed"}
    except Exception as exc:  # noqa: BLE001
        set_status(db, session_uuid, "failed", error_message=str(exc))
        raise
    finally:
        db.close()
