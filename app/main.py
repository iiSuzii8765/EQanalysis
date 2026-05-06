import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Base, engine, get_db
from app.schemas import AnalysisSubmitResponse, HealthResponse, SessionResultResponse, SessionStatusResponse
from app.services import create_session, get_session
from app.tasks import process_analysis_session

app = FastAPI(title="EQ Philosophy Backend", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.report_path.mkdir(parents=True, exist_ok=True)
    settings.openface_output_path.mkdir(parents=True, exist_ok=True)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post(
    "/api/v1/sessions/analyse",
    response_model=AnalysisSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def analyse_session(
    context: str = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AnalysisSubmitResponse:
    if not _is_supported_video(video.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file extension. Allowed: .mp4, .mov, .webm",
        )

    session_id = uuid.uuid4()
    extension = Path(video.filename or "").suffix.lower()
    target_file = settings.upload_path / f"{session_id}{extension}"

    with target_file.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    session_row = create_session(db, context=context, video_path=str(target_file))
    process_analysis_session.delay(str(session_row.id), context, str(target_file))

    return AnalysisSubmitResponse(
        session_id=session_row.id,
        status=session_row.status,
        message="Session queued for asynchronous processing.",
    )


@app.get("/api/v1/sessions/{session_id}/status", response_model=SessionStatusResponse)
def get_session_status(session_id: uuid.UUID, db: Session = Depends(get_db)) -> SessionStatusResponse:
    row = get_session(db, session_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return SessionStatusResponse(
        session_id=row.id,
        status=row.status,
        context=row.context,
        created_at=row.created_at,
        updated_at=row.updated_at,
        error_message=row.error_message,
    )


@app.get("/api/v1/sessions/{session_id}/result", response_model=SessionResultResponse)
def get_session_result(session_id: uuid.UUID, db: Session = Depends(get_db)) -> SessionResultResponse:
    row = get_session(db, session_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if row.status not in {"completed", "failed"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session not finished. Current status: {row.status}",
        )
    return SessionResultResponse(
        session_id=row.id,
        status=row.status,
        result=row.result_json,
        error_message=row.error_message,
    )


def _is_supported_video(filename: str | None) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in {".mp4", ".mov", ".webm"}
