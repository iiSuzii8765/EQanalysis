import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalysisSubmitResponse(BaseModel):
    session_id: uuid.UUID
    status: str
    message: str


class SessionStatusResponse(BaseModel):
    session_id: uuid.UUID
    status: str
    context: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


class SessionResultResponse(BaseModel):
    session_id: uuid.UUID
    status: str
    result: dict[str, Any] | None = None
    error_message: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
