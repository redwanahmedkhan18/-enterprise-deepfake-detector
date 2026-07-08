import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.job import JobStatus, MediaType


class DetectionJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    media_type: MediaType
    original_filename: str
    status: JobStatus
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class DetectionResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    is_fake: bool
    confidence: float
    branch_scores: dict | None
    explainability_uri: str | None
    model_version: str | None
    processing_time_ms: int | None


class DetectionJobWithResult(DetectionJobRead):
    result: DetectionResultRead | None = None
