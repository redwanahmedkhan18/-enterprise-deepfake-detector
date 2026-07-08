import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MediaType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"


class DetectionJob(Base):
    __tablename__ = "detection_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1000), nullable=False)  # object storage path
    file_size_bytes: Mapped[int] = mapped_column(nullable=True)

    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    owner: Mapped["User"] = relationship(back_populates="detection_jobs")  # noqa: F821
    result: Mapped["DetectionResult"] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("detection_jobs.id"), unique=True)

    is_fake: Mapped[bool] = mapped_column(nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 - 1.0

    # Per-branch scores from the fusion model (spatial, temporal, frequency, physiological)
    branch_scores: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Paths/URIs to explainability artifacts (heatmaps, attention maps)
    explainability_uri: Mapped[str] = mapped_column(String(1000), nullable=True)

    model_version: Mapped[str] = mapped_column(String(50), nullable=True)
    processing_time_ms: Mapped[int] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped["DetectionJob"] = relationship(back_populates="result")
