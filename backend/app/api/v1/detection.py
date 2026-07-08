import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.config import settings
from app.database.session import get_db
from app.exceptions.custom_exceptions import (
    FileTooLargeError,
    JobNotFoundError,
    UnsupportedMediaTypeError,
)
from app.models.job import DetectionJob, MediaType
from app.models.user import User
from app.repositories.job_repository import JobRepository
from app.schemas.detection import DetectionJobRead, DetectionJobWithResult
from app.services.storage_service import StorageService
from app.workers.detection_tasks import run_detection_job

router = APIRouter(prefix="/detection", tags=["Detection"])


def _classify_media_type(filename: str) -> MediaType:
    ext = Path(filename).suffix.lower()
    if ext in settings.ALLOWED_VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    if ext in settings.ALLOWED_IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    raise UnsupportedMediaTypeError(f"File extension '{ext}' is not supported.")


@router.post("/jobs", response_model=DetectionJobRead, status_code=201)
async def create_detection_job(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    media_type = _classify_media_type(file.filename)

    # Validate size (UploadFile doesn't give size upfront in all cases; check via content-length or stream).
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise FileTooLargeError(f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB limit.")
    await file.seek(0)

    storage = StorageService()
    key = storage.build_key(current_user.id, file.filename)
    storage.upload_fileobj(file.file, key, content_type=file.content_type)

    job = DetectionJob(
        owner_id=current_user.id,
        media_type=media_type,
        original_filename=file.filename,
        storage_key=key,
        file_size_bytes=len(contents),
    )
    job_repo = JobRepository(db)
    job = await job_repo.create(job)

    # Hand off to the background worker; API responds immediately.
    run_detection_job.delay(str(job.id))

    return job


@router.get("/jobs", response_model=list[DetectionJobRead])
async def list_detection_jobs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    job_repo = JobRepository(db)
    return await job_repo.list_for_user(current_user.id, limit=limit, offset=offset)


@router.get("/jobs/{job_id}", response_model=DetectionJobWithResult)
async def get_detection_job(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job_repo = JobRepository(db)
    job = await job_repo.get_by_id(job_id, owner_id=current_user.id)
    if job is None:
        raise JobNotFoundError(f"Detection job {job_id} not found.")
    return job
