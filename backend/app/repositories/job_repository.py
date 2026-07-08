import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.job import DetectionJob, DetectionResult, JobStatus


class JobRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, job: DetectionJob) -> DetectionJob:
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        return job

    async def get_by_id(self, job_id: uuid.UUID, owner_id: uuid.UUID | None = None) -> DetectionJob | None:
        stmt = select(DetectionJob).options(selectinload(DetectionJob.result)).where(DetectionJob.id == job_id)
        if owner_id is not None:
            stmt = stmt.where(DetectionJob.owner_id == owner_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_user(self, owner_id: uuid.UUID, limit: int = 50, offset: int = 0) -> list[DetectionJob]:
        stmt = (
            select(DetectionJob)
            .options(selectinload(DetectionJob.result))
            .where(DetectionJob.owner_id == owner_id)
            .order_by(DetectionJob.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self, job: DetectionJob, status: JobStatus, error_message: str | None = None
    ) -> DetectionJob:
        job.status = status
        if error_message:
            job.error_message = error_message
        await self.db.commit()
        await self.db.refresh(job)
        return job

    async def attach_result(self, job: DetectionJob, result: DetectionResult) -> DetectionJob:
        self.db.add(result)
        job.status = JobStatus.COMPLETED
        await self.db.commit()
        await self.db.refresh(job)
        return job
