import asyncio
import uuid
from datetime import datetime, timezone

from app.database.sync_session import get_sync_db
from app.models.job import DetectionJob, DetectionResult, JobStatus
from app.services.inference_client import InferenceClient
from app.services.storage_service import StorageService
from app.workers.celery_app import celery_app


@celery_app.task(name="detection.run_detection_job", bind=True, max_retries=2, default_retry_delay=15)
def run_detection_job(self, job_id: str) -> None:
    """
    Entry point invoked by the API layer after a file is uploaded.
    1. Loads the job
    2. Generates a presigned URL for the AI inference service to fetch the media
    3. Calls the inference service
    4. Persists the result
    """
    db = get_sync_db()
    try:
        job = db.get(DetectionJob, uuid.UUID(job_id))
        if job is None:
            return

        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        storage = StorageService()
        media_url = storage.generate_presigned_url(job.storage_key)

        # InferenceClient is async (httpx); bridge into this sync Celery task.
        client = InferenceClient()
        result_payload = asyncio.run(client.run_detection(media_url, job.media_type.value))

        result = DetectionResult(
            job_id=job.id,
            is_fake=result_payload["is_fake"],
            confidence=result_payload["confidence"],
            branch_scores=result_payload.get("branch_scores"),
            explainability_uri=result_payload.get("explainability_uri"),
            model_version=result_payload.get("model_version"),
            processing_time_ms=result_payload.get("processing_time_ms"),
        )
        db.add(result)
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        job = db.get(DetectionJob, uuid.UUID(job_id))
        if job:
            job.status = JobStatus.FAILED
            job.error_message = str(exc)[:2000]
            db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()
