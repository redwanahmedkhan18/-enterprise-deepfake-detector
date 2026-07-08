from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "deepfake_platform",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.detection_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.INFERENCE_TIMEOUT_SECONDS + 30,
    worker_prefetch_multiplier=1,
)
