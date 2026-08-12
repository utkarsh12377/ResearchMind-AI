"""Celery application for long-running ingestion and indexing work."""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "researchmind",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Fetch one task at a time: ingestion jobs are long and vary wildly in
    # duration, so prefetching would leave work queued behind a slow paper
    # while other workers sit idle.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=settings.celery_task_always_eager,
)
