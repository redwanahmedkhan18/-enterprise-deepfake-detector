"""
Celery workers run in a synchronous context, so they get their own sync
SQLAlchemy engine/session rather than sharing the async engine used by the
FastAPI app. Both point at the same database.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# Convert the async DSN (postgresql+asyncpg://...) to a sync one (postgresql+psycopg2://...)
_sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")

sync_engine = create_engine(_sync_url, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(bind=sync_engine, autoflush=False, expire_on_commit=False)


def get_sync_db() -> Session:
    return SyncSessionLocal()
