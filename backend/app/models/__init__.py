"""
Import every ORM model here so SQLAlchemy's mapper registry can resolve
string-based relationship() references (e.g. "DetectionJob" inside user.py)
regardless of which module gets imported first.
"""
from app.models.user import User, APIKey  # noqa: F401
from app.models.job import DetectionJob, DetectionResult, JobStatus, MediaType  # noqa: F401

__all__ = ["User", "APIKey", "DetectionJob", "DetectionResult", "JobStatus", "MediaType"]
