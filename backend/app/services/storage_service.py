"""
Thin wrapper around an S3-compatible object store (AWS S3, MinIO, R2, etc.)
"""
import uuid
from pathlib import Path

import boto3
from botocore.client import Config

from app.core.config import settings


class StorageService:
    def __init__(self):
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
            config=Config(signature_version="s3v4"),
        )
        self.bucket = settings.S3_BUCKET_NAME

    def build_key(self, owner_id: uuid.UUID, original_filename: str) -> str:
        ext = Path(original_filename).suffix
        return f"uploads/{owner_id}/{uuid.uuid4()}{ext}"

    def upload_fileobj(self, fileobj, key: str, content_type: str | None = None) -> str:
        extra_args = {"ContentType": content_type} if content_type else {}
        self._client.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra_args)
        return key

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_in
        )

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)
