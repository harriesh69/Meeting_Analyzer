"""Where uploaded audio waits between the API and the worker.

API and worker run in different pods, so a local temp file is not enough.
Two backends:
  * redis: audio stored as a key with a TTL. No extra infrastructure; fine
    for uploads up to tens of MB.
  * s3: any S3-compatible store (AWS S3, Google Cloud Storage's S3 API,
    MinIO). The right choice for large files or many workers.
"""

from __future__ import annotations

from typing import Protocol


class AudioStore(Protocol):
    def put(self, job_id: str, data: bytes) -> None: ...
    def get(self, job_id: str) -> bytes: ...
    def delete(self, job_id: str) -> None: ...


class RedisAudioStore:
    def __init__(self, redis, ttl: int):
        self.r, self.ttl = redis, ttl

    def _key(self, job_id: str) -> str:
        return f"ma:audio:{job_id}"

    def put(self, job_id, data):
        self.r.set(self._key(job_id), data, ex=self.ttl)

    def get(self, job_id):
        data = self.r.get(self._key(job_id))
        if data is None:
            raise FileNotFoundError(f"audio for job {job_id} has expired or was deleted")
        return data

    def delete(self, job_id):
        self.r.delete(self._key(job_id))


class S3AudioStore:
    def __init__(self, bucket: str, endpoint_url: str | None = None, client=None):
        if client is None:
            import boto3
            client = boto3.client("s3", endpoint_url=endpoint_url)
        self.s3, self.bucket = client, bucket

    def put(self, job_id, data):
        self.s3.put_object(Bucket=self.bucket, Key=f"audio/{job_id}", Body=data)

    def get(self, job_id):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=f"audio/{job_id}")["Body"].read()
        except self.s3.exceptions.NoSuchKey as e:
            raise FileNotFoundError(job_id) from e

    def delete(self, job_id):
        self.s3.delete_object(Bucket=self.bucket, Key=f"audio/{job_id}")


def make_store(settings, redis) -> AudioStore:
    if settings.storage == "s3":
        return S3AudioStore(settings.s3_bucket, settings.s3_endpoint)
    return RedisAudioStore(redis, settings.job_ttl)
