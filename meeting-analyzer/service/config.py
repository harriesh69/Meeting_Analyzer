"""Settings from environment variables (12-factor style, so the same image
runs locally, in docker compose and in Kubernetes)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    backend: str = os.getenv("ANALYZER_BACKEND", "real")       # "real" or "fake"
    storage: str = os.getenv("STORAGE_BACKEND", "redis")       # "redis" or "s3"
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_endpoint: str | None = os.getenv("S3_ENDPOINT_URL") or None
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 50)
    max_attempts: int = _int("MAX_ATTEMPTS", 3)
    heartbeat_ttl: int = _int("HEARTBEAT_TTL_SECONDS", 30)
    job_ttl: int = _int("JOB_TTL_SECONDS", 7 * 24 * 3600)
    metrics_port: int = _int("METRICS_PORT", 9100)


settings = Settings()
