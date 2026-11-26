"""HTTP API.

  POST   /v1/meetings        upload audio -> 202 with a job id
  GET    /v1/meetings/{id}   job status, and the result once done
  DELETE /v1/meetings/{id}   remove a job and its result
  GET    /healthz            liveness: the process is up
  GET    /readyz             readiness: Redis is reachable
  GET    /metrics            Prometheus metrics

Analysis takes minutes, so uploads return immediately and the work happens
in a separate worker process (see worker.py). Clients poll for the result.
"""

from __future__ import annotations

import hashlib
import os

import redis
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .config import Settings, settings as default_settings
from .jobs import DONE, JobStore
from .storage import make_store

ALLOWED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".webm", ".mp4"}
CHUNK = 1 << 20

SUBMITTED = Counter("ma_jobs_submitted_total", "Jobs created")
DEDUPED = Counter("ma_jobs_deduplicated_total", "Uploads answered by an existing job")
UPLOAD_BYTES = Histogram("ma_upload_bytes", "Size of uploaded audio",
                         buckets=[1e5, 1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8])
QUEUE_DEPTH = Gauge("ma_queue_depth", "Jobs waiting for a worker")
IN_PROGRESS = Gauge("ma_jobs_processing", "Jobs claimed by a worker")


def create_app(settings: Settings = default_settings, redis_client=None, store=None) -> FastAPI:
    r = redis_client or redis.Redis.from_url(settings.redis_url)
    jobs = JobStore(r, settings)
    audio = store or make_store(settings, r)
    app = FastAPI(title="Meeting Analyzer API", version="1.0")
    max_bytes = settings.max_upload_mb * 1024 * 1024

    @app.post("/v1/meetings", status_code=status.HTTP_202_ACCEPTED)
    async def submit(response: Response, file: UploadFile = File(...),
                     abstractive: bool = Form(True)):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"unsupported file type {ext or '(none)'}; "
                                     f"use one of {sorted(ALLOWED_EXTENSIONS)}")
        # Read in chunks so an oversized upload is rejected without
        # buffering all of it, and hash as we go for deduplication.
        digest, parts, size = hashlib.sha256(), [], 0
        while chunk := await file.read(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
            digest.update(chunk)
            parts.append(chunk)
        if size == 0:
            raise HTTPException(400, "empty file")
        UPLOAD_BYTES.observe(size)

        job_id, created = jobs.create(filename=file.filename, size=size,
                                      digest=digest.hexdigest(), abstractive=abstractive)
        if created:
            audio.put(job_id, b"".join(parts))
            jobs.enqueue(job_id)
            SUBMITTED.inc()
        else:
            DEDUPED.inc()
            response.status_code = status.HTTP_200_OK
        response.headers["Location"] = f"/v1/meetings/{job_id}"
        return {"id": job_id, "deduplicated": not created, **_public(jobs.get(job_id))}

    @app.get("/v1/meetings/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        body = _public(job)
        if job["status"] == DONE:
            body["result"] = jobs.result(job_id)
        return body

    @app.delete("/v1/meetings/{job_id}", status_code=204)
    def delete_job(job_id: str):
        if not jobs.delete(job_id):
            raise HTTPException(404, "job not found")
        audio.delete(job_id)
        return Response(status_code=204)

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz():
        return "ok"

    @app.get("/readyz")
    def readyz():
        try:
            r.ping()
        except redis.RedisError as e:
            return JSONResponse({"ready": False, "error": str(e)}, status_code=503)
        return {"ready": True}

    @app.get("/metrics")
    def metrics(request: Request):
        try:
            waiting, processing = jobs.depth()
            QUEUE_DEPTH.set(waiting)
            IN_PROGRESS.set(processing)
        except redis.RedisError:
            pass
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _public(job: dict) -> dict:
    keep = ("id", "status", "stage", "attempts", "filename", "size", "abstractive",
            "error", "created_at", "started_at", "finished_at", "queue_position")
    out = {k: job[k] for k in keep if k in job and job[k] not in ("", None)}
    if "started_at" in job and "finished_at" in job:
        out["processing_seconds"] = round(job["finished_at"] - job["started_at"], 2)
    return out


def app_factory() -> FastAPI:  # used by uvicorn --factory
    return create_app()
