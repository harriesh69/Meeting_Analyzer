"""Job records and a crash-safe work queue on Redis.

Keys (all prefixed `ma:`):
  job:{id}        hash   status, stage, attempts, timestamps, error
  result:{id}     string JSON result
  queue           list   job ids waiting for a worker
  processing      list   job ids a worker has claimed
  hb:{id}         string heartbeat, refreshed by the worker; expires if it dies
  dedupe:{hash}   string job id for identical audio + options

Claiming uses BLMOVE (queue -> processing) so a job is never in neither list:
if a worker crashes mid-job the id stays in `processing`, its heartbeat
expires, and a reaper puts it back on the queue (the "reliable queue"
pattern). After MAX_ATTEMPTS the job is marked failed instead of retrying
forever.
"""

from __future__ import annotations

import json
import time
import uuid

QUEUE = "ma:queue"
PROCESSING = "ma:processing"

QUEUED, PROCESSING_STATUS, DONE, FAILED = "queued", "processing", "done", "failed"


def _job_key(job_id: str) -> str:
    return f"ma:job:{job_id}"


class JobStore:
    def __init__(self, redis, settings):
        self.r = redis
        self.s = settings

    # ---- API side -------------------------------------------------------

    def create(self, *, filename: str, size: int, digest: str, abstractive: bool) -> tuple[str, bool]:
        """Create a job, or return the existing one for identical input.

        Returns (job_id, created). Dedupe uses SET NX, which is atomic, so two
        simultaneous uploads of the same file still produce one job.
        """
        dedupe_key = f"ma:dedupe:{digest}:{int(abstractive)}"
        job_id = uuid.uuid4().hex
        if not self.r.set(dedupe_key, job_id, nx=True, ex=self.s.job_ttl):
            existing = self.r.get(dedupe_key)
            existing = existing.decode() if isinstance(existing, bytes) else existing
            job = self.get(existing)
            if job and job["status"] != FAILED:
                return existing, False
            self.r.set(dedupe_key, job_id, ex=self.s.job_ttl)  # retry a failed one
        now = time.time()
        self.r.hset(_job_key(job_id), mapping={
            "id": job_id, "status": QUEUED, "stage": "", "attempts": 0,
            "filename": filename, "size": size, "sha256": digest,
            "abstractive": int(abstractive), "error": "",
            "created_at": now, "updated_at": now,
        })
        self.r.expire(_job_key(job_id), self.s.job_ttl)
        return job_id, True

    def enqueue(self, job_id: str) -> None:
        self.r.lpush(QUEUE, job_id)

    def get(self, job_id: str) -> dict | None:
        raw = self.r.hgetall(_job_key(job_id))
        if not raw:
            return None
        job = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
               for k, v in raw.items()}
        for f in ("attempts", "size"):
            job[f] = int(job[f])
        for f in ("created_at", "updated_at"):
            job[f] = float(job[f])
        for f in ("started_at", "finished_at"):
            if f in job:
                job[f] = float(job[f])
        job["abstractive"] = job["abstractive"] == "1"
        if job["status"] == QUEUED:
            job["queue_position"] = self.queue_position(job_id)
        return job

    def result(self, job_id: str) -> dict | None:
        raw = self.r.get(f"ma:result:{job_id}")
        return json.loads(raw) if raw else None

    def delete(self, job_id: str) -> bool:
        removed = self.r.delete(_job_key(job_id), f"ma:result:{job_id}")
        self.r.lrem(QUEUE, 0, job_id)
        return removed > 0

    def queue_position(self, job_id: str) -> int | None:
        # Workers take from the right end, so position counts from the right.
        ids = [i.decode() if isinstance(i, bytes) else i for i in self.r.lrange(QUEUE, 0, -1)]
        if job_id not in ids:
            return None
        return len(ids) - ids.index(job_id)

    def depth(self) -> tuple[int, int]:
        return self.r.llen(QUEUE), self.r.llen(PROCESSING)

    # ---- worker side ----------------------------------------------------

    def claim(self, timeout: float = 2.0) -> str | None:
        job_id = self.r.blmove(QUEUE, PROCESSING, timeout, "RIGHT", "LEFT")
        if job_id is None:
            return None
        job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
        self.heartbeat(job_id)
        if not self.r.exists(_job_key(job_id)):  # deleted while queued
            self.r.lrem(PROCESSING, 0, job_id)
            return None
        now = time.time()
        self.r.hincrby(_job_key(job_id), "attempts", 1)
        self.r.hset(_job_key(job_id), mapping={
            "status": PROCESSING_STATUS, "started_at": now, "updated_at": now, "error": ""})
        return job_id

    def heartbeat(self, job_id: str) -> None:
        self.r.set(f"ma:hb:{job_id}", 1, ex=self.s.heartbeat_ttl)

    def set_stage(self, job_id: str, stage: str) -> None:
        self.r.hset(_job_key(job_id), mapping={"stage": stage, "updated_at": time.time()})

    def complete(self, job_id: str, result: dict) -> None:
        pipe = self.r.pipeline()
        pipe.set(f"ma:result:{job_id}", json.dumps(result), ex=self.s.job_ttl)
        pipe.hset(_job_key(job_id), mapping={
            "status": DONE, "stage": "", "finished_at": time.time(), "updated_at": time.time()})
        pipe.lrem(PROCESSING, 0, job_id)
        pipe.delete(f"ma:hb:{job_id}")
        pipe.execute()

    def fail(self, job_id: str, error: str, *, retry: bool, _claimed: bool = True) -> str:
        """Record a failure. Retries go back on the queue until attempts run out."""
        job = self.get(job_id)
        attempts = job["attempts"] if job else self.s.max_attempts
        pipe = self.r.pipeline()
        if _claimed:
            pipe.lrem(PROCESSING, 0, job_id)
        pipe.delete(f"ma:hb:{job_id}")
        if retry and attempts < self.s.max_attempts:
            status = QUEUED
            pipe.hset(_job_key(job_id), mapping={"status": QUEUED, "stage": "",
                                                 "error": error, "updated_at": time.time()})
            pipe.lpush(QUEUE, job_id)
        else:
            status = FAILED
            pipe.hset(_job_key(job_id), mapping={"status": FAILED, "stage": "", "error": error,
                                                 "finished_at": time.time(), "updated_at": time.time()})
        pipe.execute()
        return status

    def find_orphans(self) -> list[str]:
        """Claimed jobs whose worker stopped sending heartbeats."""
        ids = [i.decode() if isinstance(i, bytes) else i for i in self.r.lrange(PROCESSING, 0, -1)]
        return [i for i in ids if not self.r.exists(f"ma:hb:{i}")]

    def requeue_orphan(self, job_id: str) -> str | None:
        """Requeue one orphan. LREM tells us whether *this* reaper removed it,
        so two reapers racing on the same job cannot both requeue it."""
        if self.r.lrem(PROCESSING, 0, job_id) == 0:
            return None
        if not self.r.exists(_job_key(job_id)):
            return None  # job was deleted
        return self.fail(job_id, "worker stopped responding", retry=True, _claimed=False)
