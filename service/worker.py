"""Worker process: loads the models once, then takes jobs off the queue.

Run several replicas to process meetings in parallel; each one handles a
single job at a time because a job already uses every CPU core.

Failure handling:
  * A background thread refreshes the job's heartbeat. If the pod is killed
    or the node dies, the heartbeat expires and another worker's reaper
    requeues the job.
  * A bad audio file (ffmpeg cannot decode it) fails immediately; other
    errors are retried up to MAX_ATTEMPTS.
  * SIGTERM (Kubernetes scale-down or rollout) lets the current job finish
    before exiting.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
import threading
import time

import redis
from prometheus_client import Counter, Histogram, start_http_server

from analyzer import pipeline
from .config import Settings, settings as default_settings
from .jobs import DONE, JobStore
from .storage import make_store

log = logging.getLogger("worker")

COMPLETED = Counter("ma_jobs_completed_total", "Jobs finished", ["status"])
REQUEUED = Counter("ma_jobs_requeued_total", "Jobs requeued after a worker died")
DURATION = Histogram("ma_job_duration_seconds", "Time to analyze one meeting",
                     buckets=[5, 15, 30, 60, 120, 300, 600, 1200, 3600])


class Worker:
    def __init__(self, settings: Settings = default_settings, redis_client=None,
                 store=None, analyze=None):
        self.s = settings
        self.r = redis_client or redis.Redis.from_url(settings.redis_url)
        self.jobs = JobStore(self.r, settings)
        self.store = store or make_store(settings, self.r)
        self.analyze = analyze or self._default_analyze()
        self.stopping = threading.Event()
        self._suspects: set[str] = set()
        self._last_reap = 0.0

    def _default_analyze(self):
        if self.s.backend == "fake":
            return lambda path, abstractive, progress: pipeline.analyze_fake(
                path, abstractive, progress, delay=float(os.getenv("FAKE_DELAY", "0.5")))
        from analyzer.models import Models
        models = Models()
        log.info("loading models")
        models.warm_up()
        log.info("models loaded")
        return lambda path, abstractive, progress: pipeline.analyze(path, models, abstractive, progress)

    def run(self) -> None:
        log.info("worker started")
        while not self.stopping.is_set():
            self.reap()
            job_id = self.jobs.claim(timeout=2)
            if job_id:
                self.process(job_id)
        log.info("worker stopped")

    def reap(self) -> None:
        """Requeue jobs whose worker died. A job must be missing its heartbeat
        on two passes in a row, so a job claimed a moment ago (before its
        first heartbeat) is never mistaken for an orphan."""
        now = time.monotonic()
        if now - self._last_reap < self.s.heartbeat_ttl / 2:
            return
        self._last_reap = now
        orphans = set(self.jobs.find_orphans())
        for job_id in orphans & self._suspects:
            status = self.jobs.requeue_orphan(job_id)
            if status:
                REQUEUED.inc()
                log.warning("requeued orphaned job %s -> %s", job_id, status)
        self._suspects = orphans

    def process(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        log.info("job %s started (attempt %d)", job_id, job["attempts"])
        beating = threading.Event()

        def beat():
            while not beating.wait(self.s.heartbeat_ttl / 3):
                self.jobs.heartbeat(job_id)

        threading.Thread(target=beat, daemon=True).start()
        started = time.monotonic()
        try:
            data = self.store.get(job_id)
            suffix = os.path.splitext(job["filename"])[1]
            with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                f.write(data)
                f.flush()
                result = self.analyze(f.name, job["abstractive"],
                                      lambda stage: self.jobs.set_stage(job_id, stage))
            self.jobs.complete(job_id, result)
            self.store.delete(job_id)
            COMPLETED.labels(DONE).inc()
            DURATION.observe(time.monotonic() - started)
            log.info("job %s done in %.1fs", job_id, time.monotonic() - started)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Undecodable or missing audio: retrying cannot help.
            self.jobs.fail(job_id, f"could not read audio: {e}", retry=False)
            COMPLETED.labels("failed").inc()
            log.warning("job %s failed permanently: %s", job_id, e)
        except Exception as e:  # noqa: BLE001 - any other error may be transient
            status = self.jobs.fail(job_id, f"{type(e).__name__}: {e}", retry=True)
            COMPLETED.labels(status).inc()
            log.exception("job %s failed (%s)", job_id, status)
        finally:
            beating.set()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
    start_http_server(default_settings.metrics_port)
    worker = Worker()
    signal.signal(signal.SIGTERM, lambda *_: worker.stopping.set())
    signal.signal(signal.SIGINT, lambda *_: worker.stopping.set())
    worker.run()


if __name__ == "__main__":
    main()
