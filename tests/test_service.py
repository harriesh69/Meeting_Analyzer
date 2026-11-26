import io
import threading
import time

import pytest
from fastapi.testclient import TestClient

from service.api import create_app
from service.storage import RedisAudioStore
from service.worker import Worker


def fake_analyze(path, abstractive, progress):
    progress("transcribing")
    with open(path, "rb") as f:
        size = len(f.read())
    return {"summary": "ok" if abstractive else None, "bytes": size}


@pytest.fixture
def client(settings, r):
    return TestClient(create_app(settings, redis_client=r))


def upload(client, data=b"RIFFfakeaudio", name="m.wav", **form):
    return client.post("/v1/meetings", files={"file": (name, io.BytesIO(data), "audio/wav")},
                       data={k: str(v).lower() for k, v in form.items()})


def make_worker(settings, r, analyze=fake_analyze):
    return Worker(settings, redis_client=r, store=RedisAudioStore(r, 60), analyze=analyze)


def test_submit_then_worker_completes(client, settings, r):
    resp = upload(client)
    assert resp.status_code == 202
    job_id = resp.json()["id"]
    assert resp.headers["location"] == f"/v1/meetings/{job_id}"
    assert client.get(f"/v1/meetings/{job_id}").json()["status"] == "queued"

    w = make_worker(settings, r)
    claimed = w.jobs.claim(timeout=1)
    w.process(claimed)

    body = client.get(f"/v1/meetings/{job_id}").json()
    assert body["status"] == "done"
    assert body["result"] == {"summary": "ok", "bytes": len(b"RIFFfakeaudio")}
    assert body["attempts"] == 1
    assert r.get(f"ma:audio:{job_id}") is None  # audio cleaned up after success


def test_identical_upload_is_deduplicated(client):
    first = upload(client).json()
    again = upload(client)
    assert again.status_code == 200
    assert again.json()["id"] == first["id"] and again.json()["deduplicated"]
    # Different options are a different job.
    assert upload(client, abstractive=False).json()["id"] != first["id"]


def test_rejects_bad_uploads(client):
    assert upload(client, name="notes.txt").status_code == 415
    assert upload(client, data=b"").status_code == 400
    assert upload(client, data=b"x" * (1024 * 1024 + 1)).status_code == 413


def test_unknown_and_delete(client):
    assert client.get("/v1/meetings/nope").status_code == 404
    job_id = upload(client).json()["id"]
    assert client.delete(f"/v1/meetings/{job_id}").status_code == 204
    assert client.get(f"/v1/meetings/{job_id}").status_code == 404


def test_queue_position(client):
    ids = [upload(client, data=bytes([i]) * 10).json()["id"] for i in range(3)]
    positions = [client.get(f"/v1/meetings/{i}").json()["queue_position"] for i in ids]
    assert positions == [1, 2, 3]  # first submitted is served first


def test_transient_error_is_retried_then_fails(client, settings, r):
    job_id = upload(client).json()["id"]

    def boom(*_):
        raise RuntimeError("GPU out of memory")

    w = make_worker(settings, r, analyze=boom)
    for attempt in range(1, settings.max_attempts + 1):
        w.process(w.jobs.claim(timeout=1))
        status = client.get(f"/v1/meetings/{job_id}").json()
        assert status["attempts"] == attempt
    assert status["status"] == "failed"
    assert "GPU out of memory" in status["error"]
    assert r.llen("ma:queue") == 0 and r.llen("ma:processing") == 0


def test_bad_audio_fails_without_retry(client, settings, r):
    import subprocess
    job_id = upload(client).json()["id"]

    def bad(*_):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    w = make_worker(settings, r, analyze=bad)
    w.process(w.jobs.claim(timeout=1))
    body = client.get(f"/v1/meetings/{job_id}").json()
    assert body["status"] == "failed" and body["attempts"] == 1


def test_crashed_worker_job_is_requeued(client, settings, r):
    job_id = upload(client).json()["id"]
    dead = make_worker(settings, r)
    assert dead.jobs.claim(timeout=1) == job_id  # claims, then "crashes": no heartbeats

    rescuer = make_worker(settings, r)
    rescuer._last_reap = 0
    rescuer.reap()                      # first pass: only marks it suspect
    assert r.lrange("ma:processing", 0, -1) == [job_id.encode()]
    time.sleep(settings.heartbeat_ttl + 0.2)  # heartbeat expires
    rescuer._last_reap = 0
    rescuer.reap()                      # still missing a heartbeat: requeue
    rescuer._last_reap = 0
    rescuer.reap()
    assert client.get(f"/v1/meetings/{job_id}").json()["status"] == "queued"

    rescuer.process(rescuer.jobs.claim(timeout=1))
    body = client.get(f"/v1/meetings/{job_id}").json()
    assert body["status"] == "done" and body["attempts"] == 2


def test_live_job_is_not_reaped(client, settings, r):
    job_id = upload(client).json()["id"]
    gate = threading.Event()

    def slow(path, abstractive, progress):
        gate.wait(5)
        return {"ok": True}

    w = make_worker(settings, r, analyze=slow)
    t = threading.Thread(target=w.process, args=(w.jobs.claim(timeout=1),))
    t.start()
    other = make_worker(settings, r)
    for _ in range(3):  # longer than the heartbeat TTL, but heartbeats keep it alive
        other._last_reap = 0
        other.reap()
        time.sleep(settings.heartbeat_ttl / 2 + 0.2)
    assert client.get(f"/v1/meetings/{job_id}").json()["status"] == "processing"
    gate.set()
    t.join()
    assert client.get(f"/v1/meetings/{job_id}").json()["status"] == "done"


def test_parallel_workers_never_double_process(client, settings, r):
    ids = {upload(client, data=bytes([i]) * 20).json()["id"] for i in range(40)}
    seen, lock = [], threading.Lock()

    def record(path, abstractive, progress):
        time.sleep(0.005)
        return {}

    def run():
        w = make_worker(settings, r, analyze=record)
        while (job := w.jobs.claim(timeout=0.3)) is not None:
            with lock:
                seen.append(job)
            w.process(job)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == sorted(ids)  # every job exactly once


def test_health_ready_metrics(client):
    assert client.get("/healthz").text == "ok"
    assert client.get("/readyz").json() == {"ready": True}
    upload(client)
    m = client.get("/metrics").text
    assert "ma_jobs_submitted_total" in m and "ma_queue_depth 1.0" in m


def test_fake_pipeline_end_to_end(settings, r):
    from analyzer.pipeline import analyze_fake
    client = TestClient(create_app(settings, redis_client=r))
    job_id = upload(client).json()["id"]
    w = Worker(settings, redis_client=r, store=RedisAudioStore(r, 60),
               analyze=lambda p, a, prog: analyze_fake(p, a, prog, delay=0))
    w.process(w.jobs.claim(timeout=1))
    result = client.get(f"/v1/meetings/{job_id}").json()["result"]
    assert result["action_items"] == ["Priya will send the report tomorrow."]
    assert set(result["speakers"]) == {"SPEAKER_00", "SPEAKER_01"}
