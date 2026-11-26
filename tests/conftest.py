import os
import shutil
import socket
import subprocess
import time

import pytest
import redis

from service.config import Settings


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def redis_url():
    """A real Redis: REDIS_URL if set (CI service container), else a local
    redis-server started for the test session."""
    if os.getenv("REDIS_URL"):
        yield os.environ["REDIS_URL"]
        return
    if not shutil.which("redis-server"):
        pytest.skip("redis-server not installed and REDIS_URL not set")
    port = _free_port()
    proc = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
                            stdout=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    for _ in range(50):
        try:
            redis.Redis.from_url(url).ping()
            break
        except redis.ConnectionError:
            time.sleep(0.05)
    yield url
    proc.terminate()
    proc.wait()


@pytest.fixture
def r(redis_url):
    client = redis.Redis.from_url(redis_url)
    client.flushdb()
    return client


@pytest.fixture
def settings(redis_url):
    return Settings(redis_url=redis_url, backend="fake", max_upload_mb=1,
                    heartbeat_ttl=2, max_attempts=3)
