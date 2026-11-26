import pytest

from service.storage import RedisAudioStore, S3AudioStore


def test_redis_store_roundtrip(r):
    store = RedisAudioStore(r, ttl=60)
    store.put("j1", b"\x00audio")
    assert store.get("j1") == b"\x00audio"
    assert 0 < r.ttl("ma:audio:j1") <= 60
    store.delete("j1")
    with pytest.raises(FileNotFoundError):
        store.get("j1")


def test_s3_store_roundtrip():
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="audio")
        store = S3AudioStore("audio", client=client)
        store.put("j1", b"\x00audio")
        assert store.get("j1") == b"\x00audio"
        store.delete("j1")
        with pytest.raises(FileNotFoundError):
            store.get("j1")
