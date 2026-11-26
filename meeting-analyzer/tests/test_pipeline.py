"""Runs the real `analyze()` wiring with stand-in models shaped like
Whisper, pyannote, Sentence-BERT and the HF summarizer."""

import shutil
import subprocess
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
from analyzer import pipeline  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


class StubWhisper:
    def transcribe(self, path, word_timestamps):
        assert word_timestamps
        words = ["Welcome", "everyone", "to", "the", "planning", "meeting.", "Arjun", "will",
                 "send", "the", "budget", "draft", "by", "Friday.", "Sounds", "good,", "I",
                 "will", "review", "it", "on", "Monday", "morning."]
        return {"segments": [{"words": [{"start": i * 0.5, "end": i * 0.5 + 0.4, "word": " " + w}
                                        for i, w in enumerate(words)]}]}


class StubAnnotation:
    def itertracks(self, yield_label):
        yield SimpleNamespace(start=0.0, end=7.0), None, "SPEAKER_00"
        yield SimpleNamespace(start=7.0, end=12.0), None, "SPEAKER_01"


class StubEmbedder:
    def encode(self, sentences, convert_to_numpy, normalize_embeddings):
        v = np.array([[len(s), s.count(" ") + 1, 1.0] for s in sentences], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def stub_summarizer(text, **kw):
    assert kw["num_beams"] == 4
    return [{"summary_text": "Planning meeting about the budget draft."}]


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "in.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=1",
                    "-ac", "2", "-ar", "44100", str(path)], check=True)
    return str(path)


def test_analyze_wiring(wav):
    models = SimpleNamespace(whisper=StubWhisper(), diarization=lambda p: StubAnnotation(),
                             embedder=StubEmbedder(), summarizer=stub_summarizer)
    stages = []
    out = pipeline.analyze(wav, models, abstractive=True, progress=stages.append)
    assert stages == ["converting", "transcribing", "diarizing", "aligning", "summarizing", "analytics"]
    assert out["summary"] == "Planning meeting about the budget draft."
    assert out["action_items"] == ["Arjun will send the budget draft by Friday.",
                                   "Sounds good, I will review it on Monday morning."]
    assert [s["speaker"] for s in out["sentences"]] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    assert set(out["speakers"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert set(out["timings"]) == set(stages)


def test_bad_audio_raises_called_process_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    models = SimpleNamespace()
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.analyze(str(bad), models)
