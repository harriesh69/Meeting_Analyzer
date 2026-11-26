"""End-to-end meeting analysis. Returns plain JSON-serializable data."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Callable

from . import text

ProgressFn = Callable[[str], None]


def to_wav_16k(src: str, dst: str) -> None:
    """Convert any ffmpeg-readable audio to 16 kHz mono WAV (what both models expect)."""
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
                    "-ac", "1", "-ar", "16000", dst], check=True)


def extractive_summary(sentences, embedder, k: int = 5):
    """Rank sentences by cosine similarity to the meeting's centroid embedding."""
    import numpy as np
    if not sentences:
        return []
    emb = embedder.encode([s.text for s in sentences], convert_to_numpy=True,
                          normalize_embeddings=True)
    centroid = emb.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    scores = emb @ centroid
    top = sorted(np.argsort(scores)[-k:])
    return [sentences[i] for i in top]


def abstractive_summary(extract: str, summarizer) -> str:
    prompt = ("summarize the meeting, including key decisions, "
              "introductions, and issues discussed:\n\n")
    return summarizer(prompt + extract, max_new_tokens=180, min_new_tokens=100,
                      do_sample=False, num_beams=4, repetition_penalty=1.2,
                      truncation=True)[0]["summary_text"]


def analyze(audio_path: str, models, abstractive: bool = True,
            progress: ProgressFn | None = None) -> dict:
    progress = progress or (lambda stage: None)
    timings = {}

    def stage(name):
        progress(name)
        timings[name] = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        stage("converting")
        to_wav_16k(audio_path, wav)

        stage("transcribing")
        result = models.whisper.transcribe(wav, word_timestamps=True)
        words = [(w["start"], w["end"], w["word"])
                 for seg in result["segments"] for w in seg.get("words", [])]

        stage("diarizing")
        diar = models.diarization(wav)
        annotation = getattr(diar, "speaker_diarization", diar)  # pyannote 4 wraps it
        turns = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]

    stage("aligning")
    sentences = text.split_sentences(
        text.merge_turns(text.fill_unknown(text.assign_speakers(words, turns))))

    stage("summarizing")
    candidates = [s for s in sentences if text.is_summary_candidate(s.text)]
    key = extractive_summary(candidates, models.embedder)
    summary = (abstractive_summary(" ".join(s.text for s in key), models.summarizer)
               if abstractive and key else None)

    stage("analytics")
    out = build_result(sentences, key, summary)
    names = list(timings)
    now = time.perf_counter()
    out["timings"] = {n: round((timings[names[i + 1]] if i + 1 < len(names) else now) - timings[n], 2)
                      for i, n in enumerate(names)}
    return out


def build_result(sentences, key_moments, summary) -> dict:
    as_dict = lambda s: {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}  # noqa: E731
    return {
        "summary": summary,
        "key_moments": [as_dict(s) for s in key_moments],
        "action_items": text.action_items(s.text for s in sentences),
        "speakers": text.dominance(sentences),
        "sentences": [as_dict(s) for s in sentences],
        "duration_seconds": round(max((s.end for s in sentences), default=0.0), 2),
    }


def analyze_fake(audio_path: str, abstractive: bool = True,
                 progress: ProgressFn | None = None, delay: float = 0.5) -> dict:
    """Stand-in for `analyze` used when ANALYZER_BACKEND=fake: no model
    downloads, no GPU. Lets the API, worker, Docker images and Kubernetes
    manifests be tested end to end in CI in seconds."""
    progress = progress or (lambda stage: None)
    for name in ("converting", "transcribing", "diarizing", "aligning", "summarizing"):
        progress(name)
        time.sleep(delay / 5)
    words = [(0.0, 0.5, "Hi"), (0.5, 1.0, "team."), (1.0, 1.6, "Priya"), (1.6, 2.0, "will"),
             (2.0, 2.4, "send"), (2.4, 2.9, "the"), (2.9, 3.3, "report"), (3.3, 3.8, "tomorrow."),
             (4.0, 4.4, "Thanks,"), (4.4, 4.8, "I'll"), (4.8, 5.2, "review"), (5.2, 5.6, "it.")]
    turns = [(0.0, 3.9, "SPEAKER_00"), (3.9, 6.0, "SPEAKER_01")]
    sentences = text.split_sentences(text.merge_turns(text.fill_unknown(
        text.assign_speakers(words, turns))))
    return build_result(sentences, sentences[:2],
                        "Fake summary for testing." if abstractive else None)
