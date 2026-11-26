"""Lazy loading of the ML models, independent of any UI framework.

The original loaded every model at import time through Streamlit's cache and
hard-coded the summarizer to GPU 0, which crashed on CPU-only machines.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    whisper_size: str = os.getenv("WHISPER_MODEL", "small")
    diarization_model: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    summary_model: str = os.getenv("SUMMARY_MODEL", "t5-base")
    hf_token: str | None = os.getenv("HF_TOKEN")


@dataclass
class Models:
    """Loads each model on first use and keeps it for the life of the process."""

    config: ModelConfig = field(default_factory=ModelConfig)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def _get(self, name, loader):
        with self._lock:
            if name not in self._cache:
                self._cache[name] = loader()
            return self._cache[name]

    @property
    def device(self) -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def whisper(self):
        def load():
            import whisper
            return whisper.load_model(self.config.whisper_size, device=self.device)
        return self._get("whisper", load)

    @property
    def diarization(self):
        def load():
            import torch
            from pyannote.audio import Pipeline
            try:  # pyannote >= 4 renamed the argument
                pipe = Pipeline.from_pretrained(self.config.diarization_model, token=self.config.hf_token)
            except TypeError:
                pipe = Pipeline.from_pretrained(self.config.diarization_model,
                                                use_auth_token=self.config.hf_token)
            if pipe is None:
                # from_pretrained returns None instead of raising for gated models.
                raise RuntimeError(
                    f"could not load {self.config.diarization_model}: set HF_TOKEN and "
                    "accept the model terms on huggingface.co")
            pipe.to(torch.device(self.device))
            return pipe
        return self._get("diarization", load)

    @property
    def embedder(self):
        def load():
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(self.config.embedding_model, device=self.device)
        return self._get("embedder", load)

    @property
    def summarizer(self):
        def load():
            from transformers import pipeline
            return pipeline("summarization", model=self.config.summary_model,
                            device=0 if self.device == "cuda" else -1)
        return self._get("summarizer", load)

    def warm_up(self) -> None:
        """Load everything now, so the first job does not pay the cost."""
        _ = self.whisper, self.diarization, self.embedder, self.summarizer
