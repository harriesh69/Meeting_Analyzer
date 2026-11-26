"""Meeting analysis pipeline: transcription, diarization, summarization, analytics.

`analyzer.text` holds the pure-Python stages (no ML dependencies), so they can
be unit-tested and reused anywhere. `analyzer.models` loads the ML models
lazily, and `analyzer.pipeline` wires the stages together.
"""
