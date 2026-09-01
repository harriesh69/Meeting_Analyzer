# Meeting Analyzer

Takes a meeting recording and returns a speaker-attributed transcript, a summary, extracted action items, and participation statistics. Runs locally as a Streamlit app — no API calls, all models run on your machine.

Built with Whisper for transcription, pyannote.audio for speaker diarization, Sentence-BERT for extractive ranking, and t5-base for the final summary.

## What it produces

- **Executive summary** — a short paragraph covering what the meeting was about
- **Key moments** — the five most representative sentences, each with speaker and timestamp
- **Action items** — sentences that look like commitments, pulled out by rule
- **Participation stats** — talk time, turn counts, a speaking timeline, and a per-speaker dominance rating

## How it works

The pipeline runs in six stages.

**1. Transcription.** Whisper (`small`) transcribes the audio with `word_timestamps=True`, giving start and end times for every word.

**2. Diarization.** pyannote produces speaker turns as `(start, end, speaker)` tuples. Each Whisper word is assigned to whichever turn overlaps it most in time. Words with no overlap are marked `Unknown`.

**3. Merging.** An `Unknown` word is reassigned to its neighbours' speaker when the words before and after it agree. Consecutive words from the same speaker are then collapsed into blocks, and each block is split into sentences on `.`, `!`, and `?`. Since Whisper gives word times but not sentence times, each sentence's duration is estimated by distributing the block's duration proportionally to character length.

**4. Summarization.** Two stages, extractive then abstractive:

- Every sentence is embedded with `all-MiniLM-L6-v2` and scored by cosine similarity against the centroid of all embeddings, so sentences closest to the meeting's overall topic rank highest. Fragments are penalized — a sentence without terminal punctuation is scaled by 0.3, one under six words by 0.5. The top five survive.
- Those five are concatenated and passed to `t5-base` with a prompt asking for decisions, introductions, and issues. Beam search with `num_beams=4` and a repetition penalty of 1.2.

Running t5 over the extractive output rather than the full transcript keeps the input inside t5's context window and filters out filler before it reaches the model.

**5. Action items.** No model here. A sentence qualifies if it contains a modal (`will`, `need to`, `should`, `must`, `please`, `plan to`, `follow up`, `deadline`) and an action verb (`send`, `email`, `share`, `decide`, `schedule`, `review`, `assign`, `prepare`, `update`). Sentences containing a hedge (`think`, `maybe`, `might`, `could`, `discuss`, `consider`) are dropped, since those are usually proposals rather than commitments.

**6. Analytics.** Talk time per speaker, turn counts, a Gantt-style speaking timeline, and a dominance score. Dominance is a speaker's share of total talk time, bucketed at 40% (Dominant), 25% (High), and 10% (Moderate).

## Setup

Requires Python 3.10 and ffmpeg on your PATH.

```bash
git clone https://github.com/harriesh69/Meeting_Analyzer
cd meeting-analyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install ffmpeg separately if you don't have it — `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS, or `winget install ffmpeg` on Windows.

### Hugging Face access

The pyannote diarization model is gated. Without this step the pipeline fails with `'NoneType' object is not callable`, because `Pipeline.from_pretrained` returns `None` rather than raising.

1. Accept the terms at [hf.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) and [hf.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation).
2. Create a read token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Log in: `huggingface-cli login`

### GPU

CUDA is used automatically if available. On CPU the t5 summary and Whisper transcription are considerably slower — the "Generate Abstractive Summary" checkbox in the app turns off the slowest stage.

## Running it

```bash
streamlit run app.py
```

Upload a `.wav` or `.m4a` file. Models download on first run (roughly 2 GB across Whisper, pyannote, MiniLM, and t5).

Mono 16 kHz WAV works best. To convert:

```bash
ffmpeg -i input.m4a -ac 1 -ar 16000 output.wav
```

## Repository layout

```
app.py                          Streamlit UI
pipeline.py                     Transcription, diarization, summarization, analytics
requirements.txt
notebooks/
  meeting_analyser.ipynb        Original prototype, kept for reference
```

`analyze_meeting(audio_path, generate_abstractive=True)` in `pipeline.py` is the entry point if you want to use it without the UI. It returns a dict with `sentences`, `important_sentences`, `summary`, `action_items`, and `dominance`.

## Models

| Stage | Model | Size |
|---|---|---|
| Transcription | `openai/whisper-small` | 244M |
| Diarization | `pyannote/speaker-diarization` | — |
| Sentence embedding | `all-MiniLM-L6-v2` | 22M |
| Summarization | `t5-base` | 220M |

## Limitations

Speakers are labelled `SPEAKER_00`, `SPEAKER_01` and so on — there's no name resolution. Sentence-level timestamps are estimated from character counts, not measured, so they drift within a long speaker block. Action item detection is keyword-based and will miss anything phrased indirectly. t5-base is small enough that summaries of technical discussion can be generic. Overlapping speech is a known weak point for the word-to-speaker overlap assignment.

## License

MIT. See [LICENSE](LICENSE).
