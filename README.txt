Meeting Analyzer

AI-powered tool that analyzes meeting recordings and generates structured insights like transcripts, summaries, action items, and speaker analytics.

Features
**Speaker Diarization** – Identifies “who spoke when” using Pyannote.  
**Transcription** – Converts speech to text with Whisper (OpenAI).  
**Summarization** – Extracts key highlights with HuggingFace Transformers.  
**Action Item Extraction** – Detects tasks/commitments (“will”, “must”, “shall”).  
**Speaker Analytics** – Talk-time distribution per speaker.  
**Q&A on Meetings** – Ask natural language questions about the meeting using embeddings.

## Tech Stack
- **Speech Models**: Whisper (ASR), Pyannote (Diarization)  
- **NLP Models**: HuggingFace Transformers, Sentence-BERT  
- **Frameworks**: PyTorch, Torchaudio, Transformers  