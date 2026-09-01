import torch
import whisper
from pyannote.audio import Pipeline

import os
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from collections import defaultdict


@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="t5-base",
        device=0
    )

@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

summarizer = load_summarizer()
sbert_model = load_sbert()

DEVICE = 0 if torch.cuda.is_available() else -1
whisper_model = whisper.load_model("small")



DFPipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
)
#pretty stuff
def format_time(seconds):
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    if h > 0:
        return f"{h:d}:{m:02d}:{s:05.2f}"
    else:
        return f"{m:d}:{s:05.2f}"

def transcribe_diarrize(resampled_file):
    whisper_result = whisper_model.transcribe(resampled_file, word_timestamps=True)
    diarization = DFPipeline(resampled_file)
    segments = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]

    final_transcript = []

    for segment in whisper_result["segments"]:
        for word in segment["words"]:
            w_start = word["start"]
            w_end = word["end"]
            text = word["word"]

            speaker = "Unknown"
            max_overlap = 0.0

            for s_start, s_end, s_label in segments:
                overlap = max(0, min(w_end, s_end) - max(w_start, s_start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = s_label

            final_transcript.append([speaker, w_start, w_end, text.strip()])
    return final_transcript

def merger(final_transcript):
    merged1 = []

    current_speaker = None
    current_text = []
    start_time = None
    end_time = None

    for i in range(1,len(final_transcript)):
        if final_transcript[i][0] == "Unknown":
            if final_transcript[i-1][0] == final_transcript[i+1][0]:
                final_transcript[i][0] = final_transcript[i-1][0]


    for speaker, w_start, w_end, word in final_transcript:

        if speaker != current_speaker:
            if current_text:
                merged1.append([
                    current_speaker,
                    start_time,
                    end_time,
                    " ".join(current_text)
                ])

            current_speaker = speaker
            current_text = [word]
            start_time = w_start
            end_time = w_end
        else:
            current_text.append(word)
            end_time = w_end

    # Append last segment
    if current_text:
        merged1.append([
            current_speaker,
            start_time,
            end_time,
            " ".join(current_text)
        ])
    return(merged1)

def split_into_sentences(transcript):
    """
    Split each (speaker, start, end, text) into sentence-level segments.
    Timestamps are distributed proportionally by character length.
    """
    sentence_regex = re.compile(r'(?<=[.!?])\s+')
    output = []

    for speaker, start, end, text in transcript:
        text = text.strip()
        if not text:
            continue

        sentences = sentence_regex.split(text)

        if len(sentences) == 1:
            output.append((speaker, start, end, text))
            continue

        total_chars = sum(len(s) for s in sentences)
        duration = end - start
        current_time = start

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            proportion = len(sentence) / total_chars
            sent_duration = proportion * duration
            sent_end = current_time + sent_duration

            output.append((
                speaker,
                round(current_time, 2),
                round(sent_end, 2),
                sentence
            ))

            current_time = sent_end

    return output
    
def extractive_sbert(sentence_tuples, num_sentences=5):
    """
    sentence_tuples: list of (speaker, start, end, sentence)
    returns: top-ranked sentence tuples
    """
    sentences = [s for _, _, _, s in sentence_tuples]

    embeddings = sbert_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    centroid = embeddings.mean(axis=0)

    scores = cosine_similarity([centroid], embeddings)[0]

    # Penalize junk
    for i, sent in enumerate(sentences):
        if not sent.strip().endswith((".", "?", "!")):
            scores[i] *= 0.3
        if len(sent.split()) < 6:
            scores[i] *= 0.5

    top_idx = np.argsort(scores)[-num_sentences:]
    top_idx = sorted(top_idx)

    return [sentence_tuples[i] for i in top_idx]

def is_complete_sentence(sentence):
    sentence = sentence.strip()
    return (
        len(sentence.split()) >= 6 and
        sentence[-1] in ".?!"
    )

def is_valid_sentence(sentence):
    bad_patterns = [
        "we've got used to the",
        "you know ever since",
        "sort of",
        "kind of",
    ]

    if any(p in sentence.lower() for p in bad_patterns):
        return False

    return True

def chunk_sentences(sentence_tuples, sentences_per_chunk=6):
    """
    Groups sentence tuples into batches for abstractive summarization.
    """
    chunks = []
    current = []

    for tup in sentence_tuples:
        current.append(tup)

        if len(current) >= sentences_per_chunk:
            chunks.append(current)
            current = []

    if current:
        chunks.append(current)

    return chunks

def sentences_to_text(sentence_chunk):
    return " ".join(sentence for _, _, _, sentence in sentence_chunk)

def abstractive_summary(text):
    prompt = (
        "summarize the meeting, including key decisions, "
        "introductions, and issues discussed:\n\n"
    )
    return summarizer(
        prompt + text,
        max_new_tokens=180,
        min_new_tokens=100,
        do_sample=False,
        num_beams=4,
        repetition_penalty=1.2,
        truncation=True
    )[0]["summary_text"]

def detect_action_items_rule(sentences):
    patterns = [
        r"\bwill\b", r"\bneed to\b", r"\bshould\b", r"\bmust\b",
        r"\bplease\b", r"\bplan to\b", r"\bfollow up\b", r"\bdeadline\b"
    ]

    action_verbs = [
        "send", "email", "share", "decide", "schedule",
        "review", "assign", "prepare", "update"
    ]

    hedges = ["think", "maybe", "might", "could", "discuss", "consider"]

    action_items = []

    for s in sentences:
        s_lower = s.lower()

        if any(h in s_lower for h in hedges):
            continue

        has_modal = any(re.search(p, s_lower) for p in patterns)
        has_action = any(v in s_lower for v in action_verbs)

        if has_modal and has_action:
            action_items.append(s)

    return action_items



def plot_talk_time(sentences):
    talk_time = defaultdict(float)

    for speaker, start, end, _ in sentences:
        talk_time[speaker] += (end - start)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        talk_time.values(),
        labels=talk_time.keys(),
        autopct="%1.1f%%",
        startangle=140
    )
    ax.set_title("Talk Time per Speaker")

    return fig


def plot_speaker_turns(sentences):
    turns = Counter(speaker for speaker, _, _, _ in sentences)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(turns.keys(), turns.values())
    ax.set_title("Speaking Turns per Speaker")
    ax.set_xlabel("Speaker")
    ax.set_ylabel("Turns")

    return fig



def compute_dominance(final_transcript):
    """
    final_transcript: list of (speaker, start, end, text)
    returns: dict {speaker: {"time": x, "ratio": y, "label": z}}
    """
    talk_time = defaultdict(float)

    for spk, start, end, _ in final_transcript:
        talk_time[spk] += (end - start)

    total_time = sum(talk_time.values())
    dominance = {}

    for spk, t in talk_time.items():
        ratio = t / total_time if total_time > 0 else 0

        if ratio > 0.40:
            label = "Dominant"
        elif ratio > 0.25:
            label = "High"
        elif ratio > 0.10:
            label = "Moderate"
        else:
            label = "Low"

        dominance[spk] = {
            "time": round(t, 1),
            "ratio": round(ratio * 100, 1),
            "label": label
        }

    return dominance

import matplotlib.pyplot as plt

def plot_speaking_timeline(sentences):
    """
    sentences: list of (speaker, start, end, text)
    returns: matplotlib figure
    """
    # preserve speaker order of appearance
    speakers = []
    for spk, _, _, _ in sentences:
        if spk not in speakers:
            speakers.append(spk)

    speaker_idx = {spk: i for i, spk in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(10, max(2, len(speakers) * 0.6)))

    for spk, start, end, _ in sentences:
        ax.barh(
            speaker_idx[spk],
            end - start,
            left=start,
            height=0.4
        )

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Speaking Timeline")

    ax.invert_yaxis()  # top speaker at top

    return fig

import matplotlib.pyplot as plt

def plot_dominance_breakdown(transcript):
    from collections import defaultdict, Counter

    talk_time = defaultdict(float)
    turns = Counter()

    total_time = 0
    total_turns = 0

    for spk, start, end, _ in transcript:
        talk_time[spk] += (end - start)
        turns[spk] += 1
        total_time += (end - start)
        total_turns += 1

    speakers = list(talk_time.keys())
    talk_ratio = [talk_time[s]/total_time for s in speakers]
    turn_ratio = [turns[s]/total_turns for s in speakers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(speakers, talk_ratio, label="Talk Time")
    ax.bar(speakers, turn_ratio, bottom=talk_ratio, label="Turns")

    ax.set_ylabel("Contribution Ratio")
    ax.set_title("Speaker Dominance Breakdown")
    ax.legend()

    return fig




def analyze_meeting(audio,generate_abstractive=True):
    final_transcript = transcribe_diarrize(audio)
    merged1 = merger(final_transcript)
    final_transcript = split_into_sentences(merged1)

    clean_sentences = [
        (s, st, et, sent)
        for s, st, et, sent in final_transcript
        if is_complete_sentence(sent) and is_valid_sentence(sent)
    ]

    ext_summary = extractive_sbert(
        clean_sentences,
        num_sentences=5
    )

    extractive_text = " ".join(
        sentence for _, _, _, sentence in ext_summary
    )

    if generate_abstractive:
        summary = abstractive_summary(extractive_text)
    else:
        summary = None
    sentences = [t[3] for t in final_transcript]
    action_items = detect_action_items_rule(sentences)
    dominance = compute_dominance(final_transcript)

    return {
        "sentences": final_transcript,
        "important_sentences": ext_summary,
        "summary": summary,
        "action_items": action_items,
        "dominance": dominance
    }
