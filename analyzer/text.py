"""Pure-Python pipeline stages. No ML imports, so these are fast to test."""

from __future__ import annotations

import bisect
import re
from collections import Counter, defaultdict
from typing import Iterable, NamedTuple

UNKNOWN = "Unknown"


class Word(NamedTuple):
    speaker: str
    start: float
    end: float
    text: str


class Sentence(NamedTuple):
    speaker: str
    start: float
    end: float
    text: str


def assign_speakers(words: Iterable[tuple[float, float, str]],
                    turns: list[tuple[float, float, str]]) -> list[Word]:
    """Give each Whisper word the speaker whose diarization turn overlaps it most.

    The original version compared every word with every turn (O(W x T)).
    Turns are sorted by start time here, so for each word only the turns that
    start before the word ends are checked, found with a binary search.
    A word with no overlapping turn gets UNKNOWN.
    """
    turns = sorted(turns)
    starts = [t[0] for t in turns]
    # Longest turn bounds how far back an overlapping turn can start.
    max_len = max((e - s for s, e, _ in turns), default=0.0)
    out: list[Word] = []
    for w_start, w_end, text in words:
        speaker, best = UNKNOWN, 0.0
        lo = bisect.bisect_left(starts, w_start - max_len)
        hi = bisect.bisect_right(starts, w_end)
        for s_start, s_end, label in turns[lo:hi]:
            overlap = min(w_end, s_end) - max(w_start, s_start)
            if overlap > best:
                best, speaker = overlap, label
        out.append(Word(speaker, w_start, w_end, text.strip()))
    return out


def fill_unknown(words: list[Word]) -> list[Word]:
    """Give an UNKNOWN word its neighbours' speaker when both neighbours agree.

    Fixes an IndexError in the original, which read words[i + 1] for the
    last word.
    """
    out = list(words)
    for i in range(1, len(out) - 1):
        if out[i].speaker == UNKNOWN and out[i - 1].speaker == out[i + 1].speaker:
            out[i] = out[i]._replace(speaker=out[i - 1].speaker)
    return out


def merge_turns(words: list[Word]) -> list[Sentence]:
    """Collapse consecutive words from the same speaker into one block."""
    blocks: list[Sentence] = []
    for w in words:
        if blocks and blocks[-1].speaker == w.speaker:
            b = blocks[-1]
            blocks[-1] = Sentence(b.speaker, b.start, w.end, f"{b.text} {w.text}")
        else:
            blocks.append(Sentence(w.speaker, w.start, w.end, w.text))
    return blocks


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(blocks: list[Sentence]) -> list[Sentence]:
    """Split speaker blocks into sentences.

    Whisper gives word times, not sentence times, so each sentence's span is
    estimated by sharing the block's duration in proportion to character count.
    """
    out: list[Sentence] = []
    for speaker, start, end, text in blocks:
        parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
        if not parts:
            continue
        if len(parts) == 1:
            out.append(Sentence(speaker, start, end, parts[0]))
            continue
        total = sum(len(p) for p in parts)
        t = start
        for p in parts:
            t_end = t + (end - start) * len(p) / total
            out.append(Sentence(speaker, round(t, 2), round(t_end, 2), p))
            t = t_end
    return out


_FILLER = ("we've got used to the", "you know ever since", "sort of", "kind of")


def is_summary_candidate(text: str) -> bool:
    """Complete, non-filler sentences are eligible for the extractive summary."""
    text = text.strip()
    if len(text.split()) < 6 or not text or text[-1] not in ".?!":
        return False
    lower = text.lower()
    return not any(f in lower for f in _FILLER)


_MODALS = [re.compile(p) for p in (
    r"\bwill\b", r"\bneed to\b", r"\bshould\b", r"\bmust\b",
    r"\bplease\b", r"\bplan to\b", r"\bfollow up\b", r"\bdeadline\b")]
_ACTION_VERBS = ("send", "email", "share", "decide", "schedule",
                 "review", "assign", "prepare", "update")
_HEDGES = re.compile(r"\b(think|maybe|might|could|discuss|consider)\b")


def action_items(sentences: Iterable[str]) -> list[str]:
    """Sentences with a modal and an action verb, minus hedged proposals.

    Hedges are matched as whole words: the original substring check dropped
    sentences like "Please review the rethinking doc" because of "think".
    """
    items = []
    for s in sentences:
        lower = s.lower()
        if _HEDGES.search(lower):
            continue
        if any(p.search(lower) for p in _MODALS) and any(v in lower for v in _ACTION_VERBS):
            items.append(s)
    return items


def dominance(sentences: list[Sentence]) -> dict[str, dict]:
    """Talk time, share and turn count per speaker, with a coarse label."""
    talk = defaultdict(float)
    turns = Counter()
    for spk, start, end, _ in sentences:
        talk[spk] += end - start
        turns[spk] += 1
    total = sum(talk.values())
    out = {}
    for spk, t in talk.items():
        ratio = t / total if total > 0 else 0.0
        label = ("Dominant" if ratio > 0.40 else "High" if ratio > 0.25
                 else "Moderate" if ratio > 0.10 else "Low")
        out[spk] = {"time": round(t, 1), "ratio": round(ratio * 100, 1),
                    "turns": turns[spk], "label": label}
    return out


def format_time(seconds: float) -> str:
    seconds = float(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}" if h else f"{int(m)}:{s:05.2f}"
