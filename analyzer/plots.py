"""Matplotlib charts for the Streamlit UI. Input is the API's `sentences` list."""

from collections import Counter, defaultdict

import matplotlib.pyplot as plt


def talk_time(sentences):
    totals = defaultdict(float)
    for s in sentences:
        totals[s["speaker"]] += s["end"] - s["start"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(totals.values(), labels=totals.keys(), autopct="%1.1f%%", startangle=140)
    ax.set_title("Talk time per speaker")
    return fig


def speaker_turns(sentences):
    turns = Counter(s["speaker"] for s in sentences)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(turns.keys(), turns.values())
    ax.set_title("Speaking turns per speaker")
    return fig


def timeline(sentences):
    order = list(dict.fromkeys(s["speaker"] for s in sentences))
    idx = {spk: i for i, spk in enumerate(order)}
    fig, ax = plt.subplots(figsize=(10, max(2, len(order) * 0.6)))
    for s in sentences:
        ax.barh(idx[s["speaker"]], s["end"] - s["start"], left=s["start"], height=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Speaking timeline")
    ax.invert_yaxis()
    return fig
