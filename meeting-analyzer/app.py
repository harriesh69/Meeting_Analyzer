"""Streamlit front end. It no longer runs the models itself: it uploads audio
to the API, polls the job, and renders the result."""

import os
import time

import requests
import streamlit as st

from analyzer import plots
from analyzer.text import format_time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Meeting Analyzer", layout="wide")
st.title("Meeting Analyzer")
st.caption("Upload a meeting recording to get a summary, action items and participation stats.")

uploaded = st.file_uploader("Meeting audio", type=["wav", "m4a", "mp3", "flac", "ogg", "webm", "mp4"])
abstractive = st.checkbox("Generate abstractive summary (slower)", value=True)

if uploaded and st.button("Analyze"):
    resp = requests.post(f"{API_URL}/v1/meetings",
                         files={"file": (uploaded.name, uploaded.getvalue())},
                         data={"abstractive": str(abstractive).lower()}, timeout=120)
    if not resp.ok:
        st.error(resp.json().get("detail", resp.text))
        st.stop()
    st.session_state["job_id"] = resp.json()["id"]

job_id = st.session_state.get("job_id")
if job_id:
    status_box = st.empty()
    while True:
        job = requests.get(f"{API_URL}/v1/meetings/{job_id}", timeout=10).json()
        if job["status"] == "done":
            status_box.success(f"Done in {job.get('processing_seconds', 0):.0f}s")
            break
        if job["status"] == "failed":
            status_box.error(f"Analysis failed: {job.get('error')}")
            st.stop()
        where = (f"position {job['queue_position']} in queue" if job["status"] == "queued"
                 else job.get("stage") or "starting")
        status_box.info(f"{job['status'].capitalize()}: {where}")
        time.sleep(2)

    result = job["result"]
    st.header("Summary")
    st.write(result.get("summary") or "No abstractive summary requested.")

    left, right = st.columns(2)
    with left:
        st.header("Key moments")
        for s in result["key_moments"]:
            st.markdown(f"**{s['speaker']} | {format_time(s['start'])}**  \n{s['text']}")
    with right:
        st.header("Action items")
        for item in result["action_items"] or ["None detected."]:
            st.markdown(f"- {item}")

    sentences = result["sentences"]
    if sentences:
        st.header("Participation")
        c1, c2 = st.columns(2)
        c1.pyplot(plots.talk_time(sentences))
        c2.pyplot(plots.speaker_turns(sentences))
        st.pyplot(plots.timeline(sentences))
        st.table([{"speaker": k, **v} for k, v in result["speakers"].items()])

    st.download_button("Download summary", data=result.get("summary") or "",
                       file_name="meeting_summary.txt")
