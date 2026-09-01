import streamlit as st
import tempfile
import os

from pipeline import analyze_meeting
from pipeline import plot_talk_time, plot_speaker_turns
from pipeline import plot_speaking_timeline
from pipeline import plot_dominance_breakdown


@st.cache_data(show_spinner=False)
def run_analysis(audio_path, generate_abs):
    return analyze_meeting(audio_path, generate_abs)

st.set_page_config( 
    page_title="Meeting Analyzer",
    layout="wide"
)

st.title("🎙️ Meeting Analyzer")
st.caption("Upload a meeting recording and get a structured insights")

# -----------------------------
# File uploader
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload meeting audio",
    type=["wav", "m4a"]
)
generate_abs = st.checkbox(
    "Generate Abstractive Summary (slower)",
    value=True
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.success("Audio uploaded successfully")

    # -----------------------------
    # Run pipeline
    # -----------------------------
    with st.spinner("Analyzing meeting, this may take a while"):
        result = run_analysis(audio_path,generate_abs)

    # Cleanup temp file
    os.remove(audio_path)

    # -----------------------------
    # Display results
    # -----------------------------
    st.header("Executive Summary")
    st.caption("TL;DR")
    st.write(result.get("summary", "No summary generated"))

    col1, col2 = st.columns(2)

    with col1:
        st.header("⭐ Key Moments")
        st.caption("Important notes/points of the meetings")
        ext_summary = result.get("important_sentences", [])
        for speaker, start, end, sentence in ext_summary:
            st.markdown(
                f"**{speaker} | {start:.2f}s → {end:.2f}s**  \n{sentence}"
            )


    with col2:
        st.header("✅ Action Items")
        st.caption("A to-do list")
        actions = result.get("action_items", [])
        if actions:
            for a in actions:
                st.markdown(f"- {a}")
        else:
            st.write("No action items detected.")
    
    st.header("📊 Participation Insights")
    st.caption("Based on speaker turn-taking and speaking duration")


    sentences = result.get("sentences", [])

    if sentences:
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Talk Time")
            fig1 = plot_talk_time(sentences)
            st.pyplot(fig1)

        with col4:
            st.subheader("Speaking Turns")
            fig2 = plot_speaker_turns(sentences)
            st.pyplot(fig2)
    else:
        st.write("No speaker analytics available.")


    st.header("🗣️ Speaking Timeline")
    st.caption("timeline of various speakers")

    sentences = result.get("sentences", [])

    if sentences:
        fig = plot_speaking_timeline(sentences)
        st.pyplot(fig)
    else:
        st.write("No speaking data available.")

    st.header("👑 Dominance Breakdown")

    sentences = result.get("sentences", [])

    if sentences:
        fig = plot_dominance_breakdown(sentences)
        st.pyplot(fig)

        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        st.write("No dominance data available.")
    
    st.header("Speaker Dominance")

    dominance = result["dominance"]

    for spk, info in dominance.items():
        st.markdown(
            f"""
            **{spk}**
            - Talk Time: {info['time']}s
            - Share: {info['ratio']}%
            - Level: **{info['label']}**
            """
        )

    
    st.download_button(
        "📄 Download Summary",
        data=result["summary"],
        file_name="meeting_summary.txt"
    )



