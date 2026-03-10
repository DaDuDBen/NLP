"""Streamlit dashboard for multimodal tone detection."""

from __future__ import annotations

import tempfile
import time
from queue import Queue

import cv2
import pandas as pd
import streamlit as st

from pipeline.audio import LiveAudioCapture, WhisperTranscriber, iter_audio_segments_from_file
from pipeline.fusion import ToneFusionEngine
from pipeline.nlp import NLPToneAnalyzer
from pipeline.video import (
    FaceResult,
    LiveVideoCapture,
    VideoEmotionAnalyzer,
    draw_face_overlay,
    iter_video_frames_from_file,
)

st.set_page_config(page_title="Multimodal Tone Detector", layout="wide")
st.title("🎭 Multimodal Tone Detector")

COLOR_MAP = {
    "Sarcastic": "orange",
    "Sincere/Positive": "green",
    "Joking/Playful": "gold",
    "Neutral/Sincere": "gray",
}


def tone_badge(label: str) -> str:
    color = COLOR_MAP.get(label, "gray")
    return f"<h2 style='color:{color};'>Current Tone: {label}</h2>"


def run_file_processing(video_path: str) -> pd.DataFrame:
    """Process uploaded file and return timeline DataFrame."""
    transcriber = WhisperTranscriber("base")
    nlp_analyzer = NLPToneAnalyzer()
    face_analyzer = VideoEmotionAnalyzer()
    fusion = ToneFusionEngine()

    nlp_results = []
    face_results = []

    with st.spinner("Transcribing audio chunks..."):
        for segment in iter_audio_segments_from_file(video_path, chunk_seconds=5):
            text = transcriber.transcribe_waveform(segment.waveform, segment.sample_rate)
            nlp_results.append(nlp_analyzer.analyze(text, segment.timestamp))

    with st.spinner("Analyzing facial expressions..."):
        for ts, frame in iter_video_frames_from_file(video_path, sample_hz=1.0):
            face_results.append(face_analyzer.analyze_frame(frame, ts))

    fused = fusion.fuse_timeline(nlp_results, face_results, tolerance=1.0)
    rows = [
        {
            "timestamp": round(r.timestamp, 2),
            "transcript": r.transcript,
            "facial_emotion": r.facial_emotion,
            "final_tone": r.tone_label,
            "confidence": round(r.confidence, 3),
        }
        for r in fused
    ]
    return pd.DataFrame(rows)


mode = st.sidebar.selectbox("Input Mode", ["Live Webcam", "Upload Video File"])

if mode == "Upload Video File":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            temp_path = tmp.name

        timeline_df = run_file_processing(temp_path)
        st.subheader("Timeline Replay")
        st.dataframe(timeline_df, use_container_width=True, height=350)

        if not timeline_df.empty:
            last = timeline_df.iloc[-1]
            st.markdown(tone_badge(last["final_tone"]), unsafe_allow_html=True)
            st.progress(float(last["confidence"]))

else:
    st.info("Press Start to begin live webcam + microphone analysis.")
    col1, col2 = st.columns([2, 1])

    if "running" not in st.session_state:
        st.session_state.running = False
        st.session_state.logs = []

    start_btn = st.sidebar.button("Start Live")
    stop_btn = st.sidebar.button("Stop")

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    frame_placeholder = col1.empty()
    transcript_placeholder = col2.empty()
    tone_placeholder = col2.empty()
    log_placeholder = st.empty()

    if st.session_state.running:
        audio_q: Queue = Queue()
        frame_q: Queue = Queue()

        audio_capture = LiveAudioCapture(audio_q, chunk_seconds=5)
        video_capture = LiveVideoCapture(frame_q, sample_hz=1.0)

        transcriber = WhisperTranscriber("base")
        nlp_analyzer = NLPToneAnalyzer()
        face_analyzer = VideoEmotionAnalyzer()
        fusion = ToneFusionEngine()

        latest_face = FaceResult(0.0, "unknown", 0.0, False)

        audio_capture.start()
        video_capture.start()

        try:
            while st.session_state.running:
                while not frame_q.empty():
                    ts, frame = frame_q.get()
                    latest_face = face_analyzer.analyze_frame(frame, ts)
                    overlay = draw_face_overlay(frame, latest_face)
                    if overlay is not None:
                        frame_placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")

                while not audio_q.empty():
                    segment = audio_q.get()
                    text = transcriber.transcribe_waveform(segment.waveform, segment.sample_rate)
                    nlp = nlp_analyzer.analyze(text, segment.timestamp)
                    tone = fusion.fuse(nlp, latest_face)

                    transcript_placeholder.markdown(f"**Transcript:** {tone.transcript or '[silence]'}")
                    tone_placeholder.markdown(tone_badge(tone.tone_label), unsafe_allow_html=True)
                    tone_placeholder.progress(float(tone.confidence))

                    st.session_state.logs.append(
                        {
                            "timestamp": round(tone.timestamp, 2),
                            "transcript": tone.transcript,
                            "facial_emotion": tone.facial_emotion,
                            "final_tone": tone.tone_label,
                            "confidence": round(tone.confidence, 3),
                        }
                    )

                log_placeholder.dataframe(pd.DataFrame(st.session_state.logs), use_container_width=True, height=300)
                time.sleep(0.05)
        finally:
            audio_capture.stop()
            video_capture.stop()
    else:
        if st.session_state.logs:
            log_placeholder.dataframe(pd.DataFrame(st.session_state.logs), use_container_width=True, height=300)
