"""Entry point for multimodal tone detection.

Supports live webcam/mic processing and pre-recorded video files.
"""

from __future__ import annotations

import argparse
import threading
import time
from queue import Queue

from pipeline.audio import LiveAudioCapture, WhisperTranscriber, iter_audio_segments_from_file
from pipeline.fusion import ToneFusionEngine
from pipeline.nlp import NLPToneAnalyzer
from pipeline.video import FaceResult, LiveVideoCapture, VideoEmotionAnalyzer, iter_video_frames_from_file


def run_file_mode(video_path: str) -> None:
    """Process a full media file and print timeline events."""
    transcriber = WhisperTranscriber("base")
    nlp_analyzer = NLPToneAnalyzer()
    face_analyzer = VideoEmotionAnalyzer()
    fusion = ToneFusionEngine()

    nlp_results = []
    face_results = []

    for segment in iter_audio_segments_from_file(video_path, chunk_seconds=5):
        text = transcriber.transcribe_waveform(segment.waveform, segment.sample_rate)
        nlp_results.append(nlp_analyzer.analyze(text, segment.timestamp))

    for ts, frame in iter_video_frames_from_file(video_path, sample_hz=1.0):
        face_results.append(face_analyzer.analyze_frame(frame, ts))

    fused = fusion.fuse_timeline(nlp_results, face_results, tolerance=1.0)
    for event in fused:
        print(
            f"[{event.timestamp:7.2f}s] text='{event.transcript}' | face={event.facial_emotion:<10} "
            f"| tone={event.tone_label:<16} | conf={event.confidence:.2f}"
        )


def run_live_mode(duration_seconds: int = 60) -> None:
    """Run live mode for a fixed duration for terminal usage."""
    audio_queue: Queue = Queue()
    frame_queue: Queue = Queue()

    audio_capture = LiveAudioCapture(audio_queue, chunk_seconds=5)
    video_capture = LiveVideoCapture(frame_queue, sample_hz=1.0)

    transcriber = WhisperTranscriber("base")
    nlp_analyzer = NLPToneAnalyzer()
    face_analyzer = VideoEmotionAnalyzer()
    fusion = ToneFusionEngine()

    latest_face = FaceResult(0.0, "unknown", 0.0, False)

    audio_capture.start()
    video_capture.start()

    start = time.time()
    try:
        while time.time() - start < duration_seconds:
            while not frame_queue.empty():
                ts, frame = frame_queue.get()
                latest_face = face_analyzer.analyze_frame(frame, ts)

            while not audio_queue.empty():
                segment = audio_queue.get()
                text = transcriber.transcribe_waveform(segment.waveform, segment.sample_rate)
                nlp = nlp_analyzer.analyze(text, segment.timestamp)
                tone = fusion.fuse(nlp, latest_face)
                print(
                    f"[{tone.timestamp:7.2f}s] '{tone.transcript}' -> {tone.tone_label} "
                    f"({tone.confidence:.2f}) face={tone.facial_emotion}"
                )

            time.sleep(0.05)
    finally:
        audio_capture.stop()
        video_capture.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal tone detector")
    parser.add_argument("--mode", choices=["live", "file"], default="live")
    parser.add_argument("--video", type=str, help="Path to input video for file mode")
    parser.add_argument("--duration", type=int, default=60, help="Live mode duration in seconds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "file":
        if not args.video:
            raise SystemExit("Please provide --video path in file mode.")
        run_file_mode(args.video)
    else:
        run_live_mode(args.duration)
