"""Video pipeline for frame capture and DeepFace emotion analysis."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Generator, Optional

import cv2

try:
    from deepface import DeepFace
except Exception:  # pragma: no cover
    DeepFace = None


@dataclass
class FaceResult:
    timestamp: float
    emotion_label: str
    confidence: float
    face_detected: bool
    frame: Optional[any] = None
    bbox: Optional[tuple[int, int, int, int]] = None


class VideoEmotionAnalyzer:
    """Analyzes one frame for dominant emotion using DeepFace."""

    def __init__(self) -> None:
        if DeepFace is None:
            self.available = False
        else:
            self.available = True

    def analyze_frame(self, frame, timestamp: float) -> FaceResult:
        """Analyze a frame and return facial emotion summary."""
        if not self.available:
            return FaceResult(timestamp, "unknown", 0.0, False, frame=frame)

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )
            # DeepFace returns a dict or list[dict] depending on version.
            if isinstance(result, list):
                result = result[0]

            emotions = result.get("emotion", {})
            if emotions:
                label = max(emotions, key=emotions.get)
                confidence = float(emotions[label]) / 100.0
            else:
                label, confidence = "neutral", 0.0

            region = result.get("region", {}) or {}
            bbox = (
                int(region.get("x", 0)),
                int(region.get("y", 0)),
                int(region.get("w", 0)),
                int(region.get("h", 0)),
            )
            face_detected = bbox[2] > 0 and bbox[3] > 0
            return FaceResult(
                timestamp=timestamp,
                emotion_label=label.lower(),
                confidence=confidence,
                face_detected=face_detected,
                frame=frame,
                bbox=bbox if face_detected else None,
            )
        except Exception:
            return FaceResult(timestamp, "unknown", 0.0, False, frame=frame)


class LiveVideoCapture:
    """Capture webcam frames and sample one frame per second."""

    def __init__(self, output_queue: Queue, camera_index: int = 0, sample_hz: float = 1.0) -> None:
        self.output_queue = output_queue
        self.camera_index = camera_index
        self.sample_hz = sample_hz
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        start_time = time.time()
        next_emit = 0.0

        try:
            while not self._stop_event.is_set() and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    continue

                now = time.time() - start_time
                if now >= next_emit:
                    self.output_queue.put((now, frame))
                    next_emit += 1.0 / self.sample_hz
        finally:
            cap.release()


def iter_video_frames_from_file(video_path: str | Path, sample_hz: float = 1.0) -> Generator[tuple[float, any], None, None]:
    """Yield sampled frames from a video file with timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(max(1, round(fps / sample_hz)))

    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            ts = frame_idx / fps
            yield ts, frame

        frame_idx += 1

    cap.release()


def draw_face_overlay(frame, result: FaceResult):
    """Draw bounding box and emotion label on frame for UI display."""
    if frame is None:
        return None

    overlay = frame.copy()
    if result.bbox:
        x, y, w, h = result.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{result.emotion_label} ({result.confidence:.2f})"
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return overlay
