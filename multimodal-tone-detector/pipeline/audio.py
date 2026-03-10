"""Audio pipeline for live and file-based speech transcription.

This module captures audio, chunks it into near-real-time windows, and runs
local Whisper transcription.
"""

from __future__ import annotations

import io
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Generator, Iterable, Optional

import numpy as np

try:
    import pyaudio
except Exception:  # pragma: no cover - optional dependency during CI
    pyaudio = None

try:
    import whisper
except Exception:  # pragma: no cover - optional dependency during CI
    whisper = None


@dataclass
class AudioSegment:
    """Represents one chunk of audio and its metadata."""

    timestamp: float
    duration: float
    waveform: np.ndarray
    sample_rate: int
    transcript: str = ""


class WhisperTranscriber:
    """Thin wrapper around OpenAI Whisper local model."""

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self.model = whisper.load_model(model_name) if whisper else None

    def transcribe_waveform(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Transcribe waveform chunk into text.

        If Whisper isn't available, returns an empty string.
        """
        if self.model is None or waveform.size == 0:
            return ""

        # Whisper expects float32 in [-1, 1].
        wav = waveform.astype(np.float32)
        if np.max(np.abs(wav)) > 1.0:
            wav = wav / 32768.0

        result = self.model.transcribe(wav, fp16=False, language="en")
        return result.get("text", "").strip()


class LiveAudioCapture:
    """Captures microphone audio and emits chunks via queue."""

    def __init__(
        self,
        output_queue: Queue,
        sample_rate: int = 16_000,
        channels: int = 1,
        chunk_seconds: int = 5,
        frames_per_buffer: int = 1024,
    ) -> None:
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = chunk_seconds
        self.frames_per_buffer = frames_per_buffer
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start capture loop in background thread."""
        if pyaudio is None:
            raise RuntimeError("PyAudio is not installed; cannot capture live audio.")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal capture loop to stop and wait for completion."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )

        bytes_per_sample = 2  # int16
        target_bytes = self.sample_rate * self.chunk_seconds * bytes_per_sample
        buffer = bytearray()
        start_time = time.time()

        try:
            while not self._stop_event.is_set():
                data = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                buffer.extend(data)

                if len(buffer) >= target_bytes:
                    ts = time.time() - start_time
                    raw = bytes(buffer[:target_bytes])
                    del buffer[:target_bytes]

                    waveform = np.frombuffer(raw, dtype=np.int16)
                    segment = AudioSegment(
                        timestamp=ts,
                        duration=self.chunk_seconds,
                        waveform=waveform,
                        sample_rate=self.sample_rate,
                    )
                    self.output_queue.put(segment)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


def iter_audio_segments_from_file(
    video_path: str | Path,
    chunk_seconds: int = 5,
    sample_rate: int = 16_000,
) -> Generator[AudioSegment, None, None]:
    """Yield audio chunks from a media file using ffmpeg-python.

    This decodes to mono PCM16 and slices into fixed-size chunks.
    """
    try:
        import ffmpeg
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ffmpeg-python is required for file processing") from exc

    video_path = str(video_path)
    out, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
        .run(capture_stdout=True, capture_stderr=True)
    )

    waveform = np.frombuffer(out, dtype=np.int16)
    samples_per_chunk = sample_rate * chunk_seconds

    for i in range(0, len(waveform), samples_per_chunk):
        chunk = waveform[i : i + samples_per_chunk]
        if chunk.size == 0:
            continue
        yield AudioSegment(
            timestamp=i / sample_rate,
            duration=chunk.size / sample_rate,
            waveform=chunk,
            sample_rate=sample_rate,
        )


def write_wave_chunk(path: str | Path, waveform: np.ndarray, sample_rate: int = 16_000) -> None:
    """Utility helper for debugging: write a wave chunk to disk."""
    path = Path(path)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.astype(np.int16).tobytes())
