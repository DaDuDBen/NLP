"""NLP inference module for irony and emotion classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


@dataclass
class NLPResult:
    timestamp: float
    text: str
    irony_label: str
    irony_score: float
    emotion_label: str
    emotion_score: float


class NLPToneAnalyzer:
    """Runs local HuggingFace models for irony + emotion."""

    def __init__(self) -> None:
        self.irony_pipe = (
            pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-irony",
                top_k=None,
            )
            if pipeline
            else None
        )
        self.emotion_pipe = (
            pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=1,
            )
            if pipeline
            else None
        )

    def analyze(self, text: str, timestamp: float) -> NLPResult:
        """Classify text into irony and emotion dimensions."""
        text = (text or "").strip()

        if not text:
            return NLPResult(timestamp, "", "non_irony", 0.0, "neutral", 0.0)

        irony_label, irony_score = "non_irony", 0.0
        emotion_label, emotion_score = "neutral", 0.0

        if self.irony_pipe:
            preds = self.irony_pipe(text)
            # Model labels are often LABEL_0/LABEL_1. LABEL_1 -> irony.
            ranked = sorted(preds[0], key=lambda x: x["score"], reverse=True)
            top = ranked[0]
            lbl = top["label"].lower()
            irony_label = "irony" if lbl in {"label_1", "irony"} else "non_irony"
            irony_score = float(top["score"])

        if self.emotion_pipe:
            pred = self.emotion_pipe(text)[0][0]
            emotion_label = pred["label"].lower()
            emotion_score = float(pred["score"])

        return NLPResult(
            timestamp=timestamp,
            text=text,
            irony_label=irony_label,
            irony_score=irony_score,
            emotion_label=emotion_label,
            emotion_score=emotion_score,
        )
