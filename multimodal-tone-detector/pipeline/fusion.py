"""Fusion logic combining NLP and facial expression signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List

from pipeline.nlp import NLPResult
from pipeline.video import FaceResult
from utils.sync import find_nearest_with_tolerance


NEGATIVE_EMOTIONS = {"anger", "angry", "sad", "fear", "disgust"}
POSITIVE_EMOTIONS = {"joy", "happy", "surprise"}
NEUTRAL_EMOTIONS = {"neutral", "unknown"}


@dataclass
class ToneResult:
    timestamp: float
    tone_label: str
    confidence: float
    transcript: str
    facial_emotion: str
    nlp: NLPResult
    face: FaceResult

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nlp"] = asdict(self.nlp)
        d["face"] = asdict(self.face)
        return d


class ToneFusionEngine:
    """Apply rule-based multimodal tone fusion."""

    def fuse(self, nlp_result: NLPResult, face_result: FaceResult) -> ToneResult:
        emo = (nlp_result.emotion_label or "neutral").lower()
        face_emo = (face_result.emotion_label or "unknown").lower()
        irony = nlp_result.irony_label == "irony"

        tone = "Neutral/Sincere"
        conf = 0.5

        # Rule 1: positive language, negative face => likely sarcasm.
        if emo in {"joy", "love", "optimism"} and face_emo in NEGATIVE_EMOTIONS:
            tone = "Sarcastic"
            conf = min(1.0, 0.65 + 0.35 * max(nlp_result.emotion_score, face_result.confidence))

        # Rule 2: explicit irony + smirk/neutral-like expression.
        elif irony and face_emo in {"neutral", "happy", "unknown"}:
            tone = "Sarcastic"
            conf = min(1.0, 0.6 + 0.4 * nlp_result.irony_score)

        # Rule 3: joy + happy face.
        elif emo in {"joy", "amusement"} and face_emo == "happy":
            tone = "Sincere/Positive"
            conf = min(1.0, 0.55 + 0.45 * ((nlp_result.emotion_score + face_result.confidence) / 2.0))

        # Rule 4: exaggerated face with neutral text.
        elif emo in {"neutral", ""} and face_emo in {"surprise", "happy"}:
            tone = "Joking/Playful"
            conf = min(1.0, 0.5 + 0.5 * face_result.confidence)

        # Rule 5: both neutral.
        elif emo in {"neutral", ""} and face_emo in NEUTRAL_EMOTIONS:
            tone = "Neutral/Sincere"
            conf = 0.8

        # Fallback mapping.
        else:
            if emo in {"joy", "amusement"}:
                tone = "Sincere/Positive"
            elif irony:
                tone = "Sarcastic"
            elif face_emo in {"surprise", "happy"}:
                tone = "Joking/Playful"
            else:
                tone = "Neutral/Sincere"
            conf = max(nlp_result.emotion_score, nlp_result.irony_score, face_result.confidence, 0.4)

        return ToneResult(
            timestamp=nlp_result.timestamp,
            tone_label=tone,
            confidence=float(max(0.0, min(1.0, conf))),
            transcript=nlp_result.text,
            facial_emotion=face_result.emotion_label,
            nlp=nlp_result,
            face=face_result,
        )

    def fuse_timeline(
        self,
        nlp_results: List[NLPResult],
        face_results: List[FaceResult],
        tolerance: float = 1.0,
    ) -> List[ToneResult]:
        """Match NLP events with nearest face event and fuse."""
        fused: List[ToneResult] = []
        for nlp in nlp_results:
            face = find_nearest_with_tolerance(
                nlp.timestamp,
                face_results,
                get_ts=lambda r: r.timestamp,
                tolerance=tolerance,
            )
            if face is None:
                face = FaceResult(
                    timestamp=nlp.timestamp,
                    emotion_label="unknown",
                    confidence=0.0,
                    face_detected=False,
                )
            fused.append(self.fuse(nlp, face))
        return fused
