"""Wraps LettuceDetect's span-level hallucination detector: given the
retrieved context and a candidate answer, flags the exact substrings of
the answer that aren't supported by that context."""

from dataclasses import dataclass

from lettucedetect.detectors.factory import make_detector

MODEL_PATH = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


@dataclass
class FlaggedSpan:
    text: str
    start: int
    end: int
    confidence: float


def load_detector():
    return make_detector("transformer", model_path=MODEL_PATH)


def check_answer(detector, context: list[str], question: str, answer: str) -> list[FlaggedSpan]:
    predictions = detector.predict(
        context=context, question=question, answer=answer, output_format="spans"
    )
    return [
        FlaggedSpan(text=p["text"], start=p["start"], end=p["end"], confidence=p["confidence"])
        for p in predictions
    ]
