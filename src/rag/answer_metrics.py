"""Normalized answer exact match and HotpotQA-style token overlap F1."""

import re
import string
from collections import Counter


def normalize_answer(answer: str) -> str:
    """Strip numeric citations, lowercase, remove punctuation/articles/extra spaces."""
    answer = re.sub(r"\[\d+\]", "", answer).lower()
    answer = "".join(c for c in answer if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", answer).split())


def score_answer(prediction: str, gold: str) -> dict[str, float]:
    """Score a single reference; enforce exact yes/no/noanswer agreement for F1."""
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    exact = float(predicted == expected)
    special = ("yes", "no", "noanswer")
    if predicted != expected and (predicted in special or expected in special):
        return {"em": exact, "f1": 0.0}
    overlap = sum((Counter(predicted.split()) & Counter(expected.split())).values())
    f1 = 2 * overlap / (len(predicted.split()) + len(expected.split())) if overlap else exact
    return {"em": exact, "f1": f1}
