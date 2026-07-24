from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch


REFUSAL_PATTERNS = [
    r"\bi (do not|don't|cannot|can't|could not|couldn't) (find|verify|confirm|provide|identify|locate|know)\b",
    r"\b(no|not enough|insufficient) (information|evidence|context|data)\b",
    r"\bdoes not appear to (exist|be recognized)\b",
    r"\bnot (a recognized|an established|a known|widely recognized)\b",
    r"\bmay be (fictional|invented|made up|a typo|misspelled|nonexistent)\b",
    r"\bunable to verify\b",
    r"\bi'?m not familiar with\b",
]


@dataclass
class RunningStats:
    total: int = 0
    correct: int = 0
    hallucinated: int = 0
    neither: int = 0
    refused: int = 0
    high_conf_wrong: int = 0
    conf_correct: list[float] = field(default_factory=list)
    conf_hallucinated: list[float] = field(default_factory=list)
    conf_neither: list[float] = field(default_factory=list)
    entropy_correct: list[float] = field(default_factory=list)
    entropy_hallucinated: list[float] = field(default_factory=list)
    entropy_neither: list[float] = field(default_factory=list)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u0600-\u06ff\u0900-\u097f\u4e00-\u9fff\s.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_answer(generation: str, answer: str) -> bool:
    gen = normalize(generation)
    ans = normalize(answer)
    return bool(ans and ans in gen)


def is_refusal(text: str) -> bool:
    norm = normalize(text)
    return any(re.search(pattern, norm) for pattern in REFUSAL_PATTERNS)


def classify_generation(generation: str, right_answer: str, hallucinated_answer: str) -> str:
    if contains_answer(generation, right_answer):
        return "correct"
    if contains_answer(generation, hallucinated_answer):
        return "hallucinated"
    if is_refusal(generation):
        return "refused"
    return "neither"


def token_confidence_from_scores(scores: tuple[torch.Tensor, ...]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    stacked = torch.stack(scores, dim=0)
    probs = torch.softmax(stacked.float(), dim=-1)
    top1 = probs.max(dim=-1).values.mean().item()
    entropy = (-(probs * (probs + 1e-9).log()).sum(dim=-1)).mean().item()
    return float(top1), float(entropy)


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(stats: RunningStats, language: str, model_name: str, threshold: float) -> dict[str, object]:
    total = max(stats.total, 1)
    wrong = stats.hallucinated + stats.neither + stats.refused
    return {
        "model": model_name,
        "language": language,
        "num_samples": stats.total,
        "accuracy": stats.correct / total,
        "hallucination_rate": stats.hallucinated / total,
        "neither_rate": stats.neither / total,
        "refusal_rate": stats.refused / total,
        "wrong_rate": wrong / total,
        "correct": stats.correct,
        "hallucinated": stats.hallucinated,
        "neither": stats.neither,
        "refused": stats.refused,
        "high_confidence_threshold": threshold,
        "high_conf_wrong": stats.high_conf_wrong,
        "mean_conf_correct": mean(stats.conf_correct),
        "mean_conf_hallucinated": mean(stats.conf_hallucinated),
        "mean_conf_neither": mean(stats.conf_neither),
        "mean_entropy_correct": mean(stats.entropy_correct),
        "mean_entropy_hallucinated": mean(stats.entropy_hallucinated),
        "mean_entropy_neither": mean(stats.entropy_neither),
    }

