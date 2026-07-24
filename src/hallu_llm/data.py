from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class QASample:
    index: int
    question: str
    context: str
    right_answer: str
    hallucinated_answer: str


def _clean(value: Any) -> str:
    return str(value or "").strip()


def load_halueval_qa(num_samples: int, seed: int, dataset_cfg: dict[str, Any]) -> list[QASample]:
    ds = load_dataset(
        dataset_cfg.get("name", "pminervini/HaluEval"),
        dataset_cfg.get("subset", "qa_samples"),
        split=dataset_cfg.get("split", "data"),
    )
    rows = list(ds)
    random.Random(seed).shuffle(rows)

    samples: list[QASample] = []
    for row in rows:
        question = _clean(row.get("question"))
        context = _clean(row.get("knowledge") or row.get("context"))
        right_answer = _clean(row.get("right_answer") or row.get("answer"))
        hallucinated_answer = _clean(row.get("hallucinated_answer") or row.get("hallucinated_ans"))
        if not question or not right_answer or not hallucinated_answer:
            continue
        samples.append(
            QASample(
                index=len(samples),
                question=question,
                context=context,
                right_answer=right_answer,
                hallucinated_answer=hallucinated_answer,
            )
        )
        if len(samples) >= num_samples:
            break
    return samples

