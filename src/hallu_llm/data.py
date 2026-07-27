from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class QASample:
    index: int
    question: str
    context: str
    right_answer: str
    hallucinated_answer: str


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _maybe_expand_row(row: dict[str, Any]) -> dict[str, Any]:
    """Handle HaluEval cache variants where the real sample is nested."""
    expanded = dict(row)
    for key in ("data", "sample", "example"):
        value = row.get(key)
        if isinstance(value, dict):
            expanded.update(value)
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                expanded.update(parsed)
    return expanded


def _first_present(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    normalized = {str(key).lower().replace(" ", "_"): value for key, value in row.items()}
    for key in candidates:
        value = row.get(key)
        if value:
            return _clean(value)
        value = normalized.get(key.lower().replace(" ", "_"))
        if value:
            return _clean(value)
    return ""


def load_halueval_qa(num_samples: int, seed: int, dataset_cfg: dict[str, Any]) -> list[QASample]:
    from datasets import load_dataset

    ds = load_dataset(
        dataset_cfg.get("name", "pminervini/HaluEval"),
        dataset_cfg.get("subset", "qa_samples"),
        split=dataset_cfg.get("split", "data"),
    )
    rows = list(ds)
    random.Random(seed).shuffle(rows)

    samples: list[QASample] = []
    rejected = 0
    for row in rows:
        row = _maybe_expand_row(row)
        question = _first_present(row, ("question", "query", "prompt"))
        context = _first_present(row, ("knowledge", "context", "document", "passage"))
        right_answer = _first_present(row, ("right_answer", "correct_answer", "gold_answer", "answer"))
        hallucinated_answer = _first_present(
            row,
            (
                "hallucinated_answer",
                "hallucinated_ans",
                "hallucination",
                "hallucinated_response",
                "fake_answer",
                "wrong_answer",
            ),
        )
        if not question or not right_answer or not hallucinated_answer:
            rejected += 1
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

    if not samples:
        preview_rows = [_maybe_expand_row(row) for row in rows[:3]]
        preview = [
            {
                "keys": sorted(str(key) for key in row.keys()),
                "non_empty": {str(key): _clean(value)[:120] for key, value in row.items() if _clean(value)},
            }
            for row in preview_rows
        ]
        raise ValueError(
            "Loaded 0 valid HaluEval QA samples. The dataset schema on this machine "
            "does not match the expected fields. "
            f"Rows loaded={len(rows)}, rejected={rejected}, preview={json.dumps(preview, ensure_ascii=False)}"
        )

    print(f"Accepted {len(samples)} HaluEval samples; rejected {rejected} rows with missing fields.")
    return samples
