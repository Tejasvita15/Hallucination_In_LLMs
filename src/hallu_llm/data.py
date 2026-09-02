from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QASample:
    index: int
    question: str
    context: str
    right_answer: str
    hallucinated_answer: str
    sample_id: str | None = None


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


def validate_paired_qa_schema(rows: list[dict[str, Any]], subset: str) -> None:
    if not rows:
        return
    first_row = _maybe_expand_row(rows[0])
    keys = {str(key).lower().replace(" ", "_") for key in first_row}
    has_paired_answers = "right_answer" in keys and "hallucinated_answer" in keys
    has_sample_label = "answer" in keys and "hallucination" in keys
    if has_sample_label and not has_paired_answers:
        raise ValueError(
            f"HaluEval subset {subset!r} contains one candidate answer plus a yes/no "
            "hallucination label. Paired evaluation requires the 'qa' subset with "
            "right_answer and hallucinated_answer fields."
        )


def load_halueval_qa(
    num_samples: int | None,
    seed: int,
    dataset_cfg: dict[str, Any],
) -> list[QASample]:
    if num_samples is not None and num_samples < 1:
        raise ValueError("num_samples must be at least 1 or null for all samples.")
    local_path = dataset_cfg.get("path")
    subset = dataset_cfg.get("subset", "qa")
    if local_path:
        rows = _read_local_rows(Path(str(local_path)))
        subset = str(dataset_cfg.get("format", "paired_jsonl"))
    else:
        from datasets import load_dataset

        ds = load_dataset(
            dataset_cfg.get("name", "pminervini/HaluEval"),
            subset,
            split=dataset_cfg.get("split", "data"),
        )
        rows = list(ds)
    validate_paired_qa_schema(rows, subset)
    if bool(dataset_cfg.get("shuffle", not bool(local_path))):
        random.Random(seed).shuffle(rows)

    samples: list[QASample] = []
    rejected = 0
    for row in rows:
        row = _maybe_expand_row(row)
        sample_id = _first_present(row, ("sample_id", "source_id")) or None
        if local_path and sample_id is None:
            raise ValueError(
                "Every local paired-dataset row must contain sample_id or source_id."
            )
        question = _first_present(row, ("question", "query", "prompt"))
        context = _first_present(row, ("knowledge", "context", "document", "passage"))
        right_answer = _first_present(row, ("right_answer", "correct_answer", "gold_answer", "answer"))
        hallucinated_answer = _first_present(
            row,
            (
                "hallucinated_answer",
                "hallucinated_ans",
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
                sample_id=sample_id,
            )
        )
        if num_samples is not None and len(samples) >= num_samples:
            break

    label_like_answers = sum(
        sample.hallucinated_answer.strip().lower() in {"yes", "no", "true", "false", "0", "1"}
        for sample in samples
    )
    sample_ids = [sample.sample_id for sample in samples if sample.sample_id]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Local paired dataset contains duplicate sample IDs.")
    if samples and label_like_answers >= max(5, int(0.8 * len(samples))):
        raise ValueError(
            "Most loaded hallucinated answers are binary labels rather than answer text. "
            "Use the HaluEval 'qa' subset, not 'qa_samples'."
        )

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


def _read_local_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Local paired dataset does not exist: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError("Local paired datasets must use JSONL format.")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} of {path} must contain a JSON object."
                )
            rows.append(row)
    return rows
