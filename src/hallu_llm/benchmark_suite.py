from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SPAN_JSON = re.compile(r"\{.*\}", flags=re.DOTALL)


def parse_binary_generation(
    text: str,
    positive_label: str,
    negative_label: str,
) -> tuple[bool | None, bool]:
    stripped = text.strip()
    labels = {
        positive_label.upper(): True,
        negative_label.upper(): False,
    }
    escaped = "|".join(
        re.escape(label)
        for label in sorted(labels, key=len, reverse=True)
    )
    match = re.match(
        rf"^({escaped})(?=$|[\s.,:;!?])",
        stripped,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, False
    matched = match.group(1).upper()
    return labels[matched], stripped.upper() == matched


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
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


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return read_jsonl(path)
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "samples", "examples"):
            rows = payload.get(key)
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                return rows
    raise ValueError(f"Expected a JSON array or JSONL objects in {path}.")


def normalize_halueval_qa(
    examples: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rejected = 0
    for position, example in enumerate(examples):
        question = str(example.get("question", "")).strip()
        knowledge = str(example.get("knowledge", "")).strip()
        right_answer = str(example.get("right_answer", "")).strip()
        hallucinated_answer = str(example.get("hallucinated_answer", "")).strip()
        if not question or not right_answer or not hallucinated_answer:
            rejected += 1
            continue
        rows.append(
            {
                "sample_id": f"halueval:qa:{position}",
                "question": question,
                "knowledge": knowledge,
                "right_answer": right_answer,
                "hallucinated_answer": hallucinated_answer,
            }
        )
    _validate_unique_ids(rows)
    if rejected:
        raise ValueError(
            f"Official HaluEval QA file has {rejected} rows missing paired fields."
        )
    return rows, {
        "benchmark": "halueval_qa",
        "num_rows": len(rows),
        "rejected_rows": rejected,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ragtruth(
    responses: Iterable[Mapping[str, Any]],
    sources: Iterable[Mapping[str, Any]],
    split: str | None = "test",
    include_quality_issues: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_by_id = {
        str(source["source_id"]): source for source in sources
    }
    rows: list[dict[str, Any]] = []
    excluded_quality = 0
    span_warnings = 0
    missing_sources: list[str] = []

    for response in responses:
        if split is not None and str(response.get("split")) != split:
            continue
        quality = str(response.get("quality", "good") or "good")
        if quality != "good" and not include_quality_issues:
            excluded_quality += 1
            continue
        source_id = str(response.get("source_id", ""))
        source = source_by_id.get(source_id)
        if source is None:
            missing_sources.append(source_id)
            continue
        output = str(response.get("response", ""))
        spans, warnings = normalize_spans(
            output,
            response.get("labels", []),
            text_field="text",
        )
        span_warnings += len(warnings)
        rows.append(
            {
                "sample_id": f"ragtruth:{response['id']}",
                "benchmark": "ragtruth",
                "split": str(response.get("split", "")),
                "language": "en",
                "task": str(source.get("task_type", "unknown")),
                "source_group_id": source_id,
                "source_dataset": str(source.get("source", "")),
                "model_input": str(source.get("prompt", "")),
                "context": _json_text(source.get("source_info", "")),
                "model_output_text": output,
                "generator_model": str(response.get("model", "")),
                "gold_hallucinated": bool(spans),
                "gold_spans": [list(span) for span in spans],
                "quality": quality,
                "annotation_warnings": warnings,
            }
        )

    _validate_unique_ids(rows)
    if missing_sources:
        preview = sorted(set(missing_sources))[:5]
        raise ValueError(
            "RAGTruth responses reference missing source IDs: "
            f"{preview}"
        )
    return rows, {
        "benchmark": "ragtruth",
        "num_rows": len(rows),
        "split": split,
        "excluded_quality_rows": excluded_quality,
        "span_alignment_warnings": span_warnings,
        "task_counts": _counts(rows, "task"),
        "label_counts": _boolean_counts(rows, "gold_hallucinated"),
    }


def normalize_mushroom(
    examples: Iterable[Mapping[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    span_warnings = 0
    skipped_unlabeled = 0
    for position, example in enumerate(examples):
        if "hard_labels" not in example or example.get("hard_labels") is None:
            skipped_unlabeled += 1
            continue
        output = str(example.get("model_output_text", ""))
        spans, warnings = normalize_spans(output, example.get("hard_labels", []))
        span_warnings += len(warnings)
        language = str(example.get("lang", "unknown"))
        identifier = str(example.get("id") or "row")
        rows.append(
            {
                "sample_id": (
                    f"mu_shroom:{language}:{split}:{identifier}:{position}"
                ),
                "benchmark": "mu_shroom",
                "split": split,
                "language": language,
                "task": "multilingual_span_detection",
                "model_input": str(example.get("model_input", "")),
                "context": str(example.get("wikipedia_url", "")),
                "model_output_text": output,
                "generator_model": str(example.get("model_id", "")),
                "gold_hallucinated": bool(spans),
                "gold_spans": [list(span) for span in spans],
                "soft_labels": _jsonable(example.get("soft_labels", [])),
                "annotation_warnings": warnings,
            }
        )
    _validate_unique_ids(rows)
    return rows, {
        "benchmark": "mu_shroom",
        "num_rows": len(rows),
        "split": split,
        "skipped_unlabeled_rows": skipped_unlabeled,
        "span_alignment_warnings": span_warnings,
        "language_counts": _counts(rows, "language"),
        "label_counts": _boolean_counts(rows, "gold_hallucinated"),
    }


def normalize_truthfulqa_csv(
    path: str | Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        question = str(source.get("Question", "")).strip()
        true_answer = str(source.get("Best Answer", "")).strip()
        false_answer = str(source.get("Best Incorrect Answer", "")).strip()
        if not question or not true_answer or not false_answer:
            continue
        sample_hash = hashlib.sha256(
            f"{seed}:{question}".encode("utf-8")
        ).hexdigest()
        true_is_a = int(sample_hash[:8], 16) % 2 == 0
        rows.append(
            {
                "sample_id": f"truthfulqa:{index}",
                "benchmark": "truthfulqa",
                "split": "benchmark",
                "language": "en",
                "task": "binary_multiple_choice",
                "question": question,
                "choice_a": true_answer if true_is_a else false_answer,
                "choice_b": false_answer if true_is_a else true_answer,
                "correct_label": "A" if true_is_a else "B",
                "category": str(source.get("Category", "")),
                "source": str(source.get("Source", "")),
            }
        )
    _validate_unique_ids(rows)
    return rows, {
        "benchmark": "truthfulqa",
        "num_rows": len(rows),
        "split": "benchmark",
        "category_counts": _counts(rows, "category"),
        "correct_label_counts": _counts(rows, "correct_label"),
        "choice_order_seed": seed,
    }


def normalize_spans(
    text: str,
    raw_spans: Any,
    text_field: str | None = None,
) -> tuple[list[tuple[int, int]], list[str]]:
    spans: list[tuple[int, int]] = []
    warnings: list[str] = []
    if not isinstance(raw_spans, list):
        return [], ["spans_not_a_list"]
    for position, raw_span in enumerate(raw_spans):
        if isinstance(raw_span, Mapping):
            start = raw_span.get("start")
            end = raw_span.get("end")
            expected_text = (
                str(raw_span.get(text_field, "")) if text_field else ""
            )
        elif isinstance(raw_span, (list, tuple)) and len(raw_span) >= 2:
            start, end = raw_span[0], raw_span[1]
            expected_text = ""
        else:
            warnings.append(f"span_{position}_malformed")
            continue
        try:
            start_int = int(start)
            end_int = int(end)
        except (TypeError, ValueError):
            warnings.append(f"span_{position}_non_integer")
            continue
        if start_int < 0 or end_int <= start_int or end_int > len(text):
            warnings.append(f"span_{position}_out_of_bounds")
            continue
        if expected_text and text[start_int:end_int] != expected_text:
            warnings.append(f"span_{position}_text_mismatch")
        spans.append((start_int, end_int))
    return merge_spans(spans), warnings


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(set(spans))
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def parse_span_generation(text: str, model_output: str) -> list[tuple[int, int]]:
    spans, _ = parse_span_generation_details(text, model_output)
    return spans


def parse_span_generation_details(
    text: str,
    model_output: str,
) -> tuple[list[tuple[int, int]], bool]:
    match = SPAN_JSON.search(text.strip())
    if not match:
        return [], False
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [], False
    if not isinstance(payload, Mapping) or "hallucinated_spans" not in payload:
        return [], False
    raw_spans = payload.get("hallucinated_spans", [])
    if not isinstance(raw_spans, list):
        return [], False
    direct: list[tuple[int, int]] = []
    substrings: list[str] = []
    for span in raw_spans:
        if isinstance(span, Mapping) and "start" in span and "end" in span:
            try:
                start = int(span["start"])
                end = int(span["end"])
            except (TypeError, ValueError):
                continue
            if 0 <= start < end <= len(model_output):
                direct.append((start, end))
        elif isinstance(span, str) and span:
            substrings.append(span)
    occupied = set(index for start, end in direct for index in range(start, end))
    for substring in substrings:
        search_from = 0
        while True:
            start = model_output.find(substring, search_from)
            if start < 0:
                break
            end = start + len(substring)
            if not any(index in occupied for index in range(start, end)):
                direct.append((start, end))
                occupied.update(range(start, end))
                break
            search_from = start + 1
    return merge_spans(direct), True


def span_iou(
    gold_spans: Iterable[tuple[int, int]],
    predicted_spans: Iterable[tuple[int, int]],
) -> float:
    gold = _span_indices(gold_spans)
    predicted = _span_indices(predicted_spans)
    union = gold | predicted
    if not union:
        return 1.0
    return len(gold & predicted) / len(union)


def binary_metrics(actual: list[bool], predicted: list[bool]) -> dict[str, Any]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Binary metrics require equal non-empty label lists.")
    tp = sum(a and p for a, p in zip(actual, predicted))
    fp = sum((not a) and p for a, p in zip(actual, predicted))
    tn = sum((not a) and (not p) for a, p in zip(actual, predicted))
    fn = sum(a and (not p) for a, p in zip(actual, predicted))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "num_samples": len(actual),
        "accuracy": (tp + tn) / len(actual),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": ((tp * tn - fp * fn) / denominator) if denominator else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def probability_metrics(
    actual: list[bool],
    probabilities: list[float],
    ece_bins: int = 10,
) -> dict[str, float]:
    if not actual or len(actual) != len(probabilities):
        raise ValueError("Probability metrics require equal non-empty lists.")
    clipped = [min(max(float(value), 1e-12), 1 - 1e-12) for value in probabilities]
    brier = sum((prob - float(label)) ** 2 for label, prob in zip(actual, clipped)) / len(actual)
    nll = -sum(
        float(label) * math.log(prob) + (1 - float(label)) * math.log(1 - prob)
        for label, prob in zip(actual, clipped)
    ) / len(actual)
    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        members = [
            index
            for index, probability in enumerate(clipped)
            if lower <= probability < upper or (bin_index == ece_bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        mean_probability = sum(clipped[index] for index in members) / len(members)
        empirical_rate = sum(float(actual[index]) for index in members) / len(members)
        ece += len(members) / len(actual) * abs(mean_probability - empirical_rate)
    return {"brier_score": brier, "negative_log_likelihood": nll, "ece": ece}


def binary_roc_auc(actual: list[bool], probabilities: list[float]) -> float | None:
    if not actual or len(actual) != len(probabilities):
        raise ValueError("ROC AUC requires equal non-empty lists.")
    positives = sum(actual)
    negatives = len(actual) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(
        enumerate(probabilities),
        key=lambda item: float(item[1]),
    )
    ranks = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and float(ordered[end][1]) == float(ordered[index][1]):
            end += 1
        average_rank = ((index + 1) + end) / 2
        for ranked_index in range(index, end):
            ranks[ordered[ranked_index][0]] = average_rank
        index = end
    positive_rank_sum = sum(rank for label, rank in zip(actual, ranks) if label)
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def span_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Span metrics require at least one prediction row.")
    ious: list[float] = []
    gold_characters: set[tuple[int, int]] = set()
    predicted_characters: set[tuple[int, int]] = set()
    actual_binary: list[bool] = []
    predicted_binary: list[bool] = []
    exact = 0
    for row_index, row in enumerate(materialized):
        gold = [tuple(span) for span in row.get("gold_spans", [])]
        predicted = [tuple(span) for span in row.get("predicted_spans", [])]
        ious.append(span_iou(gold, predicted))
        gold_indices = _span_indices(gold)
        predicted_indices = _span_indices(predicted)
        gold_characters.update((row_index, index) for index in gold_indices)
        predicted_characters.update((row_index, index) for index in predicted_indices)
        actual_binary.append(bool(gold))
        predicted_binary.append(bool(predicted))
        exact += int(merge_spans(gold) == merge_spans(predicted))
    overlap = len(gold_characters & predicted_characters)
    char_precision = _safe_div(overlap, len(predicted_characters))
    char_recall = _safe_div(overlap, len(gold_characters))
    return {
        "num_samples": len(materialized),
        "mean_hard_span_iou": sum(ious) / len(ious),
        "exact_span_match_rate": exact / len(materialized),
        "hallucinated_character_precision": char_precision,
        "hallucinated_character_recall": char_recall,
        "hallucinated_character_f1": _safe_div(
            2 * char_precision * char_recall,
            char_precision + char_recall,
        ),
        "response_level": binary_metrics(actual_binary, predicted_binary),
    }


def _span_indices(spans: Iterable[tuple[int, int]]) -> set[int]:
    return {
        index
        for start, end in merge_spans(spans)
        for index in range(start, end)
    }


def _validate_unique_ids(rows: list[dict[str, Any]]) -> None:
    identifiers = [str(row["sample_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Prepared benchmark contains duplicate sample IDs.")


def _counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _boolean_counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return {
        "false": sum(not bool(row.get(field)) for row in rows),
        "true": sum(bool(row.get(field)) for row in rows),
    }


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
