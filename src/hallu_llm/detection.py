from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from .data import QASample
from .prompts import halueval_detection_prompt

if TYPE_CHECKING:
    from .models import HFGenerator


@dataclass(frozen=True)
class DetectionInstance:
    sample_index: int
    sample_id: str
    question: str
    knowledge: str
    candidate_answer: str
    candidate_type: str
    is_hallucinated: bool
    excluded_from_primary_metrics: bool = False
    source_review_reason: str | None = None


def stable_sample_id(sample: QASample) -> str:
    content = "\n".join(
        (sample.question, sample.context, sample.right_answer, sample.hallucinated_answer)
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def build_detection_instances(
    samples: list[QASample],
    seed: int,
    sample_ids: dict[int, str] | None = None,
    source_reviews: Mapping[str, Any] | None = None,
) -> list[DetectionInstance]:
    instances: list[DetectionInstance] = []
    for sample in samples:
        sample_id = (sample_ids or {}).get(sample.index, stable_sample_id(sample))
        review = dict((source_reviews or {}).get(sample_id, {}))
        excluded = bool(review.get("exclude_from_primary_metrics", False))
        review_reason = (
            str(review.get("reason")).strip()
            if review.get("reason") is not None
            else None
        )
        instances.extend(
            (
                DetectionInstance(
                    sample_index=sample.index,
                    sample_id=sample_id,
                    question=sample.question,
                    knowledge=sample.context,
                    candidate_answer=sample.right_answer,
                    candidate_type="grounded",
                    is_hallucinated=False,
                    excluded_from_primary_metrics=excluded,
                    source_review_reason=review_reason,
                ),
                DetectionInstance(
                    sample_index=sample.index,
                    sample_id=sample_id,
                    question=sample.question,
                    knowledge=sample.context,
                    candidate_answer=sample.hallucinated_answer,
                    candidate_type="hallucinated",
                    is_hallucinated=True,
                    excluded_from_primary_metrics=excluded,
                    source_review_reason=review_reason,
                ),
            )
        )
    random.Random(seed).shuffle(instances)
    return instances


def parse_binary_response(text: str) -> bool | None:
    match = re.match(r"""^\s*[`"'(\[]*\s*(yes|no)\b""", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "yes"


def is_strict_binary_response(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"""\s*[`"'(\[]*\s*(yes|no)\s*[`"')\].!]*\s*""",
            text,
            flags=re.IGNORECASE,
        )
    )


def evaluate_detection_instances(
    generator: HFGenerator,
    instances: list[DetectionInstance],
    model_name: str,
    language_code: str,
    max_new_tokens: int,
    high_confidence_threshold: float,
    prompt_instruction: str | None = None,
    translation_warnings: dict[int, list[str]] | None = None,
    length_match_max_relative_difference: float = 0.25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    warnings = translation_warnings or {}

    for instance in instances:
        prompt = halueval_detection_prompt(
            knowledge=instance.knowledge,
            question=instance.question,
            candidate_answer=instance.candidate_answer,
            language_code=language_code,
            instruction=prompt_instruction,
        )
        decision = generator.generate_binary(
            prompt,
            positive_label="YES",
            negative_label="NO",
            max_new_tokens=max_new_tokens,
        )
        parsed_prediction = parse_binary_response(decision.text)
        primary_prediction = (
            parsed_prediction
            if parsed_prediction is not None
            else decision.predicted_positive
        )
        rows.append(
            {
                "sample_index": instance.sample_index,
                "sample_id": instance.sample_id,
                "model": model_name,
                "language": language_code,
                "candidate_type": instance.candidate_type,
                "actual_label": "hallucinated" if instance.is_hallucinated else "grounded",
                "predicted_label": "hallucinated" if primary_prediction else "grounded",
                "prediction_source": (
                    "generated_response"
                    if parsed_prediction is not None
                    else "logit_fallback"
                ),
                "logit_predicted_label": (
                    "hallucinated" if decision.predicted_positive else "grounded"
                ),
                "parsed_output_label": (
                    None
                    if parsed_prediction is None
                    else ("hallucinated" if parsed_prediction else "grounded")
                ),
                "prediction_correct": primary_prediction == instance.is_hallucinated,
                "excluded_from_primary_metrics": (
                    instance.excluded_from_primary_metrics
                ),
                "source_review_reason": instance.source_review_reason,
                "output_parseable": parsed_prediction is not None,
                "format_valid": is_strict_binary_response(decision.text),
                "parsed_logit_agreement": (
                    None
                    if parsed_prediction is None
                    else parsed_prediction == decision.predicted_positive
                ),
                "hallucination_probability": decision.positive_probability,
                "grounded_probability": decision.negative_probability,
                "decision_confidence": decision.decision_confidence,
                "decision_entropy": decision.decision_entropy,
                "label_probability_mass": decision.label_probability_mass,
                "first_generated_token_id": decision.first_generated_token_id,
                "translation_warnings": warnings.get(instance.sample_index, []),
                "question": instance.question,
                "knowledge": instance.knowledge,
                "candidate_answer": instance.candidate_answer,
                "candidate_character_length": _content_character_length(
                    instance.candidate_answer
                ),
                "prompt": prompt,
                "generation": decision.text,
            }
        )

    summary = summarize_detection(
        rows=rows,
        language_code=language_code,
        model_name=model_name,
        high_confidence_threshold=high_confidence_threshold,
        length_match_max_relative_difference=(
            length_match_max_relative_difference
        ),
    )
    return summary, rows


def summarize_detection(
    rows: list[dict[str, Any]],
    language_code: str,
    model_name: str,
    high_confidence_threshold: float,
    length_match_max_relative_difference: float = 0.25,
) -> dict[str, Any]:
    generated_total = len(rows)
    excluded_sample_ids = sorted(
        {
            str(row["sample_id"])
            for row in rows
            if row.get("excluded_from_primary_metrics", False)
        }
    )
    all_translation_warning_samples = len(
        {
            row["sample_index"]
            for row in rows
            if row.get("translation_warnings", [])
        }
    )
    rows = [
        row
        for row in rows
        if not row.get("excluded_from_primary_metrics", False)
    ]
    total = len(rows)
    if total == 0:
        raise ValueError("No detection rows remain for primary metrics.")

    actual = [row["actual_label"] == "hallucinated" for row in rows]
    predicted = [row["predicted_label"] == "hallucinated" for row in rows]
    logit_predicted = [row["logit_predicted_label"] == "hallucinated" for row in rows]
    tp = sum(a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = (tp + tn) / total
    balanced_accuracy = (recall + specificity) / 2
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator else 0.0

    probabilities = [float(row["hallucination_probability"]) for row in rows]
    confidence = [float(row["decision_confidence"]) for row in rows]
    correct = [bool(row["prediction_correct"]) for row in rows]
    output_parseable = [bool(row["output_parseable"]) for row in rows]
    format_valid = [bool(row["format_valid"]) for row in rows]
    valid_agreements = [
        bool(row["parsed_logit_agreement"])
        for row in rows
        if row["parsed_logit_agreement"] is not None
    ]
    high_confidence_errors = sum(
        not is_correct and conf >= high_confidence_threshold
        for is_correct, conf in zip(correct, confidence)
    )
    epsilon = 1e-12
    brier_score = sum(
        (probability - int(label)) ** 2
        for probability, label in zip(probabilities, actual)
    ) / total
    negative_log_likelihood = -sum(
        int(label) * math.log(max(probability, epsilon))
        + (1 - int(label)) * math.log(max(1 - probability, epsilon))
        for probability, label in zip(probabilities, actual)
    ) / total
    pairwise = _pairwise_metrics(rows)
    length_diagnostics = _length_diagnostics(
        rows,
        max_relative_difference=length_match_max_relative_difference,
    )

    return {
        "protocol": "paired_halueval_detection",
        "model": model_name,
        "language": language_code,
        "num_generated_source_samples": generated_total // 2,
        "num_generated_instances": generated_total,
        "num_excluded_source_samples": len(excluded_sample_ids),
        "excluded_sample_ids": excluded_sample_ids,
        "num_source_samples": total // 2,
        "num_instances": total,
        "grounded_instances": sum(not label for label in actual),
        "hallucinated_instances": sum(actual),
        "accuracy": accuracy,
        "logit_accuracy": sum(
            prediction == label
            for prediction, label in zip(logit_predicted, actual)
        )
        / total,
        "balanced_accuracy": balanced_accuracy,
        "hallucination_precision": precision,
        "hallucination_recall": recall,
        "hallucination_f1": f1,
        "specificity": specificity,
        "mcc": mcc,
        "confusion_matrix": {
            "true_hallucinated": tp,
            "false_hallucinated": fp,
            "true_grounded": tn,
            "false_grounded": fn,
        },
        "predicted_hallucination_rate": sum(predicted) / total,
        "logit_predicted_hallucination_rate": sum(logit_predicted) / total,
        "output_parseable_rate": sum(output_parseable) / total,
        "generated_prediction_fallbacks": sum(
            row["prediction_source"] == "logit_fallback" for row in rows
        ),
        "format_compliance_rate": sum(format_valid) / total,
        "parsed_logit_agreement_rate": (
            sum(valid_agreements) / len(valid_agreements) if valid_agreements else None
        ),
        "brier_score": brier_score,
        "negative_log_likelihood": negative_log_likelihood,
        "roc_auc": _roc_auc(actual, probabilities),
        "expected_calibration_error": _expected_calibration_error(confidence, correct),
        "mean_label_probability_mass": _mean(
            [float(row["label_probability_mass"]) for row in rows]
        ),
        "mean_confidence_correct": _mean(
            [conf for conf, is_correct in zip(confidence, correct) if is_correct]
        ),
        "mean_confidence_wrong": _mean(
            [conf for conf, is_correct in zip(confidence, correct) if not is_correct]
        ),
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_errors": high_confidence_errors,
        "high_confidence_error_rate": high_confidence_errors / total,
        "mean_hallucination_probability_grounded": _mean(
            [probability for probability, label in zip(probabilities, actual) if not label]
        ),
        "mean_hallucination_probability_hallucinated": _mean(
            [probability for probability, label in zip(probabilities, actual) if label]
        ),
        "samples_with_translation_warnings": len(
            {row["sample_index"] for row in rows if row["translation_warnings"]}
        ),
        "all_generated_samples_with_translation_warnings": (
            all_translation_warning_samples
        ),
        "candidate_length_diagnostics": length_diagnostics,
        **pairwise,
    }


def build_paired_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)

    audit_rows: list[dict[str, Any]] = []
    for sample_id, group in grouped.items():
        grounded = next(
            (row for row in group if row["candidate_type"] == "grounded"),
            None,
        )
        hallucinated = next(
            (row for row in group if row["candidate_type"] == "hallucinated"),
            None,
        )
        if grounded is None or hallucinated is None:
            continue
        grounded_probability = float(grounded["hallucination_probability"])
        hallucinated_probability = float(hallucinated["hallucination_probability"])
        grounded_length = int(
            grounded.get(
                "candidate_character_length",
                _content_character_length(grounded["candidate_answer"]),
            )
        )
        hallucinated_length = int(
            hallucinated.get(
                "candidate_character_length",
                _content_character_length(hallucinated["candidate_answer"]),
            )
        )
        audit_rows.append(
            {
                "sample_index": grounded["sample_index"],
                "sample_id": sample_id,
                "excluded_from_primary_metrics": bool(
                    grounded.get("excluded_from_primary_metrics", False)
                ),
                "source_review_reason": grounded.get("source_review_reason"),
                "model": grounded["model"],
                "language": grounded["language"],
                "question": grounded["question"],
                "knowledge": grounded["knowledge"],
                "grounded_candidate": grounded["candidate_answer"],
                "grounded_candidate_character_length": grounded_length,
                "grounded_prediction": grounded["predicted_label"],
                "grounded_generation": grounded["generation"],
                "grounded_hallucination_probability": grounded_probability,
                "hallucinated_candidate": hallucinated["candidate_answer"],
                "hallucinated_candidate_character_length": hallucinated_length,
                "hallucinated_to_grounded_length_ratio": (
                    hallucinated_length / grounded_length
                    if grounded_length
                    else None
                ),
                "hallucinated_prediction": hallucinated["predicted_label"],
                "hallucinated_generation": hallucinated["generation"],
                "hallucinated_hallucination_probability": hallucinated_probability,
                "pairwise_probability_margin": (
                    hallucinated_probability - grounded_probability
                ),
                "pairwise_rank_correct": (
                    hallucinated_probability > grounded_probability
                ),
                "both_candidates_classified_correctly": (
                    grounded["predicted_label"] == "grounded"
                    and hallucinated["predicted_label"] == "hallucinated"
                ),
                "translation_warnings": "|".join(
                    str(warning)
                    for warning in grounded.get("translation_warnings", [])
                ),
            }
        )
    return sorted(audit_rows, key=lambda row: int(row["sample_index"]))


def _length_diagnostics(
    rows: list[dict[str, Any]],
    max_relative_difference: float,
) -> dict[str, Any]:
    lengths = [
        int(
            row.get(
                "candidate_character_length",
                _content_character_length(row["candidate_answer"]),
            )
        )
        for row in rows
    ]
    actual = [row["actual_label"] == "hallucinated" for row in rows]
    probabilities = [float(row["hallucination_probability"]) for row in rows]
    grounded_lengths = [
        length for length, label in zip(lengths, actual) if not label
    ]
    hallucinated_lengths = [
        length for length, label in zip(lengths, actual) if label
    ]
    grounded_mean = _mean([float(length) for length in grounded_lengths])
    hallucinated_mean = _mean(
        [float(length) for length in hallucinated_lengths]
    )
    return {
        "unit": "unicode_alphanumeric_characters",
        "mean_grounded": grounded_mean,
        "mean_hallucinated": hallucinated_mean,
        "hallucinated_to_grounded_ratio": (
            hallucinated_mean / grounded_mean
            if grounded_mean and hallucinated_mean is not None
            else None
        ),
        "length_only_roc_auc": _roc_auc(
            actual,
            [float(length) for length in lengths],
        ),
        "length_probability_correlation": _pearson_correlation(
            [float(length) for length in lengths],
            probabilities,
        ),
        "matched_subset": _length_matched_metrics(
            rows,
            max_relative_difference=max_relative_difference,
        ),
    }


def _length_matched_metrics(
    rows: list[dict[str, Any]],
    max_relative_difference: float,
) -> dict[str, Any]:
    if max_relative_difference < 0:
        raise ValueError("max_relative_difference must be non-negative")
    grounded = [
        row for row in rows if row["actual_label"] == "grounded"
    ]
    hallucinated = [
        row for row in rows if row["actual_label"] == "hallucinated"
    ]
    candidates: list[tuple[float, int, int]] = []
    for grounded_index, grounded_row in enumerate(grounded):
        grounded_length = _row_candidate_length(grounded_row)
        for hallucinated_index, hallucinated_row in enumerate(hallucinated):
            hallucinated_length = _row_candidate_length(hallucinated_row)
            denominator = max(grounded_length, hallucinated_length, 1)
            relative_difference = (
                abs(grounded_length - hallucinated_length) / denominator
            )
            candidates.append(
                (
                    relative_difference,
                    grounded_index,
                    hallucinated_index,
                )
            )

    used_grounded: set[int] = set()
    used_hallucinated: set[int] = set()
    matched_rows: list[dict[str, Any]] = []
    differences: list[float] = []
    for difference, grounded_index, hallucinated_index in sorted(candidates):
        if difference > max_relative_difference:
            break
        if (
            grounded_index in used_grounded
            or hallucinated_index in used_hallucinated
        ):
            continue
        used_grounded.add(grounded_index)
        used_hallucinated.add(hallucinated_index)
        matched_rows.extend(
            (grounded[grounded_index], hallucinated[hallucinated_index])
        )
        differences.append(difference)

    matched_per_class = len(differences)
    if not matched_rows:
        return {
            "max_relative_difference": max_relative_difference,
            "matched_per_class": 0,
            "instance_coverage": 0.0,
            "mean_relative_length_difference": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "roc_auc": None,
        }

    actual = [
        row["actual_label"] == "hallucinated" for row in matched_rows
    ]
    predicted = [
        row["predicted_label"] == "hallucinated" for row in matched_rows
    ]
    probabilities = [
        float(row["hallucination_probability"]) for row in matched_rows
    ]
    true_positive = sum(
        label and prediction
        for label, prediction in zip(actual, predicted)
    )
    true_negative = sum(
        not label and not prediction
        for label, prediction in zip(actual, predicted)
    )
    recall = _safe_div(true_positive, matched_per_class)
    specificity = _safe_div(true_negative, matched_per_class)
    return {
        "max_relative_difference": max_relative_difference,
        "matched_per_class": matched_per_class,
        "instance_coverage": len(matched_rows) / len(rows),
        "mean_relative_length_difference": _mean(differences),
        "accuracy": (true_positive + true_negative) / len(matched_rows),
        "balanced_accuracy": (recall + specificity) / 2,
        "roc_auc": _roc_auc(actual, probabilities),
    }


def _row_candidate_length(row: dict[str, Any]) -> int:
    return int(
        row.get(
            "candidate_character_length",
            _content_character_length(row["candidate_answer"]),
        )
    )


def _content_character_length(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _pearson_correlation(
    left: list[float],
    right: list[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
    )
    right_scale = math.sqrt(
        sum((value - right_mean) ** 2 for value in right)
    )
    denominator = left_scale * right_scale
    return numerator / denominator if denominator else None


def _pairwise_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)

    margins: list[float] = []
    ranking_correct = 0
    ranking_ties = 0
    both_correct = 0
    for group in grouped.values():
        grounded = next(
            (row for row in group if row["candidate_type"] == "grounded"),
            None,
        )
        hallucinated = next(
            (row for row in group if row["candidate_type"] == "hallucinated"),
            None,
        )
        if grounded is None or hallucinated is None:
            continue
        margin = (
            float(hallucinated["hallucination_probability"])
            - float(grounded["hallucination_probability"])
        )
        margins.append(margin)
        if margin > 0:
            ranking_correct += 1
        elif margin == 0:
            ranking_ties += 1
        if (
            grounded["predicted_label"] == "grounded"
            and hallucinated["predicted_label"] == "hallucinated"
        ):
            both_correct += 1

    num_pairs = len(margins)
    return {
        "num_complete_pairs": num_pairs,
        "pairwise_ranking_correct": ranking_correct,
        "pairwise_ranking_ties": ranking_ties,
        "pairwise_ranking_accuracy": (
            (ranking_correct + 0.5 * ranking_ties) / num_pairs
            if num_pairs
            else None
        ),
        "pairwise_classification_accuracy": (
            both_correct / num_pairs if num_pairs else None
        ),
        "mean_pairwise_probability_margin": _mean(margins),
    }


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None

    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            int(label) for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _expected_calibration_error(
    confidence: list[float],
    correct: list[bool],
    num_bins: int = 10,
) -> float:
    total = len(confidence)
    ece = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        members = [
            index
            for index, value in enumerate(confidence)
            if value >= lower and (value <= upper if bin_index == num_bins - 1 else value < upper)
        ]
        if not members:
            continue
        bin_confidence = sum(confidence[index] for index in members) / len(members)
        bin_accuracy = sum(correct[index] for index in members) / len(members)
        ece += (len(members) / total) * abs(bin_accuracy - bin_confidence)
    return ece
