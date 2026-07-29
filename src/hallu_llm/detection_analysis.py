from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Iterable

import numpy as np


ANALYSIS_SCHEMA_VERSION = 1

METHOD_ORDER = (
    "generated_decision",
    "probability_0_5",
    "pooled_threshold",
    "reference_threshold",
    "language_threshold",
    "language_temperature",
    "pooled_platt",
    "reference_platt",
    "language_platt",
    "length_only_logistic",
    "model_and_length_logistic",
)

BOOTSTRAP_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "hallucination_recall",
    "specificity",
    "roc_auc",
    "brier_score",
    "negative_log_likelihood",
    "expected_calibration_error",
    "pairwise_ranking_accuracy",
    "pairwise_classification_accuracy",
)

COMPARISON_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "hallucination_recall",
    "specificity",
    "roc_auc",
    "brier_score",
    "negative_log_likelihood",
    "expected_calibration_error",
    "predicted_hallucination_rate",
    "pairwise_ranking_accuracy",
    "pairwise_classification_accuracy",
)


def primary_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if not bool(row.get("excluded_from_primary_metrics", False))
    ]
    validate_prediction_rows(primary)
    return sorted(
        primary,
        key=lambda row: (
            str(row["sample_id"]),
            str(row["candidate_type"]),
        ),
    )


def validate_prediction_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Prediction rows are empty.")
    required = {
        "sample_id",
        "candidate_type",
        "actual_label",
        "predicted_label",
        "hallucination_probability",
        "candidate_answer",
    }
    seen: set[tuple[str, str]] = set()
    candidate_types_by_sample: dict[str, set[str]] = {}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(
                "Prediction row is missing required fields: "
                + ", ".join(sorted(missing))
            )
        sample_id = str(row["sample_id"])
        candidate_type = str(row["candidate_type"])
        if candidate_type not in {"grounded", "hallucinated"}:
            raise ValueError(
                f"Unexpected candidate_type {candidate_type!r} for {sample_id}."
            )
        expected_label = candidate_type
        if str(row["actual_label"]) != expected_label:
            raise ValueError(
                f"Candidate and actual labels disagree for {sample_id}."
            )
        key = (sample_id, candidate_type)
        if key in seen:
            raise ValueError(f"Duplicate prediction row for {key}.")
        seen.add(key)
        candidate_types_by_sample.setdefault(sample_id, set()).add(
            candidate_type
        )
        probability = float(row["hallucination_probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Invalid hallucination probability for {key}: {probability}"
            )
    incomplete = sorted(
        sample_id
        for sample_id, candidate_types in candidate_types_by_sample.items()
        if candidate_types != {"grounded", "hallucinated"}
    )
    if incomplete:
        raise ValueError(
            "Every source sample must have one grounded and one hallucinated "
            f"candidate; incomplete samples: {incomplete[:5]}"
        )


def assign_sample_splits(
    sample_ids: Iterable[str],
    dev_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    unique_ids = sorted({str(sample_id) for sample_id in sample_ids})
    if len(unique_ids) < 2:
        raise ValueError("At least two source samples are required for a split.")
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between zero and one.")

    ranked = sorted(
        (
            hashlib.sha256(
                f"{seed}:{sample_id}".encode("utf-8")
            ).hexdigest(),
            sample_id,
        )
        for sample_id in unique_ids
    )
    num_dev = max(1, min(len(ranked) - 1, round(len(ranked) * dev_fraction)))
    dev_ids = {sample_id for _, sample_id in ranked[:num_dev]}
    rank_by_id = {
        sample_id: rank
        for rank, (_, sample_id) in enumerate(ranked, start=1)
    }
    hash_by_id = {sample_id: digest for digest, sample_id in ranked}
    return [
        {
            "sample_id": sample_id,
            "split": "development" if sample_id in dev_ids else "test",
            "split_rank": rank_by_id[sample_id],
            "split_hash": hash_by_id[sample_id],
            "seed": seed,
            "requested_dev_fraction": dev_fraction,
        }
        for sample_id in unique_ids
    ]


def fit_analysis_methods(
    development_rows_by_language: dict[str, list[dict[str, Any]]],
    reference_language: str,
    logistic_l2: float = 1e-3,
) -> dict[str, dict[str, dict[str, Any]]]:
    if reference_language not in development_rows_by_language:
        raise ValueError(
            f"Reference language {reference_language!r} is absent from predictions."
        )
    for rows in development_rows_by_language.values():
        validate_prediction_rows(rows)

    pooled_rows = [
        row
        for language in sorted(development_rows_by_language)
        for row in development_rows_by_language[language]
    ]
    reference_rows = development_rows_by_language[reference_language]
    pooled_threshold = fit_balanced_accuracy_threshold(pooled_rows)
    reference_threshold = fit_balanced_accuracy_threshold(reference_rows)
    pooled_platt = fit_logistic_calibrator(
        pooled_rows,
        feature_names=("model_logit",),
        l2=logistic_l2,
    )
    reference_platt = fit_logistic_calibrator(
        reference_rows,
        feature_names=("model_logit",),
        l2=logistic_l2,
    )

    methods: dict[str, dict[str, dict[str, Any]]] = {}
    for language, rows in sorted(development_rows_by_language.items()):
        methods[language] = {
            "generated_decision": {
                "kind": "generated",
                "threshold": None,
                "fit_scope": "none",
                "fit_language": None,
                "num_development_instances": 0,
            },
            "probability_0_5": {
                "kind": "identity",
                "threshold": 0.5,
                "fit_scope": "none",
                "fit_language": None,
                "num_development_instances": 0,
            },
            "pooled_threshold": {
                **pooled_threshold,
                "fit_scope": "pooled_languages",
                "fit_language": None,
            },
            "reference_threshold": {
                **reference_threshold,
                "fit_scope": "reference_language",
                "fit_language": reference_language,
            },
            "language_threshold": {
                **fit_balanced_accuracy_threshold(rows),
                "fit_scope": "language",
                "fit_language": language,
            },
            "language_temperature": {
                **fit_temperature_scaler(rows),
                "fit_scope": "language",
                "fit_language": language,
            },
            "pooled_platt": {
                **pooled_platt,
                "fit_scope": "pooled_languages",
                "fit_language": None,
            },
            "reference_platt": {
                **reference_platt,
                "fit_scope": "reference_language",
                "fit_language": reference_language,
            },
            "language_platt": {
                **fit_logistic_calibrator(
                    rows,
                    feature_names=("model_logit",),
                    l2=logistic_l2,
                ),
                "fit_scope": "language",
                "fit_language": language,
            },
            "length_only_logistic": {
                **fit_logistic_calibrator(
                    rows,
                    feature_names=("log_candidate_length",),
                    l2=logistic_l2,
                ),
                "fit_scope": "language",
                "fit_language": language,
            },
            "model_and_length_logistic": {
                **fit_logistic_calibrator(
                    rows,
                    feature_names=(
                        "model_logit",
                        "log_candidate_length",
                    ),
                    l2=logistic_l2,
                ),
                "fit_scope": "language",
                "fit_language": language,
            },
        }
    return methods


def fit_balanced_accuracy_threshold(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = [float(row["hallucination_probability"]) for row in rows]
    labels = [row["actual_label"] == "hallucinated" for row in rows]
    unique_scores = sorted(set(scores))
    candidates = {0.0, 0.5, 1.0}
    candidates.update(unique_scores)
    candidates.update(
        (left + right) / 2
        for left, right in zip(unique_scores, unique_scores[1:])
    )
    best: tuple[tuple[float, float, float], float, dict[str, float]] | None = None
    for threshold in sorted(candidates):
        predicted = [score >= threshold for score in scores]
        metrics = _classification_metrics(labels, predicted)
        key = (
            metrics["balanced_accuracy"],
            metrics["accuracy"],
            -abs(threshold - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    assert best is not None
    return {
        "kind": "identity",
        "threshold": best[1],
        "fit_objective": "balanced_accuracy",
        "fit_balanced_accuracy": best[2]["balanced_accuracy"],
        "fit_accuracy": best[2]["accuracy"],
        "num_development_instances": len(rows),
    }


def fit_temperature_scaler(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    logits = np.asarray(
        [_probability_logit(row["hallucination_probability"]) for row in rows],
        dtype=float,
    )
    labels = np.asarray(
        [row["actual_label"] == "hallucinated" for row in rows],
        dtype=float,
    )

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        probabilities = _sigmoid_array(logits / temperature)
        return _binary_nll_array(labels, probabilities)

    left = math.log(0.05)
    right = math.log(20.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(80):
        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    log_temperature = (left + right) / 2.0
    temperature = math.exp(log_temperature)
    return {
        "kind": "temperature",
        "temperature": temperature,
        "threshold": 0.5,
        "fit_objective": "negative_log_likelihood",
        "fit_nll_before": objective(0.0),
        "fit_nll_after": objective(log_temperature),
        "num_development_instances": len(rows),
    }


def fit_logistic_calibrator(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    l2: float = 1e-3,
    max_iterations: int = 100,
) -> dict[str, Any]:
    if l2 < 0:
        raise ValueError("Logistic L2 regularization cannot be negative.")
    features = np.asarray(
        [
            [_feature_value(row, feature_name) for feature_name in feature_names]
            for row in rows
        ],
        dtype=float,
    )
    labels = np.asarray(
        [row["actual_label"] == "hallucinated" for row in rows],
        dtype=float,
    )
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    normalized = (features - means) / scales
    design = np.column_stack((normalized, np.ones(len(rows))))
    coefficients = np.zeros(design.shape[1], dtype=float)
    prevalence = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    coefficients[-1] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.ones_like(coefficients)
    penalty[-1] = 0.0

    def objective(values: np.ndarray) -> float:
        probabilities = _sigmoid_array(design @ values)
        penalty_value = 0.5 * l2 * float(
            np.sum((values * penalty) ** 2)
        )
        return _binary_nll_array(labels, probabilities) + (
            penalty_value / len(rows)
        )

    converged = False
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        probabilities = _sigmoid_array(design @ coefficients)
        gradient = design.T @ (probabilities - labels)
        gradient += l2 * coefficients * penalty
        weights = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * weights[:, None])
        hessian += np.diag(l2 * penalty)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if float(np.linalg.norm(step)) < 1e-8:
            converged = True
            break
        current_objective = objective(coefficients)
        step_scale = 1.0
        while step_scale >= 1e-6:
            candidate = coefficients - step_scale * step
            if objective(candidate) <= current_objective:
                coefficients = candidate
                break
            step_scale *= 0.5
        else:
            break
        if float(np.linalg.norm(step_scale * step)) < 1e-8:
            converged = True
            break

    calibrated = _sigmoid_array(design @ coefficients)
    raw_probabilities = np.asarray(
        [float(row["hallucination_probability"]) for row in rows],
        dtype=float,
    )
    return {
        "kind": "logistic",
        "feature_names": list(feature_names),
        "feature_means": [float(value) for value in means],
        "feature_scales": [float(value) for value in scales],
        "feature_coefficients": [
            float(value) for value in coefficients[:-1]
        ],
        "intercept": float(coefficients[-1]),
        "threshold": 0.5,
        "l2": l2,
        "converged": converged,
        "iterations": iterations,
        "fit_objective": "negative_log_likelihood",
        "fit_nll_before": _binary_nll_array(labels, raw_probabilities),
        "fit_nll_after": _binary_nll_array(labels, calibrated),
        "num_development_instances": len(rows),
    }


def score_prediction_rows(
    rows: list[dict[str, Any]],
    method: dict[str, Any],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    probabilities = apply_probability_method(rows, method)
    for row, probability in zip(rows, probabilities):
        if method["kind"] == "generated":
            predicted = row["predicted_label"] == "hallucinated"
        else:
            predicted = probability >= float(method["threshold"])
        scored.append(
            {
                "sample_id": str(row["sample_id"]),
                "sample_index": int(row.get("sample_index", 0)),
                "candidate_type": str(row["candidate_type"]),
                "actual": row["actual_label"] == "hallucinated",
                "predicted": predicted,
                "probability": probability,
                "candidate_length": int(
                    row.get(
                        "candidate_character_length",
                        _content_character_length(
                            str(row["candidate_answer"])
                        ),
                    )
                ),
            }
        )
    return scored


def apply_probability_method(
    rows: list[dict[str, Any]],
    method: dict[str, Any],
) -> list[float]:
    kind = str(method["kind"])
    if kind in {"generated", "identity"}:
        return [
            float(row["hallucination_probability"])
            for row in rows
        ]
    if kind == "temperature":
        temperature = float(method["temperature"])
        return [
            _sigmoid(
                _probability_logit(row["hallucination_probability"])
                / temperature
            )
            for row in rows
        ]
    if kind == "logistic":
        feature_names = tuple(str(name) for name in method["feature_names"])
        means = [float(value) for value in method["feature_means"]]
        scales = [float(value) for value in method["feature_scales"]]
        coefficients = [
            float(value) for value in method["feature_coefficients"]
        ]
        intercept = float(method["intercept"])
        probabilities: list[float] = []
        for row in rows:
            normalized = [
                (_feature_value(row, name) - mean) / scale
                for name, mean, scale in zip(feature_names, means, scales)
            ]
            logit = intercept + sum(
                coefficient * value
                for coefficient, value in zip(coefficients, normalized)
            )
            probabilities.append(_sigmoid(logit))
        return probabilities
    raise ValueError(f"Unsupported analysis method kind: {kind}")


def evaluate_scored_rows(
    scored_rows: list[dict[str, Any]],
    ece_bins: int = 10,
    high_confidence_threshold: float = 0.75,
    length_match_max_relative_difference: float = 0.25,
) -> dict[str, Any]:
    groups = _group_scored_rows(scored_rows)
    return _metrics_from_groups(
        list(groups.values()),
        ece_bins=ece_bins,
        high_confidence_threshold=high_confidence_threshold,
        length_match_max_relative_difference=(
            length_match_max_relative_difference
        ),
        include_length_matching=True,
    )


def bootstrap_metric_intervals(
    scored_rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
    ece_bins: int = 10,
    high_confidence_threshold: float = 0.75,
) -> dict[str, dict[str, float]]:
    if replicates < 0:
        raise ValueError("Bootstrap replicates cannot be negative.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")
    if replicates == 0:
        return {}
    groups = list(_group_scored_rows(scored_rows).values())
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        metric: [] for metric in BOOTSTRAP_METRICS
    }
    for _ in range(replicates):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        metrics = _metrics_from_groups(
            draw,
            ece_bins=ece_bins,
            high_confidence_threshold=high_confidence_threshold,
            length_match_max_relative_difference=0.25,
            include_length_matching=False,
        )
        for metric in BOOTSTRAP_METRICS:
            value = metrics.get(metric)
            if value is not None and math.isfinite(float(value)):
                samples[metric].append(float(value))
    alpha = 1.0 - confidence_level
    return {
        metric: {
            "low": float(np.quantile(values, alpha / 2.0)),
            "high": float(np.quantile(values, 1.0 - alpha / 2.0)),
        }
        for metric, values in samples.items()
        if values
    }


def compare_scored_languages(
    reference_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
    ece_bins: int = 10,
    high_confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    reference_groups = _group_scored_rows(reference_rows)
    target_groups = _group_scored_rows(target_rows)
    common_ids = sorted(set(reference_groups) & set(target_groups))
    if not common_ids:
        raise ValueError("Languages have no common test samples.")
    if set(reference_groups) != set(target_groups):
        raise ValueError(
            "Cross-language comparison requires identical test sample IDs."
        )

    reference_group_list = [reference_groups[sample_id] for sample_id in common_ids]
    target_group_list = [target_groups[sample_id] for sample_id in common_ids]
    reference_metrics = _metrics_from_groups(
        reference_group_list,
        ece_bins=ece_bins,
        high_confidence_threshold=high_confidence_threshold,
        length_match_max_relative_difference=0.25,
        include_length_matching=False,
    )
    target_metrics = _metrics_from_groups(
        target_group_list,
        ece_bins=ece_bins,
        high_confidence_threshold=high_confidence_threshold,
        length_match_max_relative_difference=0.25,
        include_length_matching=False,
    )
    result: dict[str, Any] = {
        "num_common_test_samples": len(common_ids),
        "num_common_test_instances": 2 * len(common_ids),
    }
    for metric in COMPARISON_METRICS:
        reference_value = reference_metrics.get(metric)
        target_value = target_metrics.get(metric)
        result[f"reference_{metric}"] = reference_value
        result[f"target_{metric}"] = target_value
        result[f"delta_{metric}"] = (
            float(target_value) - float(reference_value)
            if reference_value is not None and target_value is not None
            else None
        )

    reference_by_key = _scored_by_key(reference_rows)
    target_by_key = _scored_by_key(target_rows)
    keys = sorted(reference_by_key)
    correctness_reference_only = 0
    correctness_target_only = 0
    prediction_reference_only = 0
    prediction_target_only = 0
    for key in keys:
        reference = reference_by_key[key]
        target = target_by_key[key]
        reference_correct = reference["actual"] == reference["predicted"]
        target_correct = target["actual"] == target["predicted"]
        correctness_reference_only += int(
            reference_correct and not target_correct
        )
        correctness_target_only += int(
            target_correct and not reference_correct
        )
        prediction_reference_only += int(
            reference["predicted"] and not target["predicted"]
        )
        prediction_target_only += int(
            target["predicted"] and not reference["predicted"]
        )
    result.update(
        {
            "accuracy_reference_only_correct": correctness_reference_only,
            "accuracy_target_only_correct": correctness_target_only,
            "accuracy_mcnemar_exact_p": exact_mcnemar_p(
                correctness_reference_only,
                correctness_target_only,
            ),
            "prediction_reference_only_hallucinated": (
                prediction_reference_only
            ),
            "prediction_target_only_hallucinated": prediction_target_only,
            "prediction_shift_mcnemar_exact_p": exact_mcnemar_p(
                prediction_reference_only,
                prediction_target_only,
            ),
        }
    )

    reference_pair_correct = {
        sample_id: _pairwise_rank_correct(reference_groups[sample_id])
        for sample_id in common_ids
    }
    target_pair_correct = {
        sample_id: _pairwise_rank_correct(target_groups[sample_id])
        for sample_id in common_ids
    }
    pair_reference_only = sum(
        reference_pair_correct[sample_id]
        and not target_pair_correct[sample_id]
        for sample_id in common_ids
    )
    pair_target_only = sum(
        target_pair_correct[sample_id]
        and not reference_pair_correct[sample_id]
        for sample_id in common_ids
    )
    result.update(
        {
            "pairwise_reference_only_correct": pair_reference_only,
            "pairwise_target_only_correct": pair_target_only,
            "pairwise_mcnemar_exact_p": exact_mcnemar_p(
                pair_reference_only,
                pair_target_only,
            ),
        }
    )

    if replicates > 0:
        rng = random.Random(seed)
        differences: dict[str, list[float]] = {
            metric: [] for metric in COMPARISON_METRICS
        }
        for _ in range(replicates):
            indexes = [
                rng.randrange(len(common_ids)) for _ in common_ids
            ]
            reference_draw = [reference_group_list[index] for index in indexes]
            target_draw = [target_group_list[index] for index in indexes]
            reference_bootstrap = _metrics_from_groups(
                reference_draw,
                ece_bins=ece_bins,
                high_confidence_threshold=high_confidence_threshold,
                length_match_max_relative_difference=0.25,
                include_length_matching=False,
            )
            target_bootstrap = _metrics_from_groups(
                target_draw,
                ece_bins=ece_bins,
                high_confidence_threshold=high_confidence_threshold,
                length_match_max_relative_difference=0.25,
                include_length_matching=False,
            )
            for metric in COMPARISON_METRICS:
                reference_value = reference_bootstrap.get(metric)
                target_value = target_bootstrap.get(metric)
                if reference_value is None or target_value is None:
                    continue
                difference = float(target_value) - float(reference_value)
                if math.isfinite(difference):
                    differences[metric].append(difference)
        alpha = 1.0 - confidence_level
        for metric, values in differences.items():
            if not values:
                continue
            result[f"delta_{metric}_ci_low"] = float(
                np.quantile(values, alpha / 2.0)
            )
            result[f"delta_{metric}_ci_high"] = float(
                np.quantile(values, 1.0 - alpha / 2.0)
            )
    return result


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def holm_adjusted_p_values(values: list[float]) -> list[float]:
    if not values:
        return []
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid p-value for Holm adjustment: {value}")
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [0.0] * len(values)
    running_maximum = 0.0
    num_tests = len(values)
    for rank, (original_index, value) in enumerate(ranked):
        candidate = min(1.0, (num_tests - rank) * value)
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return adjusted


def _metrics_from_groups(
    groups: list[list[dict[str, Any]]],
    ece_bins: int,
    high_confidence_threshold: float,
    length_match_max_relative_difference: float,
    include_length_matching: bool,
) -> dict[str, Any]:
    rows = [row for group in groups for row in group]
    actual = [bool(row["actual"]) for row in rows]
    predicted = [bool(row["predicted"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    classification = _classification_metrics(actual, predicted)
    correct = [
        label == prediction
        for label, prediction in zip(actual, predicted)
    ]
    confidence = [
        max(probability, 1.0 - probability)
        for probability in probabilities
    ]
    epsilon = 1e-12
    brier = sum(
        (probability - int(label)) ** 2
        for probability, label in zip(probabilities, actual)
    ) / len(rows)
    nll = -sum(
        int(label) * math.log(max(probability, epsilon))
        + (1 - int(label))
        * math.log(max(1.0 - probability, epsilon))
        for probability, label in zip(probabilities, actual)
    ) / len(rows)
    pairwise_margins: list[float] = []
    pairwise_correct = 0.0
    pairwise_both_correct = 0
    for group in groups:
        grounded = next(
            row for row in group if row["candidate_type"] == "grounded"
        )
        hallucinated = next(
            row for row in group if row["candidate_type"] == "hallucinated"
        )
        margin = (
            float(hallucinated["probability"])
            - float(grounded["probability"])
        )
        pairwise_margins.append(margin)
        pairwise_correct += 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        pairwise_both_correct += int(
            not grounded["predicted"] and hallucinated["predicted"]
        )

    lengths = [int(row["candidate_length"]) for row in rows]
    grounded_lengths = [
        length for length, label in zip(lengths, actual) if not label
    ]
    hallucinated_lengths = [
        length for length, label in zip(lengths, actual) if label
    ]
    mean_grounded_length = _mean(grounded_lengths)
    mean_hallucinated_length = _mean(hallucinated_lengths)
    high_confidence_errors = sum(
        not is_correct and certainty >= high_confidence_threshold
        for is_correct, certainty in zip(correct, confidence)
    )
    metrics: dict[str, Any] = {
        "num_source_samples": len(groups),
        "num_instances": len(rows),
        **classification,
        "predicted_hallucination_rate": sum(predicted) / len(rows),
        "brier_score": brier,
        "negative_log_likelihood": nll,
        "roc_auc": _roc_auc(actual, probabilities),
        "expected_calibration_error": _expected_calibration_error(
            confidence,
            correct,
            num_bins=ece_bins,
        ),
        "mean_confidence_correct": _mean(
            [
                certainty
                for certainty, is_correct in zip(confidence, correct)
                if is_correct
            ]
        ),
        "mean_confidence_wrong": _mean(
            [
                certainty
                for certainty, is_correct in zip(confidence, correct)
                if not is_correct
            ]
        ),
        "high_confidence_errors": high_confidence_errors,
        "high_confidence_error_rate": high_confidence_errors / len(rows),
        "mean_hallucination_probability_grounded": _mean(
            [
                probability
                for probability, label in zip(probabilities, actual)
                if not label
            ]
        ),
        "mean_hallucination_probability_hallucinated": _mean(
            [
                probability
                for probability, label in zip(probabilities, actual)
                if label
            ]
        ),
        "pairwise_ranking_accuracy": pairwise_correct / len(groups),
        "pairwise_classification_accuracy": (
            pairwise_both_correct / len(groups)
        ),
        "mean_pairwise_probability_margin": _mean(pairwise_margins),
        "mean_grounded_candidate_length": mean_grounded_length,
        "mean_hallucinated_candidate_length": mean_hallucinated_length,
        "hallucinated_to_grounded_length_ratio": (
            mean_hallucinated_length / mean_grounded_length
            if mean_grounded_length
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
    }
    if include_length_matching:
        metrics.update(
            _length_matched_metrics(
                rows,
                max_relative_difference=(
                    length_match_max_relative_difference
                ),
            )
        )
    return metrics


def _classification_metrics(
    actual: list[bool],
    predicted: list[bool],
) -> dict[str, float | dict[str, int]]:
    true_positive = sum(
        label and prediction
        for label, prediction in zip(actual, predicted)
    )
    true_negative = sum(
        not label and not prediction
        for label, prediction in zip(actual, predicted)
    )
    false_positive = sum(
        not label and prediction
        for label, prediction in zip(actual, predicted)
    )
    false_negative = sum(
        label and not prediction
        for label, prediction in zip(actual, predicted)
    )
    total = len(actual)
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    specificity = _safe_div(true_negative, true_negative + false_positive)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (
            true_positive * true_negative
            - false_positive * false_negative
        )
        / denominator
        if denominator
        else 0.0
    )
    return {
        "accuracy": (true_positive + true_negative) / total,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "hallucination_precision": precision,
        "hallucination_recall": recall,
        "hallucination_f1": f1,
        "specificity": specificity,
        "mcc": mcc,
        "confusion_matrix": {
            "true_hallucinated": true_positive,
            "false_hallucinated": false_positive,
            "true_grounded": true_negative,
            "false_grounded": false_negative,
        },
    }


def _length_matched_metrics(
    rows: list[dict[str, Any]],
    max_relative_difference: float,
) -> dict[str, Any]:
    grounded = [row for row in rows if not row["actual"]]
    hallucinated = [row for row in rows if row["actual"]]
    candidates: list[tuple[float, int, int]] = []
    for grounded_index, grounded_row in enumerate(grounded):
        for hallucinated_index, hallucinated_row in enumerate(hallucinated):
            grounded_length = int(grounded_row["candidate_length"])
            hallucinated_length = int(
                hallucinated_row["candidate_length"]
            )
            denominator = max(grounded_length, hallucinated_length, 1)
            difference = (
                abs(grounded_length - hallucinated_length) / denominator
            )
            candidates.append(
                (difference, grounded_index, hallucinated_index)
            )
    used_grounded: set[int] = set()
    used_hallucinated: set[int] = set()
    matched: list[dict[str, Any]] = []
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
        matched.extend(
            (grounded[grounded_index], hallucinated[hallucinated_index])
        )
        differences.append(difference)
    if not matched:
        return {
            "length_matched_max_relative_difference": (
                max_relative_difference
            ),
            "length_matched_per_class": 0,
            "length_matched_instance_coverage": 0.0,
            "length_matched_mean_relative_difference": None,
            "length_matched_accuracy": None,
            "length_matched_balanced_accuracy": None,
            "length_matched_roc_auc": None,
        }
    classification = _classification_metrics(
        [row["actual"] for row in matched],
        [row["predicted"] for row in matched],
    )
    return {
        "length_matched_max_relative_difference": max_relative_difference,
        "length_matched_per_class": len(differences),
        "length_matched_instance_coverage": len(matched) / len(rows),
        "length_matched_mean_relative_difference": _mean(differences),
        "length_matched_accuracy": classification["accuracy"],
        "length_matched_balanced_accuracy": classification[
            "balanced_accuracy"
        ],
        "length_matched_roc_auc": _roc_auc(
            [row["actual"] for row in matched],
            [float(row["probability"]) for row in matched],
        ),
    }


def _group_scored_rows(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    for sample_id, group in grouped.items():
        candidate_types = {row["candidate_type"] for row in group}
        if candidate_types != {"grounded", "hallucinated"} or len(group) != 2:
            raise ValueError(
                f"Incomplete scored pair for sample {sample_id}."
            )
    return grouped


def _scored_by_key(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["sample_id"]), str(row["candidate_type"])): row
        for row in rows
    }


def _pairwise_rank_correct(group: list[dict[str, Any]]) -> bool:
    grounded = next(
        row for row in group if row["candidate_type"] == "grounded"
    )
    hallucinated = next(
        row for row in group if row["candidate_type"] == "hallucinated"
    )
    return (
        float(hallucinated["probability"])
        > float(grounded["probability"])
    )


def _feature_value(row: dict[str, Any], feature_name: str) -> float:
    if feature_name == "model_logit":
        return _probability_logit(row["hallucination_probability"])
    if feature_name == "log_candidate_length":
        length = int(
            row.get(
                "candidate_character_length",
                _content_character_length(str(row["candidate_answer"])),
            )
        )
        return math.log1p(length)
    raise ValueError(f"Unknown calibration feature: {feature_name}")


def _probability_logit(value: Any) -> float:
    probability = min(max(float(value), 1e-8), 1.0 - 1e-8)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    value = min(max(value, -40.0), 40.0)
    return 1.0 / (1.0 + math.exp(-value))


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _binary_nll_array(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(
            labels * np.log(clipped)
            + (1.0 - labels) * np.log(1.0 - clipped)
        )
    )


def _content_character_length(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float] | list[int]) -> float | None:
    return sum(values) / len(values) if values else None


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
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            int(label) for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


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


def _expected_calibration_error(
    confidence: list[float],
    correct: list[bool],
    num_bins: int,
) -> float:
    if num_bins <= 0:
        raise ValueError("ECE bins must be positive.")
    total = len(confidence)
    error = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        members = [
            index
            for index, value in enumerate(confidence)
            if value >= lower
            and (
                value <= upper
                if bin_index == num_bins - 1
                else value < upper
            )
        ]
        if not members:
            continue
        mean_confidence = sum(confidence[index] for index in members) / len(
            members
        )
        accuracy = sum(correct[index] for index in members) / len(members)
        error += (len(members) / total) * abs(
            accuracy - mean_confidence
        )
    return error
