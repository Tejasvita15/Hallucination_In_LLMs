from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.detection_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    BOOTSTRAP_METRICS,
    METHOD_ORDER,
    assign_sample_splits,
    bootstrap_metric_intervals,
    compare_scored_languages,
    evaluate_scored_rows,
    fit_analysis_methods,
    holm_adjusted_p_values,
    primary_prediction_rows,
    score_prediction_rows,
)


MAIN_TABLE_METHODS = {
    "generated_decision",
    "probability_0_5",
    "pooled_threshold",
    "reference_threshold",
    "language_threshold",
    "language_temperature",
    "language_platt",
    "length_only_logistic",
    "model_and_length_logistic",
}

METRIC_FIELDS = (
    "num_source_samples",
    "num_instances",
    "accuracy",
    "balanced_accuracy",
    "hallucination_precision",
    "hallucination_recall",
    "hallucination_f1",
    "specificity",
    "mcc",
    "predicted_hallucination_rate",
    "brier_score",
    "negative_log_likelihood",
    "roc_auc",
    "expected_calibration_error",
    "mean_confidence_correct",
    "mean_confidence_wrong",
    "high_confidence_errors",
    "high_confidence_error_rate",
    "mean_hallucination_probability_grounded",
    "mean_hallucination_probability_hallucinated",
    "pairwise_ranking_accuracy",
    "pairwise_classification_accuracy",
    "mean_pairwise_probability_margin",
    "mean_grounded_candidate_length",
    "mean_hallucinated_candidate_length",
    "hallucinated_to_grounded_length_ratio",
    "length_only_roc_auc",
    "length_probability_correlation",
    "length_matched_max_relative_difference",
    "length_matched_per_class",
    "length_matched_instance_coverage",
    "length_matched_mean_relative_difference",
    "length_matched_accuracy",
    "length_matched_balanced_accuracy",
    "length_matched_roc_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze paired cross-lingual HaluEval predictions using a "
            "development/test split, calibration, length controls, and "
            "source-pair bootstrap intervals."
        )
    )
    parser.add_argument(
        "--config",
        help="Optional JSON analysis config; explicit CLI values override it.",
    )
    parser.add_argument(
        "--results-dir",
        help="Detection result directory containing summary_all.json.",
    )
    parser.add_argument(
        "--output-dir",
        help="Analysis output directory (default: <results-dir>/analysis).",
    )
    parser.add_argument(
        "--split-manifest",
        help=(
            "Optional prior split_manifest.csv to freeze development/test "
            "assignments."
        ),
    )
    parser.add_argument("--reference-language")
    parser.add_argument("--dev-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--confidence-level", type=float)
    parser.add_argument("--ece-bins", type=int)
    parser.add_argument("--high-confidence-threshold", type=float)
    parser.add_argument("--length-match-max-relative-difference", type=float)
    parser.add_argument("--logistic-l2", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_json(args.config) if args.config else {}
    settings = _effective_settings(args, config)
    results_dir = Path(settings["results_dir"]).resolve()
    output_dir = Path(
        settings.get("output_dir")
        or results_dir / "analysis"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, prediction_files = _load_prediction_runs(results_dir)
    sample_ids = _validate_common_sample_ids(predictions)
    split_records = _load_or_assign_split(sample_ids, settings)
    split_by_id = {
        row["sample_id"]: row["split"]
        for row in split_records
    }
    development_ids = {
        sample_id
        for sample_id, split in split_by_id.items()
        if split == "development"
    }
    test_ids = set(sample_ids) - development_ids
    print(
        f"Loaded {len(predictions)} model run(s), {len(sample_ids)} primary "
        f"source samples: {len(development_ids)} development / "
        f"{len(test_ids)} test"
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_comparison_rows: list[dict[str, Any]] = []
    all_parameters: dict[str, Any] = {}
    model_languages: dict[str, list[str]] = {}

    for model, rows_by_language in sorted(predictions.items()):
        reference_language = settings["reference_language"]
        if reference_language not in rows_by_language:
            raise ValueError(
                f"Model {model} has no {reference_language!r} reference run."
            )
        model_languages[model] = sorted(rows_by_language)
        development_by_language = {
            language: [
                row
                for row in rows
                if str(row["sample_id"]) in development_ids
            ]
            for language, rows in rows_by_language.items()
        }
        test_by_language = {
            language: [
                row
                for row in rows
                if str(row["sample_id"]) in test_ids
            ]
            for language, rows in rows_by_language.items()
        }
        methods_by_language = fit_analysis_methods(
            development_by_language,
            reference_language=reference_language,
            logistic_l2=settings["logistic_l2"],
        )
        all_parameters[model] = methods_by_language
        scored_by_language_method: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = {}

        for language in sorted(rows_by_language):
            bootstrap_seed = _derived_seed(
                settings["seed"],
                model,
                language,
                "metric-bootstrap",
            )
            for method_name in METHOD_ORDER:
                method = methods_by_language[language][method_name]
                scored = score_prediction_rows(
                    test_by_language[language],
                    method,
                )
                scored_by_language_method[(language, method_name)] = scored
                metrics = evaluate_scored_rows(
                    scored,
                    ece_bins=settings["ece_bins"],
                    high_confidence_threshold=(
                        settings["high_confidence_threshold"]
                    ),
                    length_match_max_relative_difference=(
                        settings[
                            "length_match_max_relative_difference"
                        ]
                    ),
                )
                intervals = bootstrap_metric_intervals(
                    scored,
                    replicates=settings["bootstrap_replicates"],
                    seed=bootstrap_seed,
                    confidence_level=settings["confidence_level"],
                    ece_bins=settings["ece_bins"],
                    high_confidence_threshold=(
                        settings["high_confidence_threshold"]
                    ),
                )
                all_metric_rows.append(
                    _metric_row(
                        model=model,
                        language=language,
                        method_name=method_name,
                        method=method,
                        metrics=metrics,
                        intervals=intervals,
                    )
                )

        for target_language in sorted(rows_by_language):
            if target_language == reference_language:
                continue
            comparison_seed = _derived_seed(
                settings["seed"],
                model,
                reference_language,
                target_language,
                "comparison-bootstrap",
            )
            for method_name in METHOD_ORDER:
                comparison = compare_scored_languages(
                    scored_by_language_method[
                        (reference_language, method_name)
                    ],
                    scored_by_language_method[
                        (target_language, method_name)
                    ],
                    replicates=settings["bootstrap_replicates"],
                    seed=comparison_seed,
                    confidence_level=settings["confidence_level"],
                    ece_bins=settings["ece_bins"],
                    high_confidence_threshold=(
                        settings["high_confidence_threshold"]
                    ),
                )
                all_comparison_rows.append(
                    {
                        "model": model,
                        "method": method_name,
                        "reference_language": reference_language,
                        "target_language": target_language,
                        **comparison,
                    }
                )

    _add_holm_adjustments(all_comparison_rows)
    effective_config = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        **settings,
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "input_prediction_fingerprint": _prediction_fingerprint(
            prediction_files,
            base_dir=results_dir,
        ),
        "confidence_interval_scope": (
            "test_source_pair_bootstrap_conditional_on_fitted_calibrators"
        ),
        "mcnemar_scope": (
            "descriptive_instance_level_exact_test; paired bootstrap is primary"
        ),
        "post_hoc_status": (
            "exploratory unless split and hypotheses were preregistered "
            "before inspecting results"
        ),
    }
    if settings["split_manifest"]:
        split_manifest_path = Path(settings["split_manifest"]).resolve()
        effective_config["split_manifest"] = str(split_manifest_path)
        effective_config["split_manifest_fingerprint"] = hashlib.sha256(
            split_manifest_path.read_bytes()
        ).hexdigest()
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "config": effective_config,
        "split": {
            "num_primary_samples": len(sample_ids),
            "num_development_samples": len(development_ids),
            "num_test_samples": len(test_ids),
        },
        "models": model_languages,
        "metrics": all_metric_rows,
        "cross_language_comparisons": all_comparison_rows,
        "calibration_parameters": all_parameters,
    }

    _write_json(output_dir / "analysis_config.json", effective_config)
    _write_json(output_dir / "analysis.json", analysis)
    _write_json(
        output_dir / "calibration_parameters.json",
        all_parameters,
    )
    _write_csv(output_dir / "split_manifest.csv", split_records)
    _write_csv(output_dir / "metrics_all.csv", all_metric_rows)
    _write_csv(
        output_dir / "table_main.csv",
        [
            _publication_metric_row(row)
            for row in all_metric_rows
            if row["method"] in MAIN_TABLE_METHODS
        ],
    )
    _write_csv(
        output_dir / "table_crosslingual.csv",
        all_comparison_rows,
    )
    (output_dir / "report.md").write_text(
        _build_markdown_report(
            effective_config,
            split_records,
            all_metric_rows,
            all_comparison_rows,
        ),
        encoding="utf-8",
    )
    print(f"Analysis complete: {output_dir}")


def _effective_settings(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    defaults = {
        "split_manifest": None,
        "reference_language": "en",
        "dev_fraction": 0.3,
        "seed": 42,
        "bootstrap_replicates": 2000,
        "confidence_level": 0.95,
        "ece_bins": 10,
        "high_confidence_threshold": 0.75,
        "length_match_max_relative_difference": 0.25,
        "logistic_l2": 1e-3,
    }
    settings: dict[str, Any] = {}
    for key, default in defaults.items():
        cli_value = getattr(args, key)
        settings[key] = (
            cli_value
            if cli_value is not None
            else config.get(key, default)
        )
    settings["results_dir"] = args.results_dir or config.get("results_dir")
    settings["output_dir"] = args.output_dir or config.get("output_dir")
    if not settings["results_dir"]:
        raise ValueError(
            "A results directory is required via --results-dir or config."
        )
    settings["reference_language"] = str(
        settings["reference_language"]
    )
    settings["split_manifest"] = (
        str(settings["split_manifest"])
        if settings["split_manifest"]
        else None
    )
    settings["dev_fraction"] = float(settings["dev_fraction"])
    settings["seed"] = int(settings["seed"])
    settings["bootstrap_replicates"] = int(
        settings["bootstrap_replicates"]
    )
    settings["confidence_level"] = float(settings["confidence_level"])
    settings["ece_bins"] = int(settings["ece_bins"])
    settings["high_confidence_threshold"] = float(
        settings["high_confidence_threshold"]
    )
    settings["length_match_max_relative_difference"] = float(
        settings["length_match_max_relative_difference"]
    )
    settings["logistic_l2"] = float(settings["logistic_l2"])
    return settings


def _load_prediction_runs(
    results_dir: Path,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    list[Path],
]:
    summary_path = results_dir / "summary_all.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing detection summary: {summary_path}"
        )
    summary = _read_json(summary_path)
    if not isinstance(summary, list):
        raise ValueError("summary_all.json must contain a list of runs.")

    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    prediction_files: list[Path] = []
    for run in summary:
        model = str(run["model"])
        language = str(run["language"])
        path = results_dir / model / language / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions: {path}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        predictions.setdefault(model, {})[language] = (
            primary_prediction_rows(rows)
        )
        prediction_files.append(path)
    return predictions, prediction_files


def _load_or_assign_split(
    sample_ids: list[str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_path = settings.get("split_manifest")
    if not manifest_path:
        return assign_sample_splits(
            sample_ids,
            dev_fraction=settings["dev_fraction"],
            seed=settings["seed"],
        )

    path = Path(manifest_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest_by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not sample_id:
            raise ValueError("Split manifest contains an empty sample_id.")
        if sample_id in manifest_by_id:
            raise ValueError(
                f"Split manifest contains duplicate sample {sample_id}."
            )
        if split not in {"development", "test"}:
            raise ValueError(
                f"Invalid split {split!r} for sample {sample_id}."
            )
        manifest_by_id[sample_id] = row

    expected = set(sample_ids)
    observed = set(manifest_by_id)
    if expected != observed:
        raise ValueError(
            "Split manifest sample IDs do not match primary predictions; "
            f"missing={sorted(expected - observed)[:5]}, "
            f"extra={sorted(observed - expected)[:5]}."
        )
    splits = {
        str(row["split"])
        for row in manifest_by_id.values()
    }
    if splits != {"development", "test"}:
        raise ValueError(
            "Split manifest must contain development and test samples."
        )
    normalized: list[dict[str, Any]] = []
    for sample_id in sorted(sample_ids):
        source = manifest_by_id[sample_id]
        normalized.append(
            {
                "sample_id": sample_id,
                "split": source["split"],
                "split_rank": _optional_int(source.get("split_rank")),
                "split_hash": source.get("split_hash") or None,
                "seed": _optional_int(source.get("seed")),
                "requested_dev_fraction": _optional_float(
                    source.get("requested_dev_fraction")
                ),
            }
        )
    return normalized


def _validate_common_sample_ids(
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[str]:
    expected: set[str] | None = None
    expected_name: str | None = None
    for model, rows_by_language in sorted(predictions.items()):
        for language, rows in sorted(rows_by_language.items()):
            sample_ids = {str(row["sample_id"]) for row in rows}
            name = f"{model}/{language}"
            if expected is None:
                expected = sample_ids
                expected_name = name
                continue
            if sample_ids != expected:
                missing = sorted(expected - sample_ids)
                extra = sorted(sample_ids - expected)
                raise ValueError(
                    f"Primary sample IDs differ between {expected_name} and "
                    f"{name}; missing={missing[:5]}, extra={extra[:5]}."
                )
    if not expected:
        raise ValueError("No primary prediction samples were found.")
    return sorted(expected)


def _metric_row(
    model: str,
    language: str,
    method_name: str,
    method: dict[str, Any],
    metrics: dict[str, Any],
    intervals: dict[str, dict[str, float]],
) -> dict[str, Any]:
    confusion = metrics["confusion_matrix"]
    row: dict[str, Any] = {
        "model": model,
        "language": language,
        "split": "test",
        "method": method_name,
        "method_kind": method["kind"],
        "fit_scope": method.get("fit_scope"),
        "fit_language": method.get("fit_language"),
        "num_development_instances": method.get(
            "num_development_instances",
            0,
        ),
        "decision_threshold": method.get("threshold"),
        "true_hallucinated": confusion["true_hallucinated"],
        "false_hallucinated": confusion["false_hallucinated"],
        "true_grounded": confusion["true_grounded"],
        "false_grounded": confusion["false_grounded"],
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field)
    for metric in BOOTSTRAP_METRICS:
        interval = intervals.get(metric, {})
        row[f"{metric}_ci_low"] = interval.get("low")
        row[f"{metric}_ci_high"] = interval.get("high")
    row["method_parameters_json"] = json.dumps(
        method,
        ensure_ascii=False,
        sort_keys=True,
    )
    return row


def _add_holm_adjustments(
    comparison_rows: list[dict[str, Any]],
) -> None:
    p_value_fields = (
        "accuracy_mcnemar_exact_p",
        "prediction_shift_mcnemar_exact_p",
        "pairwise_mcnemar_exact_p",
    )
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = {}
    for row in comparison_rows:
        key = (
            str(row["model"]),
            str(row["reference_language"]),
            str(row["target_language"]),
        )
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        for field in p_value_fields:
            adjusted = holm_adjusted_p_values(
                [float(row[field]) for row in rows]
            )
            output_field = field.replace(
                "_exact_p",
                "_holm_p",
            )
            for row, value in zip(rows, adjusted):
                row[output_field] = value


def _publication_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "model",
        "language",
        "split",
        "method",
        "fit_scope",
        "fit_language",
        "num_development_instances",
        "num_source_samples",
        "num_instances",
        "decision_threshold",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "balanced_accuracy",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "hallucination_recall",
        "hallucination_recall_ci_low",
        "hallucination_recall_ci_high",
        "specificity",
        "specificity_ci_low",
        "specificity_ci_high",
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "brier_score",
        "brier_score_ci_low",
        "brier_score_ci_high",
        "expected_calibration_error",
        "expected_calibration_error_ci_low",
        "expected_calibration_error_ci_high",
        "predicted_hallucination_rate",
        "pairwise_ranking_accuracy",
        "pairwise_ranking_accuracy_ci_low",
        "pairwise_ranking_accuracy_ci_high",
        "pairwise_classification_accuracy",
        "length_only_roc_auc",
        "length_matched_instance_coverage",
        "length_matched_accuracy",
        "length_matched_roc_auc",
    )
    return {field: row.get(field) for field in fields}


def _prediction_fingerprint(paths: list[Path], base_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(
            path.relative_to(base_dir).as_posix().encode("utf-8")
        )
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _derived_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        ":".join((str(seed), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _build_markdown_report(
    config: dict[str, Any],
    split_records: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    num_dev = sum(row["split"] == "development" for row in split_records)
    num_test = len(split_records) - num_dev
    lines = [
        "# Cross-lingual HaluEval Analysis",
        "",
        "## Protocol",
        "",
        f"- Analysis schema: {ANALYSIS_SCHEMA_VERSION}",
        f"- Reference language: `{config['reference_language']}`",
        f"- Development/test source pairs: {num_dev}/{num_test}",
        f"- Split seed: {config['seed']}",
        (
            "- Source-pair bootstrap replicates: "
            f"{config['bootstrap_replicates']}"
        ),
        (
            "- Confidence level: "
            f"{100 * config['confidence_level']:.1f}%"
        ),
        (
            "- Confidence intervals resample test source pairs and are "
            "conditional on the fitted calibrator."
        ),
        (
            "- This analysis is exploratory unless the split and hypotheses "
            "were fixed before inspecting model results."
        ),
        "",
        "## Test Results",
        "",
        (
            "| Model | Language | Method | Accuracy | Balanced accuracy | "
            "ROC-AUC | ECE | Pair rank | Pred. hallucination |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    display_methods = {
        "generated_decision",
        "probability_0_5",
        "language_threshold",
        "language_temperature",
        "language_platt",
        "length_only_logistic",
        "model_and_length_logistic",
    }
    for row in metric_rows:
        if row["method"] not in display_methods:
            continue
        lines.append(
            "| {model} | {language} | {method} | {accuracy} | "
            "{balanced} | {auc} | {ece} | {pairwise} | {rate} |".format(
                model=row["model"],
                language=row["language"],
                method=row["method"],
                accuracy=_format_rate(row["accuracy"]),
                balanced=_format_rate(row["balanced_accuracy"]),
                auc=_format_rate(row["roc_auc"]),
                ece=_format_rate(row["expected_calibration_error"]),
                pairwise=_format_rate(row["pairwise_ranking_accuracy"]),
                rate=_format_rate(row["predicted_hallucination_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Cross-language Differences",
            "",
            (
                "Deltas are target minus reference language. Exact McNemar "
                "tests are descriptive; source-pair bootstrap intervals are "
                "the primary uncertainty estimate."
            ),
            "",
            (
                "| Model | Method | Comparison | Accuracy delta | "
                "Accuracy CI | Pair-rank delta | Pair-rank CI | "
                "Prediction-shift Holm p |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    comparison_methods = {
        "generated_decision",
        "probability_0_5",
        "language_threshold",
        "language_platt",
    }
    for row in comparison_rows:
        if row["method"] not in comparison_methods:
            continue
        lines.append(
            "| {model} | {method} | {reference}->{target} | {accuracy} | "
            "[{accuracy_low}, {accuracy_high}] | {pairwise} | "
            "[{pairwise_low}, {pairwise_high}] | {shift_p} |".format(
                model=row["model"],
                method=row["method"],
                reference=row["reference_language"],
                target=row["target_language"],
                accuracy=_format_signed_rate(row["delta_accuracy"]),
                accuracy_low=_format_signed_rate(
                    row.get("delta_accuracy_ci_low")
                ),
                accuracy_high=_format_signed_rate(
                    row.get("delta_accuracy_ci_high")
                ),
                pairwise=_format_signed_rate(
                    row["delta_pairwise_ranking_accuracy"]
                ),
                pairwise_low=_format_signed_rate(
                    row.get(
                        "delta_pairwise_ranking_accuracy_ci_low"
                    )
                ),
                pairwise_high=_format_signed_rate(
                    row.get(
                        "delta_pairwise_ranking_accuracy_ci_high"
                    )
                ),
                shift_p=_format_p(
                    row["prediction_shift_mcnemar_holm_p"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            (
                "- `generated_decision` reproduces the model's saved YES/NO "
                "decision on the held-out test split."
            ),
            (
                "- `probability_0_5` uses the saved decision-token probability "
                "with no fitted calibration."
            ),
            (
                "- Pooled, reference, and language-specific calibrators use "
                "development rows only."
            ),
            (
                "- Temperature scaling changes probability calibration but "
                "does not move the 0.5 decision boundary."
            ),
            (
                "- Length-only performance quantifies benchmark leakage; the "
                "model-and-length probe measures model signal conditional on "
                "candidate length but is not a causal correction."
            ),
            (
                "- Confirm final claims across model families, languages, and "
                "a newly frozen analysis split."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_rate(value: Any) -> str:
    return "NA" if value is None else f"{100 * float(value):.1f}%"


def _format_signed_rate(value: Any) -> str:
    return "NA" if value is None else f"{100 * float(value):+.1f} pp"


def _format_p(value: Any) -> str:
    if value is None:
        return "NA"
    probability = float(value)
    return f"{probability:.2e}" if probability < 0.001 else f"{probability:.3f}"


def _optional_int(value: Any) -> int | None:
    return int(value) if value not in {None, ""} else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value not in {None, ""} else None


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
