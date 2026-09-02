from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.utils import read_json, write_csv, write_json


CORE_BENCHMARKS = ("halueval_qa", "hallulens", "ragtruth", "mu_shroom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark summaries without pooling incompatible metrics."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_suite.full.json",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write partial tables rather than failing the completeness gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(_root_path(args.config))
    model_name = config["models"][0]["name"]
    output_dir = _root_path(config["output_dir"])
    preparation_manifest = _read_optional(
        _root_path(config["prepared_dir"]) / "manifest.json"
    )
    records: dict[str, dict[str, Any]] = {}

    halueval_config = read_json(
        _root_path(config["benchmarks"]["halueval_qa"]["runner_config"])
    )
    halueval_path = (
        _root_path(halueval_config["output_dir"])
        / halueval_config["experiment_name"]
        / "summary_all.json"
    )
    halueval = _read_optional(halueval_path)
    expected_halueval_rows = _manifest_row_count(preparation_manifest, "halueval_qa")
    halueval_full = _halueval_is_full(halueval, expected_halueval_rows)
    records["halueval_qa"] = {
        "status": "complete" if halueval_full else "missing_or_partial",
        "is_full_split_run": halueval_full,
        "path": _display_path(halueval_path),
        "summary": halueval,
    }

    for benchmark in ("ragtruth", "mu_shroom", "truthfulqa"):
        path = output_dir / benchmark / model_name / "summary.json"
        summary = _read_optional(path)
        complete = bool(summary and summary.get("is_full_split_run"))
        records[benchmark] = {
            "status": "complete" if complete else "missing_or_partial",
            "is_full_split_run": complete,
            "path": _display_path(path),
            "summary": summary,
        }

    hallulens_path = output_dir / "hallulens" / "completion.json"
    hallulens = _read_optional(hallulens_path)
    hallulens_full = bool(hallulens and hallulens.get("is_full_split_run"))
    records["hallulens"] = {
        "status": "complete" if hallulens_full else "missing_or_partial",
        "is_full_split_run": hallulens_full,
        "path": _display_path(hallulens_path),
        "summary": hallulens,
    }

    missing_core = [
        benchmark
        for benchmark in CORE_BENCHMARKS
        if not records[benchmark]["is_full_split_run"]
    ]
    aggregate = {
        "suite_name": config["suite_name"],
        "protocol_version": config["protocol_version"],
        "model": model_name,
        "core_benchmarks": list(CORE_BENCHMARKS),
        "core_suite_complete": not missing_core,
        "missing_or_partial_core_benchmarks": missing_core,
        "truthfulqa_is_separate_factuality_control": True,
        "results": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "suite_summary.json", aggregate)
    write_csv(output_dir / "suite_table.csv", _table_rows(records))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if missing_core and not args.allow_incomplete:
        raise SystemExit(
            "Full-suite completeness gate failed: " + ", ".join(missing_core)
        )


def _table_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, record in records.items():
        summary = record.get("summary") or {}
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "status": record["status"],
            "is_full_split_run": record["is_full_split_run"],
            "result_path": record["path"],
            "num_samples": "",
            "num_reported_groups": "",
            "headline_metrics": "",
        }
        if benchmark == "halueval_qa" and isinstance(summary, list):
            row["num_reported_groups"] = len(summary)
            row["headline_metrics"] = json.dumps(
                [
                    {
                        "language": item.get("language"),
                        "balanced_accuracy": item.get("balanced_accuracy"),
                        "hallucination_f1": item.get("hallucination_f1"),
                        "mcc": item.get("mcc"),
                        "roc_auc": item.get("roc_auc"),
                    }
                    for item in summary
                ],
                ensure_ascii=False,
            )
        elif benchmark in {"ragtruth", "mu_shroom"} and summary:
            response = summary.get("response_level", {})
            span = summary.get("span_level", {})
            row["num_samples"] = summary.get("num_samples")
            row["headline_metrics"] = json.dumps(
                {
                    "balanced_accuracy": response.get("balanced_accuracy"),
                    "f1": response.get("f1"),
                    "mcc": response.get("mcc"),
                    "mean_hard_span_iou": span.get("mean_hard_span_iou"),
                }
            )
        elif benchmark == "truthfulqa" and summary:
            row["num_samples"] = summary.get("num_samples")
            row["headline_metrics"] = json.dumps(
                {"accuracy": summary.get("accuracy")}
            )
        elif benchmark == "hallulens" and summary:
            row["num_reported_groups"] = len(summary.get("task_results", []))
            row["headline_metrics"] = "See hashed official task outputs"
        rows.append(row)
    return rows


def _halueval_is_full(summary: Any, expected_rows: int | None) -> bool:
    if not isinstance(summary, list) or not summary or expected_rows is None:
        return False
    return all(
        int(item.get("num_source_samples", -1)) == expected_rows
        for item in summary
    )


def _manifest_row_count(manifest: Any, benchmark: str) -> int | None:
    if not isinstance(manifest, dict):
        return None
    value = manifest.get("benchmarks", {}).get(benchmark, {}).get("num_rows")
    return int(value) if isinstance(value, int) else None


def _read_optional(path: Path) -> Any:
    return read_json(path) if path.is_file() else None


def _root_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
