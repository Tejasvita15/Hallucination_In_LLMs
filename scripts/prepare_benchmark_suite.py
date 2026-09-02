from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.benchmark_suite import (
    file_sha256,
    normalize_halueval_qa,
    normalize_mushroom,
    normalize_ragtruth,
    normalize_truthfulqa_csv,
    read_json_records,
    read_jsonl,
)
from hallu_llm.utils import ensure_dir, read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen full-benchmark suite and provenance manifest."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_suite.full.json",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        help="Prepare only this benchmark; repeat to select several.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Clone repositories and download Hugging Face datasets when absent.",
    )
    parser.add_argument(
        "--download-hallulens-data",
        action="store_true",
        help="Run the official HalluLens data download script after cloning.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write an incomplete manifest instead of failing on missing assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _root_path(args.config)
    config = read_json(config_path)
    benchmark_configs = config["benchmarks"]
    selected = args.benchmark or list(benchmark_configs)
    unknown = sorted(set(selected) - set(benchmark_configs))
    if unknown:
        raise SystemExit(f"Unknown benchmark(s): {', '.join(unknown)}")

    raw_dir = ensure_dir(_root_path(config["raw_dir"]))
    prepared_dir = ensure_dir(_root_path(config["prepared_dir"]))
    manifest_path = prepared_dir / "manifest.json"
    config_hash = file_sha256(config_path)
    previous_manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    if previous_manifest.get("config_sha256") == config_hash:
        entries: dict[str, dict[str, Any]] = dict(
            previous_manifest.get("benchmarks", {})
        )
    else:
        entries = {}
    failures: list[str] = []

    for name in selected:
        print(f"\n=== Preparing {name} ===", flush=True)
        benchmark = benchmark_configs[name]
        try:
            if name == "ragtruth":
                entry = prepare_ragtruth(benchmark, raw_dir, prepared_dir, args.download)
            elif name == "mu_shroom":
                entry = prepare_mu_shroom(benchmark, prepared_dir, args.download)
            elif name == "truthfulqa":
                entry = prepare_truthfulqa(
                    benchmark,
                    raw_dir,
                    prepared_dir,
                    args.download,
                    int(config["seed"]),
                )
            elif name == "hallulens":
                entry = prepare_hallulens(
                    benchmark,
                    raw_dir,
                    args.download,
                    args.download_hallulens_data,
                )
            elif name == "halueval_qa":
                entry = prepare_halueval(
                    benchmark,
                    raw_dir,
                    prepared_dir,
                    args.download,
                )
            else:
                raise ValueError(f"No preparation adapter for {name}")
            entries[name] = {"status": "ready", **entry}
        except Exception as error:
            entries[name] = {
                "status": "missing_or_failed",
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(name)
            print(f"FAILED: {error}", file=sys.stderr, flush=True)

    full_suite_ready = set(entries) == set(benchmark_configs) and all(
        entries[name].get("status") == "ready" for name in benchmark_configs
    )
    manifest = {
        "suite_name": config["suite_name"],
        "protocol_version": config["protocol_version"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": _display_path(config_path),
        "config_sha256": config_hash,
        "selected_benchmarks": selected,
        "full_suite_ready": full_suite_ready,
        "benchmarks": entries,
    }
    write_json(manifest_path, manifest)
    print(f"\nManifest: {_display_path(manifest_path)}")
    if failures and not args.allow_missing:
        raise SystemExit(
            "Benchmark preparation failed for: " + ", ".join(failures)
        )


def prepare_ragtruth(
    config: dict[str, Any],
    raw_dir: Path,
    prepared_dir: Path,
    download: bool,
) -> dict[str, Any]:
    checkout = _ensure_repository(config, raw_dir, download)
    responses_path = checkout / config["responses_file"]
    sources_path = checkout / config["sources_file"]
    _require_files([responses_path, sources_path])
    rows, quality = normalize_ragtruth(
        read_jsonl(responses_path),
        read_jsonl(sources_path),
        split=config.get("split"),
        include_quality_issues=bool(config.get("include_quality_issues", False)),
    )
    output = prepared_dir / "ragtruth.jsonl"
    write_jsonl(output, rows)
    write_json(prepared_dir / "ragtruth.quality.json", quality)
    return {
        "num_rows": len(rows),
        "prepared_file": _display_path(output),
        "prepared_sha256": file_sha256(output),
        "source_revision": _git_revision(checkout),
        "source_files": _source_file_records([responses_path, sources_path]),
        "quality": quality,
    }


def prepare_mu_shroom(
    config: dict[str, Any],
    prepared_dir: Path,
    download: bool,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to prepare Mu-SHROOM.") from error

    download_mode = "reuse_dataset_if_exists" if download else "reuse_cache_if_exists"
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    split_quality: dict[str, Any] = {}
    for split in config["prepare_splits"]:
        dataset = load_dataset(
            config["source"],
            config.get("source_config"),
            split=split,
            download_mode=download_mode,
        )
        normalized, quality = normalize_mushroom(dataset, split)
        if not normalized:
            raise ValueError(f"Mu-SHROOM split {split!r} has no labeled examples.")
        rows_by_split[split] = normalized
        split_quality[split] = quality

    rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Mu-SHROOM splits produced duplicate sample IDs.")
    for split, split_rows in rows_by_split.items():
        write_jsonl(prepared_dir / f"mu_shroom.{split}.jsonl", split_rows)
    evaluation_split = config["evaluation_split"]
    evaluation_rows = rows_by_split.get(evaluation_split)
    if evaluation_rows is None:
        raise ValueError(
            f"Mu-SHROOM evaluation split {evaluation_split!r} was not prepared."
        )
    output = prepared_dir / "mu_shroom.jsonl"
    quality = {
        "benchmark": "mu_shroom",
        "num_rows": len(evaluation_rows),
        "num_prepared_rows": len(rows),
        "evaluation_split": evaluation_split,
        "splits": split_quality,
        "language_counts": _count(rows, "language"),
    }
    write_jsonl(output, evaluation_rows)
    write_json(prepared_dir / "mu_shroom.quality.json", quality)
    return {
        "num_rows": len(evaluation_rows),
        "num_prepared_rows": len(rows),
        "prepared_file": _display_path(output),
        "prepared_sha256": file_sha256(output),
        "source_revision": getattr(dataset, "_fingerprint", "unknown"),
        "source": config["source"],
        "source_config": config.get("source_config"),
        "quality": quality,
    }


def prepare_truthfulqa(
    config: dict[str, Any],
    raw_dir: Path,
    prepared_dir: Path,
    download: bool,
    seed: int,
) -> dict[str, Any]:
    checkout = _ensure_repository(config, raw_dir, download)
    csv_path = checkout / config["csv_file"]
    _require_files([csv_path])
    rows, quality = normalize_truthfulqa_csv(csv_path, seed)
    output = prepared_dir / "truthfulqa.jsonl"
    write_jsonl(output, rows)
    write_json(prepared_dir / "truthfulqa.quality.json", quality)
    return {
        "num_rows": len(rows),
        "prepared_file": _display_path(output),
        "prepared_sha256": file_sha256(output),
        "source_revision": _git_revision(checkout),
        "source_files": _source_file_records([csv_path]),
        "quality": quality,
    }


def prepare_hallulens(
    config: dict[str, Any],
    raw_dir: Path,
    download: bool,
    download_data: bool,
) -> dict[str, Any]:
    checkout = _ensure_repository(config, raw_dir, download)
    script = checkout / "scripts" / "download_data.sh"
    _require_files([script])
    if download_data:
        subprocess.run(
            ["bash", str(script)],
            cwd=checkout,
            check=True,
        )
    data_dir = checkout / "data"
    return {
        "source_revision": _git_revision(checkout),
        "checkout_dir": _display_path(checkout),
        "data_dir": _display_path(data_dir),
        "data_download_requested": download_data,
        "tasks": config["tasks"],
        "official_runner_scripts": [
            "scripts/task1_precisewikiqa.sh",
            "scripts/task2_longwiki.sh",
            "scripts/task3-1_mixedentities.sh",
            "scripts/task3-2_generatedentities.sh",
        ],
    }


def prepare_halueval(
    config: dict[str, Any],
    raw_dir: Path,
    prepared_dir: Path,
    download: bool,
) -> dict[str, Any]:
    runner_config = _root_path(config["runner_config"])
    _require_files([runner_config])
    checkout = _ensure_repository(config, raw_dir, download)
    source_path = checkout / config["qa_file"]
    _require_files([source_path])
    rows, quality = normalize_halueval_qa(read_json_records(source_path))
    output = prepared_dir / "halueval_qa.jsonl"
    write_jsonl(output, rows)
    write_json(prepared_dir / "halueval_qa.quality.json", quality)
    return {
        "num_rows": len(rows),
        "prepared_file": _display_path(output),
        "prepared_sha256": file_sha256(output),
        "runner_config": _display_path(runner_config),
        "runner_config_sha256": file_sha256(runner_config),
        "source_revision": _git_revision(checkout),
        "source_files": _source_file_records([source_path]),
        "quality": quality,
    }


def _ensure_repository(
    config: dict[str, Any],
    raw_dir: Path,
    download: bool,
) -> Path:
    checkout = raw_dir / config["checkout_dir"]
    if not checkout.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing {checkout}; rerun with --download from a login node."
            )
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                config.get("revision", "main"),
                config["repository"],
                str(checkout),
            ],
            check=True,
        )
    if not (checkout / ".git").exists():
        raise ValueError(f"Expected a Git checkout at {checkout}")
    return checkout


def _git_revision(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
    ).strip()


def _source_file_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": _display_path(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def _count(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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
