from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face datasets and models required by an experiment."
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to an experiment JSON; repeat to prefetch an ablation set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    configs = []
    for config_path in args.config:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            configs.append(json.load(handle))

    datasets = {
        dataset
        for cfg in configs
        for dataset in _dataset_specs(cfg)
    }
    for dataset_name, dataset_subset, dataset_split in sorted(datasets):
        print(
            f"Prefetching dataset: {dataset_name} / "
            f"{dataset_subset} / {dataset_split}"
        )
        load_dataset(dataset_name, dataset_subset, split=dataset_split)

    repositories: set[str] = set()
    for cfg in configs:
        repositories.update(
            model_cfg["path"]
            for model_cfg in cfg.get("models", [])
            if model_cfg.get("path")
        )
        repositories.update(
            _translator_repositories(cfg.get("languages", []))
        )
        generator = cfg.get("generator", {})
        if generator.get("path"):
            repositories.add(str(generator["path"]))
    for position, repository in enumerate(sorted(repositories), start=1):
        print(f"[{position}/{len(repositories)}] Prefetching model: {repository}")
        snapshot_download(repo_id=repository, repo_type="model")

    print("All Hugging Face assets are cached.")


def _translator_repositories(languages: list[dict[str, Any]]) -> set[str]:
    repositories: set[str] = set()
    for language_cfg in languages:
        translator = language_cfg.get("translator")
        if isinstance(translator, str):
            repositories.add(translator)
        elif isinstance(translator, dict) and translator.get("model"):
            repositories.add(str(translator["model"]))
    return repositories


def _dataset_specs(cfg: dict[str, Any]) -> set[tuple[str, str, str]]:
    specs: set[tuple[str, str, str]] = set()
    candidates = [cfg.get("dataset", {})]
    source = cfg.get("source", {})
    if isinstance(source, dict):
        candidates.append(source.get("dataset", {}))
    for dataset_cfg in candidates:
        if not isinstance(dataset_cfg, dict):
            continue
        if dataset_cfg.get("path") or not dataset_cfg.get("name"):
            continue
        specs.add(
            (
                str(dataset_cfg["name"]),
                str(dataset_cfg.get("subset", "qa")),
                str(dataset_cfg.get("split", "data")),
            )
        )
    benchmarks = cfg.get("benchmarks", {})
    if isinstance(benchmarks, dict):
        for benchmark_cfg in benchmarks.values():
            if not isinstance(benchmark_cfg, dict):
                continue
            dataset_name = benchmark_cfg.get("source")
            dataset_subset = benchmark_cfg.get("source_config")
            if not dataset_name or not dataset_subset:
                continue
            splits = benchmark_cfg.get("prepare_splits")
            if not isinstance(splits, list):
                splits = [benchmark_cfg.get("split", "data")]
            specs.update(
                (
                    str(dataset_name),
                    str(dataset_subset),
                    str(split),
                )
                for split in splits
            )
    return specs


if __name__ == "__main__":
    main()
