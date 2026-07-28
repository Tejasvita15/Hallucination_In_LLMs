from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download


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
    configs = []
    for config_path in args.config:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            configs.append(json.load(handle))

    datasets = {
        (
            cfg.get("dataset", {}).get("name", "pminervini/HaluEval"),
            cfg.get("dataset", {}).get("subset", "qa"),
            cfg.get("dataset", {}).get("split", "data"),
        )
        for cfg in configs
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


if __name__ == "__main__":
    main()
