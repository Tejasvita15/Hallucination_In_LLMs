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
    parser.add_argument("--config", required=True, help="Path to experiment JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    dataset_cfg = cfg.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "pminervini/HaluEval")
    dataset_subset = dataset_cfg.get("subset", "qa")
    dataset_split = dataset_cfg.get("split", "data")
    print(
        f"Prefetching dataset: {dataset_name} / {dataset_subset} / {dataset_split}"
    )
    load_dataset(dataset_name, dataset_subset, split=dataset_split)

    repositories = {
        model_cfg["path"]
        for model_cfg in cfg.get("models", [])
        if model_cfg.get("path")
    }
    repositories.update(_translator_repositories(cfg.get("languages", [])))
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
