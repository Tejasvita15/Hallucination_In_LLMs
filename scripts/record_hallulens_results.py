from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.benchmark_suite import file_sha256
from hallu_llm.utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record and hash outputs from the official HalluLens task scripts."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_suite.full.json",
    )
    parser.add_argument(
        "--task-result",
        action="append",
        required=True,
        metavar="TASK=PATH",
        help="Official task name and its result file; repeat for every task.",
    )
    parser.add_argument(
        "--full-split",
        action="store_true",
        help="Assert that no official prompt/sample limit was applied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(_root_path(args.config))
    expected = set(config["benchmarks"]["hallulens"]["tasks"])
    result_paths: dict[str, Path] = {}
    for specification in args.task_result:
        if "=" not in specification:
            raise SystemExit(f"Expected TASK=PATH, got {specification!r}")
        task, raw_path = specification.split("=", 1)
        if task in result_paths:
            raise SystemExit(f"Duplicate HalluLens task: {task}")
        result_paths[task] = _root_path(raw_path)
    unknown = sorted(set(result_paths) - expected)
    if unknown:
        raise SystemExit("Unknown HalluLens task(s): " + ", ".join(unknown))
    missing_tasks = sorted(expected - set(result_paths))
    records = []
    for task, path in sorted(result_paths.items()):
        if not path.is_file():
            raise SystemExit(f"Missing result file for {task}: {path}")
        records.append(
            {
                "task": task,
                "path": _display_path(path),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "benchmark": "hallulens",
        "official_implementation": True,
        "full_split_asserted": bool(args.full_split),
        "is_full_split_run": bool(args.full_split) and not missing_tasks,
        "expected_tasks": sorted(expected),
        "missing_tasks": missing_tasks,
        "task_results": records,
    }
    output = _root_path(config["output_dir"]) / "hallulens" / "completion.json"
    write_json(output, manifest)
    print(f"Wrote {_display_path(output)}")
    if missing_tasks:
        print("Still missing: " + ", ".join(missing_tasks))


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
