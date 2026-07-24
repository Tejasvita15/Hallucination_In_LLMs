from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin wrapper around hallulens_eval.py.")
    parser.add_argument("--task", required=True, choices=["precisewikiqa", "longwiki", "nonexistent"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="results/hallulens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "hallulens_eval.py"),
        "--task",
        args.task,
        "--model",
        args.model,
        "--model-name",
        args.model_name,
        "--output-dir",
        args.output_dir,
    ]
    if args.data:
        cmd += ["--data", args.data]
    elif args.task == "nonexistent":
        cmd += ["--make-nonexistent-samples", "--num-samples", str(args.num_samples)]
    else:
        raise SystemExit(f"--data is required for task {args.task}")
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()

