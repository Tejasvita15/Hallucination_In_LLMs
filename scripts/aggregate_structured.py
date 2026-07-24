from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate structured experiment summaries.")
    parser.add_argument("--results-dir", default="results/structured")
    parser.add_argument("--output", default="results/structured/summary_table.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = []
    for summary_path in results_dir.glob("**/summary.json"):
        with summary_path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No summary files found under {results_dir}")
        return

    keys = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

