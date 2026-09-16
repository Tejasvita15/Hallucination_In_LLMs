"""Audit saved NSCC smoke predictions without models, GPUs or extra packages."""
import argparse
import json
import re
import statistics
from pathlib import Path


def audit_rows(rows):
    if not rows:
        raise ValueError("Empty predictions file")
    missing = sum("reasoning" not in row for row in rows)
    texts = [str(row.get("reasoning", "")) for row in rows]
    # This detects obvious short answers, not the semantic quality of reasoning.
    stripped = [re.sub(r"</?think>", "", text).strip() for text in texts]
    bare = [bool(re.fullmatch(r"(?:YES|NO)[.!]?", text, re.I)) for text in stripped]
    lengths = [int(row.get("reasoning_tokens", 0)) for row in rows]
    truncated = sum(bool(row.get("reasoning_truncated")) for row in rows)
    empty = sum(not text for text in stripped)
    return {
        "instances": len(rows), "missing_reasoning_fields": missing,
        "empty_reasoning": empty, "bare_yes_no": sum(bare),
        "reasoning_under_10_words": sum(len(text.split()) < 10 for text in stripped),
        "truncated": truncated, "median_tokens": statistics.median(lengths),
        "review_flagged_instances": sum(bool(row.get("review_quality_flags")) for row in rows),
        "max_tokens": max(lengths),
        "needs_manual_review": True,
        "automatic_checks_passed": not (missing or empty or truncated or any(bare)
                                          or any(row.get("review_quality_flags") for row in rows)),
        "source_pairs": sorted({str(row["sample_id"]) for row in rows}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/detection")
    parser.add_argument("--output", default="results/detection/reasoned_smoke_audit.txt")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    report = []
    pair_sets = []
    for short, model in [("qwen", "qwen2.5-7b"), ("llama", "llama3.1-8b"),
                         ("deepseek", "deepseek-r1-7b")]:
        experiment = (f"nscc_{short}_reasoned_smoke_v1" if args.version == "v1"
                      else f"nscc_{short}_evidence_review_smoke_v2")
        path = Path(args.results_root) / experiment / model / "en/predictions.jsonl"
        if not path.is_file():
            report.append(f"MISSING: {path}\n")
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = audit_rows(rows)
        pair_sets.append(summary.pop("source_pairs"))
        report.append(f"\n=== {model} ===\n" + json.dumps(summary, indent=2))
        # All 40 responses are retained so the report supports a complete review.
        for row in rows:
            report.append(
                f"\nID: {row['sample_id']} | {row['candidate_type']}\n"
                f"Knowledge: {row.get('knowledge', '')}\n"
                f"Question: {row.get('question', '')}\n"
                f"Candidate: {row.get('candidate_answer', '')}\n"
                f"Reasoning ({row.get('reasoning_tokens')} tokens):\n{row.get('reasoning', '')}\n"
                f"Final choice: {row.get('generation')}\n"
                f"Review flags: {row.get('review_quality_flags', [])}\n"
            )
    matched = len(pair_sets) == 3 and all(ids == pair_sets[0] for ids in pair_sets)
    report.insert(0, f"All three source-pair sets match: {matched}\n"
                  "Automatic checks cannot establish valid evidence-based reasoning.\n")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Audit saved: {output}")
    print(f"All three source-pair sets match: {matched}")
    for entry in report:
        if entry.startswith("\n===") or entry.startswith("MISSING:"):
            print(entry)


if __name__ == "__main__":
    main()
