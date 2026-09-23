"""Export blinded annotation rows or analyze independently assigned semantic labels.

Mean token likelihood is an uncertainty feature, not calibrated answer confidence.
No Brier/ECE is reported for it. Annotation hashes must match the exact output.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hallu_llm.benchmark_suite import read_jsonl, binary_roc_auc


def analyze(rows, annotations):
    keyed = {a["generation_sha256"]: a for a in annotations}
    if len(keyed) != len(annotations):
        raise ValueError("Duplicate annotation hashes")
    unknown = set(keyed) - {r["generation_sha256"] for r in rows}
    if unknown:
        raise ValueError("Annotations contain generations outside this prediction file")
    usable, pending, incomplete, abstentions = [], 0, 0, 0
    for row in rows:
        if row["truncated"] or not row["final_answer_present"]:
            incomplete += 1
            continue
        label = keyed.get(row["generation_sha256"], {}).get("semantic_label")
        if label is None:
            pending += 1
            continue
        if label not in {"supported", "unsupported", "abstention", "uncertain"}:
            raise ValueError(f"Invalid semantic label: {label}")
        if label in {"abstention", "uncertain"}:
            abstentions += 1
            continue
        usable.append((label == "unsupported", row["mean_final_token_probability"], row["generated_tokens"]))
    summary = {"total": len(rows), "truncated_or_missing_final": incomplete,
               "pending_annotations": pending, "abstention_or_uncertain": abstentions,
               "analyzed_annotations": len(usable), "mean_token_likelihood_is_not_answer_probability": True}
    if usable:
        labels = [u[0] for u in usable]
        summary["unsupported_rate_among_reviewed_answers"] = sum(labels) / len(labels)
        summary["hallucination_detection_auc_from_negative_token_likelihood"] = binary_roc_auc(labels, [-u[1] for u in usable])
        summary["by_semantic_label"] = {label: {"count": len(group),
            "mean_token_probability": statistics.mean(r[1] for r in group),
            "mean_generated_tokens": statistics.mean(r[2] for r in group)}
            for label, group in [(name, [r for r in usable if r[0] == flag])
                                for name, flag in [("supported", False), ("unsupported", True)]] if group}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--annotations")
    p.add_argument("--export-annotations")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = read_jsonl(args.predictions)
    if args.export_annotations:
        with Path(args.export_annotations).open("x", encoding="utf-8") as handle:
            for r in rows:
                # Hide model name and likelihood from the reviewer.
                annotation = {"generation_sha256": r["generation_sha256"], "question": r["question"],
                    "knowledge": r["knowledge"], "answer": r["final_answer"],
                    "semantic_label": None, "reviewer": "", "rationale": ""}
                handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
    summary = analyze(rows, read_jsonl(args.annotations) if args.annotations else [])
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
