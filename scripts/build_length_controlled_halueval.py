from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.controlled_benchmark import (
    CandidateConstraints,
    build_review_rows,
    finalize_reviewed_proposals,
    generate_proposal,
    reconcile_dual_reviews,
    summarize_dual_reviews,
    summarize_reviews,
)
from hallu_llm.data import QASample, load_halueval_qa
from hallu_llm.detection import stable_sample_id
from hallu_llm.detection_analysis import assign_sample_splits
from hallu_llm.models import HFGenerator
from hallu_llm.reviews import load_review_manifest, sample_review
from hallu_llm.utils import (
    ensure_dir,
    read_json,
    set_seed,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, review, and finalize human-approved length-controlled "
            "HaluEval negative candidates."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--experiment-name")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not load a generator; validate the edited review CSV and freeze the dataset.",
    )
    mode.add_argument(
        "--restart",
        action="store_true",
        help="Regenerate all proposals instead of resuming a compatible proposal cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = read_json(config_path)
    if int(cfg.get("schema_version", 0)) != 1:
        raise ValueError("Controlled benchmark config schema_version must be 1.")

    experiment_name = args.experiment_name or str(cfg["experiment_name"])
    output_root = ensure_dir(
        _resolve_repo_path(cfg.get("output_dir", "results/benchmark"))
        / experiment_name
    )
    proposals_path = output_root / "proposals.jsonl"
    proposal_meta_path = output_root / "proposal_meta.json"
    review_cfg = dict(cfg.get("review", {}))
    review_paths = _review_paths(output_root, review_cfg)
    adjudication_path = output_root / str(
        review_cfg.get("adjudication_file", "proposal_adjudication.csv")
    )
    audit_path = output_root / "review_audit.csv"
    quality_path = output_root / "quality.json"
    final_dataset_path = output_root / "controlled_halueval.jsonl"
    split_path = output_root / "split_manifest.csv"

    constraints_cfg = cfg.get("constraints", {})
    constraints = CandidateConstraints(
        max_relative_length_difference=float(
            constraints_cfg.get("max_relative_length_difference", 0.25)
        ),
        max_sentence_count_difference=int(
            constraints_cfg.get("max_sentence_count_difference", 0)
        ),
        max_named_entity_count_difference=int(
            constraints_cfg.get("max_named_entity_count_difference", 0)
        ),
        require_answer_style_match=bool(
            constraints_cfg.get("require_answer_style_match", True)
        ),
        forbid_newlines=bool(constraints_cfg.get("forbid_newlines", True)),
    )
    effective_cfg = dict(cfg)
    effective_cfg["experiment_name"] = experiment_name
    if args.num_samples is not None:
        source_cfg = dict(effective_cfg.get("source", {}))
        source_cfg["num_samples"] = args.num_samples
        effective_cfg["source"] = source_cfg
    write_json(output_root / "config.json", effective_cfg)
    independent_reviews = {
        path.stem: _read_review_csv(path) for path in review_paths
    }
    existing_adjudications = (
        _read_review_csv(adjudication_path) if len(review_paths) == 2 else {}
    )
    if args.restart and (
        any(_has_review_content(reviews) for reviews in independent_reviews.values())
        or _has_review_content(existing_adjudications)
    ):
        raise ValueError(
            "--restart cannot reuse edited review files because their decisions "
            "refer to the old proposals. Use a new experiment name or archive the "
            "review files before restarting."
        )

    if args.finalize_only:
        proposals = _read_jsonl(proposals_path)
        if not proposals:
            raise ValueError(
                f"No proposals found at {proposals_path}; run generation first."
            )
    else:
        source_samples, excluded = _load_source_samples(
            effective_cfg.get("source", {}),
            args.num_samples,
        )
        proposal_fingerprint = _proposal_fingerprint(
            source_samples,
            cfg.get("generator", {}),
            cfg.get("generation", {}),
            constraints_cfg,
            int(cfg.get("seed", 20260901)),
        )
        proposals = []
        if proposals_path.exists() and not args.restart:
            cached_meta = read_json(proposal_meta_path)
            if cached_meta.get("proposal_fingerprint") != proposal_fingerprint:
                raise ValueError(
                    "Existing proposal cache was created from different source, "
                    "generator, or constraint settings. Use a new experiment name "
                    "or pass --restart."
                )
            proposals = _read_jsonl(proposals_path)
            print(f"Resuming {len(proposals)} existing proposals")

        proposal_metadata = {
            "schema_version": 1,
            "proposal_fingerprint": proposal_fingerprint,
            "num_source_samples": len(source_samples),
            "num_excluded_source_samples": excluded,
            "num_proposals": len(proposals),
        }
        if args.restart:
            write_jsonl(proposals_path, [])
        write_json(proposal_meta_path, proposal_metadata)

        existing_by_id = {
            str(proposal["sample_id"]): proposal for proposal in proposals
        }
        if len(existing_by_id) != len(proposals):
            raise ValueError("Proposal cache contains duplicate sample IDs.")
        generator_cfg = cfg.get("generator", {})
        generation_cfg = cfg.get("generation", {})
        base_seed = int(cfg.get("seed", 20260901))
        generator = HFGenerator.load(
            str(generator_cfg["path"]),
            dtype=str(generator_cfg.get("dtype", "bfloat16")),
            device_map=str(generator_cfg.get("device_map", "auto")),
        )
        try:
            for position, sample in enumerate(source_samples, start=1):
                sample_id = stable_sample_id(sample)
                if sample_id in existing_by_id:
                    continue
                generation_seed = _sample_generation_seed(base_seed, sample_id)
                set_seed(generation_seed)
                proposal = generate_proposal(
                    generator=generator,
                    sample=sample,
                    constraints=constraints,
                    max_attempts=int(generation_cfg.get("max_attempts", 4)),
                    max_new_tokens=int(
                        generation_cfg.get("max_new_tokens", 128)
                    ),
                    temperature=float(generation_cfg.get("temperature", 0.8)),
                    top_p=float(generation_cfg.get("top_p", 0.9)),
                )
                proposal["generation_seed"] = generation_seed
                proposals.append(proposal)
                existing_by_id[sample_id] = proposal
                if position % 10 == 0 or position == len(source_samples):
                    write_jsonl(proposals_path, proposals)
                    proposal_metadata["num_proposals"] = len(proposals)
                    write_json(proposal_meta_path, proposal_metadata)
                    print(
                        f"Generated {position}/{len(source_samples)} proposals; "
                        f"structural pass={proposal['diagnostics']['structurally_valid']}"
                    )
        finally:
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_jsonl(proposals_path, proposals)
        proposal_metadata["num_proposals"] = len(proposals)
        write_json(proposal_meta_path, proposal_metadata)

    for reviews in independent_reviews.values():
        _validate_review_ids(proposals, reviews)
    _validate_review_ids(proposals, existing_adjudications)
    if not args.finalize_only:
        for path, reviews in zip(
            review_paths,
            independent_reviews.values(),
        ):
            write_csv(path, build_review_rows(proposals, reviews))
        if len(review_paths) == 2:
            _, audit_rows, adjudication_rows = reconcile_dual_reviews(
                proposals,
                independent_reviews,
                existing_adjudications,
                constraints,
                require_distinct_reviewers=bool(
                    review_cfg.get("require_distinct_reviewers", True)
                ),
            )
            if adjudication_rows:
                write_csv(adjudication_path, adjudication_rows)
            quality = _dual_quality_summary(cfg, proposals, audit_rows)
        else:
            pending_reviews = next(iter(independent_reviews.values()))
            _, audit_rows = finalize_reviewed_proposals(
                proposals,
                pending_reviews,
                constraints,
            )
            quality = _quality_summary(cfg, proposals, audit_rows)
        write_csv(audit_path, audit_rows)
        write_json(quality_path, quality)
        print(
            "Proposal generation complete. Review the independent files: "
            + ", ".join(str(path) for path in review_paths)
        )
        return

    if len(review_paths) == 2:
        final_rows, audit_rows, adjudication_rows = reconcile_dual_reviews(
            proposals,
            independent_reviews,
            existing_adjudications,
            constraints,
            require_distinct_reviewers=bool(
                review_cfg.get("require_distinct_reviewers", True)
            ),
        )
        if adjudication_rows:
            write_csv(adjudication_path, adjudication_rows)
        quality = _dual_quality_summary(cfg, proposals, audit_rows)
    else:
        final_rows, audit_rows = finalize_reviewed_proposals(
            proposals,
            next(iter(independent_reviews.values())),
            constraints,
        )
        quality = _quality_summary(cfg, proposals, audit_rows)
    write_csv(audit_path, audit_rows)
    write_json(quality_path, quality)
    if not quality["ready_for_detection"]:
        raise RuntimeError(
            "Controlled benchmark review gate failed. Inspect "
            f"{quality_path}, the independent review files, and "
            f"{adjudication_path}."
        )

    write_jsonl(final_dataset_path, final_rows)
    split_cfg = cfg.get("split", {})
    split_rows = assign_sample_splits(
        [str(row["sample_id"]) for row in final_rows],
        dev_fraction=float(split_cfg.get("dev_fraction", 0.2)),
        seed=int(split_cfg.get("seed", 20260901)),
    )
    write_csv(split_path, split_rows)
    print(f"Finalized {len(final_rows)} controlled source pairs")
    print(f"Dataset: {final_dataset_path}")
    print(f"Frozen split: {split_path}")


def _load_source_samples(
    source_cfg: dict[str, Any],
    num_samples_override: int | None,
) -> tuple[list[QASample], int]:
    configured_num_samples = source_cfg.get("num_samples", 500)
    num_samples = (
        num_samples_override
        if num_samples_override is not None
        else (
            None
            if configured_num_samples is None
            else int(configured_num_samples)
        )
    )
    seed = int(source_cfg.get("seed", 20260901))
    if source_cfg.get("cache_path"):
        samples = _load_prepared_english_cache(
            _resolve_repo_path(source_cfg["cache_path"]),
            num_samples=num_samples,
        )
    else:
        load_count = None if num_samples is None else num_samples + 50
        samples = load_halueval_qa(
            num_samples=load_count,
            seed=seed,
            dataset_cfg=dict(source_cfg.get("dataset", {})),
        )

    review_manifest_path = source_cfg.get("review_manifest")
    manifest = load_review_manifest(
        _resolve_repo_path(review_manifest_path)
        if review_manifest_path
        else None
    )
    retained: list[QASample] = []
    excluded = 0
    for sample in samples:
        sample_id = stable_sample_id(sample)
        if sample_review(manifest, sample_id).get(
            "exclude_from_primary_metrics", False
        ):
            excluded += 1
            continue
        retained.append(sample)
        if num_samples is not None and len(retained) >= num_samples:
            break
    if not retained:
        raise ValueError("No source samples remain after source review exclusions.")
    print(
        f"Loaded {len(retained)} source samples; excluded {excluded} reviewed defects"
    )
    return retained, excluded


def _load_prepared_english_cache(
    path: Path,
    num_samples: int | None,
) -> list[QASample]:
    rows = _read_jsonl(path)
    samples: list[QASample] = []
    for row in rows:
        payload = row.get("source") or row.get("translated") or row
        samples.append(
            QASample(
                index=len(samples),
                question=str(payload["question"]),
                context=str(payload.get("knowledge", payload.get("context", ""))),
                right_answer=str(payload["right_answer"]),
                hallucinated_answer=str(payload["hallucinated_answer"]),
                sample_id=str(row.get("sample_id") or "") or None,
            )
        )
        if num_samples is not None and len(samples) >= num_samples:
            break
    return samples


def _quality_summary(
    cfg: dict[str, Any],
    proposals: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    review_cfg = cfg.get("review", {})
    return summarize_reviews(
        proposals,
        audit_rows,
        minimum_approved_samples=int(
            review_cfg.get("minimum_approved_samples", len(proposals))
        ),
        require_all_reviewed=bool(
            review_cfg.get("require_all_reviewed", True)
        ),
    )


def _dual_quality_summary(
    cfg: dict[str, Any],
    proposals: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    review_cfg = cfg.get("review", {})
    return summarize_dual_reviews(
        proposals,
        audit_rows,
        minimum_approved_samples=int(
            review_cfg.get("minimum_approved_samples", len(proposals))
        ),
        require_all_reviewed=bool(
            review_cfg.get("require_all_reviewed", True)
        ),
    )


def _proposal_fingerprint(
    samples: list[QASample],
    generator_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    constraints_cfg: dict[str, Any],
    seed: int,
) -> str:
    payload = {
        "samples": [
            {
                "sample_id": stable_sample_id(sample),
                "question": sample.question,
                "knowledge": sample.context,
                "right_answer": sample.right_answer,
                "hallucinated_answer": sample.hallucinated_answer,
            }
            for sample in samples
        ],
        "generator": generator_cfg,
        "generation": generation_cfg,
        "constraints": constraints_cfg,
        "seed": seed,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_review_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        reviews: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(rows, start=2):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(
                    f"Review CSV row {row_number} has no sample_id."
                )
            if sample_id in reviews:
                raise ValueError(
                    f"Review CSV contains duplicate sample_id {sample_id!r}."
                )
            reviews[sample_id] = {
                "review_status": str(row.get("review_status", "pending")),
                "reviewed_candidate": str(row.get("reviewed_candidate", "")),
                "review_reason": str(row.get("review_reason", "")),
                "reviewer": str(row.get("reviewer", "")),
            }
        return reviews


def _review_paths(output_root: Path, review_cfg: dict[str, Any]) -> list[Path]:
    configured = review_cfg.get("files")
    if configured is None:
        configured = [review_cfg.get("file", "proposal_review.csv")]
    if not isinstance(configured, list) or len(configured) not in {1, 2}:
        raise ValueError("review.files must contain one or two CSV filenames.")
    names = [str(value).strip() for value in configured]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Review CSV filenames must be non-empty and unique.")
    return [output_root / name for name in names]


def _validate_review_ids(
    proposals: list[dict[str, Any]],
    reviews: dict[str, dict[str, str]],
) -> None:
    proposal_ids = [str(proposal["sample_id"]) for proposal in proposals]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("Proposals contain duplicate sample IDs.")
    unknown_ids = sorted(set(reviews) - set(proposal_ids))
    if unknown_ids:
        raise ValueError(
            "Review CSV contains IDs absent from the proposal cache: "
            f"{unknown_ids[:5]}"
        )


def _has_review_content(reviews: dict[str, dict[str, str]]) -> bool:
    for review in reviews.values():
        status = review.get("review_status", "pending").strip().lower()
        if status not in {"", "pending"}:
            return True
        if any(
            review.get(field, "").strip()
            for field in ("reviewed_candidate", "review_reason", "reviewer")
        ):
            return True
    return False


def _sample_generation_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


if __name__ == "__main__":
    main()
