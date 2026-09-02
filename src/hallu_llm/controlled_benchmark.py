from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Protocol

from .data import QASample
from .detection import stable_sample_id


REVIEW_STATUSES = {"pending", "approved", "rejected"}
SENTENCE_END = re.compile(r"[.!?。！？]+")
WHITESPACE = re.compile(r"\s+")
LABEL_PREFIX = re.compile(
    r"^(?:candidate|answer|hallucinated answer|changed claim)\s*:\s*",
    flags=re.IGNORECASE,
)
CAPITALIZED_SPAN = re.compile(
    r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*\b"
)


class TextGenerator(Protocol):
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Any: ...


@dataclass(frozen=True)
class CandidateConstraints:
    max_relative_length_difference: float = 0.25
    max_sentence_count_difference: int = 0
    max_named_entity_count_difference: int = 0
    require_answer_style_match: bool = True
    forbid_newlines: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.max_relative_length_difference < 1:
            raise ValueError(
                "max_relative_length_difference must be in [0, 1)."
            )
        if self.max_sentence_count_difference < 0:
            raise ValueError(
                "max_sentence_count_difference must be non-negative."
            )
        if self.max_named_entity_count_difference < 0:
            raise ValueError(
                "max_named_entity_count_difference must be non-negative."
            )


def content_character_length(text: str) -> int:
    return sum(character.isalnum() for character in text)


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    endings = len(SENTENCE_END.findall(stripped))
    return max(1, endings)


def normalized_text(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip().casefold()


def named_entity_count(text: str) -> int:
    return len(CAPITALIZED_SPAN.findall(text))


def answer_style(text: str) -> str:
    cleaned = text.strip()
    normalized = normalized_text(cleaned)
    if normalized in {"yes", "no", "true", "false"}:
        return "boolean"
    alphanumeric = [character for character in cleaned if character.isalnum()]
    if alphanumeric and all(character.isdigit() for character in alphanumeric):
        return "numeric"
    if any(separator in cleaned for separator in (";", "|", "\n")):
        return "list"
    if cleaned.endswith((".", "!", "?", "。", "！", "？")):
        return "sentence"
    return "fragment"


def candidate_diagnostics(
    grounded_answer: str,
    candidate: str,
    constraints: CandidateConstraints,
) -> dict[str, Any]:
    grounded_length = content_character_length(grounded_answer)
    candidate_length = content_character_length(candidate)
    denominator = max(grounded_length, 1)
    relative_difference = abs(candidate_length - grounded_length) / denominator
    grounded_sentences = sentence_count(grounded_answer)
    candidate_sentences = sentence_count(candidate)
    grounded_entities = named_entity_count(grounded_answer)
    candidate_entities = named_entity_count(candidate)
    grounded_style = answer_style(grounded_answer)
    candidate_style = answer_style(candidate)
    issues: list[str] = []
    cleaned = candidate.strip()

    if not cleaned:
        issues.append("empty_candidate")
    if normalized_text(cleaned) == normalized_text(grounded_answer):
        issues.append("same_as_grounded")
    if LABEL_PREFIX.match(cleaned):
        issues.append("label_prefix_present")
    if constraints.forbid_newlines and "\n" in cleaned:
        issues.append("contains_newline")
    if relative_difference > constraints.max_relative_length_difference:
        issues.append("length_out_of_range")
    if (
        abs(candidate_sentences - grounded_sentences)
        > constraints.max_sentence_count_difference
    ):
        issues.append("sentence_count_mismatch")
    if (
        abs(candidate_entities - grounded_entities)
        > constraints.max_named_entity_count_difference
    ):
        issues.append("named_entity_count_mismatch")
    if constraints.require_answer_style_match and candidate_style != grounded_style:
        issues.append("answer_style_mismatch")

    return {
        "grounded_character_length": grounded_length,
        "candidate_character_length": candidate_length,
        "relative_length_difference": relative_difference,
        "grounded_sentence_count": grounded_sentences,
        "candidate_sentence_count": candidate_sentences,
        "grounded_named_entity_count": grounded_entities,
        "candidate_named_entity_count": candidate_entities,
        "grounded_answer_style": grounded_style,
        "candidate_answer_style": candidate_style,
        "text_similarity": SequenceMatcher(
            None,
            normalized_text(grounded_answer),
            normalized_text(cleaned),
        ).ratio(),
        "issues": issues,
        "structurally_valid": not issues,
    }


def length_bounds(
    grounded_answer: str,
    max_relative_difference: float,
) -> tuple[int, int]:
    length = max(content_character_length(grounded_answer), 1)
    lower = max(1, math.ceil(length * (1 - max_relative_difference)))
    upper = max(lower, math.floor(length * (1 + max_relative_difference)))
    return lower, upper


def generation_prompt(
    sample: QASample,
    constraints: CandidateConstraints,
    attempt: int,
    previous_issues: list[str] | None = None,
) -> str:
    lower, upper = length_bounds(
        sample.right_answer,
        constraints.max_relative_length_difference,
    )
    issue_feedback = ""
    if previous_issues:
        issue_feedback = (
            "\nThe previous attempt failed these mechanical checks: "
            + ", ".join(previous_issues)
            + ". Correct them in this attempt."
        )
    return (
        "Create one hard negative answer for a hallucination-detection benchmark.\n"
        "The new answer must directly answer the question, contain exactly one "
        "unsupported or contradicted factual detail, and otherwise remain plausible. "
        "Prefer a minimal edit of the grounded answer. Do not add explanations, "
        "qualifiers, citations, or statements about support.\n"
        f"Count only letters and digits. The candidate must contain {lower} to "
        f"{upper} such characters and differ by no more than "
        f"{constraints.max_sentence_count_difference} sentence(s) from the grounded "
        "answer. Match its answer style and named-entity count.\n"
        "Return exactly one JSON object with string fields candidate and changed_claim."
        f"{issue_feedback}\n\n"
        f"Attempt: {attempt}\n"
        f"Knowledge: {sample.context}\n\n"
        f"Question: {sample.question}\n\n"
        f"Grounded answer: {sample.right_answer}\n\n"
        f"Original HaluEval negative for semantic inspiration only: "
        f"{sample.hallucinated_answer}"
    )


def parse_generated_proposal(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            candidate = str(payload.get("candidate", "")).strip()
            changed_claim = str(payload.get("changed_claim", "")).strip()
            if candidate:
                return _strip_wrapping_quotes(candidate), changed_claim

    candidate_match = re.search(
        r"(?:^|\n)\s*CANDIDATE\s*:\s*(.+?)(?=\n\s*(?:CHANGE|CHANGED_CLAIM)\s*:|$)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    change_match = re.search(
        r"(?:^|\n)\s*(?:CHANGE|CHANGED_CLAIM)\s*:\s*(.+)$",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if candidate_match:
        return (
            _strip_wrapping_quotes(candidate_match.group(1).strip()),
            change_match.group(1).strip() if change_match else "",
        )
    return "", ""


def generate_proposal(
    generator: TextGenerator,
    sample: QASample,
    constraints: CandidateConstraints,
    max_attempts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    attempts: list[dict[str, Any]] = []
    previous_issues: list[str] = []
    for attempt_number in range(1, max_attempts + 1):
        prompt = generation_prompt(
            sample,
            constraints,
            attempt=attempt_number,
            previous_issues=previous_issues,
        )
        generation = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        candidate, changed_claim = parse_generated_proposal(generation.text)
        diagnostics = candidate_diagnostics(
            sample.right_answer,
            candidate,
            constraints,
        )
        attempt = {
            "attempt": attempt_number,
            "raw_generation": generation.text,
            "candidate": candidate,
            "changed_claim": changed_claim,
            "mean_token_confidence": generation.mean_confidence,
            "mean_token_entropy": generation.mean_entropy,
            "diagnostics": diagnostics,
        }
        attempts.append(attempt)
        if diagnostics["structurally_valid"]:
            break
        previous_issues = list(diagnostics["issues"])

    selected = min(attempts, key=_proposal_sort_key)
    sample_id = stable_sample_id(sample)
    return {
        "sample_index": sample.index,
        "sample_id": sample_id,
        "question": sample.question,
        "knowledge": sample.context,
        "grounded_answer": sample.right_answer,
        "original_hallucinated_answer": sample.hallucinated_answer,
        "proposed_hallucinated_answer": selected["candidate"],
        "changed_claim": selected["changed_claim"],
        "selected_attempt": selected["attempt"],
        "diagnostics": selected["diagnostics"],
        "attempts": attempts,
    }


def build_review_rows(
    proposals: list[dict[str, Any]],
    existing_reviews: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    reviews = existing_reviews or {}
    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        sample_id = str(proposal["sample_id"])
        review = reviews.get(sample_id, {})
        diagnostics = proposal["diagnostics"]
        rows.append(
            {
                "sample_id": sample_id,
                "sample_index": proposal["sample_index"],
                "question": proposal["question"],
                "knowledge": proposal["knowledge"],
                "grounded_answer": proposal["grounded_answer"],
                "original_hallucinated_answer": proposal[
                    "original_hallucinated_answer"
                ],
                "proposed_hallucinated_answer": proposal[
                    "proposed_hallucinated_answer"
                ],
                "changed_claim": proposal["changed_claim"],
                "grounded_character_length": diagnostics[
                    "grounded_character_length"
                ],
                "proposed_character_length": diagnostics[
                    "candidate_character_length"
                ],
                "relative_length_difference": diagnostics[
                    "relative_length_difference"
                ],
                "structurally_valid": diagnostics["structurally_valid"],
                "structural_issues": ";".join(diagnostics["issues"]),
                "grounded_named_entity_count": diagnostics[
                    "grounded_named_entity_count"
                ],
                "proposed_named_entity_count": diagnostics[
                    "candidate_named_entity_count"
                ],
                "grounded_answer_style": diagnostics["grounded_answer_style"],
                "proposed_answer_style": diagnostics["candidate_answer_style"],
                "review_status": str(review.get("review_status", "pending")),
                "reviewed_candidate": str(review.get("reviewed_candidate", "")),
                "review_reason": str(review.get("review_reason", "")),
                "reviewer": str(review.get("reviewer", "")),
            }
        )
    return rows


def finalize_reviewed_proposals(
    proposals: list[dict[str, Any]],
    reviews: Mapping[str, Mapping[str, str]],
    constraints: CandidateConstraints,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for proposal in proposals:
        sample_id = str(proposal["sample_id"])
        review = reviews.get(sample_id, {})
        status = str(review.get("review_status", "pending")).strip().lower()
        if status not in REVIEW_STATUSES:
            raise ValueError(
                f"Invalid review_status {status!r} for sample {sample_id}."
            )
        reviewed_candidate = str(review.get("reviewed_candidate", "")).strip()
        candidate = reviewed_candidate or str(
            proposal["proposed_hallucinated_answer"]
        ).strip()
        reason = str(review.get("review_reason", "")).strip()
        reviewer = str(review.get("reviewer", "")).strip()
        diagnostics = candidate_diagnostics(
            str(proposal["grounded_answer"]),
            candidate,
            constraints,
        )
        if status in {"approved", "rejected"}:
            if not reason:
                raise ValueError(
                    f"Reviewed sample {sample_id} requires a review_reason."
                )
            if not reviewer:
                raise ValueError(
                    f"Reviewed sample {sample_id} requires a reviewer."
                )
        if status == "approved" and not diagnostics["structurally_valid"]:
            raise ValueError(
                f"Approved sample {sample_id} fails structural checks: "
                f"{diagnostics['issues']}"
            )

        audit_rows.append(
            {
                "sample_id": sample_id,
                "sample_index": proposal["sample_index"],
                "review_status": status,
                "reviewer": reviewer,
                "review_reason": reason,
                "candidate_source": (
                    "review_override" if reviewed_candidate else "generated_proposal"
                ),
                "grounded_answer": proposal["grounded_answer"],
                "final_hallucinated_answer": candidate,
                "grounded_character_length": diagnostics[
                    "grounded_character_length"
                ],
                "candidate_character_length": diagnostics[
                    "candidate_character_length"
                ],
                "relative_length_difference": diagnostics[
                    "relative_length_difference"
                ],
                "structurally_valid": diagnostics["structurally_valid"],
                "structural_issues": ";".join(diagnostics["issues"]),
                "grounded_named_entity_count": diagnostics[
                    "grounded_named_entity_count"
                ],
                "candidate_named_entity_count": diagnostics[
                    "candidate_named_entity_count"
                ],
                "grounded_answer_style": diagnostics["grounded_answer_style"],
                "candidate_answer_style": diagnostics["candidate_answer_style"],
            }
        )
        if status != "approved":
            continue
        final_rows.append(
            {
                "sample_index": len(final_rows),
                "sample_id": sample_id,
                "question": proposal["question"],
                "knowledge": proposal["knowledge"],
                "right_answer": proposal["grounded_answer"],
                "hallucinated_answer": candidate,
                "benchmark_provenance": {
                    "construction": "human_approved_length_controlled_negative",
                    "generator_changed_claim": proposal["changed_claim"],
                    "reviewer": reviewer,
                    "review_reason": reason,
                    "candidate_source": (
                        "review_override"
                        if reviewed_candidate
                        else "generated_proposal"
                    ),
                    "relative_length_difference": diagnostics[
                        "relative_length_difference"
                    ],
                },
            }
        )
    return final_rows, audit_rows


def build_adjudication_rows(
    proposals: list[dict[str, Any]],
    independent_reviews: Mapping[
        str, Mapping[str, Mapping[str, str]]
    ],
    constraints: CandidateConstraints,
    existing_adjudications: Mapping[str, Mapping[str, str]] | None = None,
    require_distinct_reviewers: bool = True,
) -> list[dict[str, Any]]:
    review_slots = list(independent_reviews)
    if len(review_slots) != 2:
        raise ValueError("Dual review requires exactly two independent review files.")
    existing = existing_adjudications or {}
    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        sample_id = str(proposal["sample_id"])
        decisions = [
            _review_decision(
                proposal,
                independent_reviews[slot].get(sample_id, {}),
                constraints,
                slot,
            )
            for slot in review_slots
        ]
        _validate_reviewer_independence(
            sample_id,
            decisions,
            require_distinct_reviewers,
        )
        if any(decision["status"] == "pending" for decision in decisions):
            continue
        if _review_decisions_agree(decisions[0], decisions[1]):
            continue

        adjudication = existing.get(sample_id, {})
        approved_candidates = [
            str(decision["candidate"])
            for decision in decisions
            if decision["status"] == "approved"
        ]
        suggested_candidate = (
            approved_candidates[0]
            if len(set(approved_candidates)) == 1
            else ""
        )
        rows.append(
            {
                "sample_id": sample_id,
                "sample_index": proposal["sample_index"],
                "question": proposal["question"],
                "knowledge": proposal["knowledge"],
                "grounded_answer": proposal["grounded_answer"],
                "proposed_hallucinated_answer": proposal[
                    "proposed_hallucinated_answer"
                ],
                "review_1_status": decisions[0]["status"],
                "review_1_candidate": decisions[0]["candidate"],
                "review_1_reason": decisions[0]["reason"],
                "review_1_reviewer": decisions[0]["reviewer"],
                "review_2_status": decisions[1]["status"],
                "review_2_candidate": decisions[1]["candidate"],
                "review_2_reason": decisions[1]["reason"],
                "review_2_reviewer": decisions[1]["reviewer"],
                "review_status": str(
                    adjudication.get("review_status", "pending")
                ),
                "reviewed_candidate": str(
                    adjudication.get("reviewed_candidate", suggested_candidate)
                ),
                "review_reason": str(adjudication.get("review_reason", "")),
                "reviewer": str(adjudication.get("reviewer", "")),
            }
        )
    return rows


def reconcile_dual_reviews(
    proposals: list[dict[str, Any]],
    independent_reviews: Mapping[
        str, Mapping[str, Mapping[str, str]]
    ],
    adjudications: Mapping[str, Mapping[str, str]],
    constraints: CandidateConstraints,
    require_distinct_reviewers: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    review_slots = list(independent_reviews)
    adjudication_rows = build_adjudication_rows(
        proposals,
        independent_reviews,
        constraints,
        existing_adjudications=adjudications,
        require_distinct_reviewers=require_distinct_reviewers,
    )
    adjudication_by_id = {
        str(row["sample_id"]): row for row in adjudication_rows
    }
    final_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for proposal in proposals:
        sample_id = str(proposal["sample_id"])
        decisions = [
            _review_decision(
                proposal,
                independent_reviews[slot].get(sample_id, {}),
                constraints,
                slot,
            )
            for slot in review_slots
        ]
        _validate_reviewer_independence(
            sample_id,
            decisions,
            require_distinct_reviewers,
        )
        both_complete = all(
            decision["status"] != "pending" for decision in decisions
        )
        status_agreement = (
            both_complete and decisions[0]["status"] == decisions[1]["status"]
        )
        exact_agreement = both_complete and _review_decisions_agree(
            decisions[0], decisions[1]
        )
        adjudication_required = both_complete and not exact_agreement
        adjudicated = False
        adjudication_decision: dict[str, Any] | None = None

        if not both_complete:
            final_status = "pending"
            final_candidate = str(proposal["proposed_hallucinated_answer"])
            final_reason = ""
            final_reviewer = ""
        elif exact_agreement:
            final_status = str(decisions[0]["status"])
            final_candidate = str(decisions[0]["candidate"])
            final_reason = " | ".join(
                str(decision["reason"]) for decision in decisions
            )
            final_reviewer = " | ".join(
                str(decision["reviewer"]) for decision in decisions
            )
        else:
            adjudication_row = adjudication_by_id[sample_id]
            adjudication_decision = _review_decision(
                proposal,
                adjudication_row,
                constraints,
                "adjudication",
            )
            if adjudication_decision["status"] == "pending":
                final_status = "adjudication_pending"
            else:
                final_status = str(adjudication_decision["status"])
                adjudicated = True
            final_candidate = str(adjudication_decision["candidate"])
            final_reason = str(adjudication_decision["reason"])
            final_reviewer = str(adjudication_decision["reviewer"])

        diagnostics = candidate_diagnostics(
            str(proposal["grounded_answer"]),
            final_candidate,
            constraints,
        )
        audit_rows.append(
            {
                "sample_id": sample_id,
                "sample_index": proposal["sample_index"],
                "review_1_status": decisions[0]["status"],
                "review_1_reviewer": decisions[0]["reviewer"],
                "review_1_candidate": decisions[0]["candidate"],
                "review_2_status": decisions[1]["status"],
                "review_2_reviewer": decisions[1]["reviewer"],
                "review_2_candidate": decisions[1]["candidate"],
                "both_independent_reviews_complete": both_complete,
                "review_status_agreement": status_agreement,
                "review_exact_agreement": exact_agreement,
                "adjudication_required": adjudication_required,
                "adjudicated": adjudicated,
                "final_review_status": final_status,
                "final_reviewer": final_reviewer,
                "final_review_reason": final_reason,
                "grounded_answer": proposal["grounded_answer"],
                "final_hallucinated_answer": final_candidate,
                "relative_length_difference": diagnostics[
                    "relative_length_difference"
                ],
                "structurally_valid": diagnostics["structurally_valid"],
                "structural_issues": ";".join(diagnostics["issues"]),
                "grounded_named_entity_count": diagnostics[
                    "grounded_named_entity_count"
                ],
                "candidate_named_entity_count": diagnostics[
                    "candidate_named_entity_count"
                ],
                "grounded_answer_style": diagnostics["grounded_answer_style"],
                "candidate_answer_style": diagnostics["candidate_answer_style"],
            }
        )
        if final_status != "approved":
            continue
        final_rows.append(
            {
                "sample_index": len(final_rows),
                "sample_id": sample_id,
                "question": proposal["question"],
                "knowledge": proposal["knowledge"],
                "right_answer": proposal["grounded_answer"],
                "hallucinated_answer": final_candidate,
                "benchmark_provenance": {
                    "construction": (
                        "dual_reviewed_length_controlled_negative"
                    ),
                    "generator_changed_claim": proposal["changed_claim"],
                    "independent_reviews": [
                        {
                            "slot": review_slots[position],
                            "status": decision["status"],
                            "reviewer": decision["reviewer"],
                            "review_reason": decision["reason"],
                            "candidate": decision["candidate"],
                        }
                        for position, decision in enumerate(decisions)
                    ],
                    "adjudication": (
                        {
                            "status": adjudication_decision["status"],
                            "reviewer": adjudication_decision["reviewer"],
                            "review_reason": adjudication_decision["reason"],
                            "candidate": adjudication_decision["candidate"],
                        }
                        if adjudication_decision is not None
                        else None
                    ),
                    "relative_length_difference": diagnostics[
                        "relative_length_difference"
                    ],
                },
            }
        )
    return final_rows, audit_rows, adjudication_rows


def summarize_dual_reviews(
    proposals: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    minimum_approved_samples: int,
    require_all_reviewed: bool,
) -> dict[str, Any]:
    statuses = ("pending", "adjudication_pending", "approved", "rejected")
    counts = {
        status: sum(row["final_review_status"] == status for row in audit_rows)
        for status in statuses
    }
    complete = [
        row
        for row in audit_rows
        if row["both_independent_reviews_complete"]
    ]
    approved = [
        row for row in audit_rows if row["final_review_status"] == "approved"
    ]
    unresolved = counts["pending"] + counts["adjudication_pending"]
    ready = len(approved) >= minimum_approved_samples and (
        unresolved == 0 or not require_all_reviewed
    )
    review_1_labels = [str(row["review_1_status"]) for row in complete]
    review_2_labels = [str(row["review_2_status"]) for row in complete]
    return {
        "schema_version": 1,
        "review_protocol": "dual_independent_review_with_adjudication",
        "num_proposals": len(proposals),
        "num_structurally_valid_proposals": sum(
            bool(row["diagnostics"]["structurally_valid"])
            for row in proposals
        ),
        "review_counts": counts,
        "minimum_approved_samples": minimum_approved_samples,
        "require_all_reviewed": require_all_reviewed,
        "all_samples_reviewed": unresolved == 0,
        "ready_for_detection": ready,
        "num_complete_independent_pairs": len(complete),
        "status_percent_agreement": _mean(
            [float(row["review_status_agreement"]) for row in complete]
        ),
        "exact_percent_agreement": _mean(
            [float(row["review_exact_agreement"]) for row in complete]
        ),
        "cohen_kappa_status": _cohen_kappa(
            review_1_labels,
            review_2_labels,
        ),
        "num_adjudications_required": sum(
            bool(row["adjudication_required"]) for row in audit_rows
        ),
        "num_adjudicated": sum(bool(row["adjudicated"]) for row in audit_rows),
        "mean_approved_relative_length_difference": _mean(
            [float(row["relative_length_difference"]) for row in approved]
        ),
        "max_approved_relative_length_difference": max(
            (float(row["relative_length_difference"]) for row in approved),
            default=None,
        ),
    }


def summarize_reviews(
    proposals: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    minimum_approved_samples: int,
    require_all_reviewed: bool,
) -> dict[str, Any]:
    counts = {status: 0 for status in REVIEW_STATUSES}
    for row in audit_rows:
        counts[str(row["review_status"])] += 1
    approved_rows = [
        row for row in audit_rows if row["review_status"] == "approved"
    ]
    all_reviewed = counts["pending"] == 0
    ready = len(approved_rows) >= minimum_approved_samples and (
        all_reviewed or not require_all_reviewed
    )
    return {
        "schema_version": 1,
        "num_proposals": len(proposals),
        "num_structurally_valid_proposals": sum(
            bool(row["diagnostics"]["structurally_valid"])
            for row in proposals
        ),
        "review_counts": counts,
        "minimum_approved_samples": minimum_approved_samples,
        "require_all_reviewed": require_all_reviewed,
        "all_samples_reviewed": all_reviewed,
        "ready_for_detection": ready,
        "mean_approved_relative_length_difference": _mean(
            [float(row["relative_length_difference"]) for row in approved_rows]
        ),
        "max_approved_relative_length_difference": max(
            (
                float(row["relative_length_difference"])
                for row in approved_rows
            ),
            default=None,
        ),
    }


def _proposal_sort_key(attempt: Mapping[str, Any]) -> tuple[int, int, float, int]:
    diagnostics = attempt["diagnostics"]
    return (
        0 if diagnostics["structurally_valid"] else 1,
        len(diagnostics["issues"]),
        float(diagnostics["relative_length_difference"]),
        int(attempt["attempt"]),
    )


def _review_decision(
    proposal: Mapping[str, Any],
    review: Mapping[str, str],
    constraints: CandidateConstraints,
    role: str,
) -> dict[str, Any]:
    sample_id = str(proposal["sample_id"])
    status = str(review.get("review_status", "pending")).strip().lower()
    if status not in REVIEW_STATUSES:
        raise ValueError(
            f"Invalid review_status {status!r} for {role} on sample {sample_id}."
        )
    reviewed_candidate = str(review.get("reviewed_candidate", "")).strip()
    candidate = reviewed_candidate or str(
        proposal["proposed_hallucinated_answer"]
    ).strip()
    reason = str(review.get("review_reason", "")).strip()
    reviewer = str(review.get("reviewer", "")).strip()
    diagnostics = candidate_diagnostics(
        str(proposal["grounded_answer"]),
        candidate,
        constraints,
    )
    if status in {"approved", "rejected"}:
        if not reason:
            raise ValueError(
                f"{role} decision for sample {sample_id} requires a review_reason."
            )
        if not reviewer:
            raise ValueError(
                f"{role} decision for sample {sample_id} requires a reviewer."
            )
    if status == "approved" and not diagnostics["structurally_valid"]:
        raise ValueError(
            f"Approved {role} candidate for sample {sample_id} fails "
            f"structural checks: {diagnostics['issues']}"
        )
    return {
        "status": status,
        "candidate": candidate,
        "reason": reason,
        "reviewer": reviewer,
        "diagnostics": diagnostics,
    }


def _review_decisions_agree(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    if first["status"] != second["status"]:
        return False
    if first["status"] != "approved":
        return True
    return normalized_text(str(first["candidate"])) == normalized_text(
        str(second["candidate"])
    )


def _validate_reviewer_independence(
    sample_id: str,
    decisions: list[dict[str, Any]],
    required: bool,
) -> None:
    if not required or any(
        decision["status"] == "pending" for decision in decisions
    ):
        return
    reviewers = [normalized_text(str(decision["reviewer"])) for decision in decisions]
    if reviewers[0] == reviewers[1]:
        raise ValueError(
            f"Sample {sample_id} requires two distinct independent reviewers."
        )


def _cohen_kappa(first: list[str], second: list[str]) -> float | None:
    if not first or len(first) != len(second):
        return None
    labels = sorted(set(first) | set(second))
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    expected = sum(
        (first.count(label) / len(first)) * (second.count(label) / len(second))
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {
        "'",
        '"',
    }:
        return stripped[1:-1].strip()
    return stripped


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
