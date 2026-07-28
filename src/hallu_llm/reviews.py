from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_TRANSLATION_FIELDS = {
    "question",
    "knowledge",
    "right_answer",
    "hallucinated_answer",
}


def load_review_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"schema_version": 1, "samples": {}}
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_review_manifest(manifest)
    return manifest


def review_manifest_metadata(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    samples = manifest.get("samples", {})
    translation_reviews = sum(
        len(sample.get("translations", {}))
        for sample in samples.values()
    )
    excluded_samples = sum(
        bool(sample.get("exclude_from_primary_metrics", False))
        for sample in samples.values()
    )
    return {
        "schema_version": int(manifest.get("schema_version", 1)),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "sample_reviews": len(samples),
        "translation_reviews": translation_reviews,
        "primary_metric_exclusions": excluded_samples,
    }


def sample_review(
    manifest: Mapping[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    return dict(manifest.get("samples", {}).get(sample_id, {}))


def translation_review(
    manifest: Mapping[str, Any],
    sample_id: str,
    language_code: str,
) -> dict[str, Any]:
    review = sample_review(manifest, sample_id)
    return dict(review.get("translations", {}).get(language_code, {}))


def _validate_review_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Review manifest must contain a JSON object.")
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Review manifest schema_version must be 1.")
    samples = manifest.get("samples", {})
    if not isinstance(samples, dict):
        raise ValueError("Review manifest samples must be a JSON object.")

    for sample_id, review in samples.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Review manifest sample IDs must be non-empty strings.")
        if not isinstance(review, dict):
            raise ValueError(f"Review for sample {sample_id} must be an object.")
        if "exclude_from_primary_metrics" in review and not isinstance(
            review["exclude_from_primary_metrics"],
            bool,
        ):
            raise ValueError(
                f"exclude_from_primary_metrics for {sample_id} must be boolean."
            )
        if review.get("exclude_from_primary_metrics") and not str(
            review.get("reason", "")
        ).strip():
            raise ValueError(
                f"Excluded sample {sample_id} requires a non-empty reason."
            )

        translations = review.get("translations", {})
        if not isinstance(translations, dict):
            raise ValueError(
                f"Translations for sample {sample_id} must be an object."
            )
        for language_code, translation in translations.items():
            if not isinstance(translation, dict):
                raise ValueError(
                    f"Translation review {sample_id}/{language_code} "
                    "must be an object."
                )
            if not str(translation.get("reason", "")).strip():
                raise ValueError(
                    f"Translation review {sample_id}/{language_code} "
                    "requires a non-empty reason."
                )
            fields = translation.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError(
                    f"Translation fields {sample_id}/{language_code} "
                    "must be an object."
                )
            unsupported = set(fields) - SUPPORTED_TRANSLATION_FIELDS
            if unsupported:
                raise ValueError(
                    f"Unsupported translation fields for "
                    f"{sample_id}/{language_code}: {sorted(unsupported)}"
                )
            for field, value in fields.items():
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Translation override {sample_id}/{language_code}/"
                        f"{field} must be a non-empty string."
                    )
            if "accept_grounded_alignment" in translation and not isinstance(
                translation["accept_grounded_alignment"],
                bool,
            ):
                raise ValueError(
                    f"accept_grounded_alignment for "
                    f"{sample_id}/{language_code} must be boolean."
                )
            accepted_warnings = translation.get("accepted_warnings", [])
            if (
                not isinstance(accepted_warnings, list)
                or not all(
                    isinstance(warning, str) and warning.strip()
                    for warning in accepted_warnings
                )
            ):
                raise ValueError(
                    f"accepted_warnings for {sample_id}/{language_code} "
                    "must be a list of non-empty strings."
                )
