from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.data import QASample
from hallu_llm.reviews import (
    load_review_manifest,
    review_manifest_metadata,
    sample_review,
    translation_review,
)
from hallu_llm.translation import TranslationResult
from scripts.run_halueval_detection import _apply_translation_review
from scripts.run_halueval_detection import (
    _build_translation_cache_row,
    _can_refresh_translation_cache,
)


class ReviewManifestTests(unittest.TestCase):
    def test_loads_review_and_reports_stable_metadata(self) -> None:
        manifest = {
            "schema_version": 1,
            "samples": {
                "sample-a": {
                    "exclude_from_primary_metrics": True,
                    "reason": "Both candidates are supported.",
                    "translations": {
                        "zh": {
                            "reason": "Reviewed translation.",
                            "fields": {"right_answer": "正确答案"},
                        }
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            loaded = load_review_manifest(path)

        metadata = review_manifest_metadata(loaded)
        self.assertTrue(
            sample_review(loaded, "sample-a")[
                "exclude_from_primary_metrics"
            ]
        )
        self.assertEqual(
            translation_review(loaded, "sample-a", "zh")["fields"][
                "right_answer"
            ],
            "正确答案",
        )
        self.assertEqual(metadata["sample_reviews"], 1)
        self.assertEqual(metadata["translation_reviews"], 1)
        self.assertEqual(metadata["primary_metric_exclusions"], 1)

    def test_rejects_unexplained_metric_exclusion(self) -> None:
        manifest = {
            "schema_version": 1,
            "samples": {
                "sample-a": {
                    "exclude_from_primary_metrics": True,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires a non-empty reason"):
                load_review_manifest(path)

    def test_applies_reviewed_fields_and_retains_machine_text(self) -> None:
        translated = QASample(
            0,
            "问题",
            "机器知识",
            "机器答案",
            "错误答案",
        )
        diagnostics = {
            field: TranslationResult(
                text=text,
                source_tokens=10,
                output_tokens=8,
                input_truncated=False,
                output_truncated=False,
            )
            for field, text in {
                "question": translated.question,
                "knowledge": translated.context,
                "right_answer": translated.right_answer,
                "hallucinated_answer": translated.hallucinated_answer,
            }.items()
        }

        reviewed, reviewed_diagnostics = _apply_translation_review(
            translated,
            diagnostics,
            {
                "reason": "Corrected mistranslation.",
                "fields": {
                    "knowledge": "人工复核知识",
                    "right_answer": "人工复核答案",
                },
            },
        )

        self.assertEqual(reviewed.context, "人工复核知识")
        self.assertEqual(reviewed.right_answer, "人工复核答案")
        self.assertEqual(reviewed.question, "问题")
        self.assertEqual(
            reviewed_diagnostics["right_answer"].strategy,
            "review_override",
        )
        self.assertEqual(
            reviewed_diagnostics["right_answer"].initial_text,
            "机器答案",
        )

    def test_review_can_accept_an_exact_audited_warning(self) -> None:
        source = QASample(
            0,
            "Who died first?",
            "The malformed source date is 194018.",
            "Person A",
            "Person B",
        )
        translated = QASample(
            0,
            "谁先去世？",
            "修复后的日期是1940年和18日。",
            "Person A",
            "Person B",
        )
        diagnostics = {
            field: TranslationResult(
                text=text,
                source_tokens=10,
                output_tokens=10,
                input_truncated=False,
                output_truncated=False,
            )
            for field, text in {
                "question": translated.question,
                "knowledge": translated.context,
                "right_answer": translated.right_answer,
                "hallucinated_answer": translated.hallucinated_answer,
            }.items()
        }

        row = _build_translation_cache_row(
            source=source,
            sample_id="sample-a",
            language_code="zh",
            automatic_translated=translated,
            automatic_diagnostics=diagnostics,
            applied_review={
                "reason": "The source concatenated two date components.",
                "accepted_warnings": [
                    "knowledge:numbers_missing:194018"
                ],
            },
            quality_cfg={"require_grounded_answer_alignment": True},
        )

        self.assertIn(
            "knowledge:numbers_missing:194018",
            row["automatic_translation_warnings"],
        )
        self.assertEqual(row["translation_warnings"], [])

    def test_schema_five_cache_can_refresh_reviews_without_translation(self) -> None:
        common = {
            "language": "zh",
            "source_fingerprint": "source-hash",
            "num_samples": 150,
            "translator": {"model": "nllb"},
            "translation_max_length": 512,
            "sentence_split_fields": ["knowledge"],
            "translation_policy": {"preserve_numeric_answers": True},
            "translation_quality_rules": {
                "require_grounded_answer_alignment": True
            },
        }
        cached = {
            **common,
            "schema_version": 5,
            "review_manifest": {"sha256": "old"},
        }
        expected = {
            **common,
            "schema_version": 6,
            "review_manifest": {"sha256": "new"},
        }

        self.assertTrue(
            _can_refresh_translation_cache(cached, expected)
        )
        expected["translator"] = {"model": "different"}
        self.assertFalse(
            _can_refresh_translation_cache(cached, expected)
        )


if __name__ == "__main__":
    unittest.main()
