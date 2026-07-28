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


if __name__ == "__main__":
    unittest.main()
