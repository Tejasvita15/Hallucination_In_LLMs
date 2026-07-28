from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.data import QASample
from hallu_llm.translation_quality import (
    grounded_answer_alignment_status,
    summarize_translation_quality,
    translation_quality_warnings,
)


class TranslationQualityTests(unittest.TestCase):
    def test_detects_semantic_and_surface_translation_failures(self) -> None:
        source = QASample(
            0,
            "Who won in 2024?",
            "Ada won the 2024 final.",
            "Ada",
            "Bea",
        )
        translated = QASample(
            0,
            "The winner",
            "The final had a winner.",
            "Ada",
            "winner winner",
        )

        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=self._diagnostics(),
            language_code="en",
        )

        self.assertIn("question:numbers_missing:2024", warnings)
        self.assertIn("knowledge:numbers_missing:2024", warnings)
        self.assertIn("hallucinated_answer:repeated_output", warnings)
        self.assertIn("question:question_intent_lost", warnings)
        self.assertIn("right_answer:grounded_alignment_lost", warnings)

    def test_clean_translation_passes_default_quality_gate(self) -> None:
        rows = [
            {
                "translation_warnings": [],
                "grounded_answer_alignment": "translated_answer",
            }
            for _ in range(10)
        ]

        summary = summarize_translation_quality(rows, {})

        self.assertTrue(summary["quality_gate_passed"])
        self.assertEqual(summary["sample_warning_rate"], 0.0)
        self.assertEqual(summary["warning_counts"], {})
        self.assertEqual(
            summary["grounded_answer_alignment_counts"],
            {"translated_answer": 10},
        )

    def test_alignment_loss_can_fail_quality_gate(self) -> None:
        rows = [
            {"translation_warnings": ["right_answer:grounded_alignment_lost"]},
            {"translation_warnings": []},
        ]

        summary = summarize_translation_quality(
            rows,
            {"max_grounded_alignment_loss_rate": 0.25},
        )

        self.assertFalse(summary["quality_gate_passed"])
        self.assertEqual(summary["grounded_alignment_loss_rate"], 0.5)
        self.assertEqual(summary["warning_counts"]["grounded_alignment_lost"], 1)

    def test_preserved_source_answer_is_accepted_as_an_alias(self) -> None:
        source = QASample(
            0,
            "Who starred in the film?",
            "The film stars Russell Crowe.",
            "Russell Crowe",
            "Mel Gibson",
        )
        translated = QASample(
            0,
            "谁主演了这部电影？",
            "这部电影由 Russell Crowe 主演。",
            "拉塞尔·克罗",
            "梅尔·吉布森",
        )

        status = grounded_answer_alignment_status(source, translated)
        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=self._diagnostics(),
            language_code="zh",
        )

        self.assertEqual(status, "source_answer_alias")
        self.assertNotIn("right_answer:grounded_alignment_lost", warnings)

    def test_detects_low_coverage_translation_segment(self) -> None:
        source = QASample(
            0,
            "What happened?",
            "The complete supporting evidence is present here.",
            "Evidence",
            "Nothing",
        )
        translated = QASample(
            0,
            "发生了什么？",
            "证据。",
            "证据",
            "没有什么",
        )
        diagnostics = self._diagnostics()
        diagnostics["knowledge"].segments = (
            SimpleNamespace(
                source_text=source.context,
                translated_text=translated.context,
                source_tokens=20,
                output_tokens=2,
            ),
        )

        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=diagnostics,
            language_code="zh",
            min_segment_output_ratio=0.2,
        )

        self.assertIn("knowledge:low_segment_coverage:1:0.100", warnings)

    @staticmethod
    def _diagnostics() -> dict[str, SimpleNamespace]:
        return {
            field: SimpleNamespace(
                input_truncated=False,
                output_truncated=False,
                segments=(),
            )
            for field in (
                "question",
                "knowledge",
                "right_answer",
                "hallucinated_answer",
            )
        }


if __name__ == "__main__":
    unittest.main()
