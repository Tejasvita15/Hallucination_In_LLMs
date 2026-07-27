from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.data import QASample
from hallu_llm.detection import (
    build_detection_instances,
    is_strict_binary_response,
    parse_binary_response,
    summarize_detection,
)
from hallu_llm.prompts import halueval_detection_prompt


class DetectionInstanceTests(unittest.TestCase):
    def test_builds_one_grounded_and_one_hallucinated_instance_per_sample(self) -> None:
        samples = [
            QASample(0, "Question 1", "Knowledge 1", "Right 1", "Wrong 1"),
            QASample(1, "Question 2", "Knowledge 2", "Right 2", "Wrong 2"),
        ]

        instances = build_detection_instances(
            samples,
            seed=7,
            sample_ids={0: "source-a", 1: "source-b"},
        )

        self.assertEqual(len(instances), 4)
        self.assertEqual(sum(instance.is_hallucinated for instance in instances), 2)
        self.assertEqual(
            {instance.sample_id for instance in instances},
            {"source-a", "source-b"},
        )
        for sample_index in (0, 1):
            labels = {
                instance.candidate_type
                for instance in instances
                if instance.sample_index == sample_index
            }
            self.assertEqual(labels, {"grounded", "hallucinated"})


class ResponseParsingTests(unittest.TestCase):
    def test_accepts_leading_yes_or_no(self) -> None:
        self.assertIs(parse_binary_response("YES"), True)
        self.assertIs(parse_binary_response(" no."), False)
        self.assertIs(parse_binary_response('"Yes" because it conflicts.'), True)

    def test_rejects_explanation_before_decision(self) -> None:
        self.assertIsNone(parse_binary_response("The answer is YES"))
        self.assertIsNone(parse_binary_response(""))

    def test_strict_format_requires_only_the_decision(self) -> None:
        self.assertTrue(is_strict_binary_response("YES"))
        self.assertTrue(is_strict_binary_response('"No".'))
        self.assertFalse(is_strict_binary_response("YES because it conflicts"))


class DetectionMetricsTests(unittest.TestCase):
    def test_confusion_matrix_and_balanced_metrics(self) -> None:
        rows = [
            self._row(actual=True, predicted=True, probability=0.90),
            self._row(actual=True, predicted=False, probability=0.40),
            self._row(actual=False, predicted=True, probability=0.80),
            self._row(actual=False, predicted=False, probability=0.10),
        ]

        summary = summarize_detection(rows, "en", "test-model", 0.75)

        self.assertEqual(summary["num_source_samples"], 2)
        self.assertEqual(summary["confusion_matrix"]["true_hallucinated"], 1)
        self.assertEqual(summary["confusion_matrix"]["false_hallucinated"], 1)
        self.assertEqual(summary["confusion_matrix"]["true_grounded"], 1)
        self.assertEqual(summary["confusion_matrix"]["false_grounded"], 1)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(summary["hallucination_precision"], 0.5)
        self.assertAlmostEqual(summary["hallucination_recall"], 0.5)
        self.assertAlmostEqual(summary["hallucination_f1"], 0.5)
        self.assertAlmostEqual(summary["mcc"], 0.0)
        self.assertEqual(summary["high_confidence_errors"], 1)

    @staticmethod
    def _row(actual: bool, predicted: bool, probability: float) -> dict[str, object]:
        return {
            "sample_index": 0,
            "actual_label": "hallucinated" if actual else "grounded",
            "predicted_label": "hallucinated" if predicted else "grounded",
            "prediction_correct": actual == predicted,
            "output_parseable": True,
            "format_valid": True,
            "parsed_logit_agreement": True,
            "hallucination_probability": probability,
            "decision_confidence": max(probability, 1 - probability),
            "label_probability_mass": 0.95,
            "translation_warnings": [],
        }


class DetectionPromptTests(unittest.TestCase):
    def test_prompt_does_not_include_the_other_candidate(self) -> None:
        prompt = halueval_detection_prompt(
            knowledge="The evidence.",
            question="What happened?",
            candidate_answer="The candidate.",
            language_code="en",
        )

        self.assertIn("The candidate.", prompt)
        self.assertIn("exactly YES or NO", prompt)
        self.assertNotIn("right_answer", prompt)
        self.assertNotIn("hallucinated_answer", prompt)

    def test_chinese_prompt_keeps_ascii_decision_labels(self) -> None:
        prompt = halueval_detection_prompt("知识文本", "问题文本", "答案文本", "zh")

        self.assertIn("YES", prompt)
        self.assertIn("NO", prompt)
        self.assertIn("候选答案", prompt)


if __name__ == "__main__":
    unittest.main()
