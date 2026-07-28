from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.data import QASample, validate_paired_qa_schema
from hallu_llm.detection import (
    build_paired_audit_rows,
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

    def test_marks_reviewed_source_pair_for_primary_metric_exclusion(self) -> None:
        sample = QASample(
            0,
            "Question",
            "Knowledge",
            "Right",
            "Wrong",
        )

        instances = build_detection_instances(
            [sample],
            seed=7,
            sample_ids={0: "source-a"},
            source_reviews={
                "source-a": {
                    "exclude_from_primary_metrics": True,
                    "reason": "Both candidates are supported.",
                }
            },
        )

        self.assertTrue(
            all(instance.excluded_from_primary_metrics for instance in instances)
        )
        self.assertEqual(
            {instance.source_review_reason for instance in instances},
            {"Both candidates are supported."},
        )


class DataSchemaTests(unittest.TestCase):
    def test_rejects_single_candidate_qa_samples_schema(self) -> None:
        rows = [
            {
                "knowledge": "Evidence",
                "question": "Question",
                "answer": "Candidate answer",
                "hallucination": "yes",
            }
        ]

        with self.assertRaisesRegex(ValueError, "requires the 'qa' subset"):
            validate_paired_qa_schema(rows, "qa_samples")

    def test_accepts_full_paired_qa_schema(self) -> None:
        rows = [
            {
                "knowledge": "Evidence",
                "question": "Question",
                "right_answer": "Grounded answer",
                "hallucinated_answer": "Unsupported answer",
            }
        ]

        validate_paired_qa_schema(rows, "qa")


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
            self._row("sample-a", 0, actual=True, predicted=True, probability=0.90),
            self._row("sample-b", 1, actual=True, predicted=False, probability=0.40),
            self._row("sample-a", 0, actual=False, predicted=True, probability=0.80),
            self._row("sample-b", 1, actual=False, predicted=False, probability=0.10),
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
        self.assertAlmostEqual(summary["roc_auc"], 0.75)
        self.assertAlmostEqual(summary["pairwise_ranking_accuracy"], 1.0)
        self.assertAlmostEqual(summary["pairwise_classification_accuracy"], 0.0)
        length_diagnostics = summary["candidate_length_diagnostics"]
        self.assertEqual(length_diagnostics["mean_grounded"], 8.0)
        self.assertEqual(length_diagnostics["mean_hallucinated"], 12.0)
        self.assertEqual(length_diagnostics["length_only_roc_auc"], 1.0)
        self.assertEqual(
            length_diagnostics["matched_subset"]["matched_per_class"],
            0,
        )
        matched_summary = summarize_detection(
            rows,
            "en",
            "test-model",
            0.75,
            length_match_max_relative_difference=0.5,
        )
        matched_subset = matched_summary[
            "candidate_length_diagnostics"
        ]["matched_subset"]
        self.assertEqual(matched_subset["matched_per_class"], 2)
        self.assertEqual(matched_subset["instance_coverage"], 1.0)
        self.assertAlmostEqual(matched_subset["roc_auc"], 0.75)

    @staticmethod
    def _row(
        sample_id: str,
        sample_index: int,
        actual: bool,
        predicted: bool,
        probability: float,
    ) -> dict[str, object]:
        return {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "model": "test-model",
            "language": "en",
            "candidate_type": "hallucinated" if actual else "grounded",
            "actual_label": "hallucinated" if actual else "grounded",
            "predicted_label": "hallucinated" if predicted else "grounded",
            "prediction_source": "generated_response",
            "logit_predicted_label": "hallucinated" if predicted else "grounded",
            "prediction_correct": actual == predicted,
            "output_parseable": True,
            "format_valid": True,
            "parsed_logit_agreement": True,
            "hallucination_probability": probability,
            "decision_confidence": max(probability, 1 - probability),
            "label_probability_mass": 0.95,
            "translation_warnings": [],
            "question": f"Question {sample_id}",
            "knowledge": f"Knowledge {sample_id}",
            "candidate_answer": "Hallucinated" if actual else "Grounded",
            "generation": "YES" if predicted else "NO",
        }

    def test_builds_one_audit_row_per_pair(self) -> None:
        rows = [
            self._row("sample-a", 0, actual=True, predicted=True, probability=0.90),
            self._row("sample-a", 0, actual=False, predicted=False, probability=0.20),
        ]

        audit_rows = build_paired_audit_rows(rows)

        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(audit_rows[0]["sample_id"], "sample-a")
        self.assertTrue(audit_rows[0]["pairwise_rank_correct"])
        self.assertTrue(audit_rows[0]["both_candidates_classified_correctly"])
        self.assertEqual(
            audit_rows[0]["grounded_candidate_character_length"],
            8,
        )
        self.assertEqual(
            audit_rows[0]["hallucinated_candidate_character_length"],
            12,
        )
        self.assertEqual(
            audit_rows[0]["hallucinated_to_grounded_length_ratio"],
            1.5,
        )

    def test_excludes_reviewed_pairs_only_from_primary_metrics(self) -> None:
        rows = [
            self._row("sample-a", 0, actual=True, predicted=True, probability=0.90),
            self._row("sample-a", 0, actual=False, predicted=False, probability=0.10),
            self._row("sample-b", 1, actual=True, predicted=False, probability=0.20),
            self._row("sample-b", 1, actual=False, predicted=True, probability=0.80),
        ]
        for row in rows:
            row["excluded_from_primary_metrics"] = row["sample_id"] == "sample-b"
            row["source_review_reason"] = (
                "Ambiguous source label."
                if row["sample_id"] == "sample-b"
                else None
            )

        summary = summarize_detection(rows, "en", "test-model", 0.75)
        audit_rows = build_paired_audit_rows(rows)

        self.assertEqual(summary["num_generated_source_samples"], 2)
        self.assertEqual(summary["num_source_samples"], 1)
        self.assertEqual(summary["num_excluded_source_samples"], 1)
        self.assertEqual(summary["excluded_sample_ids"], ["sample-b"])
        self.assertEqual(summary["accuracy"], 1.0)
        excluded_audit = next(
            row for row in audit_rows if row["sample_id"] == "sample-b"
        )
        self.assertTrue(excluded_audit["excluded_from_primary_metrics"])
        self.assertEqual(
            excluded_audit["source_review_reason"],
            "Ambiguous source label.",
        )


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
