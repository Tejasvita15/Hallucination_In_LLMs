from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hallu_llm.benchmark_suite import (
    binary_metrics,
    binary_roc_auc,
    normalize_halueval_qa,
    normalize_mushroom,
    normalize_ragtruth,
    normalize_truthfulqa_csv,
    parse_binary_generation,
    parse_span_generation_details,
    probability_metrics,
    span_iou,
    span_metrics,
)

import run_external_benchmark as runner


class BenchmarkNormalizationTests(unittest.TestCase):
    def test_halueval_qa_normalizes_official_paired_rows(self) -> None:
        rows, quality = normalize_halueval_qa(
            [
                {
                    "knowledge": "France has Paris as its capital.",
                    "question": "What is the capital of France?",
                    "right_answer": "Paris.",
                    "hallucinated_answer": "Rome.",
                }
            ]
        )

        self.assertEqual(rows[0]["sample_id"], "halueval:qa:0")
        self.assertEqual(rows[0]["hallucinated_answer"], "Rome.")
        self.assertEqual(quality["num_rows"], 1)

    def test_ragtruth_joins_sources_filters_split_and_quality(self) -> None:
        sources = [
            {
                "source_id": 7,
                "task_type": "QA",
                "source": "nq",
                "prompt": "Where?",
                "source_info": {"passages": ["Paris is in France."]},
            }
        ]
        responses = [
            {
                "id": 1,
                "source_id": 7,
                "split": "test",
                "quality": "good",
                "response": "Paris is in Italy.",
                "model": "example-model",
                "labels": [{"start": 12, "end": 17, "text": "Italy"}],
            },
            {
                "id": 2,
                "source_id": 7,
                "split": "train",
                "quality": "good",
                "response": "Paris is in France.",
                "labels": [],
            },
            {
                "id": 3,
                "source_id": 7,
                "split": "test",
                "quality": "bad",
                "response": "Paris is in France.",
                "labels": [],
            },
        ]

        rows, quality = normalize_ragtruth(responses, sources)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gold_spans"], [[12, 17]])
        self.assertTrue(rows[0]["gold_hallucinated"])
        self.assertEqual(rows[0]["task"], "QA")
        self.assertEqual(quality["excluded_quality_rows"], 1)

    def test_ragtruth_rejects_missing_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing source IDs"):
            normalize_ragtruth(
                [
                    {
                        "id": 1,
                        "source_id": "absent",
                        "split": "test",
                        "response": "text",
                        "labels": [],
                    }
                ],
                [],
            )

    def test_mushroom_uses_stable_unique_fallback_for_null_ids(self) -> None:
        examples = [
            {
                "id": None,
                "lang": "EN",
                "model_input": "Question",
                "model_output_text": "Wrong city",
                "model_id": "model-a",
                "wikipedia_url": "https://example.test",
                "hard_labels": [[0, 5]],
                "soft_labels": [{"start": 0, "end": 5, "prob": 0.9}],
            },
            {
                "id": None,
                "lang": "EN",
                "model_input": "Question",
                "model_output_text": "Right city",
                "model_id": "model-a",
                "hard_labels": [],
                "soft_labels": [],
            },
        ]

        rows, quality = normalize_mushroom(examples, "test")

        self.assertEqual(len({row["sample_id"] for row in rows}), 2)
        self.assertEqual(rows[0]["gold_spans"], [[0, 5]])
        self.assertEqual(quality["label_counts"], {"false": 1, "true": 1})

    def test_truthfulqa_binary_order_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TruthfulQA.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Question",
                        "Best Answer",
                        "Best Incorrect Answer",
                        "Category",
                        "Source",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "What color is the sky on a clear day?",
                        "Best Answer": "Blue.",
                        "Best Incorrect Answer": "Green.",
                        "Category": "Indexical Error: Other",
                        "Source": "example",
                    }
                )
            first, first_quality = normalize_truthfulqa_csv(path, seed=99)
            second, second_quality = normalize_truthfulqa_csv(path, seed=99)

        self.assertEqual(first, second)
        self.assertEqual(first_quality, second_quality)
        correct = first[0]["choice_a"] if first[0]["correct_label"] == "A" else first[0]["choice_b"]
        self.assertEqual(correct, "Blue.")


class BenchmarkMetricTests(unittest.TestCase):
    def test_binary_parser_preserves_generated_decision(self) -> None:
        self.assertEqual(parse_binary_generation("YES", "YES", "NO"), (True, True))
        self.assertEqual(
            parse_binary_generation("no, because it is grounded", "YES", "NO"),
            (False, False),
        )
        self.assertEqual(parse_binary_generation("Maybe", "YES", "NO"), (None, False))

    def test_span_parser_accepts_exact_substrings_and_offsets(self) -> None:
        output = "Paris and Rome"
        spans, parseable = parse_span_generation_details(
            '{"hallucinated_spans":["Paris", {"start":10,"end":14}]}',
            output,
        )
        invalid, invalid_parseable = parse_span_generation_details("not json", output)

        self.assertTrue(parseable)
        self.assertEqual(spans, [(0, 5), (10, 14)])
        self.assertFalse(invalid_parseable)
        self.assertEqual(invalid, [])

    def test_span_and_binary_metrics(self) -> None:
        rows = [
            {"gold_spans": [[0, 4]], "predicted_spans": [[0, 2]]},
            {"gold_spans": [], "predicted_spans": []},
        ]
        metrics = span_metrics(rows)
        binary = binary_metrics([True, False], [True, False])

        self.assertAlmostEqual(span_iou([(0, 4)], [(0, 2)]), 0.5)
        self.assertAlmostEqual(metrics["mean_hard_span_iou"], 0.75)
        self.assertEqual(binary["accuracy"], 1.0)
        self.assertEqual(binary["mcc"], 1.0)

    def test_probability_metrics_and_auc(self) -> None:
        metrics = probability_metrics([False, True], [0.1, 0.9])

        self.assertAlmostEqual(metrics["brier_score"], 0.01)
        self.assertAlmostEqual(binary_roc_auc([False, True], [0.1, 0.9]), 1.0)
        self.assertIsNone(binary_roc_auc([True, True], [0.1, 0.9]))

    def test_mushroom_summary_has_micro_and_macro_language_results(self) -> None:
        rows = [
            {
                "language": "EN",
                "gold_hallucinated": True,
                "predicted_hallucinated": True,
                "gold_spans": [[0, 3]],
                "predicted_spans": [[0, 3]],
                "span_output_parseable": True,
            },
            {
                "language": "ZH",
                "gold_hallucinated": False,
                "predicted_hallucinated": False,
                "gold_spans": [],
                "predicted_spans": [],
                "span_output_parseable": True,
            },
        ]

        summary = runner.summarize_predictions(
            "mu_shroom",
            "model",
            rows,
            is_full_split_run=True,
        )

        self.assertEqual(summary["response_level"]["accuracy"], 1.0)
        self.assertEqual(set(summary["language_results"]), {"EN", "ZH"})
        self.assertEqual(summary["macro_language"]["mean_hard_span_iou"], 1.0)

    def test_ragtruth_uses_generated_label_before_logit_fallback(self) -> None:
        class FakeGenerator:
            def generate_binary(self, *args: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(
                    text="NO",
                    predicted_positive=True,
                    positive_probability=0.8,
                    decision_confidence=0.8,
                    decision_entropy=0.5,
                    label_probability_mass=0.95,
                )

            def generate(self, *args: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(
                    text='{"hallucinated_spans": []}',
                    mean_confidence=0.9,
                    mean_entropy=0.1,
                )

        row = {
            "sample_id": "ragtruth:1",
            "model_input": "Question",
            "context": "Evidence",
            "model_output_text": "Grounded answer",
            "gold_hallucinated": False,
            "gold_spans": [],
            "task": "QA",
        }
        prediction = runner.evaluate_ragtruth_row(
            FakeGenerator(),
            row,
            {
                "binary_max_new_tokens": 4,
                "span_max_new_tokens": 32,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        )

        self.assertFalse(prediction["predicted_hallucinated"])
        self.assertTrue(prediction["logit_predicted_hallucinated"])
        self.assertFalse(prediction["generated_prediction_fallback"])


if __name__ == "__main__":
    unittest.main()
