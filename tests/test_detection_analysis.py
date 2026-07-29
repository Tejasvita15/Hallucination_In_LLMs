from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.detection_analysis import (
    METHOD_ORDER,
    apply_probability_method,
    assign_sample_splits,
    bootstrap_metric_intervals,
    compare_scored_languages,
    evaluate_scored_rows,
    fit_analysis_methods,
    fit_balanced_accuracy_threshold,
    fit_logistic_calibrator,
    fit_temperature_scaler,
    holm_adjusted_p_values,
    primary_prediction_rows,
    score_prediction_rows,
)


class DetectionAnalysisTests(unittest.TestCase):
    def test_split_is_deterministic_and_keeps_requested_size(self) -> None:
        sample_ids = [f"sample-{index}" for index in range(10)]

        first = assign_sample_splits(sample_ids, dev_fraction=0.3, seed=42)
        second = assign_sample_splits(
            reversed(sample_ids),
            dev_fraction=0.3,
            seed=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            sum(row["split"] == "development" for row in first),
            3,
        )
        self.assertEqual(
            {row["sample_id"] for row in first},
            set(sample_ids),
        )

    def test_primary_rows_validate_complete_pairs_and_exclusions(self) -> None:
        rows = self._paired_rows(
            grounded_probabilities=[0.2, 0.3],
            hallucinated_probabilities=[0.7, 0.8],
        )
        rows.extend(
            self._paired_rows(
                grounded_probabilities=[0.9],
                hallucinated_probabilities=[0.1],
                sample_prefix="excluded",
                excluded=True,
            )
        )

        primary = primary_prediction_rows(rows)

        self.assertEqual(len(primary), 4)
        self.assertEqual(
            {row["sample_id"] for row in primary},
            {"sample-0", "sample-1"},
        )

    def test_fitted_threshold_corrects_shifted_probabilities(self) -> None:
        rows = self._paired_rows(
            grounded_probabilities=[0.55, 0.60, 0.58],
            hallucinated_probabilities=[0.72, 0.80, 0.75],
        )

        method = fit_balanced_accuracy_threshold(rows)
        metrics = evaluate_scored_rows(score_prediction_rows(rows, method))

        self.assertGreater(method["threshold"], 0.60)
        self.assertLessEqual(method["threshold"], 0.72)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_temperature_and_platt_improve_development_nll(self) -> None:
        rows = self._paired_rows(
            grounded_probabilities=[0.25, 0.30, 0.35, 0.40],
            hallucinated_probabilities=[0.60, 0.65, 0.70, 0.75],
        )

        temperature = fit_temperature_scaler(rows)
        platt = fit_logistic_calibrator(
            rows,
            feature_names=("model_logit",),
        )

        self.assertLessEqual(
            temperature["fit_nll_after"],
            temperature["fit_nll_before"],
        )
        self.assertLessEqual(
            platt["fit_nll_after"],
            platt["fit_nll_before"],
        )
        self.assertEqual(len(apply_probability_method(rows, platt)), 8)

    def test_method_fits_record_training_scope(self) -> None:
        english = self._paired_rows(
            grounded_probabilities=[0.2, 0.3, 0.4],
            hallucinated_probabilities=[0.6, 0.7, 0.8],
            language="en",
        )
        chinese = self._paired_rows(
            grounded_probabilities=[0.7, 0.75, 0.8],
            hallucinated_probabilities=[0.82, 0.88, 0.92],
            language="zh",
        )

        methods = fit_analysis_methods(
            {"en": english, "zh": chinese},
            reference_language="en",
        )

        self.assertEqual(tuple(methods["en"]), METHOD_ORDER)
        self.assertEqual(
            methods["zh"]["reference_threshold"]["fit_language"],
            "en",
        )
        self.assertEqual(
            methods["zh"]["language_threshold"]["fit_language"],
            "zh",
        )
        self.assertEqual(
            methods["zh"]["pooled_threshold"]["fit_scope"],
            "pooled_languages",
        )

    def test_bootstrap_and_cross_language_comparison_use_source_pairs(
        self,
    ) -> None:
        english_rows = self._paired_rows(
            grounded_probabilities=[0.2, 0.3, 0.4, 0.45],
            hallucinated_probabilities=[0.8, 0.7, 0.6, 0.55],
            language="en",
        )
        chinese_rows = self._paired_rows(
            grounded_probabilities=[0.7, 0.75, 0.8, 0.85],
            hallucinated_probabilities=[0.9, 0.92, 0.94, 0.96],
            language="zh",
        )
        identity = {"kind": "identity", "threshold": 0.5}
        english = score_prediction_rows(english_rows, identity)
        chinese = score_prediction_rows(chinese_rows, identity)

        intervals = bootstrap_metric_intervals(
            english,
            replicates=50,
            seed=7,
        )
        comparison = compare_scored_languages(
            english,
            chinese,
            replicates=50,
            seed=7,
        )

        self.assertIn("accuracy", intervals)
        self.assertEqual(comparison["num_common_test_samples"], 4)
        self.assertLess(comparison["delta_accuracy"], 0.0)
        self.assertGreater(
            comparison["prediction_target_only_hallucinated"],
            comparison["prediction_reference_only_hallucinated"],
        )
        self.assertIn("delta_accuracy_ci_low", comparison)

    def test_holm_adjustment_is_monotonic_in_ranked_p_values(self) -> None:
        adjusted = holm_adjusted_p_values([0.01, 0.04, 0.03])

        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    @staticmethod
    def _paired_rows(
        grounded_probabilities: list[float],
        hallucinated_probabilities: list[float],
        sample_prefix: str = "sample",
        language: str = "en",
        excluded: bool = False,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, (grounded, hallucinated) in enumerate(
            zip(grounded_probabilities, hallucinated_probabilities)
        ):
            sample_id = f"{sample_prefix}-{index}"
            rows.extend(
                (
                    DetectionAnalysisTests._row(
                        sample_id,
                        index,
                        actual=False,
                        probability=grounded,
                        language=language,
                        excluded=excluded,
                    ),
                    DetectionAnalysisTests._row(
                        sample_id,
                        index,
                        actual=True,
                        probability=hallucinated,
                        language=language,
                        excluded=excluded,
                    ),
                )
            )
        return rows

    @staticmethod
    def _row(
        sample_id: str,
        sample_index: int,
        actual: bool,
        probability: float,
        language: str,
        excluded: bool,
    ) -> dict[str, object]:
        predicted = probability >= 0.5
        candidate_type = "hallucinated" if actual else "grounded"
        return {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "model": "test-model",
            "language": language,
            "candidate_type": candidate_type,
            "actual_label": candidate_type,
            "predicted_label": (
                "hallucinated" if predicted else "grounded"
            ),
            "hallucination_probability": probability,
            "candidate_answer": (
                "A much longer unsupported candidate answer"
                if actual
                else "Grounded"
            ),
            "excluded_from_primary_metrics": excluded,
        }


if __name__ == "__main__":
    unittest.main()
