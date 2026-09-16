import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hallu_llm.detection import DetectionInstance, evaluate_detection_instances


class ReasonedDetectionTests(unittest.TestCase):
    def evaluate(self, truncated):
        class Generator:
            def generate_reasoned_binary(self, prompt, budget):
                assert budget == 17
                return SimpleNamespace(
                    text="YES", predicted_positive=True,
                    positive_probability=0.8, negative_probability=0.2,
                    decision_confidence=0.8, decision_entropy=0.5,
                    label_probability_mass=0.4, first_generated_token_id=None,
                    reasoning="Evidence analysis", reasoning_truncated=truncated,
                    reasoning_tokens=17,
                )

        instances = [DetectionInstance(0, "pair", "Q", "K", "A", kind, label)
                     for kind, label in [("grounded", False), ("hallucinated", True)]]
        return evaluate_detection_instances(
            Generator(), instances, "test", "en", 4, 0.75,
            decision_mode="reason_then_forced_choice", max_reasoning_tokens=17,
        )

    def test_reasoned_decisions_are_audited_and_not_reported_as_free_generation(self):
        summary, rows = self.evaluate(False)
        self.assertTrue(summary["protocol_quality_passed"])
        self.assertTrue(summary["output_parseability_is_constrained"])
        self.assertEqual(rows[0]["prediction_source"], "reason_then_forced_choice")
        self.assertEqual(rows[0]["reasoning"], "Evidence analysis")
        self.assertEqual(rows[0]["hallucination_probability"], 0.8)
        self.assertEqual(summary["generated_prediction_fallbacks"], 0)

    def test_truncation_fails_quality_gate_but_retains_audit(self):
        summary, rows = self.evaluate(True)
        self.assertFalse(summary["protocol_quality_passed"])
        self.assertEqual(summary["reasoning_truncated_instances"], 2)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
