import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from hallu_llm.benchmark_suite import binary_metrics, span_metrics, parse_span_generation_details, normalize_mushroom
from hallu_llm.token_trace import GreedyTrace, trace_generation
from analyze_token_confidence import analyze
import run_external_benchmark as runner


class RepairTests(unittest.TestCase):
    def test_invalid_negative_is_not_a_correct_negative(self):
        m = binary_metrics([False, True], [None, None])
        self.assertEqual(m["accuracy"], 0)
        self.assertEqual(m["coverage"], 0)
        self.assertEqual(m["balanced_accuracy"], 0)
        self.assertIsNone(m["mcc"])

    def test_invalid_empty_span_cannot_score_perfectly(self):
        m = span_metrics([{"gold_spans": [], "predicted_spans": [], "span_output_parseable": False}])
        self.assertEqual(m["mean_hard_span_iou"], 0)
        self.assertEqual(m["exact_span_match_rate"], 0)
        self.assertEqual(m["response_level"]["accuracy"], 0)

    def test_nonexistent_and_malformed_spans_are_invalid(self):
        for spans in [["not present"], [{"start": -1, "end": 1}], [None], [""]]:
            self.assertEqual(parse_span_generation_details(json.dumps({"hallucinated_spans": spans}), "text"), ([], False))

    def test_mushroom_preserves_original_scores_without_using_url_as_context(self):
        rows, _ = normalize_mushroom([{"hard_labels": [], "wikipedia_url": "https://example.org", "model_output_logits": [3.2], "model_output_tokens": ["x"]}], "test")
        self.assertEqual(rows[0]["context"], "")
        self.assertEqual(rows[0]["model_output_logits"], [3.2])
        self.assertIn("https://", rows[0]["reference_url"])

    def test_no_logit_fallback_for_invalid_response(self):
        class Fake:
            def generate_binary(self, *a, **kw):
                return SimpleNamespace(text="Maybe", predicted_positive=False, positive_probability=.1,
                    decision_confidence=.9, decision_entropy=.3, label_probability_mass=.2)
            def generate(self, *a, **kw):
                return SimpleNamespace(text='{"hallucinated_spans": []}')
        row = runner.evaluate_ragtruth_row(Fake(), {"model_input": "q", "model_output_text": "a", "context": "c"},
                                           {"binary_max_new_tokens": 5, "span_max_new_tokens": 5})
        self.assertIsNone(row["predicted_hallucinated"])
        self.assertFalse(row["generated_prediction_fallback"])

    def test_unlabelled_generations_do_not_become_accuracy(self):
        row = {"generation_sha256": "a", "truncated": False, "final_answer_present": True}
        m = analyze([row], [])
        self.assertEqual(m["pending_annotations"], 1)
        self.assertNotIn("unsupported_rate_among_reviewed_answers", m)
        with self.assertRaises(ValueError):
            analyze([row], [{"generation_sha256": "wrong", "semantic_label": "supported"}])

    def test_full_review_launchers_use_full_configs(self):
        for model in ["qwen", "llama", "deepseek"]:
            text = (ROOT / f"jobs/aspire2a/{model}_review_v3_full.pbs").read_text()
            self.assertNotIn(".smoke.json", text)
            path = text.split("--config ", 1)[1].strip()
            self.assertTrue((ROOT / path).exists())

    def test_new_launchers_have_no_sample_limits(self):
        for name in ["token_confidence_full", "truthfulqa_mc_full", "ragtruth_full", "mushroom_full", "qwen_crosslingual_full"]:
            text = (ROOT / f"jobs/aspire2a/{name}.pbs").read_text()
            self.assertNotIn("--max-samples", text)
            self.assertNotIn("smoke", text)


class TensorTraceTests(unittest.TestCase):
    def test_distribution_statistics_and_masked_logits(self):
        import torch
        capture = GreedyTrace([[0], [1]])
        scores = torch.tensor([[0., math.log(3), float("-inf")]])
        self.assertIs(capture(None, scores), scores)
        self.assertAlmostEqual(capture.rows[0]["probability"], .75, places=6)
        self.assertAlmostEqual(capture.rows[0]["entropy_nats"], -.25*math.log(.25)-.75*math.log(.75), places=6)
        self.assertEqual(capture.rows[0]["token_id"], 1)
        self.assertAlmostEqual(capture.label_scores["positive_probability"], .25, places=6)

    def test_reasoning_is_excluded_from_final_answer_means(self):
        import torch
        class Inputs(dict):
            def to(self, device):
                return self
        class Tokenizer:
            pad_token_id = 4
            eos_token_id = 4
            all_special_ids = [4]
            pieces = ["reason", "</think>", "answer", "unused", "<eos>"]
            def __call__(self, *args, **kw):
                return Inputs(input_ids=torch.tensor([[0]]))
            def decode(self, ids, skip_special_tokens=False, **kw):
                return "".join(self.pieces[int(i)] for i in ids if not (skip_special_tokens and int(i) == 4))
            def convert_ids_to_tokens(self, i):
                return self.pieces[i]
        class Model:
            device = "cpu"
            config = SimpleNamespace(max_position_embeddings=100)
            generation_config = SimpleNamespace(eos_token_id=4)
            def generate(self, input_ids, logits_processor, **kw):
                for i in [0, 1, 2, 4]:
                    scores = torch.full((1, 5), -3.)
                    scores[0, i] = 3.
                    logits_processor[0](input_ids, scores)
                return torch.tensor([[0, 0, 1, 2, 4]])
        generator = SimpleNamespace(model=Model(), tokenizer=Tokenizer(), model_name="deepseek-r1", format_prompt=lambda x: x)
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(LogitsProcessorList=list)}):
            result = trace_generation(generator, "prompt", 4)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["final_answer"], "answer")
        self.assertEqual([r["segment"] for r in result["token_trace"]], ["reasoning_or_unfinished"]*2 + ["final"]*2)
        self.assertAlmostEqual(result["mean_final_token_probability"], result["token_trace"][2]["probability"])


if __name__ == "__main__":
    unittest.main()
