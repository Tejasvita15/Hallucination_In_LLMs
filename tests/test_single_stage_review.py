import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hallu_llm.single_stage_review import evaluate_reviews
from hallu_llm.detection_checkpoint import DetectionCheckpoint
from hallu_llm.detection import DetectionInstance


class SingleStageTests(unittest.TestCase):
    def test_invalid_retained_and_complete_run_reuses_all_checkpoints(self):
        instances = [DetectionInstance(0, 'a', 'Where?', 'Office in Delhi.', candidate, kind, label)
                     for candidate, kind, label in [('Delhi', 'grounded', False), ('Mumbai', 'hallucinated', True)]]
        class Generator:
            calls = 0
            def generate_review(self, prompt, budget):
                self.calls += 1
                text = json.dumps(dict(explanation='The source explicitly states the office is in Delhi.',
                                       evidence_quote='Office in Delhi.', verdict='SUPPORTED'))
                return dict(text=text if self.calls == 1 else 'invalid', truncated=False, tokens=20)
        generator = Generator()
        with tempfile.TemporaryDirectory() as directory:
            with DetectionCheckpoint(directory, {}) as checkpoint:
                summary, rows = evaluate_reviews(generator, instances, 'm', 'en', 2048, checkpoint)
            self.assertEqual(generator.calls, 2)
            self.assertEqual(summary['accuracy_all_instances'], 0.5)
            self.assertEqual(summary['accuracy_valid_only'], 1.0)
            self.assertEqual(summary['invalid_instances'], 1)
            self.assertIsNone(rows[1]['predicted_label'])
            self.assertNotIn('hallucination_probability', rows[0])
            with DetectionCheckpoint(directory, {}) as checkpoint:
                summary, _ = evaluate_reviews(generator, instances, 'm', 'en', 2048, checkpoint)
            self.assertEqual(generator.calls, 2)
            self.assertEqual(summary['resumed_instances'], 2)

    def test_truncated_valid_json_is_invalid_but_quote_defect_does_not_drop_row(self):
        instance = DetectionInstance(0, 'a', 'Q', 'Evidence', 'A', 'hallucinated', True)
        class Generator:
            truncated = False
            def generate_review(self, *args):
                return dict(text=json.dumps(dict(explanation='The candidate adds a claim missing from evidence.',
                              evidence_quote='', verdict='UNSUPPORTED')), truncated=self.truncated, tokens=20)
        generator = Generator()
        for truncated in (False, True):
            generator.truncated = truncated
            with tempfile.TemporaryDirectory() as directory:
                with DetectionCheckpoint(directory, {}) as checkpoint:
                    summary, rows = evaluate_reviews(generator, [instance], 'm', 'en', 2048, checkpoint)
            self.assertEqual(summary['scored_instances'], 1)
            self.assertEqual(summary['invalid_instances'], int(truncated))
            self.assertEqual(summary['accuracy_all_instances'], 0 if truncated else 1)
            self.assertIn('evidence_quote_not_in_source', rows[0]['review_quality_flags'])


if __name__ == '__main__':
    unittest.main()
