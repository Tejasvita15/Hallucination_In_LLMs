import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hallu_llm.detection_checkpoint import DetectionCheckpoint
from hallu_llm.detection import DetectionInstance, evaluate_detection_instances


class CheckpointTests(unittest.TestCase):
    def test_interrupted_run_reuses_completed_instance(self):
        instances = [DetectionInstance(0, "a", "Q", "K", "A", kind, label)
                     for kind, label in [("grounded", False), ("hallucinated", True)]]
        class Generator:
            calls = 0
            fail = True
            def generate_binary(self, *args, **kwargs):
                self.calls += 1
                if self.fail and self.calls == 2:
                    raise RuntimeError("simulated interruption")
                return SimpleNamespace(text="YES", predicted_positive=True,
                    positive_probability=0.8, negative_probability=0.2,
                    decision_confidence=0.8, decision_entropy=0.5,
                    label_probability_mass=0.9, first_generated_token_id=1)
        generator = Generator()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with DetectionCheckpoint(directory, {"model": "test"}) as checkpoint:
                    evaluate_detection_instances(generator, instances, "test", "en", 4, 0.75, checkpoint=checkpoint)
            generator.fail = False
            with DetectionCheckpoint(directory, {"model": "test"}) as checkpoint:
                summary, rows = evaluate_detection_instances(generator, instances, "test", "en", 4, 0.75, checkpoint=checkpoint)
            self.assertEqual(generator.calls, 3)
            self.assertEqual(summary["resumed_instances"], 1)
            self.assertEqual(len(rows), 2)
            with DetectionCheckpoint(directory, {"model": "test"}) as checkpoint:
                summary, _ = evaluate_detection_instances(generator, instances, "test", "en", 4, 0.75, checkpoint=checkpoint)
            self.assertEqual(generator.calls, 3)
            self.assertEqual(summary["resumed_instances"], 2)

    def test_changed_provenance_and_concurrent_writer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with DetectionCheckpoint(directory, {"prompt": "old"}):
                with self.assertRaises(RuntimeError):
                    with DetectionCheckpoint(directory, {"prompt": "old"}):
                        pass
            with self.assertRaises(ValueError):
                with DetectionCheckpoint(directory, {"prompt": "new"}):
                    pass
            self.assertFalse((Path(directory) / "writer.lock").exists())

    def test_corrupt_row_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with DetectionCheckpoint(directory, {}) as checkpoint:
                checkpoint.save(["a", "grounded"], {"answer": "YES"})
                path = next(p for p in Path(directory).glob('*.json') if p.name != 'manifest.json')
                record = json.loads(path.read_text())
                record['row']['answer'] = 'NO'
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    checkpoint.load(["a", "grounded"])


if __name__ == '__main__':
    unittest.main()
