import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reanalyze_saved_reviews import reanalyze


class ReanalysisTests(unittest.TestCase):
    def test_mapping_is_separate_from_original_decision(self):
        review = dict(explanation='The candidate says Mumbai but the source explicitly says Delhi.',
                      evidence_quote='head office in Delhi', verdict='UNSUPPORTED')
        row = dict(sample_id='a', candidate_type='hallucinated', actual_label='hallucinated',
                   knowledge='The hotel has its head office in Delhi.', generation='NO',
                   hallucination_probability=0.01,
                   reasoning='Explanation\n```json\n' + json.dumps(review) + '\n```')
        before = json.dumps(row)
        rows, summary = reanalyze([row])
        self.assertEqual(json.dumps(row), before)
        self.assertEqual(rows[0]['review_derived_label'], 'hallucinated')
        self.assertEqual(rows[0]['original_final_choice'], 'NO')
        self.assertEqual(summary['final_decision_conflicts_or_invalid'], 1)
        self.assertEqual(summary['parseable_verdicts'], 1)
        self.assertNotIn('hallucination_probability', rows[0])

    def test_invalid_review_does_not_become_a_negative_label(self):
        row = dict(sample_id='a', candidate_type='grounded', actual_label='grounded',
                   knowledge='Evidence', reasoning='broken JSON', generation='NO')
        rows, summary = reanalyze([row])
        self.assertIsNone(rows[0]['review_derived_label'])
        self.assertIsNone(summary['review_derived_accuracy_on_parseable'])


if __name__ == '__main__':
    unittest.main()
