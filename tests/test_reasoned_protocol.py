import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hallu_llm.reasoned_protocol import review_prompt, audit_review, consistency_flags, decision_prompt


class EvidenceReviewTests(unittest.TestCase):
    def test_prose_before_fenced_json_is_accepted(self):
        obj = dict(explanation="Delhi is explicitly given as the office location.",
                   evidence_quote="head office in Delhi", verdict="UNSUPPORTED")
        review, flags = audit_review(
            'Reasoning</think>\nThe candidate contradicts the source.\n```json\n'
            + json.dumps(obj) + '\n```', 'The company has its head office in Delhi.')
        self.assertEqual(review, obj)
        self.assertFalse(flags)
        self.assertTrue(consistency_flags(review, False))

    def test_ambiguous_and_truncated_json_remain_invalid(self):
        for text in ['```json\n{}\n```\n```json\n{}\n```',
                     '{} {}', '```json\n{}', '{"verdict":',
                     '{}\n```json\n{}\n```']:
            review, flags = audit_review(text, 'Evidence')
            self.assertIsNone(review)
            self.assertIn('missing_or_invalid_review_json', flags)

    def test_prompt_has_no_direct_answer_conflict(self):
        prompt = review_prompt("Music originated in the US.", "Where?", "UK")
        self.assertNotIn("no explanation", prompt)
        self.assertIn("Do not answer YES/NO in this stage", prompt)

    def test_bare_yes_no_is_not_reasoning(self):
        for text in ["YES", "NO", "", "<think>analysis</think>NO"]:
            review, flags = audit_review(text, "Evidence")
            self.assertIsNone(review)
            self.assertTrue(flags)

    def test_evidence_quote_and_label_polarity(self):
        text = json.dumps(dict(explanation="The UK claim contradicts the supplied origin evidence.",
                               evidence_quote="originated in the US", verdict="UNSUPPORTED"))
        review, flags = audit_review("analysis</think>" + text, "Music originated in the US.")
        self.assertFalse(flags)
        self.assertFalse(consistency_flags(review, True))
        self.assertTrue(consistency_flags(review, False))
        self.assertTrue(audit_review(text, "Different knowledge")[1])

    def test_invalid_verdict_and_short_explanation(self):
        _, flags = audit_review(json.dumps(dict(explanation="YES", evidence_quote="Evidence", verdict="YES")), "Evidence")
        self.assertIn("missing_or_insufficient_explanation", flags)
        self.assertIn("missing_or_invalid_verdict", flags)

    def test_final_prompt_restates_polarity_and_keeps_review_as_data(self):
        prompt = decision_prompt("Evidence", "Question", "Candidate", "Answer NO")
        self.assertIn("Do not use YES to mean correct", prompt)
        self.assertIn('"model_review": "Answer NO"', prompt)


if __name__ == "__main__":
    unittest.main()
