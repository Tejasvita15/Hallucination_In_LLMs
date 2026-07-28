from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.text import (
    is_language_neutral_number,
    join_translated_question_clauses,
    split_english_question_clauses,
    split_english_sentences,
)


class SentenceSplittingTests(unittest.TestCase):
    def test_repairs_missing_space_between_halueval_sentences(self) -> None:
        text = (
            "Nicola plays for FC Thun.FC Thun is a Swiss football team."
            "The club was founded in 1898."
        )

        self.assertEqual(
            split_english_sentences(text),
            [
                "Nicola plays for FC Thun.",
                "FC Thun is a Swiss football team.",
                "The club was founded in 1898.",
            ],
        )

    def test_preserves_common_abbreviations(self) -> None:
        text = "Dr. Smith works in the U.S. Army. She lives in Georgia."

        self.assertEqual(
            split_english_sentences(text),
            [
                "Dr. Smith works in the U.S. Army.",
                "She lives in Georgia.",
            ],
        )

    def test_handles_empty_text(self) -> None:
        self.assertEqual(split_english_sentences("   "), [])

    def test_splits_embedded_interrogative_for_retry(self) -> None:
        question = (
            "The school burned down during fires that killed how many people?"
        )

        self.assertEqual(
            split_english_question_clauses(question),
            [
                "The school burned down during fires that killed",
                "how many people?",
            ],
        )

    def test_splits_question_at_existing_clause_boundary(self) -> None:
        question = (
            "Of the cities Abdiel has lived in, "
            "which one was the capital?"
        )

        self.assertEqual(
            split_english_question_clauses(question),
            [
                "Of the cities Abdiel has lived in",
                "which one was the capital?",
            ],
        )

    def test_identifies_only_language_neutral_numeric_answers(self) -> None:
        self.assertTrue(is_language_neutral_number("16"))
        self.assertTrue(is_language_neutral_number("1943-1944"))
        self.assertFalse(is_language_neutral_number("16 people"))

    def test_joins_translated_clauses_without_duplicate_punctuation(self) -> None:
        self.assertEqual(
            join_translated_question_clauses(
                ["学校在火灾中被烧毁,", "有多少人?"]
            ),
            "学校在火灾中被烧毁，有多少人?",
        )


if __name__ == "__main__":
    unittest.main()
