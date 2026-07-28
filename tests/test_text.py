from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.text import split_english_sentences


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


if __name__ == "__main__":
    unittest.main()
