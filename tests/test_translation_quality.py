from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.data import QASample
from hallu_llm.translation_quality import (
    grounded_answer_alignment_status,
    missing_numbers,
    summarize_translation_quality,
    translation_quality_warnings,
)


class TranslationQualityTests(unittest.TestCase):
    def test_detects_semantic_and_surface_translation_failures(self) -> None:
        source = QASample(
            0,
            "Who won in 2024?",
            "Ada won the 2024 final.",
            "Ada",
            "Bea",
        )
        translated = QASample(
            0,
            "The winner",
            "The final had a winner.",
            "Ada",
            "winner winner",
        )

        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=self._diagnostics(),
            language_code="en",
        )

        self.assertIn("question:numbers_missing:2024", warnings)
        self.assertIn("knowledge:numbers_missing:2024", warnings)
        self.assertIn("hallucinated_answer:repeated_output", warnings)
        self.assertIn("question:question_intent_lost", warnings)
        self.assertIn("right_answer:grounded_alignment_lost", warnings)

    def test_clean_translation_passes_default_quality_gate(self) -> None:
        rows = [
            {
                "translation_warnings": [],
                "grounded_answer_alignment": "translated_answer",
            }
            for _ in range(10)
        ]

        summary = summarize_translation_quality(rows, {})

        self.assertTrue(summary["quality_gate_passed"])
        self.assertEqual(summary["sample_warning_rate"], 0.0)
        self.assertEqual(summary["warning_counts"], {})
        self.assertEqual(
            summary["grounded_answer_alignment_counts"],
            {"translated_answer": 10},
        )

    def test_alignment_loss_can_fail_quality_gate(self) -> None:
        rows = [
            {"translation_warnings": ["right_answer:grounded_alignment_lost"]},
            {"translation_warnings": []},
        ]

        summary = summarize_translation_quality(
            rows,
            {"max_grounded_alignment_loss_rate": 0.25},
        )

        self.assertFalse(summary["quality_gate_passed"])
        self.assertEqual(summary["grounded_alignment_loss_rate"], 0.5)
        self.assertEqual(summary["warning_counts"]["grounded_alignment_lost"], 1)

    def test_preserved_source_answer_is_accepted_as_an_alias(self) -> None:
        source = QASample(
            0,
            "Who starred in the film?",
            "The film stars Russell Crowe.",
            "Russell Crowe",
            "Mel Gibson",
        )
        translated = QASample(
            0,
            "谁主演了这部电影？",
            "这部电影由 Russell Crowe 主演。",
            "拉塞尔·克罗",
            "梅尔·吉布森",
        )

        status = grounded_answer_alignment_status(source, translated)
        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=self._diagnostics(),
            language_code="zh",
        )

        self.assertEqual(status, "source_answer_alias")
        self.assertNotIn("right_answer:grounded_alignment_lost", warnings)

    def test_detects_low_coverage_translation_segment(self) -> None:
        source = QASample(
            0,
            "What happened?",
            "The complete supporting evidence is present here.",
            "Evidence",
            "Nothing",
        )
        translated = QASample(
            0,
            "发生了什么？",
            "证据。",
            "证据",
            "没有什么",
        )
        diagnostics = self._diagnostics()
        diagnostics["knowledge"].segments = (
            SimpleNamespace(
                source_text=source.context,
                translated_text=translated.context,
                source_tokens=20,
                output_tokens=2,
            ),
        )

        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=diagnostics,
            language_code="zh",
            min_segment_output_ratio=0.2,
        )

        self.assertIn("knowledge:low_segment_coverage:1:0.100", warnings)

    def test_accepts_multi_person_source_name_components(self) -> None:
        source = QASample(
            0,
            "Who wrote Doomquest?",
            "Doomquest was written by David Michelinie and Bob Layton.",
            "David Michelinie and Bob Layton",
            "Stan Lee and Jack Kirby",
        )
        translated = QASample(
            0,
            "谁写了 Doomquest？",
            "Doomquest 由 David Michelinie 和 Bob Layton 撰写。",
            "戴维·米切利尼和鲍勃·莱顿",
            "斯坦·李和杰克·科比",
        )

        self.assertEqual(
            grounded_answer_alignment_status(source, translated),
            "source_answer_components",
        )

    def test_accepts_small_proper_name_transliteration_variant(self) -> None:
        source = QASample(
            0,
            "Which supplier?",
            "The supplier was BorgWarner.",
            "BorgWarner",
            "General Motors",
        )
        translated = QASample(
            0,
            "哪个供应商？",
            "供应商是博格华纳。",
            "波格华纳",
            "通用汽车",
        )

        self.assertEqual(
            grounded_answer_alignment_status(source, translated),
            "translated_answer_alias",
        )

    def test_accepts_numeric_answer_embedded_in_cjk_text(self) -> None:
        source = QASample(
            0,
            "When was she born?",
            "She was born on 20 October 1946.",
            "1946",
            "1945",
        )
        translated = QASample(
            0,
            "她是哪一年出生的？",
            "她出生于1946年10月20日。",
            "1946",
            "1945",
        )

        status = grounded_answer_alignment_status(source, translated)
        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=self._diagnostics(),
            language_code="zh",
        )

        self.assertEqual(status, "translated_numeric_answer")
        self.assertNotIn("right_answer:grounded_alignment_lost", warnings)

    def test_accepts_numeric_dominant_answer_with_translated_connectors(self) -> None:
        source = QASample(
            0,
            "During which years?",
            "He served from 1957 to 1963.",
            "1957 to 1963",
            "1964 to 1970",
        )
        translated = QASample(
            0,
            "他在哪些年份任职？",
            "他从1957年到1963年任职。",
            "1957年至1963年",
            "1964年至1970年",
        )

        self.assertEqual(
            grounded_answer_alignment_status(source, translated),
            "translated_numeric_components",
        )

    def test_accepts_latin_source_alias_adjacent_to_cjk(self) -> None:
        source = QASample(
            0,
            "Who starred in the film?",
            "The film starred Lee Byung-hun.",
            "Lee Byung-hun",
            "Another actor",
        )
        translated = QASample(
            0,
            "谁主演了这部电影？",
            "这部电影由Lee Byung-hun主演。",
            "李秉宪",
            "另一位演员",
        )

        self.assertEqual(
            grounded_answer_alignment_status(source, translated),
            "source_answer_alias",
        )

    def test_numeric_alignment_does_not_match_a_larger_number(self) -> None:
        source = QASample(
            0,
            "How many people?",
            "The fire killed 16 people.",
            "16",
            "15",
        )
        translated = QASample(
            0,
            "有多少人？",
            "这座建筑于2016年被烧毁。",
            "16",
            "15",
        )

        self.assertEqual(
            grounded_answer_alignment_status(source, translated),
            "lost",
        )

    def test_quality_summary_distinguishes_reviewed_warnings(self) -> None:
        rows = [
            {
                "automatic_translation_warnings": [
                    "right_answer:grounded_alignment_lost"
                ],
                "translation_warnings": [],
                "grounded_answer_alignment": "reviewed_semantic_alignment",
                "translation_review": {
                    "reason": "Reviewed semantic entailment.",
                    "accept_grounded_alignment": True,
                },
            }
        ]

        summary = summarize_translation_quality(
            rows,
            {"max_grounded_alignment_loss_rate": 0.0},
        )

        self.assertTrue(summary["quality_gate_passed"])
        self.assertEqual(summary["samples_with_automatic_warnings"], 1)
        self.assertEqual(summary["samples_with_warnings"], 0)
        self.assertEqual(summary["reviewed_alignment_samples"], 1)

    def test_accepts_semantically_equivalent_number_formats(self) -> None:
        self.assertEqual(
            missing_numbers(
                "The prize was $6,000.",
                "奖金为6000美元。",
            ),
            [],
        )
        self.assertEqual(
            missing_numbers(
                "The 10 best players",
                "十名最佳球员",
            ),
            [],
        )
        self.assertEqual(
            missing_numbers(
                "The firm raised $6.5 billion.",
                "该公司筹集了65亿美元。",
            ),
            [],
        )
        self.assertEqual(
            missing_numbers(
                "He worked there in the 1980s.",
                "他在20世纪80年代在那里工作。",
            ),
            [],
        )

    def test_ignores_repeated_title_present_in_source(self) -> None:
        source = QASample(
            0,
            "Which song?",
            'The song samples "Suga Suga".',
            "Suga Suga",
            "Another song",
        )
        translated = QASample(
            0,
            "哪首歌？",
            '这首歌采样了"Suga Suga"。',
            "Suga Suga",
            "另一首歌",
        )

        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=self._diagnostics(),
            language_code="zh",
        )

        self.assertNotIn("knowledge:repeated_output", warnings)

    def test_detects_repeated_clause_and_non_content_answer(self) -> None:
        source = QASample(
            0,
            "What is the album name?",
            "The answer is present.",
            "Kum ba yah",
            "Another song",
        )
        translated = QASample(
            0,
            "专辑叫什么名字？专辑叫什么名字？专辑叫什么名字？",
            "答案在这里。",
            ",,,,,",
            "另一首歌",
        )

        warnings = translation_quality_warnings(
            source=source,
            translated=translated,
            diagnostics=self._diagnostics(),
            language_code="zh",
        )

        self.assertIn("question:repeated_output", warnings)
        self.assertIn("right_answer:non_content_output", warnings)

    def test_does_not_fuzzy_match_a_common_noun_mistranslation(self) -> None:
        source = QASample(
            0,
            "What kind of athlete?",
            "Claudia is a triathlete.",
            "triathlete",
            "runner",
        )
        translated = QASample(
            0,
            "她是什么运动员？",
            "克劳迪亚是一名三项运动员。",
            "三人运动员",
            "跑步运动员",
        )

        self.assertEqual(
            grounded_answer_alignment_status(source, translated),
            "lost",
        )

    def test_does_not_fuzzy_match_a_changed_award_type(self) -> None:
        source = QASample(
            0,
            "What award did she win?",
            "She won the Filmfare Award for Best Female Debut.",
            "Filmfare Award for Best Female Debut",
            "Best Actress",
        )
        translated = QASample(
            0,
            "她获得了什么奖？",
            "她获得了最佳女演员奖。",
            "电影奖最佳女演员",
            "最佳女演员",
        )

        self.assertEqual(
            grounded_answer_alignment_status(source, translated),
            "lost",
        )

    @staticmethod
    def _diagnostics() -> dict[str, SimpleNamespace]:
        return {
            field: SimpleNamespace(
                input_truncated=False,
                output_truncated=False,
                segments=(),
            )
            for field in (
                "question",
                "knowledge",
                "right_answer",
                "hallucinated_answer",
            )
        }


if __name__ == "__main__":
    unittest.main()
