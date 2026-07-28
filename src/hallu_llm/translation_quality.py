from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

from .data import QASample


NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
REPEATED_WORD_PATTERN = re.compile(r"\b([\w-]{2,})\s+\1\b", re.IGNORECASE)
REPEATED_CJK_PATTERN = re.compile(r"([\u4e00-\u9fff])\1{2,}")
EN_QUESTION_PREFIX = re.compile(
    r"^\s*(?:who|what|when|where|why|which|how|is|are|was|were|do|does|did|"
    r"can|could|will|would|should|has|have|had)\b",
    re.IGNORECASE,
)
ZH_QUESTION_MARKERS = ("什么", "哪个", "哪一", "谁", "多少", "是否", "几", "何时", "哪里", "为什么")
CRITICAL_WARNING_CODES = {
    "empty_output",
    "empty_segment",
    "input_truncated",
    "low_segment_coverage",
    "numbers_missing",
    "output_truncated",
    "question_intent_lost",
    "repeated_output",
}


def translation_quality_warnings(
    source: QASample,
    translated: QASample,
    diagnostics: Mapping[str, Any],
    language_code: str,
    require_grounded_answer_alignment: bool = True,
    min_segment_output_ratio: float = 0.2,
    min_segment_source_tokens: int = 8,
) -> list[str]:
    warnings: list[str] = []
    source_fields = {
        "question": source.question,
        "knowledge": source.context,
        "right_answer": source.right_answer,
        "hallucinated_answer": source.hallucinated_answer,
    }
    translated_fields = {
        "question": translated.question,
        "knowledge": translated.context,
        "right_answer": translated.right_answer,
        "hallucinated_answer": translated.hallucinated_answer,
    }

    for field, result in diagnostics.items():
        if result.input_truncated:
            warnings.append(f"{field}:input_truncated")
        if result.output_truncated:
            warnings.append(f"{field}:output_truncated")
        if source_fields[field].strip() and not translated_fields[field].strip():
            warnings.append(f"{field}:empty_output")
        if field not in {"question", "knowledge"}:
            continue
        for segment_index, segment in enumerate(
            getattr(result, "segments", ()),
            start=1,
        ):
            if segment.source_text.strip() and not segment.translated_text.strip():
                warnings.append(f"{field}:empty_segment:{segment_index}")
                continue
            if segment.source_tokens < min_segment_source_tokens:
                continue
            output_ratio = segment.output_tokens / segment.source_tokens
            if output_ratio < min_segment_output_ratio:
                warnings.append(
                    f"{field}:low_segment_coverage:{segment_index}:"
                    f"{output_ratio:.3f}"
                )

    for field, source_text in source_fields.items():
        translated_text = translated_fields[field]
        missing_numbers = _missing_numbers(source_text, translated_text)
        if missing_numbers:
            warnings.append(f"{field}:numbers_missing:{','.join(missing_numbers)}")
        if _has_immediate_repetition(translated_text):
            warnings.append(f"{field}:repeated_output")

    if _question_intent_lost(source.question, translated.question, language_code):
        warnings.append("question:question_intent_lost")

    if (
        require_grounded_answer_alignment
        and grounded_answer_alignment_status(source, translated) == "lost"
    ):
        warnings.append("right_answer:grounded_alignment_lost")

    return warnings


def summarize_translation_quality(
    rows: list[dict[str, Any]],
    quality_cfg: dict[str, Any],
) -> dict[str, Any]:
    total = len(rows)
    warning_counts: Counter[str] = Counter()
    warning_samples = 0
    critical_samples = 0
    alignment_loss_samples = 0
    alignment_status_counts: Counter[str] = Counter()

    for row in rows:
        alignment_status_counts[
            str(row.get("grounded_answer_alignment", "unknown"))
        ] += 1
        warnings = list(row.get("translation_warnings", []))
        if warnings:
            warning_samples += 1
        row_has_critical = False
        row_has_alignment_loss = False
        for warning in warnings:
            code = _warning_code(str(warning))
            warning_counts[code] += 1
            row_has_critical = row_has_critical or code in CRITICAL_WARNING_CODES
            row_has_alignment_loss = (
                row_has_alignment_loss or code == "grounded_alignment_lost"
            )
        critical_samples += int(row_has_critical)
        alignment_loss_samples += int(row_has_alignment_loss)

    warning_rate = warning_samples / total if total else 0.0
    critical_rate = critical_samples / total if total else 0.0
    alignment_loss_rate = alignment_loss_samples / total if total else 0.0
    max_critical_rate = float(quality_cfg.get("max_critical_warning_rate", 0.1))
    max_alignment_loss_rate = float(
        quality_cfg.get("max_grounded_alignment_loss_rate", 0.25)
    )
    passed = (
        critical_rate <= max_critical_rate
        and alignment_loss_rate <= max_alignment_loss_rate
    )

    return {
        "num_samples": total,
        "samples_with_warnings": warning_samples,
        "sample_warning_rate": warning_rate,
        "samples_with_critical_warnings": critical_samples,
        "critical_warning_rate": critical_rate,
        "grounded_alignment_loss_samples": alignment_loss_samples,
        "grounded_alignment_loss_rate": alignment_loss_rate,
        "max_critical_warning_rate": max_critical_rate,
        "max_grounded_alignment_loss_rate": max_alignment_loss_rate,
        "quality_gate_passed": passed,
        "grounded_answer_alignment_counts": dict(
            sorted(alignment_status_counts.items())
        ),
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def grounded_answer_alignment_status(
    source: QASample,
    translated: QASample,
) -> str:
    if not _contains(source.context, source.right_answer):
        return "not_applicable"
    if _contains(translated.context, translated.right_answer):
        return "translated_answer"
    if _contains(translated.context, source.right_answer):
        return "source_answer_alias"
    return "lost"


def _missing_numbers(source: str, translated: str) -> list[str]:
    source_numbers = Counter(NUMBER_PATTERN.findall(source))
    translated_numbers = Counter(NUMBER_PATTERN.findall(translated))
    missing: list[str] = []
    for number, count in source_numbers.items():
        deficit = count - translated_numbers[number]
        if deficit > 0:
            missing.extend([number] * deficit)
    return missing


def _has_immediate_repetition(text: str) -> bool:
    return bool(
        REPEATED_WORD_PATTERN.search(text)
        or REPEATED_CJK_PATTERN.search(text)
    )


def _question_intent_lost(source: str, translated: str, language_code: str) -> bool:
    source_is_question = (
        "?" in source
        or "？" in source
        or bool(EN_QUESTION_PREFIX.search(source))
    )
    if not source_is_question:
        return False
    if "?" in translated or "？" in translated:
        return False
    if language_code == "zh":
        return not any(marker in translated for marker in ZH_QUESTION_MARKERS)
    return not bool(EN_QUESTION_PREFIX.search(translated))


def _contains(container: str, value: str) -> bool:
    if re.search(r"[\u3400-\u9fff]", value):
        normalized_container = re.sub(r"[\W_]+", "", container).lower()
        normalized_value = re.sub(r"[\W_]+", "", value).lower()
        return bool(normalized_value and normalized_value in normalized_container)

    container_tokens = re.findall(r"\w+", container.lower())
    value_tokens = re.findall(r"\w+", value.lower())
    if not value_tokens:
        return False
    width = len(value_tokens)
    return any(
        container_tokens[position : position + width] == value_tokens
        for position in range(len(container_tokens) - width + 1)
    )


def _warning_code(warning: str) -> str:
    parts = warning.split(":")
    return parts[1] if len(parts) > 1 else parts[0]
