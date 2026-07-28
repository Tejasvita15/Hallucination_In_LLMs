from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Mapping

from .data import QASample
from .text import is_language_neutral_number


NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
REPEATED_WORD_PATTERN = re.compile(r"\b([\w-]{2,})\s+\1\b", re.IGNORECASE)
REPEATED_CJK_PATTERN = re.compile(r"([\u4e00-\u9fff])\1{2,}")
CLAUSE_BOUNDARY_PATTERN = re.compile(r"[.!?。！？]+")
CONTENT_PATTERN = re.compile(r"[A-Za-z0-9\u3400-\u9fff]")
EN_QUESTION_PREFIX = re.compile(
    r"^\s*(?:who|what|when|where|why|which|how|is|are|was|were|do|does|did|"
    r"can|could|will|would|should|has|have|had)\b",
    re.IGNORECASE,
)
ZH_QUESTION_MARKERS = ("什么", "哪个", "哪一", "谁", "多少", "是否", "几", "何时", "哪里", "为什么")
PROPER_NAME_CONNECTORS = {"and", "of", "or", "the"}
CRITICAL_WARNING_CODES = {
    "empty_output",
    "empty_segment",
    "input_truncated",
    "low_segment_coverage",
    "numbers_missing",
    "non_content_output",
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
        if source_text.strip() and not CONTENT_PATTERN.search(translated_text):
            warnings.append(f"{field}:non_content_output")
        absent_numbers = missing_numbers(source_text, translated_text)
        if absent_numbers:
            warnings.append(f"{field}:numbers_missing:{','.join(absent_numbers)}")
        if _has_unexpected_repetition(source_text, translated_text):
            warnings.append(f"{field}:repeated_output")

    if question_intent_lost(source.question, translated.question, language_code):
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
    automatic_warning_counts: Counter[str] = Counter()
    warning_samples = 0
    automatic_warning_samples = 0
    critical_samples = 0
    alignment_loss_samples = 0
    alignment_status_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    reviewed_samples = 0
    reviewed_alignment_samples = 0
    manual_override_samples = 0

    for row in rows:
        alignment_status_counts[
            str(row.get("grounded_answer_alignment", "unknown"))
        ] += 1
        for field, diagnostics in row.get(
            "translation_diagnostics",
            {},
        ).items():
            strategy = str(diagnostics.get("strategy", "unknown"))
            strategy_counts[f"{field}:{strategy}"] += 1
        review = row.get("translation_review")
        if review:
            reviewed_samples += 1
            reviewed_alignment_samples += int(
                bool(review.get("accept_grounded_alignment", False))
            )
            manual_override_samples += int(bool(review.get("fields")))
        automatic_warnings = list(
            row.get(
                "automatic_translation_warnings",
                row.get("translation_warnings", []),
            )
        )
        if automatic_warnings:
            automatic_warning_samples += 1
        for warning in automatic_warnings:
            automatic_warning_counts[_warning_code(str(warning))] += 1
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
        "samples_with_automatic_warnings": automatic_warning_samples,
        "automatic_sample_warning_rate": (
            automatic_warning_samples / total if total else 0.0
        ),
        "automatic_warning_counts": dict(
            sorted(automatic_warning_counts.items())
        ),
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
        "translation_strategy_counts": dict(sorted(strategy_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "reviewed_samples": reviewed_samples,
        "reviewed_alignment_samples": reviewed_alignment_samples,
        "manual_override_samples": manual_override_samples,
    }


def grounded_answer_alignment_status(
    source: QASample,
    translated: QASample,
) -> str:
    if not _contains(source.context, source.right_answer):
        return "not_applicable"
    if (
        is_language_neutral_number(translated.right_answer)
        and _contains(translated.context, translated.right_answer)
    ):
        return "translated_numeric_answer"
    if (
        _is_numeric_dominant_answer(source.right_answer)
        and not missing_numbers(source.right_answer, translated.context)
    ):
        return "translated_numeric_components"
    if _contains(translated.context, translated.right_answer):
        return "translated_answer"
    if _contains(translated.context, source.right_answer):
        return "source_answer_alias"
    if _contains_source_name_components(
        translated.context,
        source.right_answer,
    ):
        return "source_answer_components"
    if (
        _is_short_proper_name(source.right_answer)
        and _approximately_contains_cjk(
            translated.context,
            translated.right_answer,
        )
    ):
        return "translated_answer_alias"
    return "lost"


def missing_numbers(source: str, translated: str) -> list[str]:
    source_numbers = list(dict.fromkeys(NUMBER_PATTERN.findall(source)))
    translated_keys = {
        _canonical_number(number)
        for number in NUMBER_PATTERN.findall(translated)
    }
    return [
        number
        for number in source_numbers
        if not _number_is_preserved(
            number,
            source,
            translated,
            translated_keys,
        )
    ]


def _has_unexpected_repetition(source: str, translated: str) -> bool:
    source_lower = source.lower()
    for match in REPEATED_WORD_PATTERN.finditer(translated):
        if match.group(0).lower() not in source_lower:
            return True
    if REPEATED_CJK_PATTERN.search(translated):
        return True

    clauses = [
        re.sub(r"\s+", "", clause).lower()
        for clause in CLAUSE_BOUNDARY_PATTERN.split(translated)
        if len(CONTENT_PATTERN.findall(clause)) >= 4
    ]
    counts = Counter(clauses)
    return any(count >= 3 for count in counts.values())


def _number_is_preserved(
    number: str,
    source: str,
    translated: str,
    translated_keys: set[str],
) -> bool:
    canonical = _canonical_number(number)
    if canonical in translated_keys:
        return True
    if _small_number_is_translated(canonical, translated):
        return True
    if _decade_is_translated(canonical, number, source, translated):
        return True
    return _scaled_number_is_translated(canonical, number, source, translated)


def _canonical_number(number: str) -> str:
    normalized = number.replace(",", "")
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return normalized
    return format(value.normalize(), "f")


def _small_number_is_translated(
    canonical: str,
    translated: str,
) -> bool:
    try:
        value = int(canonical)
    except ValueError:
        return False
    if not 0 <= value <= 99:
        return False
    chinese = _integer_to_chinese(value)
    if not chinese:
        return False
    ordinal_or_counter = (
        f"第{chinese}",
        f"{chinese}个",
        f"{chinese}名",
        f"{chinese}号",
        f"{chinese}位",
        f"{chinese}次",
    )
    if any(variant in translated for variant in ordinal_or_counter):
        return True
    return value >= 10 and chinese in translated


def _integer_to_chinese(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value < 10:
        return digits[value]
    tens, ones = divmod(value, 10)
    prefix = "" if tens == 1 else digits[tens]
    suffix = "" if ones == 0 else digits[ones]
    return f"{prefix}十{suffix}"


def _decade_is_translated(
    canonical: str,
    original: str,
    source: str,
    translated: str,
) -> bool:
    if not re.search(
        rf"(?<!\d){re.escape(original)}s\b",
        source,
        flags=re.IGNORECASE,
    ):
        return False
    try:
        year = int(canonical)
    except ValueError:
        return False
    if not 1000 <= year <= 2999:
        return False
    century = year // 100 + 1
    decade = year % 100
    return (
        f"{century}世纪{decade}年代" in translated
        or f"{decade}年代" in translated
    )


def _scaled_number_is_translated(
    canonical: str,
    original: str,
    source: str,
    translated: str,
) -> bool:
    scale_match = re.search(
        rf"(?<!\d){re.escape(original)}\s+"
        r"(billion|million|thousand)\b",
        source,
        flags=re.IGNORECASE,
    )
    if scale_match is None:
        return False
    try:
        value = Decimal(canonical)
    except InvalidOperation:
        return False
    unit = scale_match.group(1).lower()
    multiplier, marker = {
        "billion": (Decimal(10), "亿"),
        "million": (Decimal(100), "万"),
        "thousand": (Decimal(1), "千"),
    }[unit]
    converted = _canonical_number(str(value * multiplier))
    return f"{converted}{marker}" in translated


def question_intent_lost(
    source: str,
    translated: str,
    language_code: str,
) -> bool:
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
    if is_language_neutral_number(value):
        value_numbers = Counter(NUMBER_PATTERN.findall(value))
        container_numbers = Counter(NUMBER_PATTERN.findall(container))
        return bool(
            value_numbers
            and all(
                container_numbers[number] >= count
                for number, count in value_numbers.items()
            )
        )

    if re.search(r"[\u3400-\u9fff]", value):
        normalized_container = re.sub(r"[\W_]+", "", container).lower()
        normalized_value = re.sub(r"[\W_]+", "", value).lower()
        return bool(normalized_value and normalized_value in normalized_container)

    container_tokens = re.findall(r"[a-z0-9]+", container.lower())
    value_tokens = re.findall(r"[a-z0-9]+", value.lower())
    if not value_tokens:
        return False
    width = len(value_tokens)
    return any(
        container_tokens[position : position + width] == value_tokens
        for position in range(len(container_tokens) - width + 1)
    )


def _is_numeric_dominant_answer(text: str) -> bool:
    if not NUMBER_PATTERN.search(text):
        return False
    words = {
        word.lower()
        for word in re.findall(r"[A-Za-z]+", text)
    }
    allowed_words = {
        "and",
        "from",
        "km",
        "long",
        "m",
        "mi",
        "mile",
        "miles",
        "through",
        "to",
        "year",
        "years",
    }
    return words <= allowed_words


def _contains_source_name_components(
    container: str,
    source_answer: str,
) -> bool:
    tokens = _proper_name_tokens(source_answer)
    if not tokens:
        return False
    return all(
        re.search(
            rf"(?<![A-Za-z]){re.escape(token)}(?![A-Za-z])",
            container,
            flags=re.IGNORECASE,
        )
        is not None
        for token in tokens
    )


def _is_short_proper_name(text: str) -> bool:
    return bool(_proper_name_tokens(text))


def _proper_name_tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    tokens = [
        token
        for token in raw_tokens
        if token.lower() not in PROPER_NAME_CONNECTORS
        and len(token.replace("'", "")) > 1
    ]
    if not 1 <= len(tokens) <= 4:
        return []
    if not all(token[0].isupper() for token in tokens):
        return []
    return tokens


def _approximately_contains_cjk(
    container: str,
    value: str,
    threshold: float = 0.72,
) -> bool:
    normalized_container = "".join(
        character
        for character in container
        if "\u3400" <= character <= "\u9fff"
    )
    normalized_value = "".join(
        character
        for character in value
        if "\u3400" <= character <= "\u9fff"
    )
    if not normalized_value or not normalized_container:
        return False
    minimum_width = max(1, len(normalized_value) - 2)
    maximum_width = min(
        len(normalized_container),
        len(normalized_value) + 2,
    )
    for width in range(minimum_width, maximum_width + 1):
        for start in range(len(normalized_container) - width + 1):
            candidate = normalized_container[start : start + width]
            similarity = SequenceMatcher(
                None,
                normalized_value,
                candidate,
            ).ratio()
            if similarity >= threshold:
                return True
    return False


def _warning_code(warning: str) -> str:
    parts = warning.split(":")
    return parts[1] if len(parts) > 1 else parts[0]
