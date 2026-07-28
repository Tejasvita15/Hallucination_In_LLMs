from __future__ import annotations

import re


MISSING_BOUNDARY_SPACE = re.compile(
    r"(?<=[a-z0-9)\]])(?P<punctuation>[.!?])(?=[\"'(\[]?[A-Z])"
)
SENTENCE_BOUNDARY = re.compile(
    r"(?P<punctuation>[.!?])\s+(?=[\"'(\[]?[A-Z0-9])"
)
QUESTION_CLAUSE_BOUNDARY = re.compile(r"\s*[,;]\s*")
EMBEDDED_INTERROGATIVE = re.compile(
    r"\b(?:how many|how much|which(?:\s+one)?|what|where|when|why)\b",
    re.IGNORECASE,
)
LANGUAGE_NEUTRAL_NUMBER = re.compile(r"[\d\s.,:%+/\-–—()]+")
TRAILING_CLAUSE_SEPARATOR = re.compile(r"[\s,，;；]+$")
PRECEDING_TOKEN = re.compile(r"([A-Za-z.]+)[.!?]$")
PROTECTED_ABBREVIATIONS = {
    "dr",
    "jr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "st",
    "vs",
}


def split_english_sentences(text: str) -> list[str]:
    normalized = MISSING_BOUNDARY_SPACE.sub(
        lambda match: f"{match.group('punctuation')} ",
        text.strip(),
    )
    if not normalized:
        return []

    sentences: list[str] = []
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(normalized):
        candidate = normalized[start : match.start("punctuation") + 1].strip()
        if _ends_with_protected_abbreviation(candidate):
            continue
        if candidate:
            sentences.append(candidate)
        start = match.end()

    remainder = normalized[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def split_english_question_clauses(text: str) -> list[str]:
    clauses = [
        clause.strip()
        for clause in QUESTION_CLAUSE_BOUNDARY.split(text.strip())
        if clause.strip()
    ]
    expanded: list[str] = []
    for clause in clauses:
        matches = list(EMBEDDED_INTERROGATIVE.finditer(clause))
        split_match = next(
            (
                match
                for match in reversed(matches)
                if match.start() >= 8
            ),
            None,
        )
        if split_match is None:
            expanded.append(clause)
            continue
        prefix = clause[: split_match.start()].strip()
        question_tail = clause[split_match.start() :].strip()
        if prefix:
            expanded.append(prefix)
        if question_tail:
            expanded.append(question_tail)
    return expanded


def is_language_neutral_number(text: str) -> bool:
    return bool(text.strip() and LANGUAGE_NEUTRAL_NUMBER.fullmatch(text.strip()))


def join_translated_question_clauses(clauses: list[str]) -> str:
    cleaned = [
        TRAILING_CLAUSE_SEPARATOR.sub("", clause.strip())
        for clause in clauses
        if clause.strip()
    ]
    return "，".join(clause for clause in cleaned if clause)


def _ends_with_protected_abbreviation(text: str) -> bool:
    token_match = PRECEDING_TOKEN.search(text)
    if token_match is None:
        return False
    token = token_match.group(1).rstrip(".")
    if token.lower() in PROTECTED_ABBREVIATIONS or len(token) == 1:
        return True
    return bool(re.fullmatch(r"(?:[A-Z]\.)+[A-Z]?", token_match.group(1)))
