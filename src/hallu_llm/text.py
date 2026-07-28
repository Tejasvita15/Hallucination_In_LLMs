from __future__ import annotations

import re


MISSING_BOUNDARY_SPACE = re.compile(
    r"(?<=[a-z0-9)\]])(?P<punctuation>[.!?])(?=[\"'(\[]?[A-Z])"
)
SENTENCE_BOUNDARY = re.compile(
    r"(?P<punctuation>[.!?])\s+(?=[\"'(\[]?[A-Z0-9])"
)
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


def _ends_with_protected_abbreviation(text: str) -> bool:
    token_match = PRECEDING_TOKEN.search(text)
    if token_match is None:
        return False
    token = token_match.group(1).rstrip(".")
    if token.lower() in PROTECTED_ABBREVIATIONS or len(token) == 1:
        return True
    return bool(re.fullmatch(r"(?:[A-Z]\.)+[A-Z]?", token_match.group(1)))
