from __future__ import annotations

import unicodedata

COMBINING_ACUTE = "́"

BULGARIAN_VOWELS: frozenset[str] = frozenset("аеиоуъяюАЕИОУЪЯЮ")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_accents(text: str) -> str:
    return text.replace(COMBINING_ACUTE, "")


def count_vowels(word: str) -> int:
    return sum(1 for ch in word if ch in BULGARIAN_VOWELS)


def is_monosyllabic(word: str) -> bool:
    return count_vowels(word) <= 1


def place_accent(word: str, vowel_index: int) -> str:
    vowels_seen = 0
    for i, ch in enumerate(word):
        if ch in BULGARIAN_VOWELS:
            if vowels_seen == vowel_index:
                return word[: i + 1] + COMBINING_ACUTE + word[i + 1 :]
            vowels_seen += 1
    raise ValueError(f"Vowel index {vowel_index} out of range for word '{word}'")
