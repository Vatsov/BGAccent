from __future__ import annotations

from typing import TYPE_CHECKING

from bgaccent.unicode import BULGARIAN_VOWELS, count_vowels

if TYPE_CHECKING:
    import marisa_trie

_SUFFIXES: list[str] = [
    "овете",
    "ищата",
    "етата",
    "ците",
    "ните",
    "ите",
    "ове",
    "ета",
    "ища",
    "ата",
    "ята",
    "вам",
    "ваш",
    "ват",
    "те",
    "та",
    "ът",
    "ят",
    "то",
    "ам",
    "аш",
    "ат",
    "ем",
    "еш",
    "ет",
    "им",
    "иш",
    "ям",
    "яш",
    "ят",
    "и",
]

_BASE_ENDINGS = ("а", "я", "о", "е")


class SuffixStripper:
    def __init__(self, suffixes: list[str] | None = None) -> None:
        raw = suffixes if suffixes is not None else _SUFFIXES
        self._suffixes = sorted(raw, key=len, reverse=True)

    def strip_suffix(self, word: str) -> list[str]:
        if len(word) <= 2:
            return []
        candidates: list[str] = []
        seen: set[str] = set()
        for suffix in self._suffixes:
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if len(stem) < 2 or count_vowels(stem) < 1:
                continue
            if stem[-1] not in BULGARIAN_VOWELS:
                for ending in _BASE_ENDINGS:
                    self._add_candidate(stem + ending, word, candidates, seen)
            self._add_candidate(stem, word, candidates, seen)
        return candidates

    @staticmethod
    def _add_candidate(candidate: str, original: str, out: list[str], seen: set[str]) -> None:
        if candidate != original and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)


def morphology_lookup(
    word: str,
    trie: marisa_trie.RecordTrie[tuple[int, int]],
    stripper: SuffixStripper | None = None,
) -> tuple[int, str, str] | None:
    if stripper is None:
        stripper = SuffixStripper()
    candidates = stripper.strip_suffix(word)
    for candidate in candidates:
        results = trie.get(candidate.lower())
        if results:
            vowel_ordinal = results[0][0]
            # The stripped surface suffix is the part of ``word`` past the prefix
            # it shares with the matched base form. This is correct both when the
            # base is a bare stem (``градове`` → ``град`` → "ове") and when a base
            # ending was restored (``планините`` → ``планина`` → "ите"), where the
            # candidate is *not* a prefix of the word.
            prefix_len = 0
            for word_char, candidate_char in zip(word, candidate, strict=False):
                if word_char != candidate_char:
                    break
                prefix_len += 1
            suffix = word[prefix_len:]
            return vowel_ordinal, candidate, suffix
    return None


def transfer_stress(base_ordinal: int, base_word: str, inflected_word: str) -> int | None:
    inflected_vowel_count = count_vowels(inflected_word)
    if base_ordinal >= inflected_vowel_count:
        return None
    return base_ordinal
