from __future__ import annotations

from typing import TYPE_CHECKING

from bgaccent.unicode import count_vowels, strip_accents

if TYPE_CHECKING:
    import marisa_trie

NgramTable = dict[str, dict[int, int]]

_MIN_NGRAM = 3
_MAX_NGRAM = 6


def build_ngram_table(
    trie: marisa_trie.RecordTrie[tuple[int, int]],
) -> NgramTable:
    table: NgramTable = {}
    # ``marisa_trie`` iteration yields the key once *per stored record*, while
    # ``trie.get`` already returns all records — iterate ``set(trie)`` so each
    # distinct key is processed once. A homograph contributes one count per
    # record (one per true stress), so the suffix statistics see every outcome.
    for key in set(trie):
        records = trie.get(key)
        if not records:
            continue
        word = strip_accents(key).lower()
        for n in range(_MIN_NGRAM, min(_MAX_NGRAM + 1, len(word) + 1)):
            freq = table.setdefault(word[-n:], {})
            for vowel_ordinal, _mask in records:
                freq[vowel_ordinal] = freq.get(vowel_ordinal, 0) + 1
    return table


class StressPredictor:
    def __init__(
        self,
        trie: marisa_trie.RecordTrie[tuple[int, int]],
        min_confidence: float = 0.7,
    ) -> None:
        self._trie = trie
        self._min_confidence = min_confidence
        self._table: NgramTable = {}
        self._built = False

    def _build_table(self) -> None:
        if not self._built:
            self._table = build_ngram_table(self._trie)
            self._built = True

    def predict(self, word: str) -> tuple[int, float, str] | None:
        self._build_table()
        clean = strip_accents(word).lower()
        if count_vowels(clean) < 2:
            return None

        n_vowels = count_vowels(clean)

        # Longest suffix first; return the longest one that *also* clears the
        # confidence threshold and is in range, rather than letting a single
        # low-confidence longest suffix reject otherwise-usable shorter matches.
        for n in range(min(_MAX_NGRAM, len(clean)), _MIN_NGRAM - 1, -1):
            suffix = clean[-n:]
            freq = self._table.get(suffix)
            if not freq:
                continue
            total = sum(freq.values())
            if total == 0:
                continue
            majority_ordinal = max(freq, key=lambda k: freq[k])
            confidence = freq[majority_ordinal] / total
            if confidence < self._min_confidence:
                continue
            if majority_ordinal >= n_vowels:
                continue
            return majority_ordinal, confidence, suffix

        return None
