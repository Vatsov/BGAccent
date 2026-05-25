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
    for key in trie:
        results = trie.get(key)
        if not results:
            continue
        vowel_ordinal = results[0][0]
        word = strip_accents(key).lower()
        for n in range(_MIN_NGRAM, min(_MAX_NGRAM + 1, len(word) + 1)):
            suffix = word[-n:]
            if suffix not in table:
                table[suffix] = {}
            freq = table[suffix]
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

    def predict(self, word: str) -> tuple[int, float] | None:
        self._build_table()
        clean = strip_accents(word).lower()
        if count_vowels(clean) < 2:
            return None

        best_suffix = ""
        best_ordinal = -1
        best_confidence = 0.0

        for n in range(min(_MAX_NGRAM, len(clean)), _MIN_NGRAM - 1, -1):
            suffix = clean[-n:]
            if suffix in self._table:
                freq = self._table[suffix]
                total = sum(freq.values())
                if total == 0:
                    continue
                majority_ordinal = max(freq, key=lambda k: freq[k])
                confidence = freq[majority_ordinal] / total
                if len(suffix) > len(best_suffix):
                    best_suffix = suffix
                    best_ordinal = majority_ordinal
                    best_confidence = confidence

        if not best_suffix:
            return None

        if best_confidence < self._min_confidence:
            return None

        if best_ordinal >= count_vowels(clean):
            return None

        return best_ordinal, best_confidence
