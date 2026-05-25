from __future__ import annotations

from typing import Any

HOMOGRAPH_RULES: dict[tuple[str, str], int] = {
    ("замък", "NOUN"): 0,
    ("замък", "VERB"): 1,
    ("вълна", "NOUN"): 1,
    ("пара", "NOUN"): 0,
    ("мъка", "NOUN"): 0,
    ("мъка", "VERB"): 1,
    ("поля", "NOUN"): 1,
    ("поля", "VERB"): 0,
    ("черпя", "VERB"): 1,
    ("села", "NOUN"): 1,
    ("села", "VERB"): 0,
    ("чела", "NOUN"): 1,
    ("чела", "VERB"): 0,
    ("кръста", "NOUN"): 1,
    ("кръста", "VERB"): 0,
    ("права", "NOUN"): 1,
    ("права", "ADJ"): 0,
    ("легла", "NOUN"): 1,
    ("легла", "VERB"): 0,
    ("ръка", "NOUN"): 1,
    ("ръце", "NOUN"): 1,
    ("въже", "NOUN"): 1,
    ("коса", "NOUN"): 0,
    ("места", "NOUN"): 1,
    ("места", "VERB"): 0,
}


def rule_lookup(lemma: str, pos: str) -> int | None:
    return HOMOGRAPH_RULES.get((lemma, pos))


class HomographDisambiguator:
    def __init__(self, spacy_model: Any = None) -> None:
        self._nlp = spacy_model
        self._available = spacy_model is not None

    def disambiguate(
        self,
        word: str,
        sentence_tokens: list[str],
        trie_results: list[tuple[int, int]],
    ) -> int | None:
        if not self._available:
            return None

        sentence = " ".join(sentence_tokens)
        doc = self._nlp(sentence)

        for tok in doc:
            if tok.text.lower() == word.lower():
                lemma = tok.lemma_.lower()
                pos = tok.pos_
                result = rule_lookup(lemma, pos)
                if result is not None:
                    return result
                result = rule_lookup(word.lower(), pos)
                if result is not None:
                    return result

        return None
