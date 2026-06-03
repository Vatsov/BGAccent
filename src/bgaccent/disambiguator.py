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

    def build_doc(self, words: list[str]) -> Any | None:
        """POS-tag every word in a single pass.

        Returns a tagged spaCy ``Doc`` whose token *i* corresponds to
        ``words[i]`` by construction: the model's own tokenizer is bypassed via
        ``Doc(vocab, words=...)``, so it can neither merge nor split tokens and
        the word-list/doc alignment is exact. Returns ``None`` when no spaCy
        model is configured. Call once per text; pair with :meth:`resolve_at`.
        """
        if not self._available:
            return None
        return self._tag(list(words))

    def _tag(self, words: list[str]) -> Any:
        from spacy.tokens import Doc

        doc = Doc(self._nlp.vocab, words=words)
        for _name, proc in self._nlp.pipeline:
            proc(doc)
        return doc

    def resolve_at(self, doc: Any, index: int, word: str) -> int | None:
        """Resolve the homograph at ``index`` from its own position in ``doc``.

        Each occurrence is resolved independently from the token at its own
        position, so repeated occurrences of the same form with different parts
        of speech receive different stress. ``word`` is the lowercased,
        accent-stripped surface form, used for the (form, POS) fallback when the
        lemma carries no rule. Returns ``None`` when ``doc`` is absent, the
        index is out of range, or no rule matches.
        """
        if doc is None or index >= len(doc):
            return None
        tok = doc[index]
        result = rule_lookup(tok.lemma_.lower(), tok.pos_)
        if result is not None:
            return result
        return rule_lookup(word.lower(), tok.pos_)
