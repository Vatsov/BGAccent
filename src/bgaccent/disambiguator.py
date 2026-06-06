from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


def resolve_rule(rules: dict[tuple[str, str], int], lemma: str, pos: str, word: str) -> int | None:
    """Look up a stress ordinal, lemma first then surface form, both by POS."""
    result = rules.get((lemma.lower(), pos))
    if result is not None:
        return result
    return rules.get((word.lower(), pos))


def load_pos_rules(path: str | Path, *, merge_builtin: bool = True) -> dict[tuple[str, str], int]:
    """Load a generated ``(form, POS) -> ordinal`` rule table.

    The file is the JSON produced by ``tools/homographs/build_pos_rules.py`` — a
    list of ``{"form", "pos", "ordinal", "support"}`` records mined from the
    labeled corpus, lifting POS-conditioned coverage far beyond the hand-written
    :data:`HOMOGRAPH_RULES`. With ``merge_builtin`` the built-in rules are kept
    as a base and the loaded rules override on key collisions. Pass the result
    as ``rules=`` to a disambiguator (or ``disambiguator_rules=`` to ``Accentor``).
    """
    records: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    loaded = {(r["form"], r["pos"]): int(r["ordinal"]) for r in records}
    if merge_builtin:
        return {**HOMOGRAPH_RULES, **loaded}
    return loaded


@dataclass(frozen=True, slots=True)
class TaggedToken:
    """A POS-tagged word, backend-agnostic. The disambiguation contract speaks
    in these rather than a spaCy ``Doc`` so any tagger backend can satisfy it."""

    pos: str
    lemma: str


class Disambiguator(Protocol):
    """The seam ``Accentor`` depends on. ``build_doc`` tags the whole input —
    given per-sentence token streams plus a parallel mask of which tokens are
    words — and returns one :class:`TaggedToken` per word, in reading order, so
    occurrence *i* aligns to word *i*. ``resolve_at`` reads the POS at a word's
    own position."""

    def build_doc(
        self, sentences: list[list[str]], is_word: list[list[bool]]
    ) -> list[TaggedToken] | None: ...

    def resolve_at(self, doc: Any, index: int, word: str) -> int | None: ...


class _RuleDisambiguator:
    """Shared rule-resolution; backends differ only in how they tag."""

    _rules: dict[tuple[str, str], int]

    def resolve_at(self, doc: Any, index: int, word: str) -> int | None:
        """Resolve the homograph at ``index`` from its own tagged position.

        Each occurrence is resolved independently, so repeated occurrences of
        the same form with different parts of speech receive different stress.
        ``word`` is the lowercased, accent-stripped surface form, used for the
        (form, POS) lookup when the lemma carries no rule. Returns ``None`` when
        ``doc`` is absent, the index is out of range, or no rule matches.
        """
        if doc is None or index >= len(doc):
            return None
        tok = doc[index]
        return resolve_rule(self._rules, tok.lemma, tok.pos, word)


class HomographDisambiguator(_RuleDisambiguator):
    """spaCy backend. Bulgarian has no official spaCy pipeline, so this is inert
    unless a spaCy-like model is supplied; :class:`StanzaDisambiguator` is the
    working Bulgarian path."""

    def __init__(
        self, spacy_model: Any = None, rules: dict[tuple[str, str], int] | None = None
    ) -> None:
        self._nlp = spacy_model
        self._available = spacy_model is not None
        self._rules = HOMOGRAPH_RULES if rules is None else rules

    def build_doc(
        self, sentences: list[list[str]], is_word: list[list[bool]]
    ) -> list[TaggedToken] | None:
        """Tag each sentence in its own context via ``Doc(vocab, words=...)`` —
        the model's tokenizer is bypassed so tokens stay one-to-one with the
        input and per-sentence punctuation gives the tagger real context.
        Returns one :class:`TaggedToken` per word token, in reading order."""
        if not self._available:
            return None
        from spacy.tokens import Doc

        tags: list[TaggedToken] = []
        for sent_tokens, flags in zip(sentences, is_word, strict=True):
            doc = Doc(self._nlp.vocab, words=sent_tokens)
            for _name, proc in self._nlp.pipeline:
                proc(doc)
            for tok, is_w in zip(doc, flags, strict=False):
                if is_w:
                    tags.append(TaggedToken(pos=tok.pos_, lemma=tok.lemma_))
        return tags


class StanzaDisambiguator(_RuleDisambiguator):
    """Bulgarian POS backend via Stanza (``pip install bgaccent[pos]``).

    Uses pre-tokenized input so Stanza tags exactly the supplied tokens — no
    re-segmentation — keeping output one-to-one with the word stream.
    """

    def __init__(
        self,
        rules: dict[tuple[str, str], int] | None = None,
        lang: str = "bg",
    ) -> None:
        self._rules = HOMOGRAPH_RULES if rules is None else rules
        try:
            import stanza
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "Stanza POS disambiguation requires `pip install bgaccent[pos]`"
            ) from exc
        self._nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos,lemma",
            tokenize_pretokenized=True,
            verbose=False,
            use_gpu=False,
        )
        self._available = True

    def build_doc(
        self, sentences: list[list[str]], is_word: list[list[bool]]
    ) -> list[TaggedToken] | None:
        if not sentences:
            return None
        doc = self._nlp(sentences)
        tags: list[TaggedToken] = []
        for sent, flags in zip(doc.sentences, is_word, strict=False):
            for word, is_w in zip(sent.words, flags, strict=False):
                if is_w:
                    tags.append(TaggedToken(pos=word.upos or "", lemma=word.lemma or ""))
        return tags
