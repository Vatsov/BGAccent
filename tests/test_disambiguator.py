from __future__ import annotations

from pathlib import Path
from typing import Any

from bgaccent.accentor import Accentor
from bgaccent.disambiguator import HOMOGRAPH_RULES, HomographDisambiguator, rule_lookup

FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


class _FakeTok:
    """A minimal spaCy-token stand-in carrying only the attributes the
    disambiguator reads, so tests need no real spaCy install."""

    def __init__(self, text: str, pos: str, lemma: str) -> None:
        self.text = text
        self.pos_ = pos
        self.lemma_ = lemma


class _StubDisambiguator:
    """Duck-typed disambiguator that returns a fixed pre-tagged fake doc from
    ``build_doc`` and resolves positions against it, bypassing spaCy."""

    def __init__(self, doc: list[_FakeTok]) -> None:
        self._doc = doc

    def build_doc(self, words: list[str]) -> list[_FakeTok]:
        return self._doc

    def resolve_at(self, doc: Any, index: int, word: str) -> int | None:
        tok = doc[index]
        return rule_lookup(tok.lemma_.lower(), tok.pos_)


class TestRuleTable:
    def test_zamak_noun(self) -> None:
        assert HOMOGRAPH_RULES[("замък", "NOUN")] == 0

    def test_zamak_verb(self) -> None:
        assert HOMOGRAPH_RULES[("замък", "VERB")] == 1

    def test_table_has_at_least_20_entries(self) -> None:
        assert len(HOMOGRAPH_RULES) >= 20

    def test_rule_lookup_known(self) -> None:
        assert rule_lookup("замък", "NOUN") == 0

    def test_rule_lookup_unknown_pos(self) -> None:
        assert rule_lookup("замък", "ADJ") is None

    def test_rule_lookup_unknown_lemma(self) -> None:
        assert rule_lookup("абсурд", "NOUN") is None


class TestBuildDoc:
    def test_init_without_spacy(self) -> None:
        dis = HomographDisambiguator(spacy_model=None)
        assert dis._available is False

    def test_build_doc_returns_none_without_model(self) -> None:
        dis = HomographDisambiguator(spacy_model=None)
        assert dis.build_doc(["замък"]) is None


class TestResolveAt:
    def _dis(self) -> HomographDisambiguator:
        # resolve_at operates on the doc it is handed, so model availability is
        # irrelevant — a model-less instance keeps the unit test spaCy-free.
        return HomographDisambiguator(spacy_model=None)

    def test_none_doc_returns_none(self) -> None:
        assert self._dis().resolve_at(None, 0, "замък") is None

    def test_noun_position_returns_ordinal_0(self) -> None:
        doc = [_FakeTok("замък", "NOUN", "замък")]
        assert self._dis().resolve_at(doc, 0, "замък") == 0

    def test_verb_position_returns_ordinal_1(self) -> None:
        doc = [_FakeTok("замък", "VERB", "замък")]
        assert self._dis().resolve_at(doc, 0, "замък") == 1

    def test_each_occurrence_resolved_from_its_own_position(self) -> None:
        doc = [_FakeTok("замък", "NOUN", "замък"), _FakeTok("замък", "VERB", "замък")]
        dis = self._dis()
        assert dis.resolve_at(doc, 0, "замък") == 0
        assert dis.resolve_at(doc, 1, "замък") == 1

    def test_word_form_fallback_when_lemma_misses(self) -> None:
        # lemma has no rule, but the (surface form, POS) pair does.
        doc = [_FakeTok("замък", "NOUN", "несъществуващалема")]
        assert self._dis().resolve_at(doc, 0, "замък") == 0

    def test_unknown_pos_returns_none(self) -> None:
        doc = [_FakeTok("замък", "ADJ", "замък")]
        assert self._dis().resolve_at(doc, 0, "замък") is None

    def test_index_out_of_range_returns_none(self) -> None:
        doc = [_FakeTok("замък", "NOUN", "замък")]
        assert self._dis().resolve_at(doc, 5, "замък") is None


class TestAccentorDisambiguatorIntegration:
    def test_homograph_without_spacy_uses_priority(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        result = acc.accent_with_report("замък")
        details = [d for d in result.details if "замък" in str(d.get("word", ""))]
        assert len(details) == 1
        assert details[0]["status"] == "homograph_flagged"
        assert details[0].get("disambiguation") == "priority_fallback"

    def test_homograph_resolved_by_pos_status(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        acc._disambiguator = _StubDisambiguator([_FakeTok("замък", "NOUN", "замък")])
        result = acc.accent_with_report("замък")
        details = [d for d in result.details if "замък" in str(d.get("word", ""))]
        assert len(details) == 1
        assert details[0]["status"] == "homograph_resolved"
        assert details[0]["disambiguation"] == "pos"
        assert details[0]["chosen_vowel_index"] == 0

    def test_two_occurrences_resolved_independently(self) -> None:
        # The same homograph twice: first tagged NOUN (ordinal 0), second VERB
        # (ordinal 1). Each must take its own occurrence's stress.
        doc = [_FakeTok("замък", "NOUN", "замък"), _FakeTok("замък", "VERB", "замък")]
        acc = Accentor(trie_path=FIXTURE_TRIE)
        acc._disambiguator = _StubDisambiguator(doc)
        result = acc.accent_with_report("замък замък")
        resolved = [d for d in result.details if d.get("status") == "homograph_resolved"]
        assert len(resolved) == 2
        assert resolved[0]["chosen_vowel_index"] == 0
        assert resolved[1]["chosen_vowel_index"] == 1
        assert result.text == "за́мък замъ́к"
