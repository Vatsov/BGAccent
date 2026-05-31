from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from bgaccent.accentor import Accentor
from bgaccent.disambiguator import HOMOGRAPH_RULES, HomographDisambiguator, rule_lookup

FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


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


class TestDisambiguatorWithoutSpacy:
    def test_init_without_spacy(self) -> None:
        dis = HomographDisambiguator(spacy_model=None)
        assert dis._available is False

    def test_disambiguate_returns_none_without_spacy(self) -> None:
        dis = HomographDisambiguator(spacy_model=None)
        result = dis.disambiguate("замък", ["Старият", "замък", "беше"], [(0, 1), (1, 1)])
        assert result is None


class TestDisambiguatorWithMockedSpacy:
    def _mock_nlp(self, tokens_and_pos: list[tuple[str, str]]) -> MagicMock:
        nlp = MagicMock()
        doc = MagicMock()
        mock_tokens = []
        for text, pos in tokens_and_pos:
            tok = MagicMock()
            tok.text = text
            tok.pos_ = pos
            tok.lemma_ = text.lower()
            mock_tokens.append(tok)
        doc.__iter__ = lambda self: iter(mock_tokens)
        nlp.return_value = doc
        return nlp

    def test_noun_context_returns_ordinal_0(self) -> None:
        nlp = self._mock_nlp(
            [
                ("Старият", "ADJ"),
                ("замък", "NOUN"),
                ("беше", "AUX"),
                ("красив", "ADJ"),
            ]
        )
        dis = HomographDisambiguator(spacy_model=nlp)
        result = dis.disambiguate("замък", ["Старият", "замък", "беше", "красив"], [(0, 1), (1, 1)])
        assert result == 0

    def test_verb_context_returns_ordinal_1(self) -> None:
        nlp = self._mock_nlp(
            [
                ("Той", "PRON"),
                ("замък", "VERB"),
                ("торбата", "NOUN"),
            ]
        )
        dis = HomographDisambiguator(spacy_model=nlp)
        result = dis.disambiguate("замък", ["Той", "замък", "торбата"], [(0, 1), (1, 1)])
        assert result == 1

    def test_unknown_word_returns_none(self) -> None:
        nlp = self._mock_nlp([("непозната", "ADJ")])
        dis = HomographDisambiguator(spacy_model=nlp)
        result = dis.disambiguate("непозната", ["непозната"], [(0, 1), (1, 1)])
        assert result is None


class TestAccentorDisambiguatorIntegration:
    def test_homograph_without_spacy_uses_priority(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        result = acc.accent_with_report("замък")
        details = [d for d in result.details if "замък" in str(d.get("word", ""))]
        assert len(details) == 1
        assert details[0]["status"] == "homograph_flagged"
        assert details[0].get("disambiguation") == "priority_fallback"

    def test_homograph_resolved_by_pos_status(self) -> None:
        nlp = MagicMock()
        doc = MagicMock()
        tok = MagicMock()
        tok.text = "замък"
        tok.pos_ = "NOUN"
        tok.lemma_ = "замък"
        doc.__iter__ = lambda self: iter([tok])
        nlp.return_value = doc

        acc = Accentor(trie_path=FIXTURE_TRIE, disambiguator_model=nlp)
        result = acc.accent_with_report("замък")
        details = [d for d in result.details if "замък" in str(d.get("word", ""))]
        assert len(details) == 1
        assert details[0]["status"] == "homograph_resolved"
        assert details[0]["disambiguation"] == "pos"
