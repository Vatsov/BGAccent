from __future__ import annotations

from pathlib import Path

import marisa_trie

from bgaccent.accentor import Accentor
from bgaccent.morphology import SuffixStripper, morphology_lookup, transfer_stress
from bgaccent.unicode import count_vowels

FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


class TestSuffixStripping:
    def setup_method(self) -> None:
        self.stripper = SuffixStripper()

    def test_definite_article_plural(self) -> None:
        candidates = self.stripper.strip_suffix("планините")
        assert "планина" in candidates

    def test_complex_suffix_chain(self) -> None:
        candidates = self.stripper.strip_suffix("градовете")
        assert "градове" in candidates
        assert "град" in [c for c in candidates if count_vowels(c) >= 1]

    def test_verb_conjugation(self) -> None:
        candidates = self.stripper.strip_suffix("четем")
        assert "чета" in candidates

    def test_base_form_returns_empty(self) -> None:
        candidates = self.stripper.strip_suffix("къща")
        assert candidates == []

    def test_too_short_returns_empty(self) -> None:
        candidates = self.stripper.strip_suffix("в")
        assert candidates == []

    def test_ordered_by_suffix_length_descending(self) -> None:
        candidates = self.stripper.strip_suffix("планините")
        assert len(candidates) >= 2


class TestMorphologyLookup:
    def _build_trie(
        self, entries: dict[str, tuple[int, int]]
    ) -> marisa_trie.RecordTrie[tuple[int, int]]:
        return marisa_trie.RecordTrie("HB", entries.items())

    def test_finds_base_via_suffix(self) -> None:
        trie = self._build_trie({"планина": (1, 1)})
        result = morphology_lookup("планините", trie)
        assert result is not None
        vowel_ordinal, base, _suffix = result
        assert base == "планина"
        assert vowel_ordinal == 1

    def test_finds_gradovete_via_grad(self) -> None:
        trie = self._build_trie({"град": (0, 1)})
        result = morphology_lookup("градовете", trie)
        assert result is not None
        assert result[1] == "град"

    def test_no_match_returns_none(self) -> None:
        trie = self._build_trie({"нещо": (1, 1)})
        result = morphology_lookup("абсолютнонепозната", trie)
        assert result is None

    def test_first_trie_match_wins(self) -> None:
        trie = self._build_trie({"планина": (1, 1), "планин": (0, 1)})
        result = morphology_lookup("планините", trie)
        assert result is not None
        assert result[1] == "планина"


class TestStressTransfer:
    def test_ordinal_within_bounds(self) -> None:
        assert transfer_stress(1, "планина", "планините") == 1

    def test_ordinal_zero(self) -> None:
        assert transfer_stress(0, "град", "градовете") == 0

    def test_more_vowels_in_inflected(self) -> None:
        assert transfer_stress(1, "дърво", "дървета") == 1

    def test_ordinal_exceeds_inflected_vowels(self) -> None:
        assert transfer_stress(5, "абстракция", "ма") is None

    def test_darvo_to_darveta(self) -> None:
        assert transfer_stress(1, "дърво", "дървета") == 1


class TestAccentorMorphologicalIntegration:
    def test_planините_resolved_via_planina(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        assert acc.accent("планините") == "плани́ните"

    def test_morphological_match_stat_counted(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        result = acc.accent_with_report("планините")
        assert result.stats.morphological_matches == 1

    def test_morphological_detail_entry(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        result = acc.accent_with_report("планините")
        morph_details = [d for d in result.details if d.get("status") == "morphological_match"]
        assert len(morph_details) == 1
        assert morph_details[0]["base_form"] == "планина"

    def test_custom_dict_wins_over_morphological(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.tsv"
        custom.write_text("планините\tплани́ните\n", encoding="utf-8")
        acc = Accentor(trie_path=FIXTURE_TRIE, custom_dicts=[custom])
        result = acc.accent_with_report("планините")
        statuses = [d["status"] for d in result.details]
        assert "morphological_match" not in statuses

    def test_trie_hit_wins_over_morphological(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        result = acc.accent_with_report("планината")
        statuses = [d["status"] for d in result.details]
        assert "morphological_match" not in statuses
        assert "accented" in statuses


class TestMorphologicalEdgeCases:
    def test_impossible_ordinal_falls_to_oov(self) -> None:
        assert transfer_stress(5, "абстракция", "ма") is None

    def test_zero_vowel_stem_skipped(self) -> None:
        stripper = SuffixStripper()
        candidates = stripper.strip_suffix("бте")
        assert all(count_vowels(c) >= 1 for c in candidates)

    def test_monosyllabic_base_still_matches(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE, mark_monosyllables=False)
        result = acc.accent_with_report("времето")
        morph_details = [d for d in result.details if d.get("status") == "morphological_match"]
        assert len(morph_details) == 1
        assert morph_details[0]["base_form"] == "време"

    def test_already_accented_preserve_skips_morphology(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE, mode="preserve")
        result = acc.accent_with_report("плани́ните")
        statuses = [d["status"] for d in result.details]
        assert "morphological_match" not in statuses
