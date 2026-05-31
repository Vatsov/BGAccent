from pathlib import Path

from bgaccent.accentor import Accentor


class TestPreserveMode:
    def test_existing_accent_unchanged(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("плани́ната") == "плани́ната"

    def test_mixed_manual_and_trie(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent("плани́ната е красива")
        assert result == "плани́ната е краси́ва"

    def test_already_accented_stat(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("плани́ната е красива")
        assert result.stats.already_accented == 1


class TestReplaceSafeMode:
    def test_relookup_same_result(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path, mode="replace-safe")
        assert acc.accent("плани́ната") == "плани́ната"

    def test_no_dict_match_preserves_manual(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path, mode="replace-safe")
        assert acc.accent("непо́зната") == "непо́зната"


class TestHomographs:
    def test_homograph_resolved_lowest_ordinal(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("замък") == "за́мък"

    def test_homograph_flagged_in_report(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("замък")
        assert "замък" in result.homographs

    def test_homograph_detail_entry(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("замък")
        hom_details = [d for d in result.details if d.get("status") == "homograph_flagged"]
        assert len(hom_details) == 1
        assert hom_details[0]["word"] == "замък"
        assert "alternatives" in hom_details[0]

    def test_homograph_stat_incremented(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("замък")
        assert result.stats.homographs_flagged == 1

    def test_homograph_deduplicated(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("замък е замък")
        assert result.homographs.count("замък") == 1

    def test_cross_source_priority_beats_lowest_ordinal(self, test_trie_path: Path) -> None:
        # "килим" has wiktionary@ordinal0 and bgospodinov@ordinal1.
        # bgospodinov outranks wiktionary in DEFAULT_PRIORITY, so its
        # ordinal must win even though it is the higher index.
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("килим") == "кили́м"

    def test_cross_source_chosen_mask(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("килим")
        hom = [d for d in result.details if d.get("status") == "homograph_flagged"]
        assert len(hom) == 1
        assert hom[0]["source_mask"] == 4


class TestOOVDetection:
    def test_cyrillic_oov_script(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("непозната")
        oov_details = [d for d in result.details if d["status"] == "oov"]
        assert len(oov_details) == 1
        assert oov_details[0]["script"] == "cyrillic"

    def test_latin_oov_script(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("unknown")
        oov_details = [d for d in result.details if d["status"] == "oov"]
        assert len(oov_details) == 1
        assert oov_details[0]["script"] == "latin"

    def test_oov_stat(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("непозната интересна")
        assert result.stats.oov_multisyllabic == 2


class TestCustomOverrideConflict:
    def test_custom_overrides_trie_logged(self, test_trie_path: Path, tmp_path: Path) -> None:
        tsv = tmp_path / "custom.tsv"
        tsv.write_text("планината\tпланина́та\n", encoding="utf-8")
        acc = Accentor(trie_path=test_trie_path, custom_dicts=[tsv])
        result = acc.accent_with_report("планината")
        assert result.text == "планина́та"
        override_details = [d for d in result.details if d.get("status") == "custom_override"]
        assert len(override_details) == 1
        assert "dictionary_alternative" in override_details[0]

    def test_custom_override_stat(self, test_trie_path: Path, tmp_path: Path) -> None:
        tsv = tmp_path / "custom.tsv"
        tsv.write_text("планината\tпланина́та\n", encoding="utf-8")
        acc = Accentor(trie_path=test_trie_path, custom_dicts=[tsv])
        result = acc.accent_with_report("планината")
        assert result.stats.custom_overrides == 1

    def test_custom_agrees_no_override(self, test_trie_path: Path, tmp_path: Path) -> None:
        tsv = tmp_path / "custom.tsv"
        tsv.write_text("планината\tплани́ната\n", encoding="utf-8")
        acc = Accentor(trie_path=test_trie_path, custom_dicts=[tsv])
        result = acc.accent_with_report("планината")
        override_details = [d for d in result.details if d.get("status") == "custom_override"]
        assert len(override_details) == 0
