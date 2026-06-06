import json
from pathlib import Path

from bgaccent.accentor import Accentor
from bgaccent.report import AccentResult, AccentStats


class TestDataclassStructure:
    def test_accent_stats_fields(self) -> None:
        stats = AccentStats()
        assert stats.total_tokens == 0
        assert stats.accented_tokens == 0
        assert stats.oov_multisyllabic == 0
        assert stats.skipped_monosyllabic == 0
        assert stats.already_accented == 0
        assert stats.homographs_flagged == 0
        assert stats.custom_overrides == 0

    def test_accent_result_fields(self) -> None:
        result = AccentResult(text="test")
        assert result.text == "test"
        assert isinstance(result.stats, AccentStats)
        assert result.oov_words == []
        assert result.homographs == []
        assert result.custom_overrides == []
        assert result.details == []


class TestStatsPopulation:
    def test_basic_stats(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("планината е как")
        assert result.stats.total_tokens == 3
        assert result.stats.accented_tokens == 1
        assert result.stats.skipped_monosyllabic == 2

    def test_oov_deduplication(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("непозната е непозната")
        assert result.oov_words == ["непозната"]

    def test_oov_sorted_by_first_appearance(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("бета алфа")
        assert result.oov_words == ["бета", "алфа"]


class TestDetailEntries:
    def test_accented_word_detail(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("планината")
        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["word"] == "планината"
        assert detail["accented"] == "плани́ната"
        assert detail["status"] == "accented"
        assert detail["line"] == 1
        assert detail["column"] == 1

    def test_oov_detail(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("ааа непозната")
        oov_details = [d for d in result.details if d["status"] == "oov"]
        assert len(oov_details) >= 1
        detail = oov_details[0]
        assert detail["script"] == "cyrillic"
        assert detail["line"] == 1

    def test_multiline_line_col(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("планината\nкрасива")
        assert len(result.details) == 2
        assert result.details[0]["line"] == 1
        assert result.details[0]["column"] == 1
        assert result.details[1]["line"] == 2
        assert result.details[1]["column"] == 1


class TestJSONSerialization:
    def test_roundtrip(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("планината")
        d = result.to_dict()
        serialized = json.dumps(d, ensure_ascii=False)
        deserialized = json.loads(serialized)
        assert deserialized == d

    def test_schema_version_present(self, test_trie_path: Path) -> None:
        from bgaccent.report import SCHEMA_VERSION

        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("планината")
        assert result.to_dict()["schema_version"] == SCHEMA_VERSION

    def test_latin_oov_script(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("unknown")
        latin_details = [d for d in result.details if d.get("script") == "latin"]
        assert len(latin_details) == 1

    def test_cyrillic_oov_script(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("непозната")
        cyrillic_details = [d for d in result.details if d.get("script") == "cyrillic"]
        assert len(cyrillic_details) == 1
