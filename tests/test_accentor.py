from pathlib import Path

import pytest

from bgaccent.accentor import Accentor

from .conftest import TEST_TRIE_PATH


class TestBasicAccent:
    def test_accent_single_word(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("планината") == "плани́ната"

    def test_accent_sentence(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("планината е красива") == "плани́ната е краси́ва"

    def test_unknown_word_passthrough(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent("непозната")
        assert result == "непозната"

    def test_missing_trie_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            Accentor(trie_path=Path("/nonexistent/path.marisa"))


class TestMonosyllabicAndCase:
    def test_monosyllabic_skipped(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        for word in ["как", "сте", "в", "с"]:
            assert acc.accent(word) == word

    def test_mark_monosyllables(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path, mark_monosyllables=True)
        result = acc.accent("ден")
        assert result == "де́н"

    def test_uppercase_preserved(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("ПЛАНИНАТА") == "ПЛАНИ́НАТА"

    def test_mixed_case(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("Планината") == "Плани́ната"


class TestHyphenatedCompounds:
    def test_compound_in_trie(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent("по-голям")
        assert "́" in result

    def test_compound_split_fallback(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent("много-стара")
        assert "стара" not in result or "́" in result

    def test_number_word_compound(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent("100-годишнина")
        assert "годишни́на" in result

    def test_number_monosyllabic_compound(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("3-те") == "3-те"


class TestYaYuVowels:
    def test_nyama(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("няма") == "ня́ма"

    def test_lyubov(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("любов") == "любо́в"

    def test_byagam(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("бягам") == "бя́гам"

    def test_yuzhen(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("южен") == "ю́жен"

    def test_priyatel(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        assert acc.accent("приятел") == "прия́тел"


class TestAccentWithReport:
    def test_returns_accent_result(self, test_trie_path: Path) -> None:
        acc = Accentor(trie_path=test_trie_path)
        result = acc.accent_with_report("планината")
        assert result.text == "плани́ната"


class TestModuleLevelAPI:
    def test_import_accent(self) -> None:
        from bgaccent import accent  # noqa: F401

    def test_import_accent_with_report(self) -> None:
        from bgaccent import accent_with_report  # noqa: F401

    def test_accent_via_singleton(self) -> None:
        from bgaccent import accent

        result = accent("планината", trie_path=TEST_TRIE_PATH)
        assert result == "плани́ната"

    def test_accent_with_report_via_singleton(self) -> None:
        from bgaccent import accent_with_report

        result = accent_with_report("планината", trie_path=TEST_TRIE_PATH)
        assert result.text == "плани́ната"
