from pathlib import Path

import pytest

from bgaccent.accentor import Accentor
from bgaccent.custom import CustomDict

from .conftest import TEST_TRIE_PATH


@pytest.fixture
def custom_tsv(tmp_path: Path) -> Path:
    tsv = tmp_path / "custom.tsv"
    tsv.write_text("синклер\tсинкле́р\n", encoding="utf-8")
    return tsv


class TestTSVParsing:
    def test_extract_vowel_ordinal(self, custom_tsv: Path) -> None:
        cd = CustomDict.from_tsv(custom_tsv)
        entry = cd.lookup("синклер")
        assert entry is not None
        assert entry.vowel_index == 1

    def test_three_column_ignored(self, tmp_path: Path) -> None:
        tsv = tmp_path / "custom.tsv"
        tsv.write_text("демиан\tде́миан\tbook character\n", encoding="utf-8")
        cd = CustomDict.from_tsv(tsv)
        entry = cd.lookup("демиан")
        assert entry is not None
        assert entry.vowel_index == 0

    def test_exact_match(self, custom_tsv: Path) -> None:
        acc = Accentor(trie_path=TEST_TRIE_PATH, custom_dicts=[custom_tsv])
        assert acc.accent("синклер") == "синкле́р"

    def test_case_insensitive_fallback(self, custom_tsv: Path) -> None:
        acc = Accentor(trie_path=TEST_TRIE_PATH, custom_dicts=[custom_tsv])
        assert acc.accent("Синклер") == "Синкле́р"

    def test_custom_overrides_trie(self, tmp_path: Path) -> None:
        tsv = tmp_path / "custom.tsv"
        tsv.write_text("планината\tплани́ната\n", encoding="utf-8")
        acc = Accentor(trie_path=TEST_TRIE_PATH, custom_dicts=[tsv])
        assert acc.accent("планината") == "плани́ната"


class TestStackingPriority:
    def test_explicit_overrides_project(self, tmp_path: Path) -> None:
        project = tmp_path / "project" / "custom_stress.tsv"
        project.parent.mkdir()
        project.write_text("замък\tзамъ́к\n", encoding="utf-8")
        explicit = tmp_path / "explicit.tsv"
        explicit.write_text("замък\tза́мък\n", encoding="utf-8")
        acc = Accentor(
            trie_path=TEST_TRIE_PATH,
            custom_dicts=[project, explicit],
        )
        assert acc.accent("замък") == "за́мък"

    def test_later_file_overrides_earlier(self, tmp_path: Path) -> None:
        first = tmp_path / "first.tsv"
        first.write_text("замък\tза́мък\n", encoding="utf-8")
        second = tmp_path / "second.tsv"
        second.write_text("замък\tзамъ́к\n", encoding="utf-8")
        acc = Accentor(
            trie_path=TEST_TRIE_PATH,
            custom_dicts=[first, second],
        )
        assert acc.accent("замък") == "замъ́к"


class TestValidation:
    def test_single_column_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tsv = tmp_path / "bad.tsv"
        tsv.write_text("badline\n", encoding="utf-8")
        cd = CustomDict.from_tsv(tsv)
        assert cd.lookup("badline") is None
        assert "skipping" in capsys.readouterr().err.lower()

    def test_no_accent_mark_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tsv = tmp_path / "bad.tsv"
        tsv.write_text("тест\tтест\n", encoding="utf-8")
        cd = CustomDict.from_tsv(tsv)
        assert cd.lookup("тест") is None
        assert "skipping" in capsys.readouterr().err.lower()

    def test_accent_on_non_vowel_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tsv = tmp_path / "bad.tsv"
        tsv.write_text("тест\tт́ест\n", encoding="utf-8")
        cd = CustomDict.from_tsv(tsv)
        assert cd.lookup("тест") is None
        assert "skipping" in capsys.readouterr().err.lower()

    def test_duplicate_last_wins(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        tsv = tmp_path / "dup.tsv"
        tsv.write_text("тест\tте́ст\nтест\tте́ст\n", encoding="utf-8")
        cd = CustomDict.from_tsv(tsv)
        entry = cd.lookup("тест")
        assert entry is not None
        assert "duplicate" in capsys.readouterr().err.lower()

    def test_valid_entries_survive_invalid(self, tmp_path: Path) -> None:
        tsv = tmp_path / "mixed.tsv"
        tsv.write_text("badline\nтест\tте́ст\n", encoding="utf-8")
        cd = CustomDict.from_tsv(tsv)
        assert cd.lookup("badline") is None
        assert cd.lookup("тест") is not None
