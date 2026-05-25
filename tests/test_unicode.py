import unicodedata

from bgaccent.unicode import (
    BULGARIAN_VOWELS,
    count_vowels,
    is_monosyllabic,
    normalize,
    place_accent,
    strip_accents,
)


class TestNormalize:
    def test_nfd_to_nfc(self) -> None:
        nfd = unicodedata.normalize("NFD", "планината")
        assert normalize(nfd) == "планината"
        assert all(unicodedata.is_normalized("NFC", c) for c in normalize(nfd))

    def test_y_kratko_preserved(self) -> None:
        assert normalize("й") == "й"
        assert len(normalize("й")) == 1
        assert ord(normalize("й")) == 0x0439

    def test_nfd_y_kratko_collapsed(self) -> None:
        nfd_y = "й"
        result = normalize(nfd_y)
        assert result == "й"
        assert len(result) == 1
        assert ord(result) == 0x0439


class TestStripAccents:
    def test_removes_combining_acute(self) -> None:
        assert strip_accents("плани́ната") == "планината"

    def test_preserves_breve_on_y_kratko(self) -> None:
        assert strip_accents("й") == "й"
        assert len(strip_accents("й")) == 1

    def test_no_accent_unchanged(self) -> None:
        assert strip_accents("планината") == "планината"

    def test_multiple_accents(self) -> None:
        assert strip_accents("а́б́") == "аб"


class TestBulgarianVowels:
    def test_contains_all_lowercase_vowels(self) -> None:
        for vowel in "аеиоуъяю":
            assert vowel in BULGARIAN_VOWELS

    def test_contains_all_uppercase_vowels(self) -> None:
        for vowel in "АЕИОУЪЯЮ":
            assert vowel in BULGARIAN_VOWELS

    def test_y_kratko_not_vowel(self) -> None:
        assert "й" not in BULGARIAN_VOWELS
        assert "Й" not in BULGARIAN_VOWELS

    def test_total_count(self) -> None:
        assert len(BULGARIAN_VOWELS) == 16


class TestCountVowels:
    def test_planinata(self) -> None:
        assert count_vowels("планината") == 4

    def test_ya_is_vowel(self) -> None:
        assert count_vowels("бягам") == 2

    def test_yu_is_vowel(self) -> None:
        assert count_vowels("южен") == 2

    def test_y_kratko_not_vowel(self) -> None:
        assert count_vowels("война") == 2

    def test_no_vowels(self) -> None:
        assert count_vowels("в") == 0

    def test_all_vowels(self) -> None:
        assert count_vowels("аеиоуъяю") == 8


class TestIsMonosyllabic:
    def test_single_vowel(self) -> None:
        assert is_monosyllabic("как") is True

    def test_no_vowels(self) -> None:
        assert is_monosyllabic("в") is True

    def test_multisyllabic(self) -> None:
        assert is_monosyllabic("планината") is False


class TestPlaceAccent:
    def test_second_vowel(self) -> None:
        assert place_accent("планината", 1) == "плани́ната"

    def test_first_vowel_boundary(self) -> None:
        assert place_accent("абв", 0) == "а́бв"

    def test_last_vowel_boundary(self) -> None:
        assert place_accent("града", 1) == "града́"

    def test_uppercase(self) -> None:
        assert place_accent("ПЛАНИНАТА", 1) == "ПЛАНИ́НАТА"

    def test_ya_accent(self) -> None:
        assert place_accent("бягам", 0) == "бя́гам"

    def test_yu_accent(self) -> None:
        assert place_accent("южен", 0) == "ю́жен"
