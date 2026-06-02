from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from bgaccent.cli import app
from bgaccent.ssml import format_ssml, to_ipa

runner = CliRunner()
FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


class TestSsmlWrapping:
    def test_accented_word_wrapped(self) -> None:
        result = format_ssml("плани́ната")
        assert "<phoneme" in result
        assert 'alphabet="ipa"' in result

    def test_unaccented_word_not_wrapped(self) -> None:
        result = format_ssml("как")
        assert "<phoneme" not in result
        assert result == "как"

    def test_mixed_sentence(self) -> None:
        result = format_ssml("плани́ната е краси́ва")
        assert result.count("<phoneme") == 2
        assert " е " in result

    def test_punctuation_preserved(self) -> None:
        result = format_ssml("плани́ната, краси́ва!")
        assert "," in result
        assert "!" in result

    def test_quote_in_value_stays_valid_xml(self) -> None:
        # A double-quote reaching the ph attribute must not break the XML.
        # (With tokenizer-driven rendering a quote can no longer reach `ph` —
        # the `word` regex excludes it — but the result must still be valid XML.)
        result = format_ssml('a"́')
        ET.fromstring(f"<root>{result}</root>")

    def test_trailing_punctuation_outside_phoneme(self) -> None:
        result = format_ssml("плани́ната,")
        assert result == '<phoneme alphabet="ipa" ph="plaˈninata">планината</phoneme>,'
        ET.fromstring(f"<root>{result}</root>")

    def test_leading_and_trailing_punctuation_outside_phoneme(self) -> None:
        result = format_ssml("(плани́ната)")
        assert result == '(<phoneme alphabet="ipa" ph="plaˈninata">планината</phoneme>)'
        ET.fromstring(f"<root>{result}</root>")

    def test_two_accented_words_each_wrapped(self) -> None:
        result = format_ssml("пъ́рва вто́ра")
        assert result == (
            '<phoneme alphabet="ipa" ph="ˈpɤrva">първа</phoneme>'
            ' <phoneme alphabet="ipa" ph="ˈvtɔra">втора</phoneme>'
        )

    def test_unaccented_word_with_punctuation_not_wrapped(self) -> None:
        result = format_ssml("град,")
        assert result == "град,"
        assert "<phoneme" not in result

    def test_hyphenated_compound_splits_on_hyphen(self) -> None:
        # Each hyphen-separated part is rendered independently (hyphen = TTS word
        # boundary), so only the accented part is wrapped and the hyphen stays
        # outside any phoneme element.
        result = format_ssml("бяло-че́рвен")
        assert result == 'бяло-<phoneme alphabet="ipa" ph="ˈtʃɛrvɛn">червен</phoneme>'
        ET.fromstring(f"<root>{result}</root>")

    def test_number_token_passes_through(self) -> None:
        assert format_ssml("100") == "100"

    def test_abbrev_token_not_split_on_hyphen(self) -> None:
        # "д-р" is an abbreviation, not a hyphenated word — it must pass through
        # escaped, never split into phoneme parts.
        result = format_ssml("д-р")
        assert result == "д-р"
        assert "<phoneme" not in result


class TestBulgarianIpa:
    def test_stress_marker_placed(self) -> None:
        ipa = to_ipa("планина", 1)
        assert "ˈ" in ipa

    def test_basic_vowel_mapping(self) -> None:
        ipa = to_ipa("планина", 1)
        assert "a" in ipa

    def test_planina_ipa(self) -> None:
        ipa = to_ipa("планина", 1)
        assert ipa == "plaˈnina"

    def test_krasiva_ipa(self) -> None:
        ipa = to_ipa("красива", 1)
        assert "ˈ" in ipa

    def test_initial_vowel_stress_marked(self) -> None:
        # "ю́жен": stress on the first vowel must still emit the marker.
        assert to_ipa("южен", 0) == "ˈjuʒɛn"

    def test_initial_glide_stress_marked(self) -> None:
        # "бя́гам": first-vowel stress after a single consonant onset.
        assert to_ipa("бягам", 0) == "ˈbjagam"

    def test_initial_consonant_cluster_stress_marked(self) -> None:
        # "стра́на" with stress on the first vowel: the marker walks back to
        # before the whole onset cluster.
        assert to_ipa("страна", 0) == "ˈstrana"


class TestSsmlCliIntegration:
    def test_format_ssml_flag(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината е красива", encoding="utf-8")
        result = runner.invoke(app, [str(inp), "--format", "ssml", "--trie", str(FIXTURE_TRIE)])
        assert result.exit_code == 0
        assert "<phoneme" in result.output

    def test_format_text_default(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината е красива", encoding="utf-8")
        result = runner.invoke(app, [str(inp), "--format", "text", "--trie", str(FIXTURE_TRIE)])
        assert result.exit_code == 0
        assert "́" in result.output

    def test_ssml_valid_xml(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината", encoding="utf-8")
        result = runner.invoke(app, [str(inp), "--format", "ssml", "--trie", str(FIXTURE_TRIE)])
        ET.fromstring(f"<root>{result.output}</root>")

    def test_ssml_with_file_output(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината", encoding="utf-8")
        out = tmp_path / "output.ssml"
        result = runner.invoke(
            app, [str(inp), "-o", str(out), "--format", "ssml", "--trie", str(FIXTURE_TRIE)]
        )
        assert result.exit_code == 0
        assert "<phoneme" in out.read_text(encoding="utf-8")

    def test_ssml_with_check_no_output(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината", encoding="utf-8")
        result = runner.invoke(
            app, [str(inp), "--format", "ssml", "--check", "--trie", str(FIXTURE_TRIE)]
        )
        assert result.exit_code == 0
