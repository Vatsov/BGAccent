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


class TestSsmlCliIntegration:
    def test_format_ssml_flag(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината е красива", encoding="utf-8")
        result = runner.invoke(
            app, [str(inp), "--format", "ssml", "--trie", str(FIXTURE_TRIE)]
        )
        assert result.exit_code == 0
        assert "<phoneme" in result.output

    def test_format_text_default(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината е красива", encoding="utf-8")
        result = runner.invoke(
            app, [str(inp), "--format", "text", "--trie", str(FIXTURE_TRIE)]
        )
        assert result.exit_code == 0
        assert "́" in result.output

    def test_ssml_valid_xml(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината", encoding="utf-8")
        result = runner.invoke(
            app, [str(inp), "--format", "ssml", "--trie", str(FIXTURE_TRIE)]
        )
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
