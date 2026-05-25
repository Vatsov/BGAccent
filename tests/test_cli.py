import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bgaccent.cli import app

from .conftest import TEST_TRIE_PATH

runner = CliRunner()


class TestFileIO:
    def test_file_input_output(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        out = tmp_path / "output.txt"
        inp.write_text("планината е красива", encoding="utf-8")
        result = runner.invoke(
            app, [str(inp), "-o", str(out), "--trie", str(TEST_TRIE_PATH)]
        )
        assert result.exit_code == 0
        content = out.read_text(encoding="utf-8")
        assert "плани́ната" in content

    def test_default_stdout(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината", encoding="utf-8")
        result = runner.invoke(
            app, [str(inp), "--trie", str(TEST_TRIE_PATH)]
        )
        assert result.exit_code == 0
        assert "плани́ната" in result.stdout

    def test_stdin_dash(self) -> None:
        result = runner.invoke(
            app,
            ["-", "--trie", str(TEST_TRIE_PATH)],
            input="планината",
        )
        assert result.exit_code == 0
        assert "плани́ната" in result.stdout

    def test_raw_text_positional(self) -> None:
        result = runner.invoke(
            app,
            ["планината е красива", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "плани́ната" in result.stdout

    def test_utf8_sig_bom_input(self, tmp_path: Path) -> None:
        inp = tmp_path / "bom.txt"
        inp.write_bytes(b"\xef\xbb\xbf" + "планината".encode())
        result = runner.invoke(
            app, [str(inp), "--trie", str(TEST_TRIE_PATH)]
        )
        assert result.exit_code == 0
        assert "плани́ната" in result.stdout


class TestCoreFlags:
    def test_custom_dict(self, tmp_path: Path) -> None:
        tsv = tmp_path / "custom.tsv"
        tsv.write_text("синклер\tсинкле́р\n", encoding="utf-8")
        result = runner.invoke(
            app,
            ["синклер", "--custom", str(tsv), "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "синкле́р" in result.stdout

    def test_mode_preserve(self) -> None:
        result = runner.invoke(
            app,
            ["плани́ната", "--mode", "preserve", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "плани́ната" in result.stdout

    def test_mode_replace_safe(self) -> None:
        result = runner.invoke(
            app,
            [
                "плани́ната",
                "--mode",
                "replace-safe",
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 0
        assert "плани́ната" in result.stdout

    def test_mark_monosyllables(self) -> None:
        result = runner.invoke(
            app,
            ["ден", "--mark-monosyllables", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "де́н" in result.stdout

    def test_report_json(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.json"
        result = runner.invoke(
            app,
            [
                "планината",
                "--report",
                str(report_path),
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 0
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert "stats" in data
        assert "oov_words" in data

    def test_quiet_flag(self) -> None:
        result = runner.invoke(
            app,
            ["планината", "--quiet", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0

    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout


class TestDataLoader:
    def test_meta_json_bad_version(self, tmp_path: Path) -> None:
        meta = tmp_path / "bg.marisa.meta.json"
        meta.write_text(
            json.dumps({"format_version": 99}), encoding="utf-8"
        )
        from bgaccent.data import _validate_meta

        with pytest.raises(ValueError, match="Unsupported"):
            _validate_meta(meta)

    def test_meta_json_good_version(self, tmp_path: Path) -> None:
        meta = tmp_path / "bg.marisa.meta.json"
        meta.write_text(
            json.dumps({"format_version": 1}), encoding="utf-8"
        )
        from bgaccent.data import _validate_meta

        _validate_meta(meta)
