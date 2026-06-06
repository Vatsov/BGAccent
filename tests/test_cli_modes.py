import json
from pathlib import Path

from typer.testing import CliRunner

from bgaccent.cli import app

from .conftest import TEST_TRIE_PATH

runner = CliRunner()


class TestCheckMode:
    def test_check_no_text_output(self) -> None:
        result = runner.invoke(
            app,
            ["планината", "--check", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "плани́ната" not in result.stdout

    def test_check_exits_0_no_thresholds(self) -> None:
        result = runner.invoke(
            app,
            ["планината", "--check", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0

    def test_check_fail_on_oov_exits_1(self) -> None:
        result = runner.invoke(
            app,
            [
                "непозната",
                "--check",
                "--fail-on-oov",
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 1

    def test_check_fail_on_oov_exits_0_when_clean(self) -> None:
        result = runner.invoke(
            app,
            [
                "планината",
                "--check",
                "--fail-on-oov",
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 0

    def test_check_fail_on_homographs(self) -> None:
        result = runner.invoke(
            app,
            [
                "замък",
                "--check",
                "--fail-on-homographs",
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 1

    def test_check_max_oov_rate_exceeds(self) -> None:
        result = runner.invoke(
            app,
            [
                "непозната интересна",
                "--check",
                "--max-oov-rate",
                "0.05",
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 1

    def test_check_max_oov_rate_within(self) -> None:
        result = runner.invoke(
            app,
            [
                "планината непозната",
                "--check",
                "--max-oov-rate",
                "0.9",
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 0

    def test_check_with_report(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.json"
        result = runner.invoke(
            app,
            [
                "планината",
                "--check",
                "--report",
                str(report_path),
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 0
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert "stats" in data


class TestDiffMode:
    def test_diff_output_format(self) -> None:
        result = runner.invoke(
            app,
            ["планината е красива", "--diff", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "L1:" in result.stdout
        assert "->" in result.stdout

    def test_diff_replaces_normal_output(self) -> None:
        result = runner.invoke(
            app,
            ["планината", "--diff", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "плани́ната" not in result.stdout.split("\n")[0] or "->" in result.stdout

    def test_diff_no_changes_empty(self) -> None:
        result = runner.invoke(
            app,
            ["как сте", "--diff", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == ""

    def test_diff_includes_morphological_match(self) -> None:
        # "планините" is resolved via the morphological path, which still
        # changes the surface text and must appear in --diff.
        result = runner.invoke(
            app,
            ["планините", "--diff", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "планините -> плани́ните" in result.stdout


class TestUnknownOnlyMode:
    def test_unknown_only_output(self) -> None:
        result = runner.invoke(
            app,
            [
                "непозната интересна планината",
                "--unknown-only",
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")
        assert "непозната" in lines
        assert "интересна" in lines

    def test_unknown_only_includes_latin(self) -> None:
        result = runner.invoke(
            app,
            [
                "unknown планината",
                "--unknown-only",
                "--trie",
                str(TEST_TRIE_PATH),
            ],
        )
        assert result.exit_code == 0
        assert "unknown" in result.stdout


class TestErrorHandling:
    def test_missing_input_file(self) -> None:
        result = runner.invoke(
            app,
            ["/nonexistent/file.txt", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 1

    def test_non_utf8_input(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "win1251.txt"
        bad_file.write_bytes(b"\xc0\xc1\xc2\xc3")
        result = runner.invoke(
            app,
            [str(bad_file), "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 1

    def test_bom_utf8_reads_correctly(self, tmp_path: Path) -> None:
        bom_file = tmp_path / "bom.txt"
        bom_file.write_bytes(b"\xef\xbb\xbf" + "планината".encode())
        result = runner.invoke(
            app,
            [str(bom_file), "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        assert "плани́ната" in result.stdout


class TestLogFile:
    def test_log_created_with_output(self, tmp_path: Path) -> None:
        inp = tmp_path / "input.txt"
        inp.write_text("планината", encoding="utf-8")
        out = tmp_path / "output.txt"
        result = runner.invoke(
            app,
            [str(inp), "-o", str(out), "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
        log = tmp_path / "output.log"
        assert log.exists()
        log_content = log.read_text(encoding="utf-8")
        assert "total_tokens" in log_content

    def test_no_log_without_output(self) -> None:
        result = runner.invoke(
            app,
            ["планината", "--trie", str(TEST_TRIE_PATH)],
        )
        assert result.exit_code == 0
