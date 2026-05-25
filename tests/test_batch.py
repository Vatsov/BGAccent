from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bgaccent.cli import app

runner = CliRunner()
FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


def _make_input_dir(tmp_path: Path) -> Path:
    d = tmp_path / "input"
    d.mkdir()
    (d / "chapter1.txt").write_text("планината е красива", encoding="utf-8")
    (d / "chapter2.txt").write_text("животът е хубав", encoding="utf-8")
    (d / "notes.md").write_text("not a txt file", encoding="utf-8")
    return d


class TestDirectoryInputOutput:
    def test_outdir_creates_output_files(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        out = tmp_path / "output"
        result = runner.invoke(app, [str(inp), "--out-dir", str(out), "--trie", str(FIXTURE_TRIE)])
        assert result.exit_code == 0
        assert (out / "chapter1.txt").exists()
        assert (out / "chapter2.txt").exists()

    def test_outdir_files_contain_accented_text(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        out = tmp_path / "output"
        runner.invoke(app, [str(inp), "--out-dir", str(out), "--trie", str(FIXTURE_TRIE)])
        text = (out / "chapter1.txt").read_text(encoding="utf-8")
        assert "́" in text

    def test_log_files_created(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        out = tmp_path / "output"
        runner.invoke(app, [str(inp), "--out-dir", str(out), "--trie", str(FIXTURE_TRIE)])
        assert (out / "chapter1.log").exists()
        assert (out / "chapter2.log").exists()

    def test_outdir_created_if_missing(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        out = tmp_path / "nonexistent" / "output"
        result = runner.invoke(app, [str(inp), "--out-dir", str(out), "--trie", str(FIXTURE_TRIE)])
        assert result.exit_code == 0
        assert out.exists()

    def test_non_txt_files_skipped(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        out = tmp_path / "output"
        runner.invoke(app, [str(inp), "--out-dir", str(out), "--trie", str(FIXTURE_TRIE)])
        assert not (out / "notes.md").exists()


class TestAggregateReport:
    def test_summary_json_created(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        out = tmp_path / "output"
        report_dir = tmp_path / "reports"
        runner.invoke(app, [
            str(inp), "--out-dir", str(out),
            "--report", str(report_dir),
            "--trie", str(FIXTURE_TRIE),
        ])
        assert (report_dir / "summary.json").exists()

    def test_per_file_reports_created(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        out = tmp_path / "output"
        report_dir = tmp_path / "reports"
        runner.invoke(app, [
            str(inp), "--out-dir", str(out),
            "--report", str(report_dir),
            "--trie", str(FIXTURE_TRIE),
        ])
        assert (report_dir / "chapter1.json").exists()
        assert (report_dir / "chapter2.json").exists()

    def test_summary_has_total_stats(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        out = tmp_path / "output"
        report_dir = tmp_path / "reports"
        runner.invoke(app, [
            str(inp), "--out-dir", str(out),
            "--report", str(report_dir),
            "--trie", str(FIXTURE_TRIE),
        ])
        summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        assert "total" in summary
        assert "per_file" in summary
        assert summary["total"]["total_tokens"] > 0


class TestEmptyDirAndCheck:
    def test_empty_dir_warns_exits_0(self, tmp_path: Path) -> None:
        inp = tmp_path / "empty_input"
        inp.mkdir()
        out = tmp_path / "output"
        result = runner.invoke(app, [str(inp), "--out-dir", str(out), "--trie", str(FIXTURE_TRIE)])
        assert result.exit_code == 0

    def test_check_mode_with_directory(self, tmp_path: Path) -> None:
        inp = _make_input_dir(tmp_path)
        result = runner.invoke(app, [str(inp), "--check", "--trie", str(FIXTURE_TRIE)])
        assert result.exit_code == 0

    def test_check_max_oov_rate_exits_1(self, tmp_path: Path) -> None:
        d = tmp_path / "oov_input"
        d.mkdir()
        (d / "ch.txt").write_text(
            "абракадабра фантасмагория непостижимост", encoding="utf-8"
        )
        result = runner.invoke(app, [
            str(d), "--check", "--max-oov-rate", "0.001",
            "--trie", str(FIXTURE_TRIE),
        ])
        assert result.exit_code == 1
