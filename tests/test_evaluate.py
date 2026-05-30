import re
import subprocess
import sys
from pathlib import Path

from scripts.evaluate import evaluate

from .conftest import TEST_TRIE_PATH

SAMPLE_TEXT = Path(__file__).parent / "fixtures" / "sample_bg.txt"


class TestEvaluateScript:
    def test_exits_0_on_success(self) -> None:
        result = evaluate(TEST_TRIE_PATH, SAMPLE_TEXT)
        assert "total_tokens" in result
        assert "coverage_rate" in result

    def test_subprocess_output_format(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.evaluate",
                "--trie",
                str(TEST_TRIE_PATH),
                "--text",
                str(SAMPLE_TEXT),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert re.search(r"total_tokens: \d+", result.stdout)
        assert re.search(r"accented_tokens: \d+", result.stdout)
        assert re.search(r"oov_tokens: \d+", result.stdout)
        assert re.search(r"coverage_rate: \d+\.\d+%", result.stdout)

    def test_coverage_rate_in_range(self) -> None:
        result = evaluate(TEST_TRIE_PATH, SAMPLE_TEXT)
        rate = result["coverage_rate"]
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 100.0

    def test_flagged_homographs_not_double_counted(
        self, tmp_path: Path
    ) -> None:
        # "замък" is a flagged homograph: accented_tokens already counts it,
        # so coverage must be 100%, not 200%.
        text = tmp_path / "homograph.txt"
        text.write_text("замък", encoding="utf-8")
        result = evaluate(TEST_TRIE_PATH, text)
        assert result["accented_tokens"] == 1
        assert result["coverage_rate"] == 100.0

    def test_missing_trie_exits_nonzero(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.evaluate",
                "--trie",
                str(tmp_path / "nonexistent.marisa"),
                "--text",
                str(SAMPLE_TEXT),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0
