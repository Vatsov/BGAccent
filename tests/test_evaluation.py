from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_quality import (
    EvaluationResult,
    compare_baselines,
    evaluate_quality,
)

FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


def _make_ground_truth(tmp_path: Path) -> Path:
    gt = tmp_path / "ground_truth.tsv"
    gt.write_text(
        "планината\tплани́ната\nкрасива\tкраси́ва\nживот\tживо́т\nнепозната\tнепозна́та\n",
        encoding="utf-8",
    )
    return gt


class TestEvaluateQuality:
    def test_returns_evaluation_result(self, tmp_path: Path) -> None:
        gt = _make_ground_truth(tmp_path)
        result = evaluate_quality(FIXTURE_TRIE, gt)
        assert isinstance(result, EvaluationResult)

    def test_metrics_between_0_and_1(self, tmp_path: Path) -> None:
        gt = _make_ground_truth(tmp_path)
        result = evaluate_quality(FIXTURE_TRIE, gt)
        assert 0.0 <= result.coverage <= 1.0
        assert 0.0 <= result.accuracy <= 1.0

    def test_total_counts(self, tmp_path: Path) -> None:
        gt = _make_ground_truth(tmp_path)
        result = evaluate_quality(FIXTURE_TRIE, gt)
        assert result.total_words == 4
        assert result.total_accented >= 0
        assert result.total_correct >= 0

    def test_per_layer_breakdown(self, tmp_path: Path) -> None:
        gt = _make_ground_truth(tmp_path)
        result = evaluate_quality(FIXTURE_TRIE, gt)
        assert "trie" in result.per_layer

    def test_error_list(self, tmp_path: Path) -> None:
        gt = _make_ground_truth(tmp_path)
        result = evaluate_quality(FIXTURE_TRIE, gt)
        for error in result.errors:
            assert "word" in error
            assert "predicted" in error
            assert "actual" in error


class TestBaseline:
    def test_no_regression(self, tmp_path: Path) -> None:
        gt = _make_ground_truth(tmp_path)
        result = evaluate_quality(FIXTURE_TRIE, gt)
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        comparison = compare_baselines(result, baseline_path)
        assert comparison["regressions"] == []

    def test_regression_detected(self, tmp_path: Path) -> None:
        gt = _make_ground_truth(tmp_path)
        result = evaluate_quality(FIXTURE_TRIE, gt)
        fake_baseline = result.to_dict()
        fake_baseline["accuracy"] = result.accuracy + 0.1
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(fake_baseline, ensure_ascii=False), encoding="utf-8")
        comparison = compare_baselines(result, baseline_path)
        if result.accuracy < fake_baseline["accuracy"] - 0.01:
            assert len(comparison["regressions"]) > 0

    def test_to_dict_roundtrip(self, tmp_path: Path) -> None:
        gt = _make_ground_truth(tmp_path)
        result = evaluate_quality(FIXTURE_TRIE, gt)
        d = result.to_dict()
        assert "coverage" in d
        assert "accuracy" in d
        assert "per_layer" in d
        assert "errors" in d
