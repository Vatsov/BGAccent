"""Evaluation framework — continuous quality tracking for BGAccent.

Usage:
    uv run python scripts/evaluate_quality.py --trie bg.marisa --gt gt.tsv
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bgaccent.accentor import Accentor
from bgaccent.unicode import strip_accents


@dataclass
class EvaluationResult:
    total_words: int = 0
    total_accented: int = 0
    total_correct: int = 0
    total_incorrect: int = 0
    coverage: float = 0.0
    accuracy: float = 0.0
    per_layer: dict[str, dict[str, float]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_words": self.total_words,
            "total_accented": self.total_accented,
            "total_correct": self.total_correct,
            "total_incorrect": self.total_incorrect,
            "coverage": round(self.coverage, 4),
            "accuracy": round(self.accuracy, 4),
            "per_layer": self.per_layer,
            "errors": self.errors,
        }


def evaluate_quality(trie_path: Path, ground_truth_path: Path) -> EvaluationResult:
    acc = Accentor(trie_path=trie_path)
    entries = _load_ground_truth(ground_truth_path)

    result = EvaluationResult()
    result.total_words = len(entries)
    layer_counts: dict[str, dict[str, int]] = {}

    for word, expected_accented in entries:
        report = acc.accent_with_report(word)
        predicted = report.text

        detail = report.details[0] if report.details else {}
        status = detail.get("status", "oov")
        layer = _status_to_layer(status)

        if layer not in layer_counts:
            layer_counts[layer] = {"total": 0, "correct": 0}
        layer_counts[layer]["total"] += 1

        is_accented = strip_accents(predicted) != strip_accents(word) or predicted != word

        if is_accented or status not in ("oov", "skipped_monosyllabic"):
            result.total_accented += 1
            if predicted == expected_accented:
                result.total_correct += 1
                layer_counts[layer]["correct"] += 1
            else:
                result.total_incorrect += 1
                result.errors.append(
                    {
                        "word": word,
                        "predicted": predicted,
                        "actual": expected_accented,
                        "layer": layer,
                    }
                )

    if result.total_words > 0:
        result.coverage = result.total_accented / result.total_words
    if result.total_accented > 0:
        result.accuracy = result.total_correct / result.total_accented

    for layer, counts in layer_counts.items():
        layer_acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        result.per_layer[layer] = {
            "count": float(counts["total"]),
            "accuracy": round(layer_acc, 4),
        }

    return result


def compare_baselines(current: EvaluationResult, baseline_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    regressions: list[str] = []
    improvements: list[str] = []

    for metric in ("coverage", "accuracy"):
        current_val = getattr(current, metric)
        baseline_val = baseline.get(metric, 0.0)
        diff = current_val - baseline_val
        if diff < -0.01:
            regressions.append(f"{metric}: {baseline_val:.4f} → {current_val:.4f} ({diff:+.4f})")
        elif diff > 0.01:
            improvements.append(f"{metric}: {baseline_val:.4f} → {current_val:.4f} ({diff:+.4f})")

    return {
        "regressions": regressions,
        "improvements": improvements,
    }


def _load_ground_truth(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            entries.append((parts[0], parts[1]))
    return entries


def _status_to_layer(status: str) -> str:
    mapping: dict[str, str] = {
        "accented": "trie",
        "morphological_match": "morphological",
        "neural_predicted": "neural",
        "predicted": "ngram",
        "homograph_flagged": "trie",
        "homograph_resolved": "trie",
        "custom_override": "custom",
        "already_accented": "preserved",
        "skipped_monosyllabic": "monosyllabic",
        "oov": "oov",
    }
    return mapping.get(status, "unknown")
