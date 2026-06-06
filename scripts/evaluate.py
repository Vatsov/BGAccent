"""Evaluate accent coverage on sample Bulgarian text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bgaccent.accentor import Accentor


def evaluate(trie_path: Path, text_path: Path) -> dict[str, object]:
    if not trie_path.exists():
        print(f"Error: trie not found at {trie_path}", file=sys.stderr)
        sys.exit(1)

    acc = Accentor(trie_path=trie_path)
    text = text_path.read_text(encoding="utf-8-sig")
    result = acc.accent_with_report(text)

    stats = result.stats
    total = stats.total_tokens
    # accented_tokens already includes homograph hits (both flagged and
    # resolved), so it must not be summed with homographs_flagged again.
    accented = stats.accented_tokens
    oov = stats.oov_multisyllabic
    multisyllabic = total - stats.skipped_monosyllabic
    coverage_rate = (accented / multisyllabic * 100) if multisyllabic > 0 else 0.0

    output = {
        "total_tokens": total,
        "accented_tokens": accented,
        "oov_tokens": oov,
        "coverage_rate": round(coverage_rate, 1),
    }

    print(f"total_tokens: {total}")
    print(f"accented_tokens: {accented}")
    print(f"oov_tokens: {oov}")
    print(f"coverage_rate: {coverage_rate:.1f}%")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BGAccent coverage")
    parser.add_argument("--trie", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.trie, args.text)
