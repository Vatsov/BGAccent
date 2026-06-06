"""Build a test trie fixture for BGAccent tests.

Run: uv run python tests/helpers/build_test_trie.py

Each entry: (word, vowel_ordinal, source_mask)
  vowel_ordinal: 0-based index among Bulgarian vowels in the NFC-normalized word
  source_mask: 1=bayganyu, 2=wiktionary, 4=bgospodinov

Note: some ordinals here are deliberately *not* the linguistically correct
stress used by the bundled 230K dictionary — e.g. "планината" is ordinal 1
("плани́ната") here, whereas the real dict yields "Планина́та" (ordinal 2).
This keeps fixture-based tests from silently passing against the shipped data
and decouples them from dictionary updates. Do not "correct" these to match
the README examples, which document the bundled dict.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import marisa_trie

ENTRIES: list[tuple[str, int, int]] = [
    ("планината", 1, 1),
    ("красива", 1, 1),
    ("планина", 1, 1),
    ("красив", 1, 1),
    ("голям", 1, 1),
    ("малък", 1, 1),
    ("хубав", 0, 1),
    ("добър", 0, 1),
    ("човек", 1, 1),
    ("живот", 1, 1),
    ("работа", 1, 1),
    ("време", 0, 1),
    ("книга", 1, 1),
    ("вода", 1, 1),
    ("земя", 1, 1),
    ("дума", 1, 1),
    ("глава", 1, 1),
    ("ръка", 1, 1),
    ("нога", 1, 1),
    ("очи", 1, 1),
    ("майка", 0, 1),
    ("баща", 1, 1),
    ("дете", 1, 1),
    ("жена", 1, 1),
    ("мъж", 0, 1),
    ("слънце", 0, 1),
    ("месец", 0, 1),
    ("година", 1, 1),
    ("ден", 0, 1),
    ("нощ", 0, 1),
    ("сутрин", 1, 1),
    ("вечер", 0, 1),
    ("обед", 1, 1),
    ("държава", 1, 1),
    ("столица", 1, 1),
    ("улица", 0, 1),
    ("здравейте", 1, 1),
    ("училище", 1, 1),
    ("университет", 3, 1),
    ("програма", 2, 1),
    ("компютър", 2, 1),
    # я/ю vowel coverage
    ("няма", 0, 1),
    ("любов", 1, 1),
    ("бягам", 0, 1),
    ("южен", 0, 1),
    ("приятел", 1, 1),
    ("събитие", 1, 1),
    ("въпрос", 1, 1),
    ("отговор", 2, 1),
    ("търся", 1, 1),
    # Hyphenated compound (to test compound lookup)
    ("по-голям", 2, 1),
    # More words for coverage
    ("война", 1, 1),
    ("годишнина", 2, 1),
    ("страна", 1, 1),
    ("стара", 1, 1),
    ("нова", 1, 1),
    # Homograph: "замък" with two stress positions (same source)
    ("замък", 0, 1),
    ("замък", 1, 1),
    # Cross-source homograph: bgospodinov (higher priority) carries the
    # HIGHER ordinal, so source precedence — not lowest-ordinal — must decide.
    ("килим", 0, 2),  # wiktionary -> ки́лим
    ("килим", 1, 4),  # bgospodinov -> кили́м (must win)
]


def build_trie(output_path: Path) -> None:
    keys: list[str] = []
    values: list[bytes] = []
    for word, vowel_ordinal, source_mask in ENTRIES:
        keys.append(word)
        values.append(struct.pack("HB", vowel_ordinal, source_mask))

    trie = marisa_trie.RecordTrie(
        "HB", zip(keys, [struct.unpack("HB", v) for v in values], strict=True)
    )
    trie.save(str(output_path))
    print(f"Built test trie with {len(keys)} entries at {output_path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/bg_test.marisa")
    out.parent.mkdir(parents=True, exist_ok=True)
    build_trie(out)
