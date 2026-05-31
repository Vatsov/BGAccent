from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from bgaccent.unicode import BULGARIAN_VOWELS, COMBINING_ACUTE


@dataclass(frozen=True, slots=True)
class CustomEntry:
    word: str
    vowel_index: int
    source_path: str


class CustomDict:
    def __init__(self) -> None:
        self._entries: dict[str, CustomEntry] = {}

    @classmethod
    def from_tsv(cls, path: Path) -> CustomDict:
        cd = cls()
        cd.load_tsv(path)
        return cd

    @classmethod
    def from_paths(cls, paths: list[Path]) -> CustomDict:
        cd = cls()
        for path in paths:
            if path.exists():
                cd.load_tsv(path)
        return cd

    def load_tsv(self, path: Path) -> None:
        for line_num, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 2:
                print(
                    f"Warning: {path}:{line_num}: wrong column count, skipping",
                    file=sys.stderr,
                )
                continue

            word = cols[0].strip()
            accented = cols[1].strip()

            accent_pos = accented.find(COMBINING_ACUTE)
            if accent_pos < 1:
                print(
                    f"Warning: {path}:{line_num}: no accent mark found, skipping",
                    file=sys.stderr,
                )
                continue

            accented_char = accented[accent_pos - 1]
            if accented_char.lower() not in BULGARIAN_VOWELS:
                print(
                    f"Warning: {path}:{line_num}: accent on non-vowel '{accented_char}', skipping",
                    file=sys.stderr,
                )
                continue

            stripped = accented.replace(COMBINING_ACUTE, "")
            vowel_index = 0
            for ch in stripped[: accent_pos - 1]:
                if ch in BULGARIAN_VOWELS:
                    vowel_index += 1

            key = word.lower()
            if key in self._entries:
                print(
                    f"Warning: {path}:{line_num}: duplicate entry for '{word}', overwriting",
                    file=sys.stderr,
                )

            self._entries[key] = CustomEntry(
                word=word,
                vowel_index=vowel_index,
                source_path=str(path),
            )

    def lookup(self, word: str) -> CustomEntry | None:
        exact = self._entries.get(word)
        if exact is not None:
            return exact
        return self._entries.get(word.lower())

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)
