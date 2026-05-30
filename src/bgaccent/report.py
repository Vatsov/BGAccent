from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class AccentStats:
    total_tokens: int = 0
    accented_tokens: int = 0
    oov_multisyllabic: int = 0
    skipped_monosyllabic: int = 0
    already_accented: int = 0
    homographs_flagged: int = 0
    custom_overrides: int = 0
    morphological_matches: int = 0
    predicted: int = 0
    neural_predicted: int = 0

    def __iadd__(self, other: AccentStats) -> AccentStats:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self


_LATIN_RE = re.compile(r"^[a-zA-Z\-]+$")


def detect_script(word: str) -> str:
    if _LATIN_RE.match(word):
        return "latin"
    return "cyrillic"


@dataclass
class AccentResult:
    text: str
    stats: AccentStats = field(default_factory=AccentStats)
    oov_words: list[str] = field(default_factory=list)
    homographs: list[str] = field(default_factory=list)
    custom_overrides: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "stats": asdict(self.stats),
            "oov_words": self.oov_words,
            "homographs": self.homographs,
            "custom_overrides": self.custom_overrides,
            "details": self.details,
        }
