from __future__ import annotations

import re
from dataclasses import dataclass, field
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
            "stats": {
                "total_tokens": self.stats.total_tokens,
                "accented_tokens": self.stats.accented_tokens,
                "oov_multisyllabic": self.stats.oov_multisyllabic,
                "skipped_monosyllabic": self.stats.skipped_monosyllabic,
                "already_accented": self.stats.already_accented,
                "homographs_flagged": self.stats.homographs_flagged,
                "custom_overrides": self.stats.custom_overrides,
                "morphological_matches": self.stats.morphological_matches,
            },
            "oov_words": self.oov_words,
            "homographs": self.homographs,
            "custom_overrides": self.custom_overrides,
            "details": self.details,
        }
