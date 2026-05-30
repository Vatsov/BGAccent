from __future__ import annotations

SOURCE_BITS: dict[str, int] = {
    "bayganyu": 1,
    "wiktionary": 2,
    "bgospodinov": 4,
}

DEFAULT_PRIORITY: list[str] = ["bgospodinov", "bayganyu", "wiktionary"]

SOURCE_LICENSES: dict[str, str] = {
    "bayganyu": "MIT",
    "wiktionary": "CC-BY-SA",
    "bgospodinov": "GPL-3.0",
}

MIT_COMPATIBLE_SOURCES: set[str] = {"bayganyu", "wiktionary"}


def mask_to_labels(mask: int) -> list[str]:
    return [name for name, bit in SOURCE_BITS.items() if mask & bit]


def priority_key(mask: int) -> int:
    for i, name in enumerate(DEFAULT_PRIORITY):
        if mask & SOURCE_BITS[name]:
            return -i
    return -99
