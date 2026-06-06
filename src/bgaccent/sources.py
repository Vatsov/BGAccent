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
    """Sort key for homograph candidates; lower means higher priority.

    ``DEFAULT_PRIORITY`` is ordered most-preferred first, so its index is the
    key directly: ascending ``sorted()`` then picks the highest-priority
    source. Unknown masks sort last.
    """
    for i, name in enumerate(DEFAULT_PRIORITY):
        if mask & SOURCE_BITS[name]:
            return i
    return 99


def records_by_priority(records: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort ``(vowel_ordinal, source_mask)`` records by source priority.

    Highest-priority source first, then lowest vowel ordinal. Used everywhere a
    homograph must be resolved deterministically (the bare ``records[0]`` order
    from the trie is arbitrary).
    """
    return sorted(records, key=lambda r: (priority_key(r[1]), r[0]))
