from __future__ import annotations

import re
from xml.sax.saxutils import escape, quoteattr

from bgaccent.unicode import BULGARIAN_VOWELS, COMBINING_ACUTE

_GRAPHEME_TO_IPA: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "е": "ɛ", "ж": "ʒ", "з": "z", "и": "i", "й": "j",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "ɔ",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ц": "ts", "ч": "tʃ", "ш": "ʃ",
    "щ": "ʃt", "ъ": "ɤ", "ь": "j", "ю": "ju", "я": "ja",
}

_WORD_RE = re.compile(r"(\S+)")


_IPA_VOWELS = frozenset("aɛiɔuɤ")


def to_ipa(word: str, stress_vowel_index: int) -> str:
    lower = word.lower().replace(COMBINING_ACUTE, "")
    ipa_chars: list[str] = []
    vowel_positions: list[int] = []
    for ch in lower:
        mapped = _GRAPHEME_TO_IPA.get(ch, ch)
        if ch in BULGARIAN_VOWELS:
            vowel_positions.append(len(ipa_chars))
        ipa_chars.append(mapped)

    if stress_vowel_index < len(vowel_positions):
        vowel_pos = vowel_positions[stress_vowel_index]
        insert_pos = vowel_pos
        while insert_pos > 0 and not any(
            c in _IPA_VOWELS for c in ipa_chars[insert_pos - 1]
        ):
            insert_pos -= 1
        # Always mark primary stress, including on a word-initial syllable
        # (e.g. "ю́жен", "бя́гам") — a leading marker is valid IPA.
        ipa_chars.insert(insert_pos, "ˈ")

    return "".join(ipa_chars)


def format_ssml(text: str) -> str:
    def _render_word(token: str) -> str:
        if COMBINING_ACUTE not in token:
            return escape(token)
        clean = token.replace(COMBINING_ACUTE, "")
        vowel_idx = 0
        for ch in token:
            if ch == COMBINING_ACUTE:
                break
            if ch in BULGARIAN_VOWELS:
                vowel_idx += 1
        stress_index = vowel_idx - 1 if vowel_idx > 0 else 0
        ipa = to_ipa(clean, stress_index)
        return (
            f'<phoneme alphabet="ipa" ph={quoteattr(ipa)}>'
            f"{escape(clean)}</phoneme>"
        )

    parts = _WORD_RE.split(text)
    return "".join(
        _render_word(part) if i % 2 == 1 else escape(part)
        for i, part in enumerate(parts)
    )
