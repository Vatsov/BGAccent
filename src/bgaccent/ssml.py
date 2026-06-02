from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from bgaccent.tokenizer import tokenize
from bgaccent.unicode import BULGARIAN_VOWELS, COMBINING_ACUTE

_GRAPHEME_TO_IPA: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "ɛ",
    "ж": "ʒ",
    "з": "z",
    "и": "i",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "ɔ",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ц": "ts",
    "ч": "tʃ",
    "ш": "ʃ",
    "щ": "ʃt",
    "ъ": "ɤ",
    "ь": "j",
    "ю": "ju",
    "я": "ja",
}

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
        while insert_pos > 0 and not any(c in _IPA_VOWELS for c in ipa_chars[insert_pos - 1]):
            insert_pos -= 1
        # Always mark primary stress, including on a word-initial syllable
        # (e.g. "ю́жен", "бя́гам") — a leading marker is valid IPA.
        ipa_chars.insert(insert_pos, "ˈ")

    return "".join(ipa_chars)


def _render_part(part: str) -> str:
    """Render one hyphen-free chunk: a phoneme element if it carries a stress
    mark, otherwise escaped text."""
    if COMBINING_ACUTE not in part:
        return escape(part)
    clean = part.replace(COMBINING_ACUTE, "")
    vowel_idx = 0
    for ch in part:
        if ch == COMBINING_ACUTE:
            break
        if ch in BULGARIAN_VOWELS:
            vowel_idx += 1
    stress_index = vowel_idx - 1 if vowel_idx > 0 else 0
    ipa = to_ipa(clean, stress_index)
    return f'<phoneme alphabet="ipa" ph={quoteattr(ipa)}>{escape(clean)}</phoneme>'


def format_ssml(text: str) -> str:
    out: list[str] = []
    for token in tokenize(text):
        if token.kind != "word":
            out.append(escape(token.text))
            continue
        # Each hyphen-separated part is rendered independently: the hyphen is a
        # TTS word boundary, so a compound like "бяло-че́рвен" yields one phoneme
        # per accented part with the hyphen left outside any element (and never
        # fed into the IPA). Heuristic: this does not special-case comparative
        # single-stress-domain forms such as "по-голя́м".
        out.append(escape("-").join(_render_part(p) for p in token.text.split("-")))
    return "".join(out)
