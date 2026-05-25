from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import marisa_trie

from bgaccent.custom import CustomDict
from bgaccent.tokenizer import Token, detokenize, tokenize
from bgaccent.unicode import (
    count_vowels,
    is_monosyllabic,
    normalize,
    place_accent,
    strip_accents,
)


@dataclass
class AccentResult:
    text: str
    stats: dict[str, int] = field(default_factory=dict)
    oov_words: list[str] = field(default_factory=list)
    homographs: list[str] = field(default_factory=list)
    custom_overrides: list[str] = field(default_factory=list)
    details: list[dict[str, object]] = field(default_factory=list)


class Accentor:
    def __init__(
        self,
        trie_path: Path,
        mark_monosyllables: bool = False,
        custom_dicts: list[Path] | None = None,
    ) -> None:
        if not trie_path.exists():
            raise FileNotFoundError(f"Trie file not found: {trie_path}")
        self._trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        self._trie.load(str(trie_path))
        self._mark_monosyllables = mark_monosyllables
        self._custom = CustomDict.from_paths(custom_dicts) if custom_dicts else CustomDict()

    def accent(self, text: str) -> str:
        return self.accent_with_report(text).text

    def accent_with_report(self, text: str) -> AccentResult:
        normalized = normalize(text)
        tokens = tokenize(normalized)
        result_tokens: list[Token] = []

        for token in tokens:
            if token.kind != "word":
                result_tokens.append(token)
                continue

            accented = self._process_word_token(token)
            result_tokens.append(Token(
                kind=token.kind,
                text=accented,
                original=accented,
                line=token.line,
                col=token.col,
            ))

        return AccentResult(text=detokenize(result_tokens))

    def _process_word_token(self, token: Token) -> str:
        word = token.text

        if "-" in word:
            return self._process_hyphenated(word)

        return self._accent_single_word(word)

    def _process_hyphenated(self, word: str) -> str:
        lookup_key = strip_accents(word).lower()
        results = self._trie.get(lookup_key)
        if results:
            vowel_ordinal, _source_mask = results[0]
            return place_accent(word, vowel_ordinal)

        parts = word.split("-")
        accented_parts: list[str] = []
        for part in parts:
            if part.isdigit():
                accented_parts.append(part)
            else:
                accented_parts.append(self._accent_single_word(part))
        return "-".join(accented_parts)

    def _accent_single_word(self, word: str) -> str:
        clean = strip_accents(word)

        if is_monosyllabic(clean):
            if self._mark_monosyllables and count_vowels(clean) == 1:
                return place_accent(word, 0)
            return word

        lookup_key = clean.lower()

        custom_entry = self._custom.lookup(lookup_key)
        if custom_entry is not None:
            return place_accent(word, custom_entry.vowel_index)

        results = self._trie.get(lookup_key)

        if not results:
            return word

        vowel_ordinal, _source_mask = results[0]
        return place_accent(word, vowel_ordinal)
