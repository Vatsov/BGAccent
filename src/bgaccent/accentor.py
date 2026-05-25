from __future__ import annotations

from pathlib import Path
from typing import Any

import marisa_trie

from bgaccent.custom import CustomDict
from bgaccent.report import AccentResult, AccentStats, detect_script
from bgaccent.tokenizer import Token, detokenize, tokenize
from bgaccent.unicode import (
    count_vowels,
    normalize,
    place_accent,
    strip_accents,
)


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
        stats = AccentStats()
        details: list[dict[str, Any]] = []
        oov_seen: set[str] = set()
        oov_words: list[str] = []

        for token in tokens:
            if token.kind != "word":
                result_tokens.append(token)
                continue

            stats.total_tokens += 1
            accented, detail = self._process_word_token(token)

            result_tokens.append(Token(
                kind=token.kind,
                text=accented,
                original=accented,
                line=token.line,
                col=token.col,
            ))

            if detail is not None:
                details.append(detail)
                status = detail["status"]
                if status == "accented":
                    stats.accented_tokens += 1
                elif status == "oov":
                    stats.oov_multisyllabic += 1
                    word = detail["word"]
                    assert isinstance(word, str)
                    if word not in oov_seen:
                        oov_seen.add(word)
                        oov_words.append(word)
                elif status == "skipped_monosyllabic":
                    stats.skipped_monosyllabic += 1

        return AccentResult(
            text=detokenize(result_tokens),
            stats=stats,
            oov_words=oov_words,
            details=details,
        )

    def _process_word_token(
        self, token: Token
    ) -> tuple[str, dict[str, Any] | None]:
        word = token.text

        if "-" in word:
            accented = self._process_hyphenated(word)
            if accented != word:
                return accented, {
                    "word": word,
                    "accented": accented,
                    "status": "accented",
                    "line": token.line,
                    "column": token.col,
                }
            return accented, None

        return self._accent_single_word_with_detail(word, token.line, token.col)

    def _accent_single_word_with_detail(
        self, word: str, line: int, col: int
    ) -> tuple[str, dict[str, Any] | None]:
        clean = strip_accents(word)

        vowel_count = count_vowels(clean)
        if vowel_count == 0:
            return word, {
                "word": word,
                "status": "oov",
                "script": detect_script(word),
                "line": line,
                "column": col,
            }

        if vowel_count <= 1:
            if self._mark_monosyllables and vowel_count == 1:
                accented = place_accent(word, 0)
                return accented, {
                    "word": word,
                    "accented": accented,
                    "status": "accented",
                    "line": line,
                    "column": col,
                }
            return word, {
                "word": word,
                "status": "skipped_monosyllabic",
                "line": line,
                "column": col,
            }

        lookup_key = clean.lower()

        custom_entry = self._custom.lookup(lookup_key)
        if custom_entry is not None:
            accented = place_accent(word, custom_entry.vowel_index)
            return accented, {
                "word": word,
                "accented": accented,
                "source": "custom",
                "status": "accented",
                "line": line,
                "column": col,
            }

        results = self._trie.get(lookup_key)

        if not results:
            return word, {
                "word": word,
                "status": "oov",
                "script": detect_script(word),
                "line": line,
                "column": col,
            }

        vowel_ordinal, source_mask = results[0]
        accented = place_accent(word, vowel_ordinal)
        return accented, {
            "word": word,
            "accented": accented,
            "source": "bayganyu",
            "source_mask": source_mask,
            "status": "accented",
            "line": line,
            "column": col,
        }

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
                accented, _detail = self._accent_single_word_with_detail(part, 0, 0)
                accented_parts.append(accented)
        return "-".join(accented_parts)
