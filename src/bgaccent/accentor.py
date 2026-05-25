from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import marisa_trie

from bgaccent.custom import CustomDict
from bgaccent.morphology import SuffixStripper, morphology_lookup, transfer_stress
from bgaccent.report import AccentResult, AccentStats, detect_script
from bgaccent.tokenizer import Token, detokenize, tokenize
from bgaccent.unicode import (
    COMBINING_ACUTE,
    count_vowels,
    normalize,
    place_accent,
    strip_accents,
)

Mode = Literal["preserve", "replace-safe"]


class Accentor:
    def __init__(
        self,
        trie_path: Path,
        mark_monosyllables: bool = False,
        custom_dicts: list[Path] | None = None,
        mode: Mode = "preserve",
    ) -> None:
        if not trie_path.exists():
            raise FileNotFoundError(f"Trie file not found: {trie_path}")
        self._trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        self._trie.load(str(trie_path))
        self._mark_monosyllables = mark_monosyllables
        self._custom = CustomDict.from_paths(custom_dicts) if custom_dicts else CustomDict()
        self._mode = mode
        self._stripper = SuffixStripper()

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
        homograph_seen: set[str] = set()
        homographs: list[str] = []
        override_seen: set[str] = set()
        custom_overrides_list: list[str] = []

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
                elif status == "already_accented":
                    stats.already_accented += 1
                elif status == "homograph_flagged":
                    stats.accented_tokens += 1
                    stats.homographs_flagged += 1
                    word = detail["word"]
                    assert isinstance(word, str)
                    if word not in homograph_seen:
                        homograph_seen.add(word)
                        homographs.append(word)
                elif status == "morphological_match":
                    stats.accented_tokens += 1
                    stats.morphological_matches += 1
                elif status == "custom_override":
                    stats.accented_tokens += 1
                    stats.custom_overrides += 1
                    word = detail["word"]
                    assert isinstance(word, str)
                    if word not in override_seen:
                        override_seen.add(word)
                        custom_overrides_list.append(word)

        return AccentResult(
            text=detokenize(result_tokens),
            stats=stats,
            oov_words=oov_words,
            homographs=homographs,
            custom_overrides=custom_overrides_list,
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
        has_existing_accent = COMBINING_ACUTE in word
        clean = strip_accents(word)

        if has_existing_accent and self._mode == "preserve":
            return word, {
                "word": clean,
                "status": "already_accented",
                "line": line,
                "column": col,
            }

        if has_existing_accent and self._mode == "replace-safe":
            lookup_key = clean.lower()
            custom_entry = self._custom.lookup(lookup_key)
            results = self._trie.get(lookup_key)
            if custom_entry is not None:
                accented = place_accent(clean, custom_entry.vowel_index)
                return accented, {
                    "word": clean,
                    "accented": accented,
                    "source": "custom",
                    "status": "accented",
                    "line": line,
                    "column": col,
                }
            if results:
                vowel_ordinal, source_mask = results[0]
                accented = place_accent(clean, vowel_ordinal)
                return accented, {
                    "word": clean,
                    "accented": accented,
                    "source": "bayganyu",
                    "source_mask": source_mask,
                    "status": "accented",
                    "line": line,
                    "column": col,
                }
            return word, {
                "word": clean,
                "status": "already_accented",
                "line": line,
                "column": col,
            }

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
            trie_results = self._trie.get(lookup_key)
            if trie_results:
                trie_ordinal = trie_results[0][0]
                if trie_ordinal != custom_entry.vowel_index:
                    trie_accented = place_accent(word, trie_ordinal)
                    return accented, {
                        "word": word,
                        "accented": accented,
                        "source": "custom",
                        "dictionary_alternative": trie_accented,
                        "status": "custom_override",
                        "line": line,
                        "column": col,
                    }
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
            morph = morphology_lookup(clean, self._trie, self._stripper)
            if morph is not None:
                base_ordinal, base_form, suffix_stripped = morph
                transferred = transfer_stress(base_ordinal, base_form, word)
                if transferred is not None:
                    accented = place_accent(word, transferred)
                    return accented, {
                        "word": word,
                        "accented": accented,
                        "base_form": base_form,
                        "suffix_stripped": suffix_stripped,
                        "status": "morphological_match",
                        "line": line,
                        "column": col,
                    }
            return word, {
                "word": word,
                "status": "oov",
                "script": detect_script(word),
                "line": line,
                "column": col,
            }

        if len(results) > 1:
            sorted_results = sorted(results, key=lambda r: (-(r[1] & 0xF0), r[0]))
            chosen_ordinal, chosen_mask = sorted_results[0]
            accented = place_accent(word, chosen_ordinal)
            alternatives = [
                {"vowel_index": r[0], "source_mask": r[1]}
                for r in sorted_results[1:]
            ]
            return accented, {
                "word": word,
                "accented": accented,
                "chosen_vowel_index": chosen_ordinal,
                "source_mask": chosen_mask,
                "alternatives": alternatives,
                "status": "homograph_flagged",
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
