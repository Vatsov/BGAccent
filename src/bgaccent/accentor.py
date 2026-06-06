from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import marisa_trie

from bgaccent.custom import CustomDict
from bgaccent.disambiguator import (
    Disambiguator,
    HomographDisambiguator,
    StanzaDisambiguator,
    load_pos_rules,
)
from bgaccent.morphology import SuffixStripper, morphology_lookup, transfer_stress
from bgaccent.neural_predictor import NeuralStressPredictor
from bgaccent.ngram_predictor import StressPredictor
from bgaccent.report import AccentResult, AccentStats, detect_script
from bgaccent.sources import mask_to_labels, records_by_priority
from bgaccent.tokenizer import Token, detokenize, tokenize
from bgaccent.unicode import (
    COMBINING_ACUTE,
    count_vowels,
    normalize,
    place_accent,
    strip_accents,
)

Mode = Literal["preserve", "replace-safe"]

_SENTENCE_END = {".", "!", "?", "…"}


def _segment_for_tagging(tokens: list[Token]) -> tuple[list[list[str]], list[list[bool]]]:
    """Split tokens into per-sentence streams for the POS tagger.

    Each sentence keeps its non-space tokens (words, numbers, punctuation) so
    the tagger has clause context; the parallel ``is_word`` mask marks which are
    accentable words, so the disambiguator emits one tag per word in reading
    order. Sentence boundaries fall after ``.!?…`` punctuation.
    """
    sentences: list[list[str]] = []
    flags: list[list[bool]] = []
    current: list[str] = []
    current_flags: list[bool] = []
    for token in tokens:
        if token.kind == "space":
            continue
        current.append(token.text)
        current_flags.append(token.kind == "word")
        if token.kind == "punct" and token.text in _SENTENCE_END:
            sentences.append(current)
            flags.append(current_flags)
            current = []
            current_flags = []
    if current:
        sentences.append(current)
        flags.append(current_flags)
    return sentences, flags


class Accentor:
    def __init__(
        self,
        trie_path: Path,
        mark_monosyllables: bool = False,
        custom_dicts: list[Path] | None = None,
        mode: Mode = "preserve",
        disambiguator: Literal["stanza", "spacy"] | None = None,
        disambiguator_model: Any = None,
        disambiguator_rules: dict[tuple[str, str], int] | Path | str | None = None,
        enable_prediction: bool = False,
        neural_model_path: Path | None = None,
        neural_vocab_path: Path | None = None,
    ) -> None:
        if not trie_path.exists():
            raise FileNotFoundError(f"Trie file not found: {trie_path}")
        self._trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        self._trie.load(str(trie_path))
        self._mark_monosyllables = mark_monosyllables
        self._custom = CustomDict.from_paths(custom_dicts) if custom_dicts else CustomDict()
        self._mode = mode
        self._stripper = SuffixStripper()
        rules = (
            load_pos_rules(disambiguator_rules)
            if isinstance(disambiguator_rules, str | Path)
            else disambiguator_rules
        )
        self._disambiguator: Disambiguator
        if disambiguator == "stanza":
            self._disambiguator = StanzaDisambiguator(rules=rules)
        else:
            self._disambiguator = HomographDisambiguator(
                spacy_model=disambiguator_model, rules=rules
            )
        self._predictor: StressPredictor | None = (
            StressPredictor(self._trie) if enable_prediction else None
        )
        self._neural: NeuralStressPredictor | None = (
            NeuralStressPredictor(neural_model_path, neural_vocab_path)
            if neural_model_path is not None and neural_vocab_path is not None
            else None
        )

    def accent(self, text: str) -> str:
        return self.accent_with_report(text).text

    def accent_with_report(self, text: str) -> AccentResult:
        normalized = normalize(text)
        tokens = tokenize(normalized)
        # Tag once, per sentence and with punctuation retained, so the tagger
        # sees real clause context instead of one document-long word blur. The
        # returned tags carry one entry per word in reading order, so the
        # running ``word_index`` below indexes each occurrence's own POS.
        sentences, is_word = _segment_for_tagging(tokens)
        tagged_doc = self._disambiguator.build_doc(sentences, is_word)
        word_index = 0
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
            accented, detail = self._process_word_token(token, tagged_doc, word_index)
            word_index += 1

            result_tokens.append(
                Token(
                    kind=token.kind,
                    text=accented,
                    original=accented,
                    line=token.line,
                    col=token.col,
                )
            )

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
                elif status in ("homograph_flagged", "homograph_resolved"):
                    stats.accented_tokens += 1
                    if status == "homograph_flagged":
                        stats.homographs_flagged += 1
                    word = detail["word"]
                    assert isinstance(word, str)
                    if word not in homograph_seen:
                        homograph_seen.add(word)
                        homographs.append(word)
                elif status == "morphological_match":
                    stats.accented_tokens += 1
                    stats.morphological_matches += 1
                elif status == "neural_predicted":
                    stats.accented_tokens += 1
                    stats.neural_predicted += 1
                elif status == "predicted":
                    stats.accented_tokens += 1
                    stats.predicted += 1
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
        self, token: Token, tagged_doc: Any = None, word_index: int | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        word = token.text

        if COMBINING_ACUTE in word and self._mode == "preserve":
            return word, {
                "word": strip_accents(word),
                "status": "already_accented",
                "line": token.line,
                "column": token.col,
            }

        if "-" in word:
            return self._process_hyphenated(word, token.line, token.col)

        return self._accent_single_word_with_detail(
            word, token.line, token.col, tagged_doc, word_index
        )

    def _accent_single_word_with_detail(
        self,
        word: str,
        line: int,
        col: int,
        tagged_doc: Any = None,
        word_index: int | None = None,
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
                    "sources": ["custom"],
                    "status": "accented",
                    "line": line,
                    "column": col,
                }
            if results:
                vowel_ordinal, source_mask = records_by_priority(results)[0]
                accented = place_accent(clean, vowel_ordinal)
                return accented, {
                    "word": clean,
                    "accented": accented,
                    "sources": mask_to_labels(source_mask),
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
                trie_ordinal = records_by_priority(trie_results)[0][0]
                if trie_ordinal != custom_entry.vowel_index:
                    trie_accented = place_accent(word, trie_ordinal)
                    return accented, {
                        "word": word,
                        "accented": accented,
                        "sources": ["custom"],
                        "dictionary_alternative": trie_accented,
                        "status": "custom_override",
                        "line": line,
                        "column": col,
                    }
            return accented, {
                "word": word,
                "accented": accented,
                "sources": ["custom"],
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
            if self._neural is not None:
                neural_result = self._neural.predict(clean)
                if neural_result is not None:
                    n_ordinal, n_conf = neural_result
                    accented = place_accent(word, n_ordinal)
                    return accented, {
                        "word": word,
                        "accented": accented,
                        "prediction_confidence": round(n_conf, 3),
                        "prediction_method": "neural",
                        "status": "neural_predicted",
                        "line": line,
                        "column": col,
                    }
            if self._predictor is not None:
                prediction = self._predictor.predict(clean)
                if prediction is not None:
                    pred_ordinal, pred_confidence, matched_suffix = prediction
                    accented = place_accent(word, pred_ordinal)
                    return accented, {
                        "word": word,
                        "accented": accented,
                        "prediction_confidence": round(pred_confidence, 3),
                        "matching_suffix": matched_suffix,
                        "status": "predicted",
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
            resolved = None
            if tagged_doc is not None and word_index is not None:
                resolved = self._disambiguator.resolve_at(
                    tagged_doc, word_index, strip_accents(word).lower()
                )
            if resolved is not None and resolved in {r[0] for r in results}:
                accented = place_accent(word, resolved)
                return accented, {
                    "word": word,
                    "accented": accented,
                    "chosen_vowel_index": resolved,
                    "status": "homograph_resolved",
                    "disambiguation": "pos",
                    "line": line,
                    "column": col,
                }
            sorted_results = records_by_priority(results)
            chosen_ordinal, chosen_mask = sorted_results[0]
            accented = place_accent(word, chosen_ordinal)
            alternatives = [{"vowel_index": r[0], "source_mask": r[1]} for r in sorted_results[1:]]
            return accented, {
                "word": word,
                "accented": accented,
                "chosen_vowel_index": chosen_ordinal,
                "source_mask": chosen_mask,
                "alternatives": alternatives,
                "status": "homograph_flagged",
                "disambiguation": "priority_fallback",
                "line": line,
                "column": col,
            }

        vowel_ordinal, source_mask = results[0]
        accented = place_accent(word, vowel_ordinal)
        return accented, {
            "word": word,
            "accented": accented,
            "sources": mask_to_labels(source_mask),
            "source_mask": source_mask,
            "status": "accented",
            "line": line,
            "column": col,
        }

    def _process_hyphenated(self, word: str, line: int, col: int) -> tuple[str, dict[str, Any]]:
        """Accent a hyphenated token and return ``(accented, detail)``.

        Whole-token matches (custom dict, then trie) emit a flat detail with the
        same provenance shape as the non-hyphenated path (``sources`` and, for
        trie hits, the real scalar ``source_mask``).

        When no whole-token match exists, each constituent is accented on its
        own and the detail is a *compound* one: ``{"compound": True, "parts":
        [<per-part detail>, ...]}``. A compound token carries **no** top-level
        ``source_mask`` because its surface is several independent accent
        decisions with potentially different sources — the per-part details hold
        the real provenance. ``status`` is ``accented`` (surface changed),
        ``oov`` (unchanged, a multisyllabic part missed), or
        ``skipped_monosyllabic`` (unchanged, nothing accentable).
        """
        clean = strip_accents(word)
        lookup_key = clean.lower()

        custom_entry = self._custom.lookup(lookup_key)
        if custom_entry is not None:
            accented = place_accent(clean, custom_entry.vowel_index)
            if accented != clean:
                return accented, {
                    "word": word,
                    "accented": accented,
                    "sources": ["custom"],
                    "status": "accented",
                    "line": line,
                    "column": col,
                }
            return accented, {
                "word": word,
                "status": "skipped_monosyllabic",
                "line": line,
                "column": col,
            }

        results = self._trie.get(lookup_key)
        if results:
            vowel_ordinal, source_mask = records_by_priority(results)[0]
            accented = place_accent(clean, vowel_ordinal)
            if accented != clean:
                return accented, {
                    "word": word,
                    "accented": accented,
                    "sources": mask_to_labels(source_mask),
                    "source_mask": source_mask,
                    "status": "accented",
                    "line": line,
                    "column": col,
                }
            return accented, {
                "word": word,
                "status": "skipped_monosyllabic",
                "line": line,
                "column": col,
            }

        parts = clean.split("-")
        accented_parts: list[str] = []
        part_details: list[dict[str, Any]] = []
        any_oov = False
        for part in parts:
            if part.isdigit():
                accented_parts.append(part)
                continue
            # Hyphen constituents are resolved in isolation: no tagged doc /
            # word index is threaded, so a homograph part falls back to source
            # priority rather than POS disambiguation. This is a deliberate gap
            # — a compound part has no standalone position in the word stream.
            accented_part, detail = self._accent_single_word_with_detail(part, line, col)
            accented_parts.append(accented_part)
            if detail is not None:
                part_details.append(detail)
                if detail.get("status") == "oov":
                    any_oov = True

        joined = "-".join(accented_parts)
        if joined != clean:
            status = "accented"
        elif any_oov:
            status = "oov"
        else:
            status = "skipped_monosyllabic"

        compound_detail: dict[str, Any] = {
            "word": word,
            "accented": joined,
            "compound": True,
            "parts": part_details,
            "status": status,
            "line": line,
            "column": col,
        }
        if status == "oov":
            compound_detail["script"] = detect_script(word)
        return joined, compound_detail
