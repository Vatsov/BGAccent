"""Build a marisa RecordTrie from bayganyu CSV source."""

from __future__ import annotations

import json
import sqlite3
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import marisa_trie

from bgaccent.sources import (
    DEFAULT_PRIORITY,
    MIT_COMPATIBLE_SOURCES,
    SOURCE_BITS,
    SOURCE_LICENSES,
)

BULGARIAN_VOWELS = frozenset("аеиоуъяюАЕИОУЪЯЮ")


def parse_bayganyu_csv(csv_path: Path) -> list[tuple[str, int, int]]:
    entries: list[tuple[str, int, int]] = []

    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    first_line = lines[0].strip() if lines else ""
    start_idx = 1 if first_line.startswith("word") else 0

    for line_num, raw_line in enumerate(lines[start_idx:], start=start_idx + 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(",")
        if len(parts) < 2:
            print(f"Warning: {csv_path}:{line_num}: malformed line, skipping", file=sys.stderr)
            continue

        base_word = unicodedata.normalize("NFC", parts[0].strip())
        stressed_form = parts[1].strip()

        if not stressed_form:
            continue

        apos_pos = stressed_form.find("'")
        if apos_pos < 0:
            continue

        if apos_pos + 1 >= len(stressed_form):
            print(
                f"Warning: {csv_path}:{line_num}: apostrophe at end of word, skipping",
                file=sys.stderr,
            )
            continue

        stressed_char = stressed_form[apos_pos + 1]
        if stressed_char.lower() not in BULGARIAN_VOWELS:
            print(
                f"Warning: {csv_path}:{line_num}: stress on non-vowel '{stressed_char}', skipping",
                file=sys.stderr,
            )
            continue

        clean = stressed_form.replace("'", "")
        clean = unicodedata.normalize("NFC", clean)
        vowel_index = 0
        for ch in clean[:apos_pos]:
            if ch.lower() in BULGARIAN_VOWELS:
                vowel_index += 1

        nfc_vowels = sum(1 for c in base_word if c.lower() in BULGARIAN_VOWELS)
        nfd_word = unicodedata.normalize("NFD", base_word)
        nfd_vowels = sum(1 for c in nfd_word if c.lower() in BULGARIAN_VOWELS)
        if nfc_vowels != nfd_vowels:
            print(
                f"Warning: {csv_path}:{line_num}: NFC/NFD vowel count mismatch "
                f"({nfc_vowels} vs {nfd_vowels}), skipping",
                file=sys.stderr,
            )
            continue

        entries.append((base_word.lower(), vowel_index, 1))

    return entries


def build_trie(
    csv_path: Path,
    output_path: Path,
    source_name: str = "bayganyu",
    source_commit: str = "",
    source_license: str = "MIT",
    allow_invalid: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = parse_bayganyu_csv(csv_path)

    all_lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    first_line = all_lines[0].strip() if all_lines else ""
    header_offset = 1 if first_line.startswith("word") else 0
    total_lines = sum(
        1
        for raw in all_lines[header_offset:]
        if raw.strip() and not raw.strip().startswith("#")
    )
    invalid_count = total_lines - len(entries)

    if invalid_count > 0 and not allow_invalid:
        print(
            f"Error: {invalid_count} invalid entries found. "
            "Use --allow-invalid-entries to proceed.",
            file=sys.stderr,
        )
        sys.exit(1)

    word_variants: dict[str, list[tuple[int, int]]] = {}
    for word, vowel_idx, source_mask in entries:
        if word not in word_variants:
            word_variants[word] = []
        variant = (vowel_idx, source_mask)
        if variant not in word_variants[word]:
            word_variants[word].append(variant)

    homographs: dict[str, list[tuple[int, int]]] = {
        w: v for w, v in word_variants.items() if len(v) > 1
    }
    if homographs:
        log_path = output_path.parent / "build_homographs.log"
        with log_path.open("w", encoding="utf-8") as f:
            for word, variants in sorted(homographs.items()):
                ordinals = [str(v[0]) for v in variants]
                f.write(f"{word}: ordinals={','.join(ordinals)}\n")

    keys: list[str] = []
    values: list[tuple[int, int]] = []
    for word, variants in word_variants.items():
        for vowel_idx, source_mask in variants:
            keys.append(word)
            values.append((vowel_idx, source_mask))

    trie = marisa_trie.RecordTrie("HB", zip(keys, values, strict=True))
    trie.save(str(output_path))

    meta = {
        "format_version": 1,
        "record_format": "HB",
        "vowel_indexing": "zero_based_vowel_ordinal",
        "normalization": "NFC",
        "source": source_name,
        "source_commit": source_commit,
        "source_license": source_license,
        "created_at": datetime.now(UTC).isoformat(),
    }
    meta_path = output_path.parent / (output_path.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"Built trie: {len(keys)} entries ({len(word_variants)} unique words, "
        f"{len(homographs)} homographs) at {output_path}"
    )


def parse_bgospodinov_db(db_path: Path) -> list[tuple[str, int, int]]:
    """Parse bgospodinov SQLite database with backtick stress encoding."""
    entries: list[tuple[str, int, int]] = []
    conn = sqlite3.connect(str(db_path))

    cursor = conn.execute(
        "SELECT wordform, wordform_stressed FROM wordform "
        "WHERE wordform_stressed IS NOT NULL"
    )

    for wordform, stressed in cursor:
        wordform = unicodedata.normalize("NFC", wordform.strip())
        stressed = stressed.strip()

        backtick_pos = stressed.find("`")
        if backtick_pos < 1:
            continue

        if stressed[backtick_pos - 1].lower() not in BULGARIAN_VOWELS:
            print(
                f"Warning: backtick not after vowel in '{wordform}', skipping",
                file=sys.stderr,
            )
            continue

        vowel_count = 0
        for ch in stressed[:backtick_pos]:
            if ch.lower() in BULGARIAN_VOWELS:
                vowel_count += 1

        if vowel_count == 0:
            continue

        entries.append((wordform.lower(), vowel_count - 1, 4))

    conn.close()
    return entries


def parse_wiktionary_jsonl(jsonl_path: Path) -> list[tuple[str, int, int]]:
    """Parse Wiktionary JSONL with U+0301 combining acute stress marks."""
    entries: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int]] = set()

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            if obj.get("lang_code") != "bg":
                continue

            forms: list[str] = []
            word = obj.get("word", "")
            if word:
                forms.append(word)
            for form_entry in obj.get("forms", []):
                form_text = form_entry.get("form", "")
                if form_text:
                    forms.append(form_text)

            for form in forms:
                form = unicodedata.normalize("NFC", form)
                if "́" not in form:
                    continue

                plain = form.replace("́", "")

                vowel_count = 0
                for ch in form:
                    if ch == "́":
                        break
                    if ch.lower() in BULGARIAN_VOWELS:
                        vowel_count += 1

                if vowel_count == 0:
                    continue

                ordinal = vowel_count - 1
                key = (plain.lower(), ordinal)
                if key not in seen:
                    seen.add(key)
                    entries.append((plain.lower(), ordinal, 2))

    return entries


def merge_entries(
    entries_by_source: dict[str, list[tuple[str, int, int]]],
    priority: list[str] | None = None,
) -> tuple[dict[str, list[tuple[int, int]]], list[str]]:
    """Merge entries from multiple sources with conflict resolution.

    Returns (word_variants, conflict_lines).
    """
    if priority is None:
        priority = DEFAULT_PRIORITY

    word_data: dict[str, dict[str, set[int]]] = {}
    for source_name, source_entries in entries_by_source.items():
        for word, ordinal, _ in source_entries:
            word_data.setdefault(word, {}).setdefault(source_name, set()).add(ordinal)

    result: dict[str, list[tuple[int, int]]] = {}
    conflicts: list[str] = []

    for word, sources in word_data.items():
        if len(sources) == 1:
            source_name = next(iter(sources))
            bit = SOURCE_BITS.get(source_name, 0)
            result[word] = [(o, bit) for o in sorted(sources[source_name])]
            continue

        all_ordinal_sets = list(sources.values())
        if all(s == all_ordinal_sets[0] for s in all_ordinal_sets[1:]):
            combined_mask = 0
            for source_name in sources:
                combined_mask |= SOURCE_BITS.get(source_name, 0)
            result[word] = [(o, combined_mask) for o in sorted(all_ordinal_sets[0])]
        else:
            winner = None
            for source_name in priority:
                if source_name in sources:
                    winner = source_name
                    break

            if winner is None:
                continue

            winning_ordinals = sources[winner]
            entries_for_word: list[tuple[int, int]] = []
            for ordinal in sorted(winning_ordinals):
                mask = SOURCE_BITS.get(winner, 0)
                for other_name, other_ordinals in sources.items():
                    if other_name != winner and ordinal in other_ordinals:
                        mask |= SOURCE_BITS.get(other_name, 0)
                entries_for_word.append((ordinal, mask))
            result[word] = entries_for_word

            parts = []
            for source_name in priority:
                if source_name in sources:
                    ords = ",".join(str(o) for o in sorted(sources[source_name]))
                    parts.append(f"{source_name}={ords}")
            conflicts.append(f"{word}: {', '.join(parts)}")

    return result, conflicts


def build_multi_source_trie(
    sources_dir: Path,
    output_path: Path,
    license_filter: str = "mit",
) -> None:
    """Build trie from multiple sources with license filtering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed = MIT_COMPATIBLE_SOURCES if license_filter == "mit" else set(SOURCE_BITS.keys())

    all_entries: dict[str, list[tuple[str, int, int]]] = {}

    bayganyu_csv = sources_dir / "bayganyu.csv"
    if bayganyu_csv.exists() and "bayganyu" in allowed:
        all_entries["bayganyu"] = parse_bayganyu_csv(bayganyu_csv)

    bgospodinov_db = sources_dir / "bgospodinov.db"
    if bgospodinov_db.exists() and "bgospodinov" in allowed:
        all_entries["bgospodinov"] = parse_bgospodinov_db(bgospodinov_db)

    wiktionary_jsonl = sources_dir / "wiktionary.jsonl"
    if wiktionary_jsonl.exists() and "wiktionary" in allowed:
        all_entries["wiktionary"] = parse_wiktionary_jsonl(wiktionary_jsonl)

    if not all_entries:
        print("Error: no source data found", file=sys.stderr)
        sys.exit(1)

    word_variants, conflict_lines = merge_entries(all_entries)

    if conflict_lines:
        conflict_log = output_path.parent / "merge_conflicts.log"
        conflict_log.write_text("\n".join(conflict_lines) + "\n", encoding="utf-8")

    keys: list[str] = []
    values: list[tuple[int, int]] = []
    for word, variants in word_variants.items():
        for ordinal, mask in variants:
            keys.append(word)
            values.append((ordinal, mask))

    trie = marisa_trie.RecordTrie("HB", zip(keys, values, strict=True))
    trie.save(str(output_path))

    source_names = sorted(all_entries.keys())
    licenses = [SOURCE_LICENSES[s] for s in source_names]
    license_str = "+".join(licenses)

    meta = {
        "format_version": 1,
        "record_format": "HB",
        "vowel_indexing": "zero_based_vowel_ordinal",
        "normalization": "NFC",
        "sources": source_names,
        "source_license": license_str,
        "created_at": datetime.now(UTC).isoformat(),
    }
    meta_path = output_path.parent / (output_path.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    homographs = sum(1 for v in word_variants.values() if len(v) > 1)
    print(
        f"Built trie: {len(keys)} entries ({len(word_variants)} unique words, "
        f"{homographs} homographs, {len(conflict_lines)} conflicts) at {output_path}"
    )
    print(f"Sources: {', '.join(source_names)} ({license_filter})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BGAccent trie")
    parser.add_argument("--sources-dir", type=Path, default=Path("./sources"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-invalid-entries", action="store_true")
    parser.add_argument("--license", choices=["mit", "gpl"], default="mit")
    args = parser.parse_args()

    build_multi_source_trie(
        args.sources_dir,
        args.out,
        license_filter=args.license,
    )
