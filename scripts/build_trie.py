"""Build a marisa RecordTrie from bayganyu CSV source."""

from __future__ import annotations

import json
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import marisa_trie

BULGARIAN_VOWELS = frozenset("аеиоуъяюАЕИОУЪЯЮ")


def parse_bayganyu_csv(csv_path: Path) -> list[tuple[str, int, int]]:
    entries: list[tuple[str, int, int]] = []
    invalid_count = 0

    for line_num, raw_line in enumerate(
        csv_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(",")
        if len(parts) < 2:
            print(f"Warning: {csv_path}:{line_num}: malformed line, skipping", file=sys.stderr)
            invalid_count += 1
            continue

        base_word = unicodedata.normalize("NFC", parts[0].strip())
        stressed_form = parts[-1].strip()

        apos_pos = stressed_form.find("'")
        if apos_pos < 1:
            print(
                f"Warning: {csv_path}:{line_num}: no apostrophe found, skipping",
                file=sys.stderr,
            )
            invalid_count += 1
            continue

        stressed_char = stressed_form[apos_pos - 1]
        if stressed_char.lower() not in BULGARIAN_VOWELS:
            print(
                f"Warning: {csv_path}:{line_num}: stress on non-vowel '{stressed_char}', skipping",
                file=sys.stderr,
            )
            invalid_count += 1
            continue

        clean = stressed_form.replace("'", "")
        clean = unicodedata.normalize("NFC", clean)
        vowel_index = 0
        for ch in clean[: apos_pos - 1]:
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
            invalid_count += 1
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
    entries = parse_bayganyu_csv(csv_path)

    all_lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    total_lines = sum(
        1 for raw in all_lines if raw.strip() and not raw.strip().startswith("#")
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BGAccent trie")
    parser.add_argument("--sources-dir", type=Path, default=Path("./sources"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-invalid-entries", action="store_true")
    args = parser.parse_args()

    csv_file = args.sources_dir / "bayganyu.csv"
    build_trie(
        csv_file,
        args.out,
        allow_invalid=args.allow_invalid_entries,
    )
