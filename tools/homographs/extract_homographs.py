"""Extract form-level accentual homographs from a kaikki.org Bulgarian dump.

A surface form is an accentual homograph if, across all dictionary entries,
it carries an accent on >=2 distinct vowel positions (vowel ordinals).
Output: data/homographs/homographs.jsonl  (one record per ambiguous surface form)
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import defaultdict

VOWELS = frozenset("аеиоуъяю")  # matches src/bgaccent/unicode.py
ACUTE, GRAVE = "́", "̀"


def strip_marks(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if c not in (ACUTE, GRAVE)))


def stress_ordinal(form: str) -> int | None:
    """0-based index of the accented vowel, or None if unmarked."""
    nfd = unicodedata.normalize("NFD", form).lower()
    vi = -1
    for ch in nfd:
        if ch in VOWELS:
            vi += 1
        elif ch in (ACUTE, GRAVE):
            return vi
    return None


def main(dump: str, out: str) -> None:
    # surface(lower, stripped) -> {ordinal -> [(lemma, pos, gloss)]}
    table: dict[str, dict[int, list[tuple[str, str, str]]]] = defaultdict(lambda: defaultdict(list))

    with open(dump, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            lemma = d.get("word")
            if not lemma:
                continue
            pos = d.get("pos", "?")
            gloss = ""
            for s in d.get("senses", []):
                gl = s.get("glosses")
                if gl:
                    gloss = gl[0]
                    break
            for fm in d.get("forms", []):
                form = fm.get("form")
                if not form or form in ("-", "no-table-tags") or " " in form:
                    continue
                if "table-tags" in fm.get("tags", []) or "inflection-template" in fm.get(
                    "tags", []
                ):
                    continue
                ordn = stress_ordinal(form)
                if ordn is None:
                    continue
                key = strip_marks(form).lower()
                if not key or not any(c in VOWELS for c in key):
                    continue
                table[key][ordn].append((lemma, pos, gloss))

    homographs = {k: v for k, v in table.items() if len(v) >= 2}

    with open(out, "w", encoding="utf-8") as fh:
        for surface in sorted(homographs):
            variants = []
            for ordn, refs in sorted(homographs[surface].items()):
                # dedupe lemma/pos/gloss refs
                seen = []
                for r in refs:
                    if r not in seen:
                        seen.append(r)
                variants.append(
                    {
                        "ordinal": ordn,
                        "lemmas": sorted({r[0] for r in seen}),
                        "pos": sorted({r[1] for r in seen}),
                        "glosses": [r[2] for r in seen if r[2]][:3],
                    }
                )
            fh.write(
                json.dumps({"surface": surface, "variants": variants}, ensure_ascii=False) + "\n"
            )

    print(f"form-level homograph surfaces: {len(homographs)}")
    # quick distribution
    by_n = defaultdict(int)
    for v in homographs.values():
        by_n[len(v)] += 1
    print("по брой различни позиции:", dict(sorted(by_n.items())))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
