"""Scan the corpus for homograph surface forms and rank by frequency."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[А-Яа-яЁёЪъЬь]+", re.UNICODE)


def load_surfaces(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["surface"]] = d
    return out


def main(homographs: str, corpus_dir: str) -> None:
    surfaces = load_surfaces(homographs)
    freq: Counter[str] = Counter()
    total_tokens = 0
    for book in sorted(Path(corpus_dir).glob("*.txt")):
        text = unicodedata.normalize("NFC", book.read_text(encoding="utf-8", errors="replace"))
        for m in WORD_RE.finditer(text.lower()):
            total_tokens += 1
            w = m.group()
            if w in surfaces:
                freq[w] += 1

    hit = len(freq)
    print(f"корпусни токени: {total_tokens:,}")
    print(f"уникални хомограф-форми, срещнати в корпуса: {hit} / {len(surfaces)}")
    print(f"общо хомограф-появи: {sum(freq.values()):,}\n")
    print("ТОП 40 по честота (форма  поява  | варианти):")
    for w, c in freq.most_common(40):
        var = surfaces[w]["variants"]
        parts = []
        for v in var:
            gloss = v["glosses"][0][:30] if v["glosses"] else "?"
            parts.append(f"ord{v['ordinal']} {'/'.join(v['pos'])} [{gloss}]")
        desc = "  ::  ".join(parts)
        print(f"  {w:14} {c:5}  | {desc}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
