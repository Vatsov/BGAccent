"""3a: freeze the genuine-homograph target list with corpus frequencies."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[А-Яа-яЁёЪъЬь]+")


def is_genuine(variants):
    lemma_sets = [set(v["lemmas"]) for v in variants]
    alll = set().union(*lemma_sets)
    if len(alll) < 2:
        return False
    return not set.intersection(*lemma_sets)


def main(homographs, corpus_dir, out):
    surf = {}
    with open(homographs, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            surf[d["surface"]] = d["variants"]
    freq = Counter()
    for book in sorted(Path(corpus_dir).glob("*.txt")):
        text = unicodedata.normalize(
            "NFC", book.read_text(encoding="utf-8", errors="replace")
        ).lower()
        for m in WORD_RE.finditer(text):
            if m.group() in surf:
                freq[m.group()] += 1
    genuine = {w: v for w, v in surf.items() if is_genuine(v)}
    targets = [(w, c) for w, c in freq.most_common() if w in genuine]
    with open(out, "w", encoding="utf-8") as fh:
        for w, c in targets:
            fh.write(
                json.dumps(
                    {"surface": w, "corpus_freq": c, "variants": genuine[w]}, ensure_ascii=False
                )
                + "\n"
            )
    # coverage buckets
    buckets = {"≥200": 0, "100-199": 0, "30-99": 0, "10-29": 0, "<10": 0}
    for _, c in targets:
        if c >= 200:
            buckets["≥200"] += 1
        elif c >= 100:
            buckets["100-199"] += 1
        elif c >= 30:
            buckets["30-99"] += 1
        elif c >= 10:
            buckets["10-29"] += 1
        else:
            buckets["<10"] += 1
    print(f"genuine хомографи, срещнати в корпуса: {len(targets)}")
    print(f"общо появи: {sum(c for _, c in targets):,}")
    print("покритие по честота:", buckets)
    print(f"\nзаписано -> {out}")


if __name__ == "__main__":
    main("data/homographs/homographs.jsonl", "data/corpus", "data/homographs/targets.jsonl")
