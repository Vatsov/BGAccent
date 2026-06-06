"""Extract real corpus sentences containing given homograph surface forms."""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

WORD_RE = re.compile(r"[А-Яа-яЁёЪъЬь]+")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def main(corpus_dir: str, targets: list[str], per: int = 7) -> None:
    want = set(targets)
    found: dict[str, list[str]] = {t: [] for t in targets}
    for book in sorted(Path(corpus_dir).glob("*.txt")):
        text = unicodedata.normalize("NFC", book.read_text(encoding="utf-8", errors="replace"))
        text = re.sub(r"\s+", " ", text)
        for sent in SENT_SPLIT.split(text):
            low = sent.lower()
            toks = set(WORD_RE.findall(low))
            for t in want:
                clean = sent.strip()
                if (
                    t in toks
                    and len(found[t]) < per
                    and 6 <= len(sent.split()) <= 35
                    and clean not in found[t]
                ):
                    found[t].append(clean)
        if all(len(found[t]) >= per for t in targets):
            break
    for t in targets:
        print(f"\n===== {t}  ({len(found[t])} изр.) =====")
        for i, s in enumerate(found[t], 1):
            # highlight the target token
            hl = re.sub(rf"\b({t})\b", r">>>\1<<<", s, flags=re.IGNORECASE)
            print(f"  [{i}] {hl}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["говори", "работи", "боя"])
