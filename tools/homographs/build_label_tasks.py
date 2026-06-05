"""Build per-homograph labeling task files for the workflow agents.
Each task file: {idx, surface, options:[{ordinal,pos,lemma,gloss}], occurrences:[{id, marked}]}
The 'marked' sentence wraps the specific target occurrence in « ».
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

CAP = 60
WORD_RE = re.compile(r"[А-Яа-яЁёЪъЬь]+")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "data/homographs/tasks"
LABELS = ROOT / "data/homographs/labels"


def main():
    TASKS.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    for old in TASKS.glob("*.json"):
        old.unlink()

    targets = []
    with open(ROOT / "data/homographs/targets.jsonl", encoding="utf-8") as fh:
        for line in fh:
            targets.append(json.loads(line))
    want = {t["surface"]: t for t in targets}
    collected = {s: [] for s in want}

    # stream corpus sentences
    for book in sorted((ROOT / "data/corpus").glob("*.txt")):
        text = unicodedata.normalize("NFC", book.read_text(encoding="utf-8", errors="replace"))
        text = re.sub(r"\s+", " ", text)
        for sent in SENT_SPLIT.split(text):
            n = len(sent.split())
            if not (5 <= n <= 40):
                continue
            toks = WORD_RE.findall(sent.lower())
            present = want.keys() & set(toks)
            for surf in present:
                if len(collected[surf]) >= CAP:
                    continue
                # mark first unmarked occurrence of surf in this sentence
                marked = re.sub(rf"\b({surf})\b", r"«\1»", sent, count=1, flags=re.IGNORECASE)
                if marked not in [o["marked"] for o in collected[surf]]:
                    collected[surf].append(
                        {"id": f"{surf}-{len(collected[surf])}", "marked": marked.strip()}
                    )

    manifest = []
    for idx, t in enumerate(targets):
        surf = t["surface"]
        occ = collected[surf]
        if not occ:
            continue
        options = []
        for v in t["variants"]:
            options.append(
                {
                    "ordinal": v["ordinal"],
                    "pos": "/".join(v["pos"]),
                    "lemma": "/".join(v["lemmas"][:2]),
                    "gloss": (v["glosses"][0] if v["glosses"] else ""),
                }
            )
        task = {"idx": idx, "surface": surf, "options": options, "occurrences": occ}
        (TASKS / f"{idx:04d}.json").write_text(
            json.dumps(task, ensure_ascii=False), encoding="utf-8"
        )
        manifest.append({"idx": idx, "surface": surf, "n": len(occ)})

    (ROOT / "data/homographs/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    print(f"task файлове: {len(manifest)}")
    print(f"общо появи за етикетиране: {sum(m['n'] for m in manifest):,}")
    print("примерен манифест:", manifest[:3])


if __name__ == "__main__":
    main()
