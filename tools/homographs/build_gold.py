"""Sample a stratified gold evaluation set (~200 occurrences) from task files."""

from __future__ import annotations

import json
import random
from pathlib import Path

P2 = Path(__file__).resolve().parents[2] / "data" / "homographs"
random.seed(42)
N = 200

tasks = []
for f in sorted((P2 / "tasks").glob("*.json")):
    tasks.append(json.loads(f.read_text(encoding="utf-8")))

# same-POS homographs are hardest -> oversample
same_pos = [t for t in tasks if len({o["pos"] for o in t["options"]}) == 1]
diff_pos = [t for t in tasks if len({o["pos"] for o in t["options"]}) > 1]
random.shuffle(same_pos)
random.shuffle(diff_pos)

picked = []
# 1 occurrence from each same-POS (hard) first
for t in same_pos:
    o = random.choice(t["occurrences"])
    picked.append((t, o))
# then round-robin from diff-POS
for t in diff_pos:
    o = random.choice(t["occurrences"])
    picked.append((t, o))
    if len(picked) >= N:
        break

picked = picked[:N]
with open(P2 / "gold_unlabeled.jsonl", "w", encoding="utf-8") as fh:
    for t, o in picked:
        fh.write(
            json.dumps(
                {
                    "id": o["id"],
                    "surface": t["surface"],
                    "marked": o["marked"],
                    "options": [
                        {
                            "ordinal": x["ordinal"],
                            "pos": x["pos"],
                            "lemma": x["lemma"],
                            "gloss": x["gloss"],
                        }
                        for x in t["options"]
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )

print(f"gold извадка: {len(picked)}  (same-POS трудни: {min(len(same_pos), len(picked))})")
print(f"-> {P2 / 'gold_unlabeled.jsonl'}")
