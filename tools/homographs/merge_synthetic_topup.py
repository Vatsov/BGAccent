"""Append gated top-up sentences into dataset_synthetic.jsonl up to each variant's floor.

Reads a top-up raw file (default data/homographs/augment_topup_raw.json, same shape as
augment_raw.json) and the worklist (augment_worklist.json) to know each surface's target
synthetic count (`need`). Appends only the deficit, applying the identical gate as
finalize_synthetic.py and continuing the per-surface id index.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finalize_synthetic import EXISTING, OUT, has_stress_mark, key, marker_ok, norm

DATA = OUT.parent
TOPUP = DATA / "augment_topup_raw.json"
WORKLIST = DATA / "augment_worklist.json"


def main() -> int:
    if not TOPUP.exists():
        print(f"missing {TOPUP}", file=sys.stderr)
        return 1

    need = {w["surface"]: w["need"] for w in json.loads(WORKLIST.read_text())}
    synth = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]

    counts: dict[str, int] = {}
    next_idx: dict[str, int] = {}
    for r in synth:
        counts[r["surface"]] = counts.get(r["surface"], 0) + 1
        n = int(r["id"].rsplit("-", 1)[1])
        next_idx[r["surface"]] = max(next_idx.get(r["surface"], -1), n)

    seen = {key(r["marked"]) for r in synth}
    for line in EXISTING.read_text().splitlines():
        if line.strip():
            seen.add(key(json.loads(line)["marked"]))

    sense_meta = {}
    for line in EXISTING.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            sense_meta.setdefault((r["surface"], r["ordinal"]), (r["pos"], r["lemma"]))

    added = {}
    for v in json.loads(TOPUP.read_text())["variants"]:
        surface = v["surface"]
        deficit = need.get(surface, 0) - counts.get(surface, 0)
        if deficit <= 0:
            continue
        pos, lemma = sense_meta.get((surface, v["ord"]), (None, None))
        for raw in v["sentences"]:
            if deficit <= 0:
                break
            s = norm(raw).strip()
            if not s or not marker_ok(s, surface) or has_stress_mark(s):
                continue
            k = key(s)
            if k in seen:
                continue
            seen.add(k)
            next_idx[surface] = next_idx.get(surface, -1) + 1
            synth.append(
                {
                    "id": f"{surface}-synth-{next_idx[surface]}",
                    "surface": surface,
                    "marked": s,
                    "ordinal": v["ord"],
                    "pos": pos,
                    "lemma": lemma,
                    "confidence": "synthetic",
                    "synthetic": True,
                }
            )
            deficit -= 1
            added[surface] = added.get(surface, 0) + 1

    with OUT.open("w") as f:
        for rec in synth:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"appended: {added or 'nothing'}")
    print(f"total synthetic records: {len(synth)}")
    for surface in need:
        c = sum(1 for r in synth if r["surface"] == surface)
        if c < need[surface]:
            print(f"  still short: {surface} {c}/{need[surface]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
