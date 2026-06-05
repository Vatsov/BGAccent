"""Finalize synthetic minority-homograph augmentation.

Reads the workflow output augment_raw.json ({"variants": [{surface, ord, need, sentences}]})
plus the existing dataset, applies a strict deterministic gate to every generated
sentence, and writes data/homographs/dataset_synthetic.jsonl in the same record schema.

Gate (a sentence is kept only if ALL hold):
  1. contains the exact «{surface}» substring exactly once (surface wrapped in guillemets);
  2. no stress marks at all — no combining acute U+0301, no precomposed accented letters
     (à/á/â/…, ѝ U+045D, и́, etc.) anywhere;
  3. NFC-stable and non-empty;
  4. not a duplicate of an existing dataset sentence or an earlier-kept synthetic one
     (compared on guillemet-stripped, whitespace-normalised text).
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "homographs"
RAW = DATA / "augment_raw.json"
EXISTING = DATA / "dataset.jsonl"
OUT = DATA / "dataset_synthetic.jsonl"

COMBINING_ACUTE = "́"  # the engine's stress mark; the ONLY thing banned in the text
GUILLEMET = re.compile(r"«([^»]*)»")


def norm(s: str) -> str:
    # NFC, and fold the grave-accented pronoun ѝ/ѐ to и/е to match the corpus,
    # which uses the undecorated forms (0 occurrences of ѝ in the dataset).
    s = unicodedata.normalize("NFC", s)
    return s.replace("ѝ", "и").replace("Ѝ", "И").replace("ѐ", "е").replace("Ѐ", "Е")


def has_stress_mark(s: str) -> bool:
    # A stress mark is a combining acute (incl. any precomposed acute letter, which
    # decomposes to base+U+0301). The grave on ѝ/ѐ is U+0300 and is NOT flagged.
    return COMBINING_ACUTE in unicodedata.normalize("NFD", s)


def marker_ok(s: str, surface: str) -> bool:
    # Exactly one guillemet span, and it is the surface (case-insensitive — the
    # original-cased output may be sentence-initial, as in 421 existing records).
    spans = GUILLEMET.findall(s)
    return len(spans) == 1 and spans[0].lower() == surface.lower()


def key(text: str) -> str:
    return " ".join(text.replace("«", "").replace("»", "").split()).lower()


def main() -> int:
    if not RAW.exists():
        print(f"missing {RAW}", file=sys.stderr)
        return 1

    variants = json.loads(RAW.read_text())["variants"]
    seen = set()
    for line in EXISTING.read_text().splitlines():
        if line.strip():
            seen.add(key(json.loads(line)["marked"]))

    records = []
    per_variant = {}
    dropped = {"no_marker": 0, "bad_marker_count": 0, "stress_mark": 0, "dup": 0, "empty": 0}

    for v in variants:
        surface = v["surface"]
        kept = 0
        for raw in v["sentences"]:
            s = norm(raw).strip()
            if not s:
                dropped["empty"] += 1
                continue
            if not marker_ok(s, surface):
                dropped["no_marker"] += 1
                continue
            if has_stress_mark(s):
                dropped["stress_mark"] += 1
                continue
            k = key(s)
            if k in seen:
                dropped["dup"] += 1
                continue
            seen.add(k)
            records.append(
                {
                    "id": f"{surface}-synth-{kept}",
                    "surface": surface,
                    "marked": s,
                    "ordinal": v["ord"],
                    "pos": None,  # filled below from existing sense metadata
                    "lemma": None,
                    "confidence": "synthetic",
                    "synthetic": True,
                }
            )
            kept += 1
        per_variant[surface] = (kept, v["need"])

    # Backfill pos/lemma from the existing dataset's records for the same (surface, ordinal).
    sense_meta = {}
    for line in EXISTING.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sense_meta.setdefault((r["surface"], r["ordinal"]), (r["pos"], r["lemma"]))
    for rec in records:
        pos, lemma = sense_meta.get((rec["surface"], rec["ordinal"]), (None, None))
        rec["pos"], rec["lemma"] = pos, lemma

    with OUT.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    short = {s: (k, n) for s, (k, n) in per_variant.items() if k < n}
    print(f"variants: {len(per_variant)}  kept sentences: {len(records)}")
    print(f"dropped: {dropped}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    if short:
        print(f"\n{len(short)} variants below floor 20 after gating:")
        for s, (k, n) in sorted(short.items(), key=lambda x: x[1][0]):
            print(f"  {s:14s} kept {k:2d}/{n} synthetic (existing+{k})")
    else:
        print("\nall variants reached their top-up target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
