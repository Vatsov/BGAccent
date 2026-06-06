"""Aggregate workflow labels into a dataset and evaluate quality.

Reads:
  data/homographs/tasks/*.json      (occurrences + options)
  data/homographs/labels/*.jsonl    (primary Sonnet labels)
  data/homographs/labels_agreement/*.jsonl  (Haiku agreement sample)
  data/homographs/gold.jsonl        (optional gold standard: {id, ordinal, escape})

Writes:
  data/homographs/dataset_raw.jsonl     (final token-classification records)
Prints:
  diagnostics, cross-model agreement, and (if gold present) accuracy.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "data/homographs"


def load_jsonl_dir(d: Path) -> dict[str, dict]:
    """id -> label record, tolerant of malformed lines."""
    out: dict[str, dict] = {}
    bad = 0
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["id"]] = rec
            except Exception:
                bad += 1
    return out, bad


def main() -> None:
    tasks = {}
    occ_meta = {}
    for f in sorted((P2 / "tasks").glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        tasks[t["idx"]] = t
        opt_pos = {o["ordinal"]: o["pos"] for o in t["options"]}
        same_pos = len({o["pos"] for o in t["options"]}) == 1
        for o in t["occurrences"]:
            occ_meta[o["id"]] = {
                "surface": t["surface"],
                "marked": o["marked"],
                "opt_pos": opt_pos,
                "same_pos": same_pos,
            }

    labels, bad_a = load_jsonl_dir(P2 / "labels")
    labels_agreement, bad_b = load_jsonl_dir(P2 / "labels_agreement")

    total = len(labels)
    expected = len(occ_meta)
    conf = Counter(label.get("confidence", "?") for label in labels.values())
    escapes = [i for i, label in labels.items() if label.get("escape")]
    esc_by_surf = Counter(occ_meta[i]["surface"] for i in escapes if i in occ_meta)

    # write dataset (non-escape, joined)
    with open(P2 / "dataset_raw.jsonl", "w", encoding="utf-8") as fh:
        for i, label in labels.items():
            if label.get("escape") or i not in occ_meta:
                continue
            m = occ_meta[i]
            fh.write(
                json.dumps(
                    {
                        "id": i,
                        "surface": m["surface"],
                        "marked": m["marked"],
                        "ordinal": label.get("ordinal"),
                        "confidence": label.get("confidence"),
                        "pos": m["opt_pos"].get(label.get("ordinal")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=== ПОКРИТИЕ ===")
    print(f"очаквани появи:        {expected}")
    print(f"етикетирани (labels):  {total}  (липсват {expected - total})")
    print(f"невалидни редове A/B:   {bad_a}/{bad_b}")
    print(f"\n=== CONFIDENCE ===\n{dict(conf)}")
    print("\n=== ESCAPE (kaikki пропуски -> 3b) ===")
    print(f"общо escape: {len(escapes)} ({100 * len(escapes) / max(total, 1):.1f}%)")
    print("топ хомографи по escape:", dict(esc_by_surf.most_common(15)))

    # cross-model agreement on overlap
    overlap = [i for i in labels_agreement if i in labels]
    if overlap:
        agree = sum(
            1
            for i in overlap
            if labels[i].get("ordinal") == labels_agreement[i].get("ordinal")
            and bool(labels[i].get("escape")) == bool(labels_agreement[i].get("escape"))
        )
        print(f"\n=== AGREEMENT (Sonnet vs Haiku, n={len(overlap)}) ===")
        print(f"съгласие: {100 * agree / len(overlap):.1f}%")
        # disagreements queue
        disq = [
            i for i in overlap if labels[i].get("ordinal") != labels_agreement[i].get("ordinal")
        ]
        (P2 / "review_disagreements.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": i,
                        "marked": occ_meta[i]["marked"],
                        "sonnet": labels[i].get("ordinal"),
                        "haiku": labels_agreement[i].get("ordinal"),
                    },
                    ensure_ascii=False,
                )
                for i in disq
            ),
            encoding="utf-8",
        )
        print(f"несъгласия -> review_disagreements.jsonl ({len(disq)})")

    # gold accuracy
    gold_f = P2 / "gold.jsonl"
    if gold_f.exists():
        gold = {}
        for line in gold_f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                gold[g["id"]] = g
        ev = [i for i in gold if i in labels]
        if ev:
            correct = 0
            by_conf = defaultdict(lambda: [0, 0])
            by_type = defaultdict(lambda: [0, 0])
            for i in ev:
                ok = labels[i].get("ordinal") == gold[i].get("ordinal") and bool(
                    labels[i].get("escape")
                ) == bool(gold[i].get("escape"))
                correct += ok
                c = labels[i].get("confidence", "?")
                by_conf[c][0] += ok
                by_conf[c][1] += 1
                tp = "same-POS" if occ_meta[i]["same_pos"] else "diff-POS"
                by_type[tp][0] += ok
                by_type[tp][1] += 1
            print(f"\n=== GOLD ACCURACY (n={len(ev)}) ===")
            print(f"обща точност: {100 * correct / len(ev):.1f}%")
            print(
                "по confidence:",
                {k: f"{100 * v[0] / v[1]:.0f}% ({v[1]})" for k, v in by_conf.items()},
            )
            print("по тип:", {k: f"{100 * v[0] / v[1]:.0f}% ({v[1]})" for k, v in by_type.items()})
    else:
        print("\n(няма gold.jsonl — пропускам accuracy)")


if __name__ == "__main__":
    main()
