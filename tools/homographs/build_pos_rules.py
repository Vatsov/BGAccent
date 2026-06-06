"""Mine (form, POS) -> stress-ordinal disambiguation rules from the dataset.

The per-surface majority dict tops out at ~0.82 on the by-record split because
~98 homograph surfaces flip stress with part of speech (NOUN ``за́мък`` vs VERB
``замъ́к``). Conditioning on POS lifts the oracle ceiling to ~0.97. This script
mines a ``(form, UPOS) -> ordinal`` table from the labeled corpus — the
data-derived successor to the hand-written ``HOMOGRAPH_RULES`` in
``src/bgaccent/disambiguator.py`` — and reports the held-out gain so the win is
auditable.

Usage:
    uv run python tools/homographs/build_pos_rules.py \
        --data data/homographs/dataset.jsonl \
        --data data/homographs/dataset_synthetic.jsonl \
        --out data/homographs/pos_rules.json

The output is derived from kaikki (CC-BY-SA) + the internal corpus; treat its
licensing like the dictionary data (NOT MIT core). It is written under
``data/homographs/`` (gitignored) — bundling/licensing into ``src/`` is a
separate decision.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Dataset POS scheme -> spaCy UPOS (so the table keys match tok.pos_ at runtime).
# Compound labels (adj/noun, noun/verb, ...) mark occurrences the labelers found
# POS-ambiguous; they cannot be keyed to a single clean POS and are skipped.
UPOS = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ", "name": "PROPN"}


def load(paths: list[Path]) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pos = UPOS.get((rec.get("pos") or "").strip())
                if pos is None:
                    continue
                rows.append((rec["surface"].lower(), pos, int(rec["ordinal"])))
    return rows


def build_rules(rows: list[tuple[str, str, int]]) -> dict[tuple[str, str], tuple[int, int]]:
    """``(form, POS) -> (majority_ordinal, support)`` for POS-dependent forms only.

    A form is kept only when its stress actually differs across POS in the data;
    otherwise the trie's default stress already covers it and a rule is redundant.
    """
    per_key: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    per_form: dict[str, set[int]] = defaultdict(set)
    for form, pos, ordinal in rows:
        per_key[(form, pos)][ordinal] += 1
        per_form[form].add(ordinal)
    rules: dict[tuple[str, str], tuple[int, int]] = {}
    for (form, pos), counts in per_key.items():
        if len(per_form[form]) < 2:
            continue  # single stress overall — POS adds nothing
        ordinal, support = counts.most_common(1)[0]
        rules[(form, pos)] = (ordinal, support)
    return rules


def evaluate(rows: list[tuple[str, str, int]], seed: int) -> None:
    """Report surface-only vs (form, POS)-oracle accuracy on a by-record split."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * 0.1)
    test = shuffled[:n_test]
    train = shuffled[int(n * 0.2) :]

    surf: dict[str, Counter[int]] = defaultdict(Counter)
    for form, _pos, ordinal in train:
        surf[form][ordinal] += 1
    surf_pred = {f: c.most_common(1)[0][0] for f, c in surf.items()}
    glob = Counter(o for _, _, o in train).most_common(1)[0][0]
    rules = build_rules(train)

    surf_ok = pos_ok = 0
    for form, pos, ordinal in test:
        base = surf_pred.get(form, glob)
        surf_ok += base == ordinal
        rule = rules.get((form, pos))
        pos_ok += (rule[0] if rule is not None else base) == ordinal
    n_test = len(test)
    print(
        f"held-out (by-record, seed {seed}): surface-only={surf_ok / n_test:.4f}  "
        f"(form,POS)-oracle={pos_ok / n_test:.4f}  gain={(pos_ok - surf_ok) / n_test:+.4f}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append")
    parser.add_argument("--out", type=Path, default=Path("data/homographs/pos_rules.json"))
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_paths: list[Path] = args.data or [
        Path("data/homographs/dataset.jsonl"),
        Path("data/homographs/dataset_synthetic.jsonl"),
    ]
    rows = load(data_paths)
    evaluate(rows, args.seed)

    rules = build_rules(rows)
    kept = {k: v for k, v in rules.items() if v[1] >= args.min_support}
    forms = {form for form, _ in kept}
    payload = sorted(
        ({"form": f, "pos": p, "ordinal": o, "support": s} for (f, p), (o, s) in kept.items()),
        key=lambda r: (r["form"], r["pos"]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {len(payload)} (form,POS) rules over {len(forms)} "
        f"POS-dependent forms -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
