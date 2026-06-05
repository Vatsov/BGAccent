"""3b refined genuine-homograph filter (POS-aware) + gold re-evaluation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

P2 = Path("data/homographs")

FORM_OF = re.compile(
    r"(inflection of|alternative form of|alternative spelling of|plural of|singular of|"
    r"participle of|imperative of|indicative of|aorist|imperfect|diminutive of|vocative|"
    r"comparative of|superlative of|\bform of|feminine\b.*\bof|masculine\b.*\bof|"
    r"neuter\b.*\bof|definite\b.*\bof|indefinite\b.*\bof)",
    re.I,
)

STOP = set(
    [
        "a",
        "an",
        "the",
        "to",
        "of",
        "or",
        "and",
        "for",
        "in",
        "on",
        "with",
        "something",
        "someone",
        "esp",
        "etc",
        "that",
        "which",
    ]
)


def toks(gloss):
    g = re.sub(r"\(.*?\)", " ", gloss.lower())
    g = re.sub(r"[^a-zа-я ]", " ", g)
    return frozenset(w for w in g.split() if w not in STOP and len(w) > 2)


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def count_concepts(variants):
    """Distinct (POS, meaning) concepts. Different POS => different concepts.
    Within a POS: cluster non-form-of glosses by synonymy; if only form-of forms
    exist for that POS, it still represents 1 word."""
    by_pos = defaultdict(list)
    for v in variants:
        by_pos["/".join(v["pos"])].append(v["glosses"][0] if v.get("glosses") else "")
    total = 0
    for glosses in by_pos.values():
        non_form = [g for g in glosses if g and not FORM_OF.search(g)]
        if not non_form:
            total += 1
            continue
        clusters = []
        for g in non_form:
            t = toks(g)
            if t and not any(jac(t, c) >= 0.5 for c in clusters):
                clusters.append(t)
        total += max(1, len(clusters))
    return total


def is_genuine(variants):
    return count_concepts(variants) >= 2


def main():
    seen = {}
    with open(P2 / "targets.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            seen[d["surface"]] = d["variants"]

    genuine = {s for s, v in seen.items() if is_genuine(v)}
    dropped = [s for s in seen if s not in genuine]
    print(f"corpus-seen: {len(seen)}   genuine: {len(genuine)}   отпаднали: {len(dropped)}\n")

    # spot-check that previously-flagged genuine homographs survive
    must_keep = [
        "отговори",
        "работи",
        "опита",
        "заема",
        "боя",
        "коса",
        "удари",
        "разказа",
        "говори",
        "постави",
    ]
    print(
        "проверка ключови истински:",
        {s: ("✓" if s in genuine else "✗ ДРОПНАТ") for s in must_keep if s in seen},
    )

    print("\n=== ОТПАДНАЛИ (псевдо) — извадка ===")
    for s in dropped[:22]:
        gl = [(v["glosses"][0][:30] if v.get("glosses") else "∅") for v in seen[s]]
        pos = ["/".join(v["pos"]) for v in seen[s]]
        print(f"  {s:13} :: " + "  |  ".join(f"{p}:{g}" for p, g in zip(pos, gl, strict=False)))

    (P2 / "genuine.json").write_text(
        json.dumps(sorted(genuine), ensure_ascii=False), encoding="utf-8"
    )

    # ---- gold re-evaluation on v2 subset ----
    labels = {}
    for f in (P2 / "labels").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    labels[r["id"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    gold = {}
    for line in (P2 / "gold.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            g = json.loads(line)
            gold[g["id"]] = g

    def surf_of(i):
        return i.rsplit("-", 1)[0]

    v1 = [i for i in gold if i in labels]
    v2 = [i for i in v1 if surf_of(i) in genuine]

    def acc(ids):
        ok = sum(
            1
            for i in ids
            if labels[i].get("ordinal") == gold[i].get("ordinal")
            and bool(labels[i].get("escape")) == bool(gold[i].get("escape"))
        )
        return ok, len(ids)

    o1, n1 = acc(v1)
    o2, n2 = acc(v2)
    print("\n=== GOLD ACCURACY ===")
    print(f"всички genuine-v1: {100 * o1 / n1:.1f}%  (n={n1})")
    print(
        f"само genuine-v2:   {100 * o2 / n2:.1f}%  (n={n2})  [{n1 - n2} псевдо изключени от gold]"
    )


if __name__ == "__main__":
    main()
