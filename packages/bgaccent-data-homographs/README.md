# bgaccent-data-homographs

Bulgarian homograph **(form, POS) → stress ordinal** rules for
[BGAccent](../../README.md). Conditioning stress on part of speech
(за́мък/NOUN vs замъ́к/VERB) lifts homograph accuracy from ~0.82 (per-surface
majority) to ~0.89 with real Stanza POS tagging.

Distributed separately from the MIT `bgaccent` core because the data is derived
from Bulgarian Wiktionary (CC-BY-SA-4.0). See [NOTICE](NOTICE).

## The data is generated, not committed

`pos_rules.json` is **not** checked into the source tree — it is mined from the
labeled corpus. Generate it before building/publishing:

```bash
# from the BGAccent repo root
uv run python tools/homographs/build_pos_rules.py \
    --out packages/bgaccent-data-homographs/src/bgaccent_data_homographs/pos_rules.json
```

## Usage

Once installed alongside `bgaccent[pos]`, the rules load automatically:

```bash
pip install bgaccent[pos] bgaccent-data-homographs
```

```python
from bgaccent.accentor import Accentor
from bgaccent.data import get_trie_path

# disambiguator_rules left unset → the installed data package is found lazily
acc = Accentor(trie_path=get_trie_path(), disambiguator="stanza")
```

Without this package, BGAccent falls back to its small built-in `HOMOGRAPH_RULES`.
