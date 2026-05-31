# BGAccent

<p align="center">
  <img src="assets/hero.png" alt="BGAccent — Bulgarian text accent placement" width="100%">
</p>

Pre-TTS QA tool for Bulgarian text — automatic accent placement via trie-based dictionary lookup with morphological fallback, homograph detection, and OOV reporting.

Bundled dictionary: **230K words** (bayganyu MIT + Wiktionary CC-BY-SA). Optional GPL expansion via bgospodinov.

## Installation

```bash
pip install bgaccent
```

Optional dependencies:

```bash
pip install bgaccent[nlp]     # spaCy POS-based homograph disambiguation
pip install bgaccent[neural]  # onnxruntime: run a pre-trained ONNX stress model (no torch)
pip install bgaccent[train]   # PyTorch + ONNX to train/export a model
```

## Quick start

```bash
# Accent a file
bgaccent input.txt -o output.txt

# Pipe from stdin
echo "Планината беше покрита със сняг." | bgaccent -

# JSON report with stats, OOV list, homographs
bgaccent input.txt -o output.txt --report report.json

# Dry-run — stats only, no output
bgaccent input.txt --check
```

## CLI reference

```
bgaccent [OPTIONS] [INPUT]
```

| Flag | Description |
|------|-------------|
| `-o`, `--output PATH` | Output file (default: stdout) |
| `--custom PATH` | Custom dictionary TSV (repeatable, later files override earlier) |
| `--mode {preserve,replace-safe}` | `preserve` (default): keep existing accents. `replace-safe`: re-accent if dictionary match exists |
| `--format {text,ssml}` | Output format. `ssml` wraps accented words in `<phoneme>` tags with IPA |
| `--report PATH` | Write JSON report (stats, OOV words, homographs, per-word details) |
| `--check` | Dry-run: print stats, no text output |
| `--diff` | Show per-occurrence changes (`L3:10 много -> мно́го`) |
| `--unknown-only` | Print OOV words only, one per line |
| `--out-dir PATH` | Batch mode: output directory for multiple input files |
| `--trie PATH` | Custom .marisa trie (default: bundled) |
| `--mark-monosyllables` | Place accent on monosyllabic words |
| `--fail-on-oov` | Exit 1 if OOV words found (use with `--check`) |
| `--fail-on-homographs` | Exit 1 if homographs flagged |
| `--max-oov-rate FLOAT` | Exit 1 if OOV rate exceeds threshold |
| `-q`, `--quiet` | Suppress non-error stderr output |
| `--version` | Show version |

## Examples

### Batch processing (audiobook chapters)

```bash
bgaccent chapters/ --out-dir accented/
```

Processes every `*.txt` file in the input directory and writes accented copies
to `--out-dir`.

### Custom dictionary for names and places

Create a TSV file (`custom.tsv`):

```tsv
Пенчо	Пе́нчо
Славейков	Славе́йков
```

Column 1: word. Column 2: accented form with U+0301 combining acute on the
stressed vowel.

```bash
bgaccent input.txt --custom names.tsv --custom places.tsv -o output.txt
```

### SSML output for TTS

```bash
bgaccent input.txt --format ssml -o output.ssml
```

Produces:

```xml
<phoneme alphabet="ipa" ph="planiˈnata">Планината</phoneme> беше покрита
```

### CI quality gate

```bash
bgaccent manuscript.txt --check --fail-on-oov --max-oov-rate 0.05
```

Exits with code 1 if OOV rate exceeds 5%.

## Python API

```python
from bgaccent import accent, accent_with_report

# Plain string — for simple cases
print(accent("Планината беше покрита със сняг."))
# → Планина́та бе́ше покри́та със сняг.

# Full report — stats, OOV list, per-token details
result = accent_with_report("Планината беше покрита със сняг.")
print(result.text)        # Планина́та бе́ше покри́та със сняг.
print(result.stats)       # AccentStats(accented_tokens=3, oov_multisyllabic=0, ...)
print(result.oov_words)   # []
```

## Accent resolution pipeline

Words are resolved in this order — first match wins:

1. **Custom dictionary** — user-provided TSV overrides
2. **Trie lookup** — bundled 230K-word dictionary
3. **Morphological fallback** — suffix stripping to find base forms
4. **Neural prediction** — BiLSTM/ONNX stress predictor (Python API only,
   requires `bgaccent[neural]` to run a pre-trained model and
   `neural_model_path`/`neural_vocab_path` passed to `Accentor`; training a
   model needs `bgaccent[train]`)
5. **N-gram analogy** — suffix-based stress prediction from known patterns
   (Python API only, requires `enable_prediction=True` on `Accentor`)
6. **OOV** — word reported as out-of-vocabulary

Homographs (words with multiple valid stress positions) are flagged in the
report for manual review. POS-based disambiguation (spaCy) is Python-API-only:
it runs only when you pass a loaded spaCy pipeline via
`Accentor(disambiguator_model=...)`. Installing `bgaccent[nlp]` provides the
dependency but does not enable disambiguation on its own, and the CLI never
loads spaCy. A transformer-based disambiguator is scaffolded for Phase 2 and is
not wired into the runtime pipeline.

> **Known limitation:** POS disambiguation resolves a homograph from the first
> occurrence of that word form in the sentence. If the same homograph appears
> more than once in one sentence with different parts of speech, every
> occurrence currently receives the first occurrence's stress. Per-occurrence
> resolution is deferred to Phase 2.

## Building the dictionary trie

The bundled trie ships with the package. To rebuild or expand it:

```bash
# Download sources
python scripts/download_sources.py

# Build MIT-compatible trie (bayganyu + wiktionary)
python scripts/build_trie.py --sources-dir sources --out src/bgaccent/data/bg.marisa --license mit

# Build GPL trie (adds bgospodinov if available)
python scripts/build_trie.py --sources-dir sources --out bg-gpl.marisa --license gpl
```

### Adding bgospodinov (optional, GPL-3.0)

The [bgospodinov/bulgarian_dictionary](https://github.com/bgospodinov/bulgarian_dictionary) doesn't distribute a pre-built database. To include it:

1. Clone the repo and build `dictionary.db` following their README (`./build --no-slovnik`)
2. Copy `dictionary.db` to `sources/bgospodinov.db`
3. Rebuild with `--license gpl`

Note: this changes the trie license to GPL-3.0.

## Development

```bash
make dev          # install deps (uv sync --dev)
make test         # uv run pytest (295 tests)
make lint         # uv run ruff check src/ tests/
make format       # uv run ruff format src/ tests/
make typecheck    # uv run mypy src/
```

## License

MIT — see [LICENSE](LICENSE) for details.

Dictionary sources carry their own licenses:
- **bayganyu**: MIT
- **Wiktionary**: CC-BY-SA-4.0
- **bgospodinov**: GPL-3.0 (optional, not bundled)
