# BGAccent

Bulgarian text accent placement engine for pre-TTS quality assurance.

Normalizes, analyzes, and marks lexical stress in Bulgarian text via dictionary lookup and rule-based fallbacks, with OOV reporting and custom dictionary support.

## Installation

```bash
pip install bgaccent
```

## Usage

```bash
bgaccent process input.txt -o output.txt
```

## Features

- Automatic accent placement via trie-based dictionary lookup
- Out-of-vocabulary (OOV) word reporting
- Custom dictionary support for names, places, and domain-specific terms
- Batch file processing for audiobook chapters
- Rich terminal output with progress tracking

## License

MIT — see [LICENSE](LICENSE) for details.

Dictionary data packages carry their own licenses (CC-BY-SA, GFDL, etc.) and are distributed separately.
