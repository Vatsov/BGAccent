from __future__ import annotations

from pathlib import Path

_DATA_DIR = Path(__file__).parent
_BUNDLED_TRIE = _DATA_DIR / "bg.marisa"


def get_trie_path() -> Path:
    if _BUNDLED_TRIE.exists():
        return _BUNDLED_TRIE
    raise FileNotFoundError(
        f"Trie file not found at {_BUNDLED_TRIE}. "
        "Reinstall with: pip install bgaccent"
    )
