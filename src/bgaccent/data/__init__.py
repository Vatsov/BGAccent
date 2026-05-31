from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_TRIE_FORMAT_VERSION = 1

_DATA_DIR = Path(__file__).parent
_BUNDLED_TRIE = _DATA_DIR / "bg.marisa"
_META_PATH = _DATA_DIR / "bg.marisa.meta.json"


def _validate_meta(meta_path: Path) -> None:
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    version = meta.get("format_version")
    if version is not None and version != SUPPORTED_TRIE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported BGAccent trie format version {version}. "
            "Rebuild with: bgaccent-build, or reinstall: pip install bgaccent"
        )


def get_trie_path() -> Path:
    try:
        from bgaccent_data_full_gpl import get_trie_path as gpl_path

        return gpl_path()  # type: ignore[no-any-return]
    except ImportError:
        pass

    if _BUNDLED_TRIE.exists():
        _validate_meta(_META_PATH)
        return _BUNDLED_TRIE

    raise FileNotFoundError(
        f"Trie file not found at {_BUNDLED_TRIE}. Reinstall with: pip install bgaccent"
    )
