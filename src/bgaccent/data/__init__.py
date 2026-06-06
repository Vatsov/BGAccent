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
            "Reinstall with: pip install --upgrade --force-reinstall bgaccent"
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


def get_pos_rules_path() -> Path | None:
    """Locate the optional homograph (form, POS) rule table.

    Returns the path shipped by the separately-licensed ``bgaccent-data-homographs``
    package when it is installed and its data is present, else ``None`` so callers
    fall back to the built-in ``HOMOGRAPH_RULES``. Mirrors :func:`get_trie_path`'s
    lazy lookup of the optional dictionary package.
    """
    try:
        from bgaccent_data_homographs import get_pos_rules_path as ext_path
    except ImportError:
        return None
    path = ext_path()
    return path if path.exists() else None
