from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgaccent.accentor import Accentor, AccentResult

_lock = threading.Lock()
_instance: Accentor | None = None


def _get_accentor(trie_path: Path | None = None) -> Accentor:
    global _instance
    if _instance is not None and trie_path is None:
        return _instance
    with _lock:
        if _instance is not None and trie_path is None:
            return _instance
        from bgaccent.accentor import Accentor as _Accentor

        if trie_path is not None:
            return _Accentor(trie_path=trie_path)

        from bgaccent.data import get_trie_path

        resolved = get_trie_path()
        _instance = _Accentor(trie_path=resolved)
        return _instance


def accent(text: str, trie_path: Path | None = None) -> str:
    return _get_accentor(trie_path).accent(text)


def accent_with_report(text: str, trie_path: Path | None = None) -> AccentResult:
    return _get_accentor(trie_path).accent_with_report(text)
