"""Resolver for the bundled Bulgarian homograph stress-rule table.

This package ships only data (CC-BY-SA-4.0). The MIT ``bgaccent`` core finds it
lazily — mirroring how ``bgaccent.data.get_trie_path`` finds the optional
``bgaccent_data_full_gpl`` dictionary — so the rules are used automatically when
this package is installed and ignored otherwise.
"""

from __future__ import annotations

from pathlib import Path

_DATA = Path(__file__).parent / "pos_rules.json"


def get_pos_rules_path() -> Path:
    """Path to ``pos_rules.json`` (may not exist until the data is generated)."""
    return _DATA
