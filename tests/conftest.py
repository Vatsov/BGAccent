from pathlib import Path

import pytest

TEST_TRIE_PATH = Path(__file__).parent / "fixtures" / "bg_test.marisa"


@pytest.fixture
def test_trie_path() -> Path:
    assert TEST_TRIE_PATH.exists(), f"Test trie not found at {TEST_TRIE_PATH}"
    return TEST_TRIE_PATH
