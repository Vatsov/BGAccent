"""Tests for multi-source dictionary merging and license-separated builds."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import marisa_trie

from scripts.build_trie import (
    build_multi_source_trie,
    merge_entries,
    parse_bgospodinov_db,
    parse_wiktionary_jsonl,
)
from scripts.download_sources import should_skip


def _make_bgospodinov_db(path: Path, rows: list[tuple[str, str | None]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE wordform ("
        "  wordform TEXT, wordform_stressed TEXT, tag TEXT, pos TEXT"
        ")"
    )
    for plain, stressed in rows:
        conn.execute(
            "INSERT INTO wordform (wordform, wordform_stressed) VALUES (?, ?)",
            (plain, stressed),
        )
    conn.commit()
    conn.close()


def _make_wiktionary_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Cycle 1: sources.lock and multi-source download ──


class TestMultiSourceLock:
    def test_sources_lock_contains_all_sources(self) -> None:
        lock_path = Path("scripts/sources.lock")
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        for name in ["bayganyu", "bgospodinov", "wiktionary"]:
            assert name in data, f"Missing source: {name}"
            for key in ["url", "sha256", "license"]:
                assert key in data[name], f"Missing '{key}' for '{name}'"


class TestDownloadSkip:
    def test_skip_when_checksum_matches(self, tmp_path: Path) -> None:
        dest = tmp_path / "test.csv"
        dest.write_text("hello", encoding="utf-8")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert should_skip(dest, expected) is True

    def test_no_skip_when_file_missing(self, tmp_path: Path) -> None:
        dest = tmp_path / "missing.csv"
        assert should_skip(dest, "deadbeef") is False

    def test_no_skip_when_checksum_differs(self, tmp_path: Path) -> None:
        dest = tmp_path / "test.csv"
        dest.write_text("hello", encoding="utf-8")
        assert should_skip(dest, "0" * 64) is False


# ── Cycle 2: Multi-format parsing ──


class TestBgospodinovParsing:
    def test_parse_backtick_stress(self, tmp_path: Path) -> None:
        db = tmp_path / "bgospodinov.db"
        # "планина" vowels: а(0), и(1), а(2). Backtick after и → ordinal 1
        _make_bgospodinov_db(db, [("планина", "плани`на")])
        entries = parse_bgospodinov_db(db)
        assert len(entries) == 1
        assert entries[0] == ("планина", 1, 4)

    def test_parse_stress_at_word_end(self, tmp_path: Path) -> None:
        db = tmp_path / "bgospodinov.db"
        # "вода" vowels: о(0), а(1). Backtick after а → ordinal 1
        _make_bgospodinov_db(db, [("вода", "вода`")])
        entries = parse_bgospodinov_db(db)
        assert entries[0] == ("вода", 1, 4)

    def test_source_mask_is_4(self, tmp_path: Path) -> None:
        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("река", "ре`ка")])
        entries = parse_bgospodinov_db(db)
        assert entries[0][2] == 4

    def test_skip_null_stress(self, tmp_path: Path) -> None:
        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("тест", None)])
        entries = parse_bgospodinov_db(db)
        assert len(entries) == 0

    def test_skip_backtick_after_non_vowel(self, tmp_path: Path) -> None:
        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("тест", "тес`т")])
        entries = parse_bgospodinov_db(db)
        assert len(entries) == 0

    def test_nfc_normalization(self, tmp_path: Path) -> None:
        import unicodedata

        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("красива", "краси`ва")])
        entries = parse_bgospodinov_db(db)
        assert unicodedata.is_normalized("NFC", entries[0][0])

    def test_lowercase_key(self, tmp_path: Path) -> None:
        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("Планина", "Плани`на")])
        entries = parse_bgospodinov_db(db)
        assert entries[0][0] == "планина"


class TestWiktionaryParsing:
    def test_parse_u0301_stress(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "wiktionary.jsonl"
        # "библиотека" vowels: и(0),и(1),о(2),е(3),а(4). U+0301 after е → ordinal 3
        _make_wiktionary_jsonl(
            jsonl,
            [
                {
                    "word": "библиотека",
                    "lang_code": "bg",
                    "pos": "noun",
                    "forms": [{"form": "библиоте́ка", "tags": ["singular"]}],
                }
            ],
        )
        entries = parse_wiktionary_jsonl(jsonl)
        found = [e for e in entries if e[0] == "библиотека"]
        assert len(found) == 1
        assert found[0][1] == 3

    def test_source_mask_is_2(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "wiktionary.jsonl"
        _make_wiktionary_jsonl(
            jsonl,
            [{"word": "вода́", "lang_code": "bg", "pos": "noun", "forms": []}],
        )
        entries = parse_wiktionary_jsonl(jsonl)
        assert entries[0][2] == 2

    def test_skip_non_bulgarian(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "wiktionary.jsonl"
        _make_wiktionary_jsonl(
            jsonl,
            [{"word": "hello", "lang_code": "en", "pos": "noun", "forms": []}],
        )
        entries = parse_wiktionary_jsonl(jsonl)
        assert len(entries) == 0

    def test_parses_word_and_forms(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "wiktionary.jsonl"
        _make_wiktionary_jsonl(
            jsonl,
            [
                {
                    "word": "вода́",
                    "lang_code": "bg",
                    "pos": "noun",
                    "forms": [
                        {"form": "вода́та", "tags": ["singular", "definite"]},
                        {"form": "води́", "tags": ["plural"]},
                    ],
                }
            ],
        )
        entries = parse_wiktionary_jsonl(jsonl)
        words = {e[0] for e in entries}
        assert "вода" in words
        assert "водата" in words
        assert "води" in words

    def test_deduplicates(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "wiktionary.jsonl"
        _make_wiktionary_jsonl(
            jsonl,
            [
                {"word": "вода́", "lang_code": "bg", "pos": "noun", "forms": []},
                {"word": "вода́", "lang_code": "bg", "pos": "noun", "forms": []},
            ],
        )
        entries = parse_wiktionary_jsonl(jsonl)
        assert len([e for e in entries if e[0] == "вода"]) == 1


# ── Cycle 3: Cross-source conflict resolution ──


class TestCrossSourceMerge:
    def test_conflict_higher_priority_wins(self) -> None:
        entries_by_source = {
            "bayganyu": [("дума", 0, 1)],
            "bgospodinov": [("дума", 1, 4)],
        }
        result, conflicts = merge_entries(entries_by_source)
        assert result["дума"] == [(1, 4)]
        assert len(conflicts) == 1

    def test_agreement_ors_masks(self) -> None:
        entries_by_source = {
            "bayganyu": [("вода", 1, 1)],
            "bgospodinov": [("вода", 1, 4)],
        }
        result, conflicts = merge_entries(entries_by_source)
        assert result["вода"] == [(1, 5)]
        assert len(conflicts) == 0

    def test_conflict_log_contains_word(self) -> None:
        entries_by_source = {
            "bayganyu": [("дума", 0, 1)],
            "bgospodinov": [("дума", 1, 4)],
        }
        _, conflicts = merge_entries(entries_by_source)
        assert any("дума" in line for line in conflicts)

    def test_unique_source_entries_preserved(self) -> None:
        entries_by_source = {
            "bayganyu": [("планина", 1, 1)],
            "wiktionary": [("река", 0, 2)],
        }
        result, conflicts = merge_entries(entries_by_source)
        assert "планина" in result
        assert "река" in result
        assert len(conflicts) == 0

    def test_three_source_full_agreement(self) -> None:
        entries_by_source = {
            "bayganyu": [("вода", 1, 1), ("планина", 1, 1)],
            "wiktionary": [("вода", 1, 2), ("река", 0, 2)],
            "bgospodinov": [("вода", 1, 4)],
        }
        result, conflicts = merge_entries(entries_by_source)
        assert result["вода"] == [(1, 7)]  # 1|2|4 = 7
        assert result["планина"] == [(1, 1)]
        assert result["река"] == [(0, 2)]
        assert len(conflicts) == 0


# ── Cycle 4: License-separated builds ──


class TestLicenseSeparation:
    def test_mit_excludes_gpl(self, tmp_path: Path) -> None:
        csv = tmp_path / "bayganyu.csv"
        csv.write_text("планината,план'ината\n", encoding="utf-8")
        jsonl = tmp_path / "wiktionary.jsonl"
        _make_wiktionary_jsonl(
            jsonl,
            [{"word": "река́", "lang_code": "bg", "pos": "noun", "forms": []}],
        )
        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("вода", "вода`")])

        out = tmp_path / "mit.marisa"
        build_multi_source_trie(tmp_path, out, license_filter="mit")
        trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        trie.load(str(out))
        assert "планината" in trie
        assert "река" in trie
        assert "вода" not in trie

    def test_gpl_includes_all(self, tmp_path: Path) -> None:
        csv = tmp_path / "bayganyu.csv"
        csv.write_text("планината,план'ината\n", encoding="utf-8")
        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("вода", "вода`")])

        out = tmp_path / "gpl.marisa"
        build_multi_source_trie(tmp_path, out, license_filter="gpl")
        trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        trie.load(str(out))
        assert "планината" in trie
        assert "вода" in trie

    def test_mit_meta_license(self, tmp_path: Path) -> None:
        csv = tmp_path / "bayganyu.csv"
        csv.write_text("планината,план'ината\n", encoding="utf-8")
        jsonl = tmp_path / "wiktionary.jsonl"
        _make_wiktionary_jsonl(
            jsonl,
            [{"word": "река́", "lang_code": "bg", "pos": "noun", "forms": []}],
        )
        out = tmp_path / "mit.marisa"
        build_multi_source_trie(tmp_path, out, license_filter="mit")
        meta_path = tmp_path / "mit.marisa.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "MIT" in meta["source_license"]
        assert "GPL" not in meta["source_license"]

    def test_gpl_meta_license(self, tmp_path: Path) -> None:
        csv = tmp_path / "bayganyu.csv"
        csv.write_text("планината,план'ината\n", encoding="utf-8")
        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("вода", "вода`")])
        out = tmp_path / "gpl.marisa"
        build_multi_source_trie(tmp_path, out, license_filter="gpl")
        meta_path = tmp_path / "gpl.marisa.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "GPL" in meta["source_license"]

    def test_mit_fewer_entries_than_gpl(self, tmp_path: Path) -> None:
        csv = tmp_path / "bayganyu.csv"
        csv.write_text("планината,план'ината\nгорите,гор'ите\n", encoding="utf-8")
        jsonl = tmp_path / "wiktionary.jsonl"
        _make_wiktionary_jsonl(
            jsonl,
            [{"word": "земя́", "lang_code": "bg", "pos": "noun", "forms": []}],
        )
        db = tmp_path / "bgospodinov.db"
        _make_bgospodinov_db(db, [("река", "ре`ка"), ("език", "ези`к")])

        out_mit = tmp_path / "mit.marisa"
        out_gpl = tmp_path / "gpl.marisa"
        build_multi_source_trie(tmp_path, out_mit, license_filter="mit")
        build_multi_source_trie(tmp_path, out_gpl, license_filter="gpl")

        trie_mit: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        trie_mit.load(str(out_mit))
        trie_gpl: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        trie_gpl.load(str(out_gpl))
        assert len(trie_mit) < len(trie_gpl)
