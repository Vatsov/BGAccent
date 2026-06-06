import hashlib
import json
from pathlib import Path

import marisa_trie
import pytest

from scripts.build_trie import build_trie, parse_bayganyu_csv
from scripts.download_sources import parse_sources_lock, verify_checksum


class TestSourcesLock:
    def test_parse_lock(self, tmp_path: Path) -> None:
        lock = tmp_path / "sources.lock"
        lock.write_text(
            json.dumps(
                {
                    "bayganyu": {
                        "url": "https://example.com/bg.csv",
                        "commit": "abc123",
                        "sha256": "deadbeef",
                        "license": "MIT",
                    }
                }
            ),
            encoding="utf-8",
        )
        data = parse_sources_lock(lock)
        assert data["bayganyu"]["sha256"] == "deadbeef"

    def test_manual_source_is_skipped(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import scripts.download_sources as ds

        lock = tmp_path / "sources.lock"
        lock.write_text(
            json.dumps(
                {
                    "bgospodinov": {
                        "url": "not a real url (build from source)",
                        "manual": True,
                        "sha256": "deadbeef",
                        "filename": "bgospodinov.db",
                    }
                }
            ),
            encoding="utf-8",
        )
        attempted: list[str] = []
        monkeypatch.setattr(ds, "download_source", lambda url, dest: attempted.append(url))
        ds.main(lock, tmp_path / "out")
        assert attempted == []

    def test_missing_sha256_raises(self, tmp_path: Path) -> None:
        lock = tmp_path / "sources.lock"
        lock.write_text(
            json.dumps({"bayganyu": {"url": "https://example.com/bg.csv"}}),
            encoding="utf-8",
        )
        with pytest.raises((KeyError, ValueError)):
            data = parse_sources_lock(lock)
            verify_checksum(tmp_path / "dummy.csv", data["bayganyu"]["sha256"])

    def test_checksum_match(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.csv"
        test_file.write_text("hello", encoding="utf-8")
        expected = hashlib.sha256(b"hello").hexdigest()
        verify_checksum(test_file, expected)

    def test_checksum_mismatch(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.csv"
        test_file.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="SHA256"):
            verify_checksum(test_file, "0" * 64)


class TestBayganyuCSVParsing:
    def test_parse_stress_position(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        # apostrophe BEFORE stressed vowel: plan'inata = stress on и (2nd vowel)
        csv.write_text("планината,план'ината\n", encoding="utf-8")
        entries = parse_bayganyu_csv(csv)
        assert len(entries) == 1
        assert entries[0][0] == "планината"
        assert entries[0][1] == 1

    def test_nfc_normalization(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        csv.write_text("красива,крас'ива\n", encoding="utf-8")
        entries = parse_bayganyu_csv(csv)
        assert len(entries) == 1
        import unicodedata

        assert unicodedata.is_normalized("NFC", entries[0][0])

    def test_meta_json_generated(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        csv.write_text("планината,план'ината\n", encoding="utf-8")
        out = tmp_path / "output.marisa"
        build_trie(csv, out, source_name="bayganyu", source_commit="abc")
        meta_path = out.parent / (out.name + ".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["format_version"] == 1
        assert meta["record_format"] == "HB"
        assert meta["source"] == "bayganyu"

    def test_built_trie_lookup(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        csv.write_text("планината,план'ината\n", encoding="utf-8")
        out = tmp_path / "output.marisa"
        build_trie(csv, out, source_name="bayganyu", source_commit="abc")
        trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        trie.load(str(out))
        results = trie["планината"]
        assert len(results) == 1
        assert results[0] == (1, 1)


class TestBayganyuValidation:
    def test_word_with_yot_is_kept(self, tmp_path: Path) -> None:
        # "й" is not a vowel in NFC but decomposes to и + breve in NFD; the old
        # NFC/NFD vowel-count guard wrongly skipped every word containing it.
        csv = tmp_path / "bg.csv"
        csv.write_text("случай,сл'учай\n", encoding="utf-8")
        entries = parse_bayganyu_csv(csv)
        assert len(entries) == 1
        assert entries[0][0] == "случай"
        assert entries[0][1] == 0

    def test_stressed_form_must_match_base_word(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        csv.write_text("планина,в'ода\n", encoding="utf-8")
        entries = parse_bayganyu_csv(csv)
        assert entries == []

    def test_second_vowel_stress_ordinal(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        # apostrophe before final а: stress on 2nd vowel (ordinal 1)
        csv.write_text("вода,вод'а\n", encoding="utf-8")
        entries = parse_bayganyu_csv(csv)
        assert len(entries) == 1
        assert entries[0][1] == 1


class TestBuildSafety:
    def test_invalid_entry_stress_on_non_vowel(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        # apostrophe before т (non-vowel)
        csv.write_text("тест,тес'т\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            build_trie(csv, tmp_path / "out.marisa", source_name="test", source_commit="x")

    def test_allow_invalid_entries(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        csv.write_text("тест,тес'т\nпланината,план'ината\n", encoding="utf-8")
        out = tmp_path / "out.marisa"
        build_trie(csv, out, source_name="test", source_commit="x", allow_invalid=True)
        trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        trie.load(str(out))
        assert "планината" in trie

    def test_homograph_all_variants_kept(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        # stress on а (idx 0) vs ъ (idx 1)
        csv.write_text("замък,з'амък\nзамък,зам'ък\n", encoding="utf-8")
        out = tmp_path / "out.marisa"
        build_trie(csv, out, source_name="test", source_commit="x")
        trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
        trie.load(str(out))
        results = trie["замък"]
        ordinals = {r[0] for r in results}
        assert 0 in ordinals
        assert 1 in ordinals

    def test_homograph_log(self, tmp_path: Path) -> None:
        csv = tmp_path / "bg.csv"
        csv.write_text("замък,з'амък\nзамък,зам'ък\n", encoding="utf-8")
        out = tmp_path / "out.marisa"
        build_trie(csv, out, source_name="test", source_commit="x")
        log = tmp_path / "build_homographs.log"
        assert log.exists()
        assert "замък" in log.read_text(encoding="utf-8")
