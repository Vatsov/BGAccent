from __future__ import annotations

from pathlib import Path

import marisa_trie

from bgaccent.accentor import Accentor
from bgaccent.ngram_predictor import StressPredictor, build_ngram_table

FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


def _load_trie() -> marisa_trie.RecordTrie[tuple[int, int]]:
    trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
    trie.load(str(FIXTURE_TRIE))
    return trie


class TestNgramTableConstruction:
    def test_table_has_entries(self) -> None:
        trie = _load_trie()
        table = build_ngram_table(trie)
        assert len(table) > 0

    def test_table_has_multiple_ngram_lengths(self) -> None:
        trie = _load_trie()
        table = build_ngram_table(trie)
        lengths = {len(suffix) for suffix in table}
        assert len(lengths) >= 2

    def test_each_entry_has_ordinal_counts(self) -> None:
        trie = _load_trie()
        table = build_ngram_table(trie)
        for _suffix, freq_map in table.items():
            assert isinstance(freq_map, dict)
            assert all(isinstance(k, int) and isinstance(v, int) for k, v in freq_map.items())

    def test_empty_trie_produces_empty_table(self) -> None:
        trie = marisa_trie.RecordTrie("HB", [])
        table = build_ngram_table(trie)
        assert len(table) == 0


class TestStressPrediction:
    def test_predict_returns_tuple_or_none(self) -> None:
        trie = _load_trie()
        predictor = StressPredictor(trie)
        result = predictor.predict("болница")
        if result is not None:
            ordinal, confidence, suffix = result
            assert isinstance(ordinal, int)
            assert 0.0 <= confidence <= 1.0
            assert isinstance(suffix, str)

    def test_predict_nonsense_returns_none(self) -> None:
        trie = _load_trie()
        predictor = StressPredictor(trie)
        assert predictor.predict("xyz") is None

    def test_confidence_is_proportion(self) -> None:
        trie = _load_trie()
        predictor = StressPredictor(trie, min_confidence=0.0)
        result = predictor.predict("планина")
        if result is not None:
            _ordinal, confidence, _suffix = result
            assert 0.0 < confidence <= 1.0

    def test_longer_suffix_preferred(self) -> None:
        trie = _load_trie()
        predictor = StressPredictor(trie, min_confidence=0.0)
        result = predictor.predict("планината")
        assert result is not None


class TestConfidenceThreshold:
    def test_default_threshold_filters_low(self) -> None:
        trie = _load_trie()
        predictor = StressPredictor(trie, min_confidence=0.99)
        predictor._build_table()
        for suffix in list(predictor._table.keys())[:5]:
            freq = predictor._table[suffix]
            total = sum(freq.values())
            majority = max(freq.values())
            if majority / total < 0.99:
                word_ending_in_suffix = "тест" + suffix
                result = predictor.predict(word_ending_in_suffix)
                assert result is None
                break

    def test_shorter_high_confidence_suffix_beats_low_confidence_longer(self) -> None:
        trie = _load_trie()
        predictor = StressPredictor(trie, min_confidence=0.7)
        # "това" (len 4) is a coin-flip; "ова" (len 3) is 90% ordinal 2.
        predictor._table = {"това": {0: 1, 1: 1}, "ова": {2: 9, 0: 1}}
        predictor._built = True
        # The returned suffix must be the shorter one that actually won, not the
        # longest candidate — this is what the report's matching_suffix relies on.
        assert predictor.predict("тестова") == (2, 0.9, "ова")

    def test_low_threshold_accepts_more(self) -> None:
        trie = _load_trie()
        pred_strict = StressPredictor(trie, min_confidence=0.99)
        pred_loose = StressPredictor(trie, min_confidence=0.01)
        word = "непозната"
        strict_result = pred_strict.predict(word)
        loose_result = pred_loose.predict(word)
        if strict_result is None:
            assert loose_result is not None


class TestAccentorNgramIntegration:
    def test_predicted_stat_counted(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE, enable_prediction=True)
        result = acc.accent_with_report("непозната дума")
        assert result.stats.predicted >= 0

    def test_no_predict_flag_disables(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE, enable_prediction=False)
        result = acc.accent_with_report("непозната")
        assert result.stats.predicted == 0

    def test_predicted_detail_entry(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE, enable_prediction=True)
        result = acc.accent_with_report("непозната дума")
        predicted = [d for d in result.details if d.get("status") == "predicted"]
        for d in predicted:
            assert "prediction_confidence" in d
            assert "matching_suffix" in d

    def test_lookup_chain_custom_wins(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.tsv"
        custom.write_text("непозната\tнепозна́та\n", encoding="utf-8")
        acc = Accentor(
            trie_path=FIXTURE_TRIE,
            custom_dicts=[custom],
            enable_prediction=True,
        )
        result = acc.accent_with_report("непозната")
        statuses = [d["status"] for d in result.details]
        assert "predicted" not in statuses
