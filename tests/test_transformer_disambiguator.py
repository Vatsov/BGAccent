from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bgaccent.accentor import Accentor
from bgaccent.transformer_disambiguator import (
    HomographSample,
    TransformerDisambiguator,
    load_training_data,
)

FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


class TestTrainingDataFormat:
    def test_homograph_sample_fields(self) -> None:
        sample = HomographSample(
            sentence="Старият замък беше красив",
            target_word="замък",
            target_index=1,
            correct_ordinal=0,
        )
        assert sample.sentence == "Старият замък беше красив"
        assert sample.correct_ordinal == 0

    def test_load_training_data(self, tmp_path: Path) -> None:
        import json

        data = [
            {
                "sentence": "Старият замък",
                "target_word": "замък",
                "target_index": 1,
                "correct_ordinal": 0,
            },
            {
                "sentence": "Той замък торбата",
                "target_word": "замък",
                "target_index": 1,
                "correct_ordinal": 1,
            },
        ]
        path = tmp_path / "train.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        samples = load_training_data(path)
        assert len(samples) == 2
        assert isinstance(samples[0], HomographSample)


class TestTransformerDisambiguator:
    def test_unavailable_returns_none(self) -> None:
        dis = TransformerDisambiguator(model_path=None)
        result = dis.disambiguate("замък", ["Старият", "замък"], [(0, 1), (1, 1)])
        assert result is None

    def test_with_mock_session(self) -> None:
        np = pytest.importorskip("numpy")
        session = MagicMock()
        session.run.return_value = [np.array([[0.9, 0.1]])]
        dis = TransformerDisambiguator(model_path=None)
        dis._session = session
        dis._available = True
        result = dis.disambiguate("замък", ["Старият", "замък"], [(0, 1), (1, 1)])
        assert result == 0

    def test_large_logits_do_not_overflow(self) -> None:
        np = pytest.importorskip("numpy")
        session = MagicMock()
        # Raw model logits with large positive values: a naive
        # exp(x) softmax overflows here; the stabilized softmax must not.
        session.run.return_value = [np.array([[1000.0, 999.0]])]
        dis = TransformerDisambiguator(model_path=None)
        dis._session = session
        dis._available = True
        result = dis.disambiguate("замък", ["Старият", "замък"], [(0, 1), (1, 1)])
        assert result == 0

    def test_inference_failure_warns_and_returns_none(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytest.importorskip("numpy")
        session = MagicMock()
        session.run.side_effect = RuntimeError("input shape mismatch")
        dis = TransformerDisambiguator(model_path=None)
        dis._session = session
        dis._available = True
        result = dis.disambiguate("замък", ["Старият", "замък"], [(0, 1), (1, 1)])
        assert result is None
        assert (
            "Warning: transformer inference failed: input shape mismatch" in capsys.readouterr().err
        )


class TestThreeTierFallback:
    def test_priority_fallback_without_models(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        result = acc.accent_with_report("замък")
        details = [d for d in result.details if d.get("word") == "замък"]
        assert len(details) == 1
        assert details[0]["disambiguation"] == "priority_fallback"

    def test_pos_fallback_with_mock_spacy(self) -> None:
        nlp = MagicMock()
        doc = MagicMock()
        tok = MagicMock()
        tok.text = "замък"
        tok.pos_ = "NOUN"
        tok.lemma_ = "замък"
        doc.__iter__ = lambda self: iter([tok])
        nlp.return_value = doc

        acc = Accentor(trie_path=FIXTURE_TRIE, disambiguator_model=nlp)
        result = acc.accent_with_report("замък")
        details = [d for d in result.details if d.get("word") == "замък"]
        assert details[0]["disambiguation"] == "pos"
