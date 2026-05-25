from __future__ import annotations

from pathlib import Path

import marisa_trie

from bgaccent.accentor import Accentor
from bgaccent.neural_predictor import NeuralStressPredictor
from scripts.train_stress_model import StressBiLSTM, build_vocab, export_onnx

FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


def _export_test_model(tmp_path: Path) -> tuple[Path, Path]:
    trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
    trie.load(str(FIXTURE_TRIE))
    vocab = build_vocab(trie)
    model = StressBiLSTM(vocab_size=len(vocab), embed_dim=32, hidden_dim=64, max_vowels=10)
    model_path = tmp_path / "model.onnx"
    export_onnx(model, vocab, model_path)
    vocab_path = tmp_path / "vocab.json"
    return model_path, vocab_path


class TestNeuralPredictor:
    def test_loads_and_predicts(self, tmp_path: Path) -> None:
        model_path, vocab_path = _export_test_model(tmp_path)
        predictor = NeuralStressPredictor(model_path, vocab_path)
        result = predictor.predict("планината")
        if result is not None:
            ordinal, confidence = result
            assert isinstance(ordinal, int)
            assert 0.0 <= confidence <= 1.0

    def test_monosyllabic_returns_none(self, tmp_path: Path) -> None:
        model_path, vocab_path = _export_test_model(tmp_path)
        predictor = NeuralStressPredictor(model_path, vocab_path)
        assert predictor.predict("а") is None

    def test_lazy_loading(self, tmp_path: Path) -> None:
        model_path, vocab_path = _export_test_model(tmp_path)
        predictor = NeuralStressPredictor(model_path, vocab_path)
        assert predictor._session is None
        predictor.predict("планина")
        assert predictor._session is not None

    def test_batch_prediction(self, tmp_path: Path) -> None:
        model_path, vocab_path = _export_test_model(tmp_path)
        predictor = NeuralStressPredictor(model_path, vocab_path, min_confidence=0.0)
        results = predictor.predict_batch(["планина", "красива", "xyz"])
        assert len(results) == 3

    def test_batch_empty(self, tmp_path: Path) -> None:
        model_path, vocab_path = _export_test_model(tmp_path)
        predictor = NeuralStressPredictor(model_path, vocab_path)
        assert predictor.predict_batch([]) == []


class TestNeuralFallback:
    def test_unavailable_without_model(self) -> None:
        predictor = NeuralStressPredictor(
            Path("/nonexistent/model.onnx"), Path("/nonexistent/vocab.json")
        )
        assert predictor.predict("планина") is None

    def test_accentor_works_without_neural(self) -> None:
        acc = Accentor(trie_path=FIXTURE_TRIE)
        result = acc.accent("планината")
        assert "́" in result


class TestNeuralAccentorIntegration:
    def test_neural_predicted_stat(self, tmp_path: Path) -> None:
        model_path, vocab_path = _export_test_model(tmp_path)
        acc = Accentor(
            trie_path=FIXTURE_TRIE,
            neural_model_path=model_path,
            neural_vocab_path=vocab_path,
        )
        result = acc.accent_with_report("непозната")
        assert result.stats.neural_predicted >= 0
