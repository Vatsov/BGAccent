from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from bgaccent.unicode import count_vowels, strip_accents

try:
    import onnxruntime
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


class NeuralStressPredictor:
    def __init__(
        self,
        model_path: Path,
        vocab_path: Path,
        min_confidence: float = 0.6,
        max_len: int = 30,
    ) -> None:
        self._model_path = model_path
        self._vocab_path = vocab_path
        self._min_confidence = min_confidence
        self._max_len = max_len
        self._session: object | None = None
        self._vocab: dict[str, int] | None = None
        self._available = _HAS_ONNX
        self._warned = False

    def _load(self) -> bool:
        if not self._available:
            return False
        if not self._model_path.exists() or not self._vocab_path.exists():
            if not self._warned:
                print(
                    f"Warning: neural model not found at {self._model_path}",
                    file=sys.stderr,
                )
                self._warned = True
            return False
        self._vocab = json.loads(self._vocab_path.read_text(encoding="utf-8"))
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(str(self._model_path), opts)
        return True

    def _encode(self, word: str) -> list[int]:
        assert self._vocab is not None
        unk = self._vocab.get("<UNK>", 1)
        pad = self._vocab.get("<PAD>", 0)
        clean = strip_accents(word).lower()
        encoded = [self._vocab.get(ch, unk) for ch in clean]
        encoded = encoded[: self._max_len] + [pad] * max(0, self._max_len - len(encoded))
        return encoded

    def predict(self, word: str) -> tuple[int, float] | None:
        clean = strip_accents(word).lower()
        if count_vowels(clean) < 2:
            return None

        if self._session is None and not self._load():
            return None

        encoded = self._encode(clean)
        import numpy as np
        input_array = np.array([encoded], dtype=np.int64)
        outputs = self._session.run(None, {"input": input_array})  # type: ignore[union-attr]
        log_probs = outputs[0][0]
        probs = [math.exp(lp) for lp in log_probs]
        max_idx = max(range(len(probs)), key=lambda i: probs[i])
        confidence = probs[max_idx] / sum(probs) if sum(probs) > 0 else 0.0

        if confidence < self._min_confidence:
            return None
        if max_idx >= count_vowels(clean):
            return None
        return max_idx, confidence

    def predict_batch(
        self, words: list[str]
    ) -> list[tuple[int, float] | None]:
        if not words:
            return []
        return [self.predict(w) for w in words]
