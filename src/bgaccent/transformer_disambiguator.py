from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class HomographSample:
    sentence: str
    target_word: str
    target_index: int
    correct_ordinal: int


def load_training_data(path: Path) -> list[HomographSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        HomographSample(
            sentence=item["sentence"],
            target_word=item["target_word"],
            target_index=item["target_index"],
            correct_ordinal=item["correct_ordinal"],
        )
        for item in raw
    ]


class TransformerDisambiguator:
    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path
        self._session: Any = None
        self._available = False
        if model_path is not None and model_path.exists():
            try:
                import onnxruntime

                self._session = onnxruntime.InferenceSession(str(model_path))
                self._available = True
            except ImportError:
                pass
            except Exception as exc:
                print(
                    f"Warning: transformer disambiguator unavailable: {exc}",
                    file=sys.stderr,
                )

    def disambiguate(
        self,
        word: str,
        sentence_tokens: list[str],
        trie_results: list[tuple[int, int]],
    ) -> int | None:
        if not self._available or self._session is None:
            return None

        import numpy as np

        sentence = " ".join(sentence_tokens)
        encoded = np.array([[ord(c) % 128 for c in sentence[:512]]], dtype=np.int64)

        try:
            outputs = self._session.run(None, {"input": encoded})
            logits = outputs[0][0]
            probs = [math.exp(x) for x in logits]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            max_idx = max(range(len(probs)), key=lambda i: probs[i])
            valid_ordinals = {r[0] for r in trie_results}
            if max_idx in valid_ordinals:
                return max_idx
        except Exception:
            pass

        return None
