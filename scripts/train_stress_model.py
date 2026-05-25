"""BiLSTM stress prediction model — training pipeline.

Usage:
    uv run python scripts/train_stress_model.py --trie src/bgaccent/data/bg.marisa --out models/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import marisa_trie
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split


def build_vocab(trie: marisa_trie.RecordTrie[tuple[int, int]]) -> dict[str, int]:
    chars: set[str] = set()
    for key in trie:
        for ch in key:
            chars.add(ch)
            chars.add(ch.upper())
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_word(word: str, vocab: dict[str, int], max_len: int = 0) -> list[int]:
    unk = vocab["<UNK>"]
    encoded = [vocab.get(ch, unk) for ch in word]
    if max_len > 0:
        pad = vocab["<PAD>"]
        encoded = encoded[:max_len] + [pad] * max(0, max_len - len(encoded))
    return encoded


class StressDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        trie: marisa_trie.RecordTrie[tuple[int, int]],
        vocab: dict[str, int],
        max_len: int = 30,
    ) -> None:
        self._items: list[tuple[list[int], int]] = []
        self._max_len = max_len
        for key in trie:
            results = trie.get(key)
            if results:
                vowel_ordinal = results[0][0]
                encoded = encode_word(key, vocab, max_len)
                self._items.append((encoded, vowel_ordinal))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        encoded, label = self._items[idx]
        return torch.tensor(encoded, dtype=torch.long), label


def train_val_test_split(
    dataset: StressDataset,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[Any, Any, Any]:
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=gen)


class StressBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 68,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        max_vowels: int = 10,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2, max_vowels)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = lstm_out.mean(dim=1)
        return self.log_softmax(self.fc(pooled))


def export_onnx(
    model: StressBiLSTM,
    vocab: dict[str, int],
    output_path: Path,
) -> None:
    model.eval()
    dummy = torch.randint(0, len(vocab), (1, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "seq_len"}, "output": {0: "batch"}},
        opset_version=14,
    )
    vocab_path = output_path.parent / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
