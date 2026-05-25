from __future__ import annotations

import json
from pathlib import Path

import marisa_trie
import torch

from scripts.train_stress_model import (
    StressBiLSTM,
    StressDataset,
    build_vocab,
    encode_word,
    export_onnx,
    train_val_test_split,
)

FIXTURE_TRIE = Path(__file__).parent / "fixtures" / "bg_test.marisa"


def _load_trie() -> marisa_trie.RecordTrie[tuple[int, int]]:
    trie: marisa_trie.RecordTrie[tuple[int, int]] = marisa_trie.RecordTrie("HB")
    trie.load(str(FIXTURE_TRIE))
    return trie


class TestVocab:
    def test_build_vocab_covers_cyrillic(self) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        assert "а" in vocab
        assert "я" in vocab
        assert "А" in vocab

    def test_vocab_has_pad_and_unk(self) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1

    def test_encode_word_length(self) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        encoded = encode_word("планина", vocab)
        assert len(encoded) == len("планина")

    def test_encode_word_unknown_char(self) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        encoded = encode_word("test", vocab)
        assert all(idx == vocab["<UNK>"] for idx in encoded)

    def test_vocab_json_roundtrip(self, tmp_path: Path) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(vocab), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == vocab


class TestDataset:
    def test_dataset_size(self) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        ds = StressDataset(trie, vocab, max_len=30)
        assert len(ds) > 0

    def test_dataset_item_shape(self) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        ds = StressDataset(trie, vocab, max_len=30)
        word_tensor, label = ds[0]
        assert isinstance(word_tensor, torch.Tensor)
        assert isinstance(label, int)

    def test_split_reproducible(self) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        ds = StressDataset(trie, vocab, max_len=30)
        s1 = train_val_test_split(ds, seed=42)
        s2 = train_val_test_split(ds, seed=42)
        assert len(s1[0]) == len(s2[0])

    def test_split_ratios(self) -> None:
        trie = _load_trie()
        vocab = build_vocab(trie)
        ds = StressDataset(trie, vocab, max_len=30)
        train, val, test = train_val_test_split(ds, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(ds)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0


class TestModel:
    def test_instantiation(self) -> None:
        model = StressBiLSTM(vocab_size=68, embed_dim=64, hidden_dim=128, max_vowels=10)
        assert model is not None

    def test_forward_shape(self) -> None:
        model = StressBiLSTM(vocab_size=68, embed_dim=64, hidden_dim=128, max_vowels=10)
        x = torch.randint(0, 68, (32, 20))
        out = model(x)
        assert out.shape == (32, 10)

    def test_output_is_log_probs(self) -> None:
        model = StressBiLSTM(vocab_size=68, embed_dim=64, hidden_dim=128, max_vowels=10)
        x = torch.randint(0, 68, (4, 10))
        out = model(x)
        probs = torch.exp(out)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_param_count(self) -> None:
        model = StressBiLSTM(vocab_size=68, embed_dim=64, hidden_dim=128, max_vowels=10)
        count = sum(p.numel() for p in model.parameters())
        assert count < 2_000_000


class TestOnnxExport:
    def test_export_creates_file(self, tmp_path: Path) -> None:
        model = StressBiLSTM(vocab_size=68, embed_dim=64, hidden_dim=128, max_vowels=10)
        trie = _load_trie()
        vocab = build_vocab(trie)
        out = tmp_path / "model.onnx"
        export_onnx(model, vocab, out)
        assert out.exists()

    def test_onnx_loadable(self, tmp_path: Path) -> None:
        import onnxruntime

        model = StressBiLSTM(vocab_size=68, embed_dim=64, hidden_dim=128, max_vowels=10)
        trie = _load_trie()
        vocab = build_vocab(trie)
        out = tmp_path / "model.onnx"
        export_onnx(model, vocab, out)
        session = onnxruntime.InferenceSession(str(out))
        assert session is not None

    def test_vocab_json_saved(self, tmp_path: Path) -> None:
        model = StressBiLSTM(vocab_size=68, embed_dim=64, hidden_dim=128, max_vowels=10)
        trie = _load_trie()
        vocab = build_vocab(trie)
        out = tmp_path / "model.onnx"
        export_onnx(model, vocab, out)
        assert (tmp_path / "vocab.json").exists()
