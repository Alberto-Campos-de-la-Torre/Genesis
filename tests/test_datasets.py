"""Tests for genesis/data/datasets.py and preprocessing.py."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from genesis.data.preprocessing import (
    TextPreprocessor,
    AudioPreprocessor,
    pad_sequence,
    create_attention_mask,
)


# ── TextPreprocessor ──────────────────────────────────────────────────────────

class TestTextPreprocessor:

    def test_lowercase(self):
        p = TextPreprocessor(lowercase=True)
        assert p.preprocess("Hello World") == "hello world"

    def test_no_lowercase_preserves_case(self):
        p = TextPreprocessor(lowercase=False)
        assert p.preprocess("Hello World") == "Hello World"

    def test_normalize_whitespace_collapses_spaces(self):
        p = TextPreprocessor(normalize_whitespace=True)
        result = p.preprocess("hello   world")
        assert result == "hello world"

    def test_normalize_whitespace_removes_newlines(self):
        p = TextPreprocessor(normalize_whitespace=True)
        result = p.preprocess("hello\nworld")
        assert "\n" not in result

    def test_normalize_whitespace_removes_tabs(self):
        p = TextPreprocessor(normalize_whitespace=True)
        result = p.preprocess("hello\tworld")
        assert "\t" not in result

    def test_remove_special_chars_removes_uncommon(self):
        # regex keeps a-zA-Z0-9 and .,!?;:'"-  — removes @, #, etc.
        p = TextPreprocessor(remove_special_chars=True)
        result = p.preprocess("Hello @World #2024")
        assert "@" not in result
        assert "#" not in result

    def test_remove_special_chars_keeps_allowed(self):
        p = TextPreprocessor(remove_special_chars=True)
        result = p.preprocess("Hello, World!")
        assert "," in result  # comma is in the allowed set
        assert "!" in result  # exclamation is in the allowed set

    def test_preprocess_returns_string(self):
        p = TextPreprocessor()
        result = p.preprocess("  some text  ")
        assert isinstance(result, str)

    def test_tokenize_no_tokenizer_raises(self):
        p = TextPreprocessor()
        with pytest.raises(ValueError, match="Tokenizer"):
            p.tokenize("hello")

    def test_tokenize_with_mock_tokenizer(self):
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        p = TextPreprocessor(tokenizer=mock_tok, max_length=16)
        result = p.tokenize("hello world")
        assert "input_ids" in result

    def test_batch_tokenize_no_tokenizer_raises(self):
        p = TextPreprocessor()
        with pytest.raises(ValueError, match="Tokenizer"):
            p.batch_tokenize(["hello", "world"])

    def test_batch_tokenize_with_tokenizer(self):
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        p = TextPreprocessor(tokenizer=mock_tok, max_length=16)
        result = p.batch_tokenize(["hello", "world"])
        assert "input_ids" in result

    def test_empty_string(self):
        p = TextPreprocessor()
        result = p.preprocess("")
        assert isinstance(result, str)

    def test_combined_lower_and_whitespace(self):
        p = TextPreprocessor(lowercase=True, normalize_whitespace=True)
        result = p.preprocess("  HELLO   WORLD  ")
        assert result == "hello world"

    def test_create_qa_input_requires_tokenizer(self):
        p = TextPreprocessor()
        with pytest.raises(ValueError, match="Tokenizer"):
            p.create_qa_input("Q?", "Context")

    def test_create_qa_input_with_tokenizer(self):
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.randint(0, 100, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        p = TextPreprocessor(tokenizer=mock_tok)
        result = p.create_qa_input("What?", "Context here", answer="Yes")
        assert "target_text" in result
        assert result["target_text"] == "Yes"


# ── AudioPreprocessor ─────────────────────────────────────────────────────────

class TestAudioPreprocessor:

    def test_default_init(self):
        proc = AudioPreprocessor()
        assert proc.sample_rate == 22050
        assert proc.n_mels == 80

    def test_custom_params(self):
        proc = AudioPreprocessor(sample_rate=16000, n_mels=40)
        assert proc.sample_rate == 16000
        assert proc.n_mels == 40

    def test_normalize_mel_range(self):
        proc = AudioPreprocessor()
        mel = torch.randn(80, 50)
        normed = proc.normalize_mel(mel)
        assert normed.min() >= 0.0 - 1e-6
        assert normed.max() <= 1.0 + 1e-6

    def test_normalize_mel_uniform_input(self):
        proc = AudioPreprocessor()
        mel = torch.ones(80, 50)
        normed = proc.normalize_mel(mel)
        # All same value → normalized to 0 (or near 0 due to eps)
        assert normed.max().item() < 1e-3

    def test_denormalize_mel_restores_range(self):
        proc = AudioPreprocessor()
        mel = torch.rand(80, 50)
        denormed = proc.denormalize_mel(mel, mel_min=-100, mel_max=0)
        assert denormed.min().item() >= -100 - 1e-3
        assert denormed.max().item() <= 0 + 1e-3

    def test_fmax_default(self):
        proc = AudioPreprocessor(sample_rate=22050)
        assert proc.fmax == 22050 / 2

    def test_compute_mel_without_librosa_raises(self):
        proc = AudioPreprocessor()
        audio = np.random.randn(22050)
        # If librosa is not installed, should raise ImportError or ModuleNotFoundError
        try:
            import librosa
            # If librosa is installed, test passes without error
        except ImportError:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                proc.compute_mel_spectrogram(audio)


# ── pad_sequence ──────────────────────────────────────────────────────────────

class TestPadSequence:

    def test_1d_sequences_padded_to_max(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded = pad_sequence(seqs)
        assert padded.shape == (2, 3)
        assert padded[1, 2] == 0.0  # padded with zeros

    def test_custom_padding_value(self):
        seqs = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        padded = pad_sequence(seqs, padding_value=-1.0)
        assert padded[1, 1] == -1.0

    def test_max_length_truncates(self):
        seqs = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([6, 7])]
        padded = pad_sequence(seqs, max_length=3)
        assert padded.shape == (2, 3)

    def test_2d_sequences(self):
        seqs = [torch.randn(4, 10), torch.randn(4, 7)]
        padded = pad_sequence(seqs)
        assert padded.shape == (2, 4, 10)

    def test_same_length_no_padding(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        padded = pad_sequence(seqs)
        assert padded.shape == (2, 3)
        assert torch.equal(padded[0], torch.tensor([1, 2, 3]))


# ── create_attention_mask ─────────────────────────────────────────────────────

class TestCreateAttentionMask:

    def test_no_padding(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = create_attention_mask(ids, pad_token_id=0)
        assert torch.equal(mask, torch.ones(1, 4, dtype=torch.long))

    def test_with_padding(self):
        ids = torch.tensor([[1, 2, 0, 0]])
        mask = create_attention_mask(ids, pad_token_id=0)
        expected = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_all_padding(self):
        ids = torch.tensor([[0, 0, 0]])
        mask = create_attention_mask(ids, pad_token_id=0)
        assert mask.sum() == 0

    def test_custom_pad_token_id(self):
        ids = torch.tensor([[1, 2, -1, -1]])
        mask = create_attention_mask(ids, pad_token_id=-1)
        expected = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_batch_dimension(self):
        ids = torch.tensor([[1, 0], [1, 2]])
        mask = create_attention_mask(ids)
        assert mask.shape == (2, 2)
        assert mask[0, 1] == 0
        assert mask[1, 1] == 1


# ── TTSDataset ────────────────────────────────────────────────────────────────

class TestTTSDataset:

    def test_empty_directory(self, tmp_path):
        from genesis.data.datasets import TTSDataset
        dataset = TTSDataset(data_dir=str(tmp_path))
        assert len(dataset) == 0

    def test_loads_from_metadata_file(self, tmp_path):
        from genesis.data.datasets import TTSDataset
        meta = tmp_path / "metadata.csv"
        meta.write_text("audio1.wav|Hello world\naudio2.wav|Goodbye world\n")
        (tmp_path / "audio1.wav").write_bytes(b"\x00" * 100)
        (tmp_path / "audio2.wav").write_bytes(b"\x00" * 100)

        dataset = TTSDataset(data_dir=str(tmp_path))
        assert len(dataset) == 2

    def test_max_samples_respected(self, tmp_path):
        from genesis.data.datasets import TTSDataset
        meta = tmp_path / "metadata.csv"
        meta.write_text("a.wav|t1\nb.wav|t2\nc.wav|t3\n")
        dataset = TTSDataset(data_dir=str(tmp_path), max_samples=2)
        assert len(dataset) == 2

    def test_scans_directory_for_audio_files(self, tmp_path):
        from genesis.data.datasets import TTSDataset
        (tmp_path / "clip1.wav").write_bytes(b"\x00" * 100)
        (tmp_path / "clip2.wav").write_bytes(b"\x00" * 100)
        (tmp_path / "notes.txt").write_text("not audio")

        dataset = TTSDataset(data_dir=str(tmp_path))
        assert len(dataset) == 2

    def test_text_set_from_metadata(self, tmp_path):
        from genesis.data.datasets import TTSDataset
        meta = tmp_path / "metadata.csv"
        meta.write_text("audio1.wav|Hello world\n")
        (tmp_path / "audio1.wav").write_bytes(b"\x00" * 100)

        dataset = TTSDataset(data_dir=str(tmp_path))
        assert dataset._samples[0]["text"] == "Hello world"


# ── DatasetLoader ─────────────────────────────────────────────────────────────

class TestDatasetLoader:

    def test_init_stores_params(self):
        from genesis.data.datasets import DatasetLoader
        loader = DatasetLoader(
            dataset_name="test_dataset",
            max_length=256,
            split="validation",
            max_samples=100,
        )
        assert loader.dataset_name == "test_dataset"
        assert loader.max_length == 256
        assert loader.split == "validation"
        assert loader.max_samples == 100

    @patch("genesis.data.datasets.load_dataset")
    def test_load_calls_huggingface(self, mock_load):
        from genesis.data.datasets import DatasetLoader
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 50
        mock_ds.select.return_value = mock_ds
        mock_load.return_value = mock_ds

        loader = DatasetLoader(dataset_name="wikitext", max_samples=10)
        loader.load()
        mock_load.assert_called_once_with("wikitext", split="train", cache_dir=None)

    @patch("genesis.data.datasets.load_dataset")
    def test_load_applies_max_samples(self, mock_load):
        from genesis.data.datasets import DatasetLoader
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 100
        mock_ds.select.return_value = mock_ds
        mock_load.return_value = mock_ds

        loader = DatasetLoader(dataset_name="wikitext", max_samples=20)
        loader.load()
        mock_ds.select.assert_called_once()

    @patch("genesis.data.datasets.load_dataset")
    def test_load_no_max_samples(self, mock_load):
        from genesis.data.datasets import DatasetLoader
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 100
        mock_load.return_value = mock_ds

        loader = DatasetLoader(dataset_name="wikitext")
        loader.load()
        mock_ds.select.assert_not_called()

    def test_default_collate_raises_without_tokenizer(self):
        from genesis.data.datasets import DatasetLoader
        loader = DatasetLoader("test")
        with pytest.raises(ValueError, match="Tokenizer"):
            loader._default_collate([{"text": "hello"}])

    def test_default_collate_with_text_field(self):
        from genesis.data.datasets import DatasetLoader
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        loader = DatasetLoader("test", tokenizer=mock_tok)
        batch = [{"text": "hello"}, {"text": "world"}]
        result = loader._default_collate(batch)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_default_collate_question_context_format(self):
        from genesis.data.datasets import DatasetLoader
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.randint(0, 100, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        loader = DatasetLoader("test", tokenizer=mock_tok)
        batch = [{"question": "What is 2+2?", "context": "math basics"}]
        result = loader._default_collate(batch)
        assert "input_ids" in result

    def test_default_collate_fallback_format(self):
        from genesis.data.datasets import DatasetLoader
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.randint(0, 100, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        loader = DatasetLoader("test", tokenizer=mock_tok)
        batch = [{"answer": "Paris", "category": "geography"}]
        result = loader._default_collate(batch)
        assert "input_ids" in result

    @patch("genesis.data.datasets.load_dataset")
    def test_load_sets_dataset(self, mock_load):
        from genesis.data.datasets import DatasetLoader
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 5
        mock_load.return_value = mock_ds

        loader = DatasetLoader("wikitext")
        assert loader._dataset is None
        loader.load()
        assert loader._dataset is not None
