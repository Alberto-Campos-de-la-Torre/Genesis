"""Dataset loaders for Genesis experiments."""

from typing import Any, Callable, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Unified dataset loader for various data sources.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any = None,
        max_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize dataset loader.

        Args:
            dataset_name: Name of dataset (HuggingFace or local)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            split: Dataset split
            max_samples: Maximum number of samples
            cache_dir: Cache directory for downloads
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        self._dataset = None

    def load(self) -> Dataset:
        """Load the dataset."""
        logger.info(f"Loading dataset: {self.dataset_name}")

        # Try to load from HuggingFace
        try:
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
            )

            if self.max_samples:
                self._dataset = self._dataset.select(range(min(self.max_samples, len(self._dataset))))

            logger.info(f"Loaded {len(self._dataset)} samples")
            return self._dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def get_dataloader(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        """
        Create a DataLoader from the dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers
            collate_fn: Custom collation function

        Returns:
            DataLoader instance
        """
        if self._dataset is None:
            self.load()

        return DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn or self._default_collate,
            pin_memory=True,
        )

    def _default_collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Default collation function."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for default collation")

        # Handle different dataset formats
        if "text" in batch[0]:
            texts = [item["text"] for item in batch]
        elif "question" in batch[0] and "context" in batch[0]:
            texts = [f"{item['question']} {item['context']}" for item in batch]
        else:
            # Fallback: concatenate all string fields
            texts = [" ".join(str(v) for v in item.values() if isinstance(v, str)) for item in batch]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone(),
        }


class PubMedQADataset(Dataset):
    """
    Dataset for PubMedQA medical question answering.
    """

    def __init__(
        self,
        tokenizer: Any,
        split: str = "train",
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize PubMedQA dataset.

        Args:
            tokenizer: Tokenizer for text processing
            split: Dataset split
            max_length: Maximum sequence length
            max_samples: Maximum number of samples
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dataset
        logger.info("Loading PubMedQA dataset...")
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.data = dataset
        logger.info(f"Loaded {len(self.data)} PubMedQA samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # Format: Question + Context -> Answer
        question = item["question"]
        context = " ".join(item["context"]["contexts"])
        answer = item["final_decision"]  # yes/no/maybe

        # Create input text
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"
        target_text = answer

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length - 10,  # Leave room for answer
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=10,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # For causal LM, labels are the same as input shifted
        full_input = torch.cat([
            input_encoding["input_ids"].squeeze(),
            target_encoding["input_ids"].squeeze()[:5],  # First 5 tokens of answer
        ])

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": input_encoding["input_ids"].squeeze(),
            "target_text": target_text,
        }


class TTSDataset(Dataset):
    """
    Dataset for TTS experiments with text-audio pairs.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 22050,
        max_samples: Optional[int] = None,
        mel_config: Optional[dict] = None,
    ):
        """
        Initialize TTS dataset.

        Args:
            data_dir: Directory containing audio and text files
            sample_rate: Audio sample rate
            max_samples: Maximum number of samples
            mel_config: Mel spectrogram configuration
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.mel_config = mel_config or {
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 80,
        }

        self._samples: list[dict] = []
        self._load_metadata()

        if max_samples:
            self._samples = self._samples[:max_samples]

    def _load_metadata(self) -> None:
        """Load dataset metadata from data directory."""
        import os

        # Look for common metadata files
        metadata_files = ["metadata.csv", "transcript.txt", "filelist.txt"]

        for meta_file in metadata_files:
            meta_path = os.path.join(self.data_dir, meta_file)
            if os.path.exists(meta_path):
                self._parse_metadata(meta_path)
                return

        # Fallback: scan directory for audio files
        for filename in os.listdir(self.data_dir):
            if filename.endswith((".wav", ".mp3", ".flac")):
                audio_path = os.path.join(self.data_dir, filename)
                text_path = audio_path.rsplit(".", 1)[0] + ".txt"

                text = ""
                if os.path.exists(text_path):
                    with open(text_path, "r") as f:
                        text = f.read().strip()

                self._samples.append({
                    "audio_path": audio_path,
                    "text": text,
                })

    def _parse_metadata(self, path: str) -> None:
        """Parse metadata file."""
        import os

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    audio_file = parts[0]
                    text = parts[1]

                    audio_path = os.path.join(self.data_dir, audio_file)
                    if not audio_path.endswith((".wav", ".mp3", ".flac")):
                        audio_path += ".wav"

                    self._samples.append({
                        "audio_path": audio_path,
                        "text": text,
                    })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]

        # Load audio and compute mel spectrogram
        mel = self._load_and_process_audio(sample["audio_path"])

        return {
            "mel_spectrogram": mel,
            "text": sample["text"],
            "audio_path": sample["audio_path"],
        }

    def _load_and_process_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio and compute mel spectrogram."""
        try:
            import librosa

            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.mel_config["n_fft"],
                hop_length=self.mel_config["hop_length"],
                win_length=self.mel_config["win_length"],
                n_mels=self.mel_config["n_mels"],
            )

            # Convert to log scale
            mel = librosa.power_to_db(mel, ref=np.max)

            return torch.from_numpy(mel).float()

        except ImportError:
            logger.warning("librosa not installed. Returning dummy mel spectrogram.")
            return torch.randn(self.mel_config["n_mels"], 100)
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return torch.randn(self.mel_config["n_mels"], 100)


def create_dataloader(
    dataset_name: str,
    tokenizer: Any,
    batch_size: int = 8,
    split: str = "train",
    max_samples: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """
    Factory function to create dataloaders.

    Args:
        dataset_name: Name of dataset
        tokenizer: Tokenizer for text processing
        batch_size: Batch size
        split: Dataset split
        max_samples: Maximum samples
        **kwargs: Additional arguments

    Returns:
        DataLoader instance
    """
    if dataset_name == "pubmed_qa":
        dataset = PubMedQADataset(
            tokenizer=tokenizer,
            split=split,
            max_samples=max_samples,
        )
    else:
        loader = DatasetLoader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=split,
            max_samples=max_samples,
        )
        dataset = loader.load()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=True,
    )


