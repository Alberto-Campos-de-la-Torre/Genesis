"""Data preprocessing utilities for Genesis."""

from typing import Any, Optional
import torch
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing for NLP tasks.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_length: int = 512,
        lowercase: bool = False,
        remove_special_chars: bool = False,
        normalize_whitespace: bool = True,
    ):
        """
        Initialize text preprocessor.

        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            lowercase: Convert to lowercase
            remove_special_chars: Remove special characters
            normalize_whitespace: Normalize whitespace
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.normalize_whitespace = normalize_whitespace

    def preprocess(self, text: str) -> str:
        """
        Apply preprocessing steps to text.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if self.lowercase:
            text = text.lower()

        if self.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", "", text)

        if self.normalize_whitespace:
            text = " ".join(text.split())

        return text

    def tokenize(
        self,
        text: str,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize preprocessed text.

        Args:
            text: Input text
            return_tensors: Return format

        Returns:
            Tokenized output
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        text = self.preprocess(text)

        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )

    def batch_tokenize(
        self,
        texts: list[str],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of input texts
            return_tensors: Return format

        Returns:
            Batch tokenized output
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        texts = [self.preprocess(t) for t in texts]

        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )

    def create_qa_input(
        self,
        question: str,
        context: str,
        answer: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create input for QA tasks.

        Args:
            question: Question text
            context: Context text
            answer: Optional answer text

        Returns:
            Formatted input dictionary
        """
        question = self.preprocess(question)
        context = self.preprocess(context)

        # Format input
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"

        result = self.tokenize(input_text)

        if answer:
            result["target_text"] = self.preprocess(answer)

        return result


class AudioPreprocessor:
    """
    Audio preprocessing for TTS tasks.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        normalize: bool = True,
    ):
        """
        Initialize audio preprocessor.

        Args:
            sample_rate: Target sample rate
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window length
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            normalize: Normalize audio
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2
        self.normalize = normalize

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio from file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as numpy array
        """
        try:
            import librosa

            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            if self.normalize:
                audio = audio / (np.abs(audio).max() + 1e-8)

            return audio

        except ImportError:
            logger.error("librosa not installed")
            raise
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise

    def compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        return_tensor: bool = True,
    ) -> torch.Tensor:
        """
        Compute mel spectrogram from audio.

        Args:
            audio: Audio waveform
            return_tensor: Return as PyTorch tensor

        Returns:
            Mel spectrogram
        """
        try:
            import librosa

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )

            # Convert to log scale
            mel = librosa.power_to_db(mel, ref=np.max)

            if return_tensor:
                return torch.from_numpy(mel).float()
            return mel

        except ImportError:
            logger.error("librosa not installed")
            raise

    def compute_f0(
        self,
        audio: np.ndarray,
        method: str = "pyin",
    ) -> np.ndarray:
        """
        Compute fundamental frequency (F0) from audio.

        Args:
            audio: Audio waveform
            method: F0 estimation method ('pyin', 'yin', 'swipe')

        Returns:
            F0 contour
        """
        try:
            import librosa

            if method == "pyin":
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                )
            else:
                # Fallback to basic method
                f0 = librosa.yin(
                    audio,
                    fmin=80,
                    fmax=600,
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                )

            return np.nan_to_num(f0)

        except ImportError:
            logger.error("librosa not installed")
            raise

    def process_audio_file(self, audio_path: str) -> dict[str, torch.Tensor]:
        """
        Process audio file and extract features.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with extracted features
        """
        audio = self.load_audio(audio_path)
        mel = self.compute_mel_spectrogram(audio)
        f0 = self.compute_f0(audio)

        return {
            "audio": torch.from_numpy(audio).float(),
            "mel_spectrogram": mel,
            "f0": torch.from_numpy(f0).float(),
            "duration": len(audio) / self.sample_rate,
        }

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Normalize mel spectrogram.

        Args:
            mel: Mel spectrogram

        Returns:
            Normalized mel spectrogram
        """
        mel_min = mel.min()
        mel_max = mel.max()
        return (mel - mel_min) / (mel_max - mel_min + 1e-8)

    def denormalize_mel(
        self,
        mel: torch.Tensor,
        mel_min: float = -100,
        mel_max: float = 0,
    ) -> torch.Tensor:
        """
        Denormalize mel spectrogram.

        Args:
            mel: Normalized mel spectrogram
            mel_min: Original minimum value
            mel_max: Original maximum value

        Returns:
            Denormalized mel spectrogram
        """
        return mel * (mel_max - mel_min) + mel_min


def pad_sequence(
    sequences: list[torch.Tensor],
    padding_value: float = 0.0,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Pad sequences to same length.

    Args:
        sequences: List of tensors
        padding_value: Value to use for padding
        max_length: Maximum length (uses max sequence length if None)

    Returns:
        Padded tensor
    """
    if max_length is None:
        max_length = max(s.size(-1) for s in sequences)

    padded = []
    for seq in sequences:
        if seq.size(-1) < max_length:
            pad_size = max_length - seq.size(-1)
            if seq.dim() == 1:
                pad = torch.full((pad_size,), padding_value, dtype=seq.dtype)
                seq = torch.cat([seq, pad])
            else:
                pad = torch.full((*seq.shape[:-1], pad_size), padding_value, dtype=seq.dtype)
                seq = torch.cat([seq, pad], dim=-1)
        elif seq.size(-1) > max_length:
            seq = seq[..., :max_length]
        padded.append(seq)

    return torch.stack(padded)


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Create attention mask from input IDs.

    Args:
        input_ids: Input token IDs
        pad_token_id: Padding token ID

    Returns:
        Attention mask (1 for real tokens, 0 for padding)
    """
    return (input_ids != pad_token_id).long()
