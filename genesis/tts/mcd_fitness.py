"""MCD-based fitness evaluation for TTS models."""

from typing import Optional
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_mcd(
    reference_mel: torch.Tensor,
    synthesized_mel: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Mel Cepstral Distortion (MCD) between two mel spectrograms.

    MCD is a common metric for evaluating TTS quality, measuring the
    difference between reference and synthesized speech in the mel-cepstral domain.

    Args:
        reference_mel: Reference mel spectrogram [batch, mel_dim, time]
        synthesized_mel: Synthesized mel spectrogram [batch, mel_dim, time]
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        MCD value (lower is better)
    """
    # Ensure same length (truncate to shorter)
    min_len = min(reference_mel.size(-1), synthesized_mel.size(-1))
    reference_mel = reference_mel[..., :min_len]
    synthesized_mel = synthesized_mel[..., :min_len]

    # Convert to numpy for DCT (if needed)
    if reference_mel.is_cuda:
        ref_np = reference_mel.cpu().numpy()
        syn_np = synthesized_mel.cpu().numpy()
    else:
        ref_np = reference_mel.numpy()
        syn_np = synthesized_mel.numpy()

    # Compute mel cepstral coefficients using DCT
    # MCC = DCT(log mel spectrogram)
    ref_mcc = _mel_to_mcc(ref_np)
    syn_mcc = _mel_to_mcc(syn_np)

    # MCD formula: (10 * sqrt(2) / ln(10)) * sqrt(sum((ref_mcc - syn_mcc)^2))
    # Typically computed over MCC coefficients 1-13 (excluding c0)
    diff = ref_mcc[..., 1:14, :] - syn_mcc[..., 1:14, :]
    frame_mcd = np.sqrt(np.sum(diff ** 2, axis=-2))

    # Scale factor: 10 * sqrt(2) / ln(10) ≈ 6.1415
    scale = 10 * np.sqrt(2) / np.log(10)
    mcd = scale * frame_mcd

    if reduction == "mean":
        return torch.tensor(np.mean(mcd))
    elif reduction == "sum":
        return torch.tensor(np.sum(mcd))
    else:
        return torch.tensor(mcd)


def _mel_to_mcc(mel_spectrogram: np.ndarray, n_mcc: int = 25) -> np.ndarray:
    """
    Convert mel spectrogram to mel cepstral coefficients using DCT.

    Args:
        mel_spectrogram: Mel spectrogram [batch, mel_dim, time]
        n_mcc: Number of MCC coefficients to compute

    Returns:
        MCC coefficients [batch, n_mcc, time]
    """
    from scipy.fftpack import dct

    # Apply log (add small epsilon to avoid log(0))
    log_mel = np.log(mel_spectrogram + 1e-8)

    # Apply DCT along mel dimension
    # DCT-II with orthonormal normalization
    mcc = dct(log_mel, type=2, axis=-2, norm="ortho")

    # Keep first n_mcc coefficients
    return mcc[..., :n_mcc, :]


class MCDFitness:
    """
    Fitness evaluator based on Mel Cepstral Distortion.

    Lower MCD indicates better match to reference, so fitness
    is computed as inverse of MCD.
    """

    def __init__(
        self,
        reference_mels: Optional[list[torch.Tensor]] = None,
        target_mcd: float = 5.0,
        weight_naturalness: float = 0.5,
        weight_similarity: float = 0.5,
        device: str = "cuda",
    ):
        """
        Initialize MCD fitness evaluator.

        Args:
            reference_mels: Optional list of reference mel spectrograms
            target_mcd: Target MCD value for fitness scaling
            weight_naturalness: Weight for naturalness score
            weight_similarity: Weight for similarity score
            device: Device to run on
        """
        self.reference_mels = reference_mels or []
        self.target_mcd = target_mcd
        self.weight_naturalness = weight_naturalness
        self.weight_similarity = weight_similarity
        self.device = device

    def add_reference(self, mel: torch.Tensor) -> None:
        """Add a reference mel spectrogram."""
        self.reference_mels.append(mel)

    def evaluate(
        self,
        synthesized_mel: torch.Tensor,
        reference_idx: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Evaluate fitness of synthesized mel spectrogram.

        Args:
            synthesized_mel: Synthesized mel spectrogram
            reference_idx: Optional specific reference index

        Returns:
            Dictionary with fitness scores
        """
        if not self.reference_mels:
            raise ValueError("No reference mel spectrograms provided")

        # Compute MCD against reference(s)
        if reference_idx is not None:
            references = [self.reference_mels[reference_idx]]
        else:
            references = self.reference_mels

        mcds = []
        for ref in references:
            mcd = compute_mcd(ref, synthesized_mel, reduction="mean")
            mcds.append(mcd.item())

        avg_mcd = np.mean(mcds)
        min_mcd = np.min(mcds)

        # Convert MCD to fitness (lower MCD = higher fitness)
        # Using sigmoid-like transformation
        similarity_fitness = 1.0 / (1.0 + avg_mcd / self.target_mcd)

        # Naturalness score (based on mel statistics)
        naturalness_fitness = self._compute_naturalness(synthesized_mel)

        # Combined fitness
        total_fitness = (
            self.weight_similarity * similarity_fitness
            + self.weight_naturalness * naturalness_fitness
        )

        return {
            "fitness": total_fitness,
            "mcd": avg_mcd,
            "min_mcd": min_mcd,
            "similarity_fitness": similarity_fitness,
            "naturalness_fitness": naturalness_fitness,
        }

    def _compute_naturalness(self, mel: torch.Tensor) -> float:
        """
        Compute naturalness score based on mel statistics.

        Checks if mel spectrogram has reasonable statistics.
        """
        # Convert to numpy
        if mel.is_cuda:
            mel_np = mel.cpu().numpy()
        else:
            mel_np = mel.numpy()

        # Check for reasonable dynamic range
        dynamic_range = mel_np.max() - mel_np.min()
        range_score = min(1.0, dynamic_range / 80.0)  # Expect ~80dB range

        # Check for smoothness (penalize excessive noise)
        diff = np.diff(mel_np, axis=-1)
        smoothness = 1.0 / (1.0 + np.std(diff))

        # Check for reasonable energy distribution
        energy = np.mean(mel_np ** 2)
        energy_score = 1.0 / (1.0 + np.abs(energy - 0.5))

        # Combine scores
        naturalness = 0.4 * range_score + 0.3 * smoothness + 0.3 * energy_score

        return float(naturalness)

    def batch_evaluate(
        self,
        synthesized_mels: list[torch.Tensor],
    ) -> list[dict[str, float]]:
        """
        Evaluate multiple synthesized mel spectrograms.

        Args:
            synthesized_mels: List of synthesized mel spectrograms

        Returns:
            List of fitness dictionaries
        """
        results = []
        for mel in synthesized_mels:
            result = self.evaluate(mel)
            results.append(result)
        return results

    def rank_population(
        self,
        synthesized_mels: list[torch.Tensor],
    ) -> list[tuple[int, float]]:
        """
        Rank population by fitness.

        Args:
            synthesized_mels: List of synthesized mel spectrograms

        Returns:
            List of (index, fitness) tuples sorted by fitness descending
        """
        results = self.batch_evaluate(synthesized_mels)
        indexed_fitness = [(i, r["fitness"]) for i, r in enumerate(results)]
        indexed_fitness.sort(key=lambda x: x[1], reverse=True)
        return indexed_fitness


def compute_f0_rmse(
    reference_f0: torch.Tensor,
    synthesized_f0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RMSE of F0 (fundamental frequency) contours.

    Args:
        reference_f0: Reference F0 contour
        synthesized_f0: Synthesized F0 contour

    Returns:
        F0 RMSE (lower is better)
    """
    # Align lengths
    min_len = min(reference_f0.size(-1), synthesized_f0.size(-1))
    ref = reference_f0[..., :min_len]
    syn = synthesized_f0[..., :min_len]

    # Only compare voiced regions (F0 > 0)
    voiced_mask = (ref > 0) & (syn > 0)

    if voiced_mask.sum() == 0:
        return torch.tensor(float("inf"))

    # Convert to log scale for perceptually meaningful comparison
    ref_log = torch.log(ref[voiced_mask] + 1e-8)
    syn_log = torch.log(syn[voiced_mask] + 1e-8)

    rmse = torch.sqrt(torch.mean((ref_log - syn_log) ** 2))

    # Convert back to cents (100 cents = 1 semitone)
    rmse_cents = rmse * 1200 / np.log(2)

    return rmse_cents


def compute_vuv_error(
    reference_f0: torch.Tensor,
    synthesized_f0: torch.Tensor,
) -> float:
    """
    Compute voiced/unvoiced error rate.

    Args:
        reference_f0: Reference F0 contour
        synthesized_f0: Synthesized F0 contour

    Returns:
        VUV error rate (0-1, lower is better)
    """
    min_len = min(reference_f0.size(-1), synthesized_f0.size(-1))
    ref = reference_f0[..., :min_len]
    syn = synthesized_f0[..., :min_len]

    ref_voiced = ref > 0
    syn_voiced = syn > 0

    errors = (ref_voiced != syn_voiced).float().mean()

    return errors.item()
