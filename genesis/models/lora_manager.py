"""LoRA adapter management for Genesis."""

from dataclasses import dataclass, field
from typing import Any, Optional
import torch
import torch.nn as nn
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[list[str]] = None

    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        from peft import LoraConfig as PeftLoraConfig, TaskType

        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
        }

        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
            modules_to_save=self.modules_to_save,
        )


class LoRAManager:
    """
    Manager for LoRA adapters supporting evolutionary operations.

    Handles extraction, manipulation, and merging of LoRA weights
    for use in genetic algorithms.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[LoRAConfig] = None,
    ):
        """
        Initialize LoRA manager.

        Args:
            model: PEFT model with LoRA adapters
            config: LoRA configuration
        """
        self.model = model
        self.config = config or LoRAConfig()
        self._lora_modules: dict[str, nn.Module] = {}
        self._discover_lora_modules()

    def _discover_lora_modules(self) -> None:
        """Discover all LoRA modules in the model."""
        for name, module in self.model.named_modules():
            # Check for PEFT LoRA layers
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                self._lora_modules[name] = module
            # Alternative: check for Linear layers with LoRA weights
            elif "lora_" in name.lower():
                self._lora_modules[name] = module

        logger.info(f"Discovered {len(self._lora_modules)} LoRA modules")

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Extract only LoRA parameters from model.

        Returns:
            State dictionary containing only LoRA weights
        """
        lora_state = {}

        for name, param in self.model.named_parameters():
            if "lora_" in name.lower() or any(
                target in name for target in self.config.target_modules
            ):
                if param.requires_grad:
                    lora_state[name] = param.data.clone()

        return lora_state

    def set_lora_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = False,
    ) -> None:
        """
        Load LoRA parameters into model.

        Args:
            state_dict: LoRA state dictionary
            strict: Whether to require exact key matching
        """
        model_state = self.model.state_dict()

        for key, value in state_dict.items():
            if key in model_state:
                model_state[key] = value
            elif not strict:
                logger.warning(f"Key {key} not found in model state dict")
            else:
                raise KeyError(f"Key {key} not found in model state dict")

        self.model.load_state_dict(model_state)

    def clone_lora_weights(self) -> dict[str, torch.Tensor]:
        """Create a deep copy of LoRA weights."""
        return deepcopy(self.get_lora_state_dict())

    def interpolate_lora(
        self,
        state_dict1: dict[str, torch.Tensor],
        state_dict2: dict[str, torch.Tensor],
        ratio: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Linear interpolation between two LoRA state dicts.

        Args:
            state_dict1: First LoRA state dict
            state_dict2: Second LoRA state dict
            ratio: Interpolation ratio (0 = state_dict1, 1 = state_dict2)

        Returns:
            Interpolated LoRA state dict
        """
        result = {}

        for key in state_dict1.keys():
            if key in state_dict2:
                w1 = state_dict1[key]
                w2 = state_dict2[key]
                result[key] = (1 - ratio) * w1 + ratio * w2
            else:
                result[key] = state_dict1[key]

        return result

    def merge_multiple_lora(
        self,
        state_dicts: list[dict[str, torch.Tensor]],
        weights: Optional[list[float]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Merge multiple LoRA state dicts with weights.

        Args:
            state_dicts: List of LoRA state dictionaries
            weights: Optional weights for each state dict

        Returns:
            Merged LoRA state dict
        """
        if not state_dicts:
            raise ValueError("At least one state dict required")

        if weights is None:
            weights = [1.0 / len(state_dicts)] * len(state_dicts)

        assert len(weights) == len(state_dicts)

        result = {}
        keys = set(state_dicts[0].keys())

        for key in keys:
            tensors = [sd.get(key) for sd in state_dicts if key in sd]
            if len(tensors) == len(state_dicts):
                merged = sum(w * t for w, t in zip(weights, tensors))
                result[key] = merged
            else:
                result[key] = state_dicts[0][key]

        return result

    def compute_lora_similarity(
        self,
        state_dict1: dict[str, torch.Tensor],
        state_dict2: dict[str, torch.Tensor],
    ) -> float:
        """
        Compute cosine similarity between two LoRA state dicts.

        Args:
            state_dict1: First LoRA state dict
            state_dict2: Second LoRA state dict

        Returns:
            Average cosine similarity across all parameters
        """
        similarities = []

        for key in state_dict1.keys():
            if key in state_dict2:
                v1 = state_dict1[key].flatten().float()
                v2 = state_dict2[key].flatten().float()

                cos_sim = torch.nn.functional.cosine_similarity(
                    v1.unsqueeze(0),
                    v2.unsqueeze(0),
                ).item()
                similarities.append(cos_sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def get_lora_magnitude(self, state_dict: Optional[dict[str, torch.Tensor]] = None) -> float:
        """
        Compute total magnitude of LoRA weights.

        Args:
            state_dict: Optional state dict (uses current model if None)

        Returns:
            L2 norm of all LoRA parameters
        """
        if state_dict is None:
            state_dict = self.get_lora_state_dict()

        total_norm = 0.0
        for tensor in state_dict.values():
            total_norm += tensor.float().norm().item() ** 2

        return total_norm**0.5

    def scale_lora_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        scale: float,
    ) -> dict[str, torch.Tensor]:
        """
        Scale all LoRA weights by a factor.

        Args:
            state_dict: LoRA state dict
            scale: Scaling factor

        Returns:
            Scaled state dict
        """
        return {key: value * scale for key, value in state_dict.items()}

    def add_noise_to_lora(
        self,
        state_dict: dict[str, torch.Tensor],
        noise_scale: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        """
        Add Gaussian noise to LoRA weights.

        Args:
            state_dict: LoRA state dict
            noise_scale: Standard deviation of noise

        Returns:
            Noisy state dict
        """
        result = {}
        for key, value in state_dict.items():
            noise = torch.randn_like(value) * noise_scale
            result[key] = value + noise
        return result

    def freeze_base_model(self) -> None:
        """Freeze all non-LoRA parameters."""
        for name, param in self.model.named_parameters():
            if "lora_" not in name.lower():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_param_count(self) -> tuple[int, int]:
        """
        Get trainable parameter statistics.

        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def save_lora(self, path: str) -> None:
        """Save LoRA weights to file."""
        state_dict = self.get_lora_state_dict()
        torch.save(state_dict, path)
        logger.info(f"LoRA weights saved to {path}")

    def load_lora(self, path: str) -> None:
        """Load LoRA weights from file."""
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.set_lora_state_dict(state_dict)
        logger.info(f"LoRA weights loaded from {path}")


def create_lora_config_for_model(
    model_type: str,
    r: int = 16,
    alpha: int = 32,
) -> LoRAConfig:
    """
    Create a LoRA config optimized for specific model types.

    Args:
        model_type: Type of model ('llama', 'mistral', 'gpt2', 'bert')
        r: LoRA rank
        alpha: LoRA alpha

    Returns:
        LoRAConfig instance
    """
    target_modules_map = {
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "bert": ["query", "key", "value", "dense"],
        "t5": ["q", "k", "v", "o", "wi", "wo"],
    }

    target_modules = target_modules_map.get(model_type.lower(), ["q_proj", "v_proj"])

    return LoRAConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
    )
