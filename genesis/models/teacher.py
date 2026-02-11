"""Teacher model wrapper for knowledge distillation."""

from typing import Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import logging

logger = logging.getLogger(__name__)


class TeacherModel:
    """
    Wrapper for teacher model in knowledge distillation.

    The teacher model is typically a larger, well-trained model that provides
    soft targets for training the student model. It runs on a dedicated GPU
    and produces logits/hidden states for distillation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ):
        """
        Initialize the teacher model.

        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to load model on
            dtype: Model dtype (float16, bfloat16, float32)
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            trust_remote_code: Trust remote code from HuggingFace
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer = None
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code

    def load(self) -> None:
        """Load the teacher model and tokenizer."""
        logger.info(f"Loading teacher model: {self.model_name_or_path}")

        # Determine quantization config
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
            )

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            device_map=self.device if quantization_config else None,
            quantization_config=quantization_config,
            trust_remote_code=self.trust_remote_code,
        )

        if quantization_config is None:
            self._model = self._model.to(self.device)

        # Set to eval mode (teacher is never trained)
        self._model.eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Teacher model loaded on {self.device}")

    @property
    def model(self) -> PreTrainedModel:
        """Get the underlying model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass through the teacher model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            output_hidden_states: Whether to output hidden states

        Returns:
            Dictionary containing logits and optionally hidden states
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        result = {"logits": outputs.logits}

        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        return result

    @torch.no_grad()
    def get_soft_targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Get soft probability targets from teacher model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            temperature: Temperature for softmax scaling

        Returns:
            Soft probability distribution over vocabulary
        """
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs["logits"]

        # Apply temperature scaling
        soft_targets = torch.softmax(logits / temperature, dim=-1)

        return soft_targets

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_indices: Optional[list[int]] = None,
    ) -> list[torch.Tensor]:
        """
        Get hidden states from specified layers.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layer_indices: Indices of layers to extract (negative indexing supported)

        Returns:
            List of hidden state tensors
        """
        outputs = self.forward(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = outputs["hidden_states"]

        if layer_indices is None:
            return list(hidden_states)

        return [hidden_states[i] for i in layer_indices]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text using the teacher model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "dtype": str(self.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "config": self.model.config.to_dict() if hasattr(self.model, "config") else {},
        }

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Teacher model unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()
