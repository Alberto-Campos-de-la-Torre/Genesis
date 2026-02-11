"""Student model wrapper for knowledge distillation."""

from typing import Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
import logging

from genesis.models.lora_manager import LoRAManager, LoRAConfig as GenesisLoRAConfig

logger = logging.getLogger(__name__)


class StudentModel:
    """
    Wrapper for student model in knowledge distillation.

    The student model is a smaller or modified version of the teacher that
    learns from soft targets. It supports LoRA adapters for efficient training.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:1",
        dtype: torch.dtype = torch.float16,
        use_lora: bool = True,
        lora_config: Optional[GenesisLoRAConfig] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ):
        """
        Initialize the student model.

        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to load model on
            dtype: Model dtype
            use_lora: Whether to use LoRA adapters
            lora_config: LoRA configuration
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            trust_remote_code: Trust remote code from HuggingFace
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.use_lora = use_lora
        self.lora_config = lora_config or GenesisLoRAConfig()
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer = None
        self._lora_manager: Optional[LoRAManager] = None
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code

    def load(self) -> None:
        """Load the student model and optionally apply LoRA."""
        logger.info(f"Loading student model: {self.model_name_or_path}")

        # Determine quantization config
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
            )

        # Load base model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            device_map=self.device if quantization_config else None,
            quantization_config=quantization_config,
            trust_remote_code=self.trust_remote_code,
        )

        if quantization_config is None:
            self._model = self._model.to(self.device)

        # Apply LoRA if enabled
        if self.use_lora:
            self._apply_lora()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Student model loaded on {self.device}")
        self._log_trainable_params()

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the model."""
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        self._model = get_peft_model(self._model, peft_config)
        self._lora_manager = LoRAManager(self._model, self.lora_config)

        logger.info("LoRA adapters applied")

    def _log_trainable_params(self) -> None:
        """Log trainable parameter statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

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

    @property
    def lora_manager(self) -> Optional[LoRAManager]:
        """Get LoRA manager if available."""
        return self._lora_manager

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass through the student model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for computing loss
            output_hidden_states: Whether to output hidden states

        Returns:
            Dictionary containing logits, loss, and optionally hidden states
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

        result = {"logits": outputs.logits}

        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["loss"] = outputs.loss

        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        return result

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single training step with distillation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            teacher_logits: Soft targets from teacher model
            labels: Hard labels (optional)
            temperature: Temperature for soft targets
            alpha: Weight for distillation loss

        Returns:
            Dictionary with total loss and component losses
        """
        self.model.train()

        # Forward pass
        outputs = self.forward(input_ids, attention_mask, labels, output_hidden_states=False)
        student_logits = outputs["logits"]

        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}

        # Distillation loss
        if teacher_logits is not None:
            teacher_logits = teacher_logits.to(self.device)

            # KL divergence loss
            student_soft = torch.log_softmax(student_logits / temperature, dim=-1)
            teacher_soft = torch.softmax(teacher_logits / temperature, dim=-1)

            kd_loss = torch.nn.functional.kl_div(
                student_soft,
                teacher_soft,
                reduction="batchmean",
            ) * (temperature**2)

            loss_dict["kd_loss"] = kd_loss
            total_loss = total_loss + alpha * kd_loss

        # Hard label loss
        if labels is not None and "loss" in outputs:
            hard_loss = outputs["loss"]
            loss_dict["hard_loss"] = hard_loss
            total_loss = total_loss + (1 - alpha) * hard_loss

        loss_dict["total_loss"] = total_loss
        return loss_dict

    def get_state_dict(self, lora_only: bool = True) -> dict[str, torch.Tensor]:
        """
        Get model state dictionary.

        Args:
            lora_only: If True and using LoRA, return only LoRA parameters

        Returns:
            State dictionary
        """
        if self.use_lora and lora_only and self._lora_manager:
            return self._lora_manager.get_lora_state_dict()
        return self.model.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = False,
    ) -> None:
        """
        Load state dictionary into model.

        Args:
            state_dict: State dictionary to load
            strict: Whether to require strict key matching
        """
        self.model.load_state_dict(state_dict, strict=strict)

    def merge_lora(self) -> None:
        """Merge LoRA weights into base model."""
        if self.use_lora and hasattr(self.model, "merge_and_unload"):
            self._model = self.model.merge_and_unload()
            self.use_lora = False
            logger.info("LoRA weights merged into base model")

    def save(self, path: str, merge_lora: bool = False) -> None:
        """
        Save model to disk.

        Args:
            path: Save path
            merge_lora: Whether to merge LoRA before saving
        """
        if merge_lora and self.use_lora:
            self.merge_lora()

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def get_config(self) -> dict:
        """Get model configuration."""
        config = {
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "dtype": str(self.dtype),
            "use_lora": self.use_lora,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }

        if self.use_lora:
            config["lora_config"] = {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "target_modules": self.lora_config.target_modules,
            }

        return config

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._lora_manager is not None:
            del self._lora_manager
            self._lora_manager = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Student model unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()
