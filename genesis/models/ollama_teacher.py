"""Teacher model backed by Ollama for inference-only knowledge distillation.

Instead of loading a large model locally, this teacher delegates all
inference to an Ollama server. Since all Qwen3 models share the same
tokenizer (vocab_size=151936), soft targets from the Ollama teacher
are directly usable in token-level KD loss with a Qwen3 student.
"""

import math
import logging
import time
from typing import Optional

import torch
import requests

logger = logging.getLogger(__name__)


class _OllamaConfig:
    """Minimal config shim so teacher.model.config.vocab_size works."""

    def __init__(self, vocab_size: int, model_name: str):
        self.vocab_size = vocab_size
        self.model_type = "ollama"
        self._model_name = model_name


class _OllamaModelShim:
    """Shim so teacher.model.parameters() / .training work as expected."""

    training = False

    def __init__(self, config: _OllamaConfig):
        self.config = config

    def parameters(self):
        return iter([])

    def named_parameters(self):
        return iter([])

    def eval(self):
        """No-op — remote model is always in inference mode."""
        return self

    def train(self, mode: bool = True):
        """No-op — remote model cannot be put in training mode."""
        return self


class OllamaTeacher:
    """
    Teacher model backed by a local (or remote) Ollama server.

    All Qwen3 variants share the same BPE tokenizer, so soft targets
    produced by, e.g., ``qwen3.5`` on Ollama can be used directly
    for KD loss against a Qwen3-1.7B student that uses the same vocab.

    Args:
        model_name: Ollama model tag (e.g. ``"qwen3.5"``, ``"qwen3:32b"``).
        base_url: Ollama server base URL.
        tokenizer_path: Local path or HF repo ID for the shared tokenizer
            (used to decode ``input_ids`` → text before sending to Ollama).
        vocab_size: Shared vocabulary size (151936 for all Qwen3 models).
        top_logprobs: Number of top-logprob tokens to request per position.
    """

    def __init__(
        self,
        model_name: str = "qwen3.5",
        base_url: str = "http://localhost:11434",
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 151936,
        top_logprobs: int = 20,
        fallback_model: Optional[str] = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._tokenizer_path = tokenizer_path
        self._vocab_size = vocab_size
        self.top_logprobs = top_logprobs
        self.fallback_model = fallback_model

        self._tokenizer = None
        self._logprobs_supported: bool = False  # set by load() after probing
        # Expose a .model attribute so teacher.model.config.vocab_size works
        self.model = _OllamaModelShim(
            _OllamaConfig(vocab_size=vocab_size, model_name=model_name)
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Verify Ollama connectivity, probe logprobs support, and load the tokenizer."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info(f"Ollama server reachable. Available models: {models}")
            if not any(self.model_name in m for m in models):
                logger.warning(
                    f"Model '{self.model_name}' not in Ollama. "
                    f"Run: ollama pull {self.model_name}"
                )
        except requests.RequestException as e:
            raise RuntimeError(
                f"Cannot reach Ollama server at {self.base_url}. "
                f"Start it with: ollama serve\nError: {e}"
            )

        if self._tokenizer_path:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
            logger.info(f"Tokenizer loaded from {self._tokenizer_path}")

        self._logprobs_supported = self._probe_logprobs()
        if self._logprobs_supported:
            logger.info("Ollama logprobs: SUPPORTED — KD soft-target loss enabled.")
        else:
            logger.warning(
                "Ollama logprobs: NOT SUPPORTED — KD soft-target loss will be "
                "disabled. Training will use hard-label cross-entropy only."
            )

    def _probe_logprobs(self) -> bool:
        """Send a tiny test request and check whether logprobs are returned."""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": "Hi",
                    "max_tokens": 1,
                    "logprobs": True,
                    "top_logprobs": 1,
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [{}])
            lp = (choices[0].get("logprobs") or {}) if choices else {}
            return bool(lp.get("top_logprobs") or lp.get("token_logprobs"))
        except Exception:
            return False

    def unload(self) -> None:
        """Release local resources (tokenizer)."""
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ids_to_text(self, input_ids: torch.Tensor) -> list[str]:
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Pass tokenizer_path= to OllamaTeacher."
            )
        return self._tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def _build_distribution(
        self,
        top_logprobs: dict,  # {token_str: logprob}
        temperature: float,
    ) -> torch.Tensor:
        """
        Convert Ollama top-k logprobs to a full-vocab probability vector.

        Known top-k tokens receive their temperature-scaled probabilities;
        the residual mass is spread uniformly over the remaining vocab.
        """
        result = torch.zeros(self._vocab_size)

        if not top_logprobs:
            return torch.full((self._vocab_size,), 1.0 / self._vocab_size)

        # Temperature-scale in log space then softmax over the top-k set
        scaled = {tok: lp / max(temperature, 1e-6) for tok, lp in top_logprobs.items()}
        max_lp = max(scaled.values())
        exp_vals = {tok: math.exp(lp - max_lp) for tok, lp in scaled.items()}
        denom = sum(exp_vals.values())

        known_prob_sum = 0.0
        for tok_str, prob in exp_vals.items():
            prob_norm = prob / denom
            if self._tokenizer is not None:
                tok_ids = self._tokenizer.encode(tok_str, add_special_tokens=False)
                if tok_ids and 0 <= tok_ids[0] < self._vocab_size:
                    result[tok_ids[0]] += prob_norm
                    known_prob_sum += prob_norm

        # Spread residual mass uniformly
        residual = max(0.0, 1.0 - known_prob_sum)
        result += residual / self._vocab_size

        return result

    # ------------------------------------------------------------------
    # Teacher interface
    # ------------------------------------------------------------------

    def get_soft_targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Return soft-target distributions for each token position.

        Decodes ``input_ids`` to text, sends the prompt to Ollama requesting
        ``top_logprobs`` per generated token, and reconstructs an approximate
        full-vocabulary distribution.

        Returns:
            Tensor of shape ``(batch, seq_len, vocab_size)``.
        """
        texts = self._ids_to_text(input_ids)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Pre-fill with uniform distribution so any uncovered position is valid
        uniform_val = 1.0 / self._vocab_size
        soft_targets = torch.full((batch_size, seq_len, self._vocab_size), uniform_val)

        for i, text in enumerate(texts):
            # Try primary model, then fallback (if configured), then uniform
            models_to_try = [self.model_name]
            if self.fallback_model:
                models_to_try.append(self.fallback_model)

            success = False
            for model_name in models_to_try:
                if success:
                    break
                backoff = 5.0
                for attempt in range(3):
                    try:
                        response = requests.post(
                            f"{self.base_url}/v1/completions",
                            json={
                                "model": model_name,
                                "prompt": text,
                                "max_tokens": seq_len,
                                "temperature": temperature,
                                "logprobs": self.top_logprobs,
                                "stream": False,
                            },
                            timeout=120,
                        )
                        if response.status_code == 429:
                            if self.fallback_model and model_name != self.fallback_model:
                                logger.warning(
                                    f"Ollama 429 on '{model_name}'. Switching to fallback '{self.fallback_model}'."
                                )
                            else:
                                retry_after = float(response.headers.get("Retry-After", backoff))
                                wait = max(retry_after, backoff)
                                logger.warning(
                                    f"Ollama 429 on '{model_name}'. Retrying in {wait:.1f}s (attempt {attempt+1}/3)."
                                )
                                time.sleep(wait)
                                backoff = min(backoff * 2, 60.0)
                            break  # move to next model (or next attempt)
                        response.raise_for_status()
                        data = response.json()

                        choices = data.get("choices", [{}])
                        if choices:
                            logprobs_data = choices[0].get("logprobs", {}) or {}
                            token_logprobs = logprobs_data.get("top_logprobs", []) or []
                            for t, tok_lp in enumerate(token_logprobs[:seq_len]):
                                if tok_lp:
                                    dist = self._build_distribution(tok_lp, temperature)
                                    if dist.sum() > 0:
                                        soft_targets[i, t] = dist
                        success = True
                        break  # success — no need to try fallback

                    except requests.RequestException as e:
                        if attempt < 2:
                            logger.warning(f"Ollama request to '{model_name}' failed: {e}. Retrying in {backoff:.1f}s.")
                            time.sleep(backoff)
                            backoff = min(backoff * 2, 60.0)
                        else:
                            logger.warning(f"Ollama request to '{model_name}' failed after 3 attempts: {e}.")

            if not success:
                logger.warning("All teacher models exhausted. Using uniform distribution for this sample.")
                # soft_targets[i] already uniform from pre-fill

        return soft_targets.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass via Ollama, returning a logits-compatible dict.

        When the Ollama server does not return logprobs (``has_logprobs=False``
        in the returned dict) the caller should skip the KD term and rely only
        on hard-label cross-entropy, because using a uniform soft-target
        distribution would actively corrupt the student model.
        """
        if not self._logprobs_supported:
            return {"logits": None, "has_logprobs": False}

        soft = self.get_soft_targets(input_ids, attention_mask, temperature=1.0)
        # Detect if all positions are still uniform (every API call failed)
        uniform_val = 1.0 / self._vocab_size
        is_uniform = (soft - uniform_val).abs().max().item() < 1e-7
        if is_uniform:
            return {"logits": None, "has_logprobs": False}

        logits = torch.log(soft.clamp(min=1e-9))
        return {"logits": logits.to(input_ids.device), "has_logprobs": True}

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        """Plain text generation via Ollama."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "think": False,   # disable Qwen3 thinking mode so response is non-empty
                "options": {"num_predict": max_tokens},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")
        # Fallback: some Ollama versions nest output differently
        if not text and "message" in data:
            text = data["message"].get("content", "")
        return text

    def get_config(self) -> dict:
        """Return config compatible with TeacherModel.get_config()."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "vocab_size": self._vocab_size,
            "num_parameters": 0,   # remote model
            "dtype": "remote",
        }
