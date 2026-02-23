"""
Real integration tests using Qwen3 models.

Teacher : Ollama qwen3.5  (remote/local Ollama server — no GPU required)
Student : Qwen/Qwen3-1.7B (GPU 1 — RTX 5090, LoRA rank-16)

Skipped automatically when Ollama is not reachable or CUDA is unavailable.
"""

import os
import pytest
import torch
import requests
from pathlib import Path

# ── Model paths ───────────────────────────────────────────────────────────────

MODEL_DIR  = Path("/media/ttech-main/42A4266DA426639F/Models")
STUDENT_ID = "Qwen/Qwen3-1.7B"

OLLAMA_URL        = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "qwen3.5")


def _find_local(model_id: str) -> str | None:
    """Resolve a model ID to a local snapshot directory, or None if not cached."""
    # 1. Simple named subdirectory: Models/Qwen3-1.7B/, etc.
    model_short = model_id.split("/")[-1]
    flat_dir = MODEL_DIR / model_short
    if flat_dir.exists() and any(flat_dir.glob("*.safetensors")):
        return str(flat_dir)

    # 2. HF hub cache structure: hub/models--<org>--<name>/snapshots/<hash>/
    safe_name = model_id.replace("/", "--")
    pattern = MODEL_DIR / "hub" / f"models--{safe_name}" / "snapshots"
    if pattern.exists():
        snapshots = sorted(pattern.iterdir())
        if snapshots:
            return str(snapshots[-1])

    return None


def _ollama_reachable() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.ok
    except Exception:
        return False


def _ollama_model_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if not r.ok:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False


STUDENT_PATH = _find_local(STUDENT_ID)

# Skip markers
skip_no_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama server not reachable at {OLLAMA_URL}",
)
skip_no_ollama_model = pytest.mark.skipif(
    not _ollama_model_available(),
    reason=f"Ollama model '{OLLAMA_MODEL}' not pulled yet — run: ollama pull {OLLAMA_MODEL}",
)
skip_no_student = pytest.mark.skipif(
    STUDENT_PATH is None,
    reason=f"{STUDENT_ID} not downloaded yet",
)
skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available — needs RTX 5090",
)
skip_no_models = pytest.mark.skipif(
    not _ollama_model_available() or STUDENT_PATH is None,
    reason="Ollama teacher or local student model not available",
)

# ── Qwen3 LoRA config ─────────────────────────────────────────────────────────

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_RANK           = 16
SEQ_LEN             = 64
BATCH_SIZE          = 1


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def teacher_model():
    """Ollama-backed teacher — no local GPU memory required."""
    from genesis.models.ollama_teacher import OllamaTeacher
    t = OllamaTeacher(
        model_name=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        tokenizer_path=STUDENT_PATH,   # shared Qwen3 tokenizer
        vocab_size=151936,
        top_logprobs=20,
    )
    t.load()
    yield t
    t.unload()


@pytest.fixture(scope="module")
def student_model():
    """Load Qwen3-1.7B student with LoRA on GPU 1."""
    from genesis.models.student import StudentModel
    from genesis.models.lora_manager import LoRAConfig

    lora_cfg = LoRAConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    s = StudentModel(
        model_name_or_path=STUDENT_PATH,
        device="cuda:0",
        dtype=torch.bfloat16,
        use_lora=True,
        lora_config=lora_cfg,
    )
    s.load()
    yield s


def _dummy_batch(vocab_size: int, device: str = "cuda:0"):
    """Tiny synthetic batch for quick forward-pass validation."""
    ids  = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
    mask = torch.ones_like(ids)
    return ids, mask


# ── OllamaTeacher tests ───────────────────────────────────────────────────────

@skip_no_ollama
@skip_no_ollama_model
class TestOllamaTeacher:

    def test_ollama_server_reachable(self, teacher_model):
        assert _ollama_reachable()

    def test_model_available(self, teacher_model):
        assert _ollama_model_available()

    def test_get_config_keys(self, teacher_model):
        cfg = teacher_model.get_config()
        assert "model_name" in cfg
        assert "vocab_size" in cfg
        assert cfg["vocab_size"] == 151936

    def test_model_shim_not_training(self, teacher_model):
        assert not teacher_model.model.training

    def test_vocab_size_matches_student(self, teacher_model):
        assert teacher_model.model.config.vocab_size == 151936

    def test_generate_returns_string(self, teacher_model):
        out = teacher_model.generate("Hello, how are you?", max_tokens=20)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_soft_targets_shape(self, teacher_model):
        ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN))
        soft = teacher_model.get_soft_targets(ids, temperature=1.0)
        assert soft.shape == (BATCH_SIZE, SEQ_LEN, 151936)

    def test_soft_targets_sum_to_one(self, teacher_model):
        ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN))
        soft = teacher_model.get_soft_targets(ids, temperature=1.0)
        sums = soft.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)

    def test_soft_targets_non_negative(self, teacher_model):
        ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN))
        soft = teacher_model.get_soft_targets(ids, temperature=1.0)
        assert (soft >= 0).all()

    def test_forward_logits_shape(self, teacher_model):
        ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN))
        out = teacher_model.forward(ids)
        assert "logits" in out
        assert out["logits"].shape == (BATCH_SIZE, SEQ_LEN, 151936)


# ── StudentModel (Qwen3-1.7B + LoRA) ─────────────────────────────────────────

@skip_no_student
@skip_no_cuda
class TestQwen3Student:

    def test_model_on_correct_device(self, student_model):
        devices = {p.device for p in student_model.model.parameters()}
        assert any(d.type == "cuda" for d in devices)

    def test_lora_manager_present(self, student_model):
        assert student_model.lora_manager is not None

    def test_trainable_params_are_lora_only(self, student_model):
        trainable = [n for n, p in student_model.model.named_parameters()
                     if p.requires_grad]
        assert len(trainable) > 0
        for name in trainable:
            assert "lora_" in name, f"Unexpected trainable param: {name}"

    def test_forward_shape(self, student_model):
        vocab = student_model.model.config.vocab_size
        ids, mask = _dummy_batch(vocab, "cuda:0")
        out = student_model.forward(ids, mask)
        assert out["logits"].shape == (BATCH_SIZE, SEQ_LEN, vocab)

    def test_forward_with_labels_loss_finite(self, student_model):
        vocab = student_model.model.config.vocab_size
        ids, mask = _dummy_batch(vocab, "cuda:0")
        out = student_model.forward(ids, mask, labels=ids)
        assert out["loss"] is not None
        assert torch.isfinite(out["loss"])

    def test_lora_state_dict_keys(self, student_model):
        sd = student_model.lora_manager.get_lora_state_dict()
        assert len(sd) > 0
        for key in sd:
            assert "lora_" in key

    def test_lora_state_dict_roundtrip(self, student_model):
        original = student_model.lora_manager.get_lora_state_dict()
        zeroed   = {k: torch.zeros_like(v) for k, v in original.items()}
        student_model.lora_manager.set_lora_state_dict(zeroed)
        student_model.lora_manager.set_lora_state_dict(original)
        restored = student_model.lora_manager.get_lora_state_dict()
        for k in original:
            assert torch.allclose(original[k].cpu(), restored[k].cpu())


# ── KD Loss (Ollama teacher → GPU student) ────────────────────────────────────

@skip_no_models
@skip_no_cuda
class TestQwen3KDLoss:

    def test_kd_loss_with_ollama_teacher(self, teacher_model, student_model):
        from genesis.distillation.kd_loss import KDLoss

        vocab = student_model.model.config.vocab_size
        ids, mask = _dummy_batch(vocab, "cuda:0")

        # Get soft targets from Ollama teacher (CPU tensors)
        ids_cpu = ids.cpu()
        teacher_soft = teacher_model.get_soft_targets(ids_cpu, temperature=2.0)
        # Convert to logits and move to student device
        teacher_logits = torch.log(teacher_soft.clamp(min=1e-9)).to("cuda:0")

        s_out = student_model.forward(ids, mask)

        kd = KDLoss(temperature=2.0, alpha=0.5)
        losses = kd(
            student_logits=s_out["logits"],
            teacher_logits=teacher_logits,
            hard_labels=ids,
        )
        assert "total_loss" in losses
        assert torch.isfinite(losses["total_loss"])
        assert losses["total_loss"].item() > 0


# ── Distillation training loop ────────────────────────────────────────────────

@skip_no_models
@skip_no_cuda
class TestQwen3DistillationTrainer:

    def _make_dataloader(self, vocab: int, n_batches: int = 4):
        from torch.utils.data import DataLoader, TensorDataset
        ids  = torch.randint(0, vocab, (n_batches * BATCH_SIZE, SEQ_LEN))
        mask = torch.ones_like(ids)

        def collate(batch):
            id_b, m_b, lab_b = zip(*batch)
            return {
                "input_ids":      torch.stack(id_b),
                "attention_mask": torch.stack(m_b),
                "labels":         torch.stack(lab_b),
            }

        ds = TensorDataset(ids, mask, ids.clone())
        return DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate)

    def test_distillation_train_steps(self, teacher_model, student_model):
        from genesis.distillation.trainer import DistillationTrainer, TrainingConfig

        vocab = student_model.model.config.vocab_size
        dl    = self._make_dataloader(vocab, n_batches=4)

        cfg = TrainingConfig(
            learning_rate=1e-4,
            max_steps=5,
            warmup_steps=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            mixed_precision="bf16",
        )

        trainer = DistillationTrainer(
            teacher=teacher_model,
            student=student_model,
            train_dataloader=list(dl),
            config=cfg,
        )

        results = trainer.train()
        assert results["global_step"] == 5
        for entry in results["training_logs"]:
            assert torch.isfinite(torch.tensor(entry["loss"]))


# ── Population evolution with real Qwen3-1.7B LoRA weights ───────────────────

@skip_no_student
@skip_no_cuda
class TestQwen3Population:

    def test_initialize_and_evolve(self, student_model):
        from genesis.core.population import Population
        from genesis.core.genetics import Genetics

        genetics = Genetics(
            mutation_rate=1.0,
            mutation_scale=0.01,
            crossover_rate=0.7,
        )
        pop = Population(size=4, genetics=genetics, elite_size=1)
        pop.initialize_from_model(student_model.model, perturbation_scale=0.01)

        assert len(pop) == 4

        for i, ind in enumerate(pop):
            ind.fitness = float(i) / len(pop)

        pop.evolve()
        assert len(pop) == 4

    def test_perplexity_fitness_on_qwen3(self, student_model):
        from genesis.core.fitness import PerplexityFitness

        vocab = student_model.model.config.vocab_size
        ids   = torch.randint(0, vocab, (BATCH_SIZE * 2, SEQ_LEN))
        mask  = torch.ones_like(ids)

        batches = [
            {"input_ids": ids[i:i+BATCH_SIZE],
             "attention_mask": mask[i:i+BATCH_SIZE],
             "labels": ids[i:i+BATCH_SIZE]}
            for i in range(0, len(ids), BATCH_SIZE)
        ]

        evaluator = PerplexityFitness(
            dataloader=batches,
            device="cuda:0",
            max_samples=BATCH_SIZE * 2,
        )

        result = evaluator.evaluate(student_model.model)
        assert 0 <= result.score <= 1
        assert torch.isfinite(torch.tensor(result.score))
        assert result.metrics["perplexity"] > 1.0
