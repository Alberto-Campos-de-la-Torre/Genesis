"""
End-to-end integration tests using real HuggingFace models.

Uses sshleifer/tiny-gpt2 (~11MB) — the smallest GPT-2 variant — so that the
full Genesis pipeline can be validated on CPU in seconds without requiring GPUs.

All tests are skipped automatically when transformers / peft / internet are
unavailable.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Skip the whole module if the heavy deps aren't installed
transformers = pytest.importorskip("transformers", reason="transformers not installed")
peft = pytest.importorskip("peft", reason="peft not installed")

MODEL = "sshleifer/tiny-gpt2"  # 4 layers, vocab=50257, ~11MB download
VOCAB = 50257
SEQ = 16
BATCH = 2


# ── shared fixtures (module-scope = loaded once per session) ─────────────────

def _batches(num=3):
    """Synthetic token batches for the tiny model."""
    ids = torch.randint(0, VOCAB, (num * BATCH, SEQ))
    mask = torch.ones_like(ids)
    labels = ids.clone()
    ds = TensorDataset(ids, mask, labels)

    def collate(batch):
        ids_b, mask_b, lab_b = zip(*batch)
        return {
            "input_ids": torch.stack(ids_b),
            "attention_mask": torch.stack(mask_b),
            "labels": torch.stack(lab_b),
        }

    return DataLoader(ds, batch_size=BATCH, collate_fn=collate)


@pytest.fixture(scope="module")
def teacher():
    from genesis.models.teacher import TeacherModel
    try:
        t = TeacherModel(MODEL, device="cpu", dtype=torch.float32)
        t.load()
        return t
    except Exception as e:
        pytest.skip(f"Could not load teacher model: {e}")


@pytest.fixture(scope="module")
def student():
    from genesis.models.student import StudentModel
    from genesis.models.lora_manager import LoRAConfig
    lora_cfg = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
    )
    try:
        s = StudentModel(
            MODEL,
            device="cpu",
            dtype=torch.float32,
            use_lora=True,
            lora_config=lora_cfg,
        )
        s.load()
        return s
    except Exception as e:
        pytest.skip(f"Could not load student model: {e}")


# ── TeacherModel ──────────────────────────────────────────────────────────────

class TestTeacherModel:

    def test_model_loaded(self, teacher):
        assert teacher.model is not None

    def test_tokenizer_loaded(self, teacher):
        assert teacher.tokenizer is not None
        assert teacher.tokenizer.pad_token is not None

    def test_forward_returns_logits(self, teacher):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        mask = torch.ones_like(ids)
        out = teacher.forward(ids, mask)
        assert "logits" in out
        assert out["logits"].shape == (BATCH, SEQ, VOCAB)

    def test_forward_with_hidden_states(self, teacher):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        out = teacher.forward(ids, output_hidden_states=True)
        assert "hidden_states" in out
        assert len(out["hidden_states"]) > 0

    def test_get_soft_targets_shape(self, teacher):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        soft = teacher.get_soft_targets(ids, temperature=2.0)
        assert soft.shape == (BATCH, SEQ, VOCAB)
        # Each position should sum to ~1.0
        assert torch.allclose(soft.sum(dim=-1), torch.ones(BATCH, SEQ), atol=1e-4)

    def test_get_soft_targets_temperature_effect(self, teacher):
        ids = torch.randint(0, VOCAB, (1, SEQ))
        sharp = teacher.get_soft_targets(ids, temperature=0.1)
        smooth = teacher.get_soft_targets(ids, temperature=10.0)
        # Smoother distribution has higher entropy (more uniform)
        entropy_sharp = -(sharp * sharp.clamp(min=1e-9).log()).sum(-1).mean()
        entropy_smooth = -(smooth * smooth.clamp(min=1e-9).log()).sum(-1).mean()
        assert entropy_smooth > entropy_sharp

    def test_get_config(self, teacher):
        cfg = teacher.get_config()
        assert "model_name_or_path" in cfg
        assert "num_parameters" in cfg
        assert cfg["num_parameters"] > 0

    def test_teacher_is_in_eval_mode(self, teacher):
        assert not teacher.model.training


# ── StudentModel ──────────────────────────────────────────────────────────────

class TestStudentModel:

    def test_model_loaded(self, student):
        assert student.model is not None

    def test_lora_manager_created(self, student):
        assert student.lora_manager is not None

    def test_lora_params_trainable(self, student):
        trainable = sum(p.numel() for p in student.model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_base_params_frozen(self, student):
        # LoRA: base weights should be frozen, only lora_A/lora_B trainable
        frozen = sum(p.numel() for p in student.model.parameters() if not p.requires_grad)
        assert frozen > 0

    def test_forward_returns_logits(self, student):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        mask = torch.ones_like(ids)
        out = student.forward(ids, mask)
        assert "logits" in out
        assert out["logits"].shape == (BATCH, SEQ, VOCAB)

    def test_forward_with_labels_returns_loss(self, student):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        out = student.forward(ids, labels=ids)
        assert "loss" in out
        assert out["loss"] is not None
        assert out["loss"].item() > 0

    def test_forward_with_hidden_states(self, student):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        out = student.forward(ids, output_hidden_states=True)
        assert "hidden_states" in out

    def test_get_lora_state_dict(self, student):
        sd = student.lora_manager.get_lora_state_dict()
        assert len(sd) > 0
        for key in sd:
            assert "lora" in key.lower()

    def test_set_and_get_state_dict_roundtrip(self, student):
        original = student.lora_manager.get_lora_state_dict()
        # Zero out LoRA weights
        zeroed = {k: torch.zeros_like(v) for k, v in original.items()}
        student.lora_manager.set_lora_state_dict(zeroed)
        # Restore
        student.lora_manager.set_lora_state_dict(original)
        restored = student.lora_manager.get_lora_state_dict()
        for k in original:
            assert torch.allclose(original[k], restored[k])


# ── KD Loss with real logits ──────────────────────────────────────────────────

class TestKDLossIntegration:

    def test_kd_loss_with_real_logits(self, teacher, student):
        from genesis.distillation.kd_loss import KDLoss
        ids = torch.randint(0, VOCAB, (BATCH, SEQ))
        mask = torch.ones_like(ids)

        with torch.no_grad():
            teacher_out = teacher.forward(ids, mask)

        student_out = student.forward(ids, mask, labels=ids)

        kd = KDLoss(temperature=2.0, alpha=0.5)
        losses = kd(
            student_logits=student_out["logits"],
            teacher_logits=teacher_out["logits"],
            hard_labels=ids,
        )
        assert "total_loss" in losses
        total = losses["total_loss"]
        assert torch.isfinite(total)
        assert total.item() > 0


# ── DistillationTrainer with real models ──────────────────────────────────────

class TestDistillationTrainerIntegration:

    def test_train_loop_runs(self, teacher, student):
        from genesis.distillation.trainer import DistillationTrainer, TrainingConfig

        cfg = TrainingConfig(
            learning_rate=1e-4,
            max_steps=3,
            warmup_steps=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            mixed_precision="fp32",
        )

        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            train_dataloader=list(_batches(num=4)),
            config=cfg,
        )

        results = trainer.train()
        assert results["global_step"] == 3
        assert torch.isfinite(torch.tensor(results["best_eval_loss"]
                                           if results["best_eval_loss"] != float("inf")
                                           else 0.0))

    def test_logged_loss_is_finite(self, teacher, student):
        from genesis.distillation.trainer import DistillationTrainer, TrainingConfig

        cfg = TrainingConfig(
            learning_rate=1e-4,
            max_steps=2,
            warmup_steps=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            mixed_precision="fp32",
        )

        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            train_dataloader=list(_batches(num=4)),
            config=cfg,
        )

        results = trainer.train()
        for entry in results["training_logs"]:
            assert torch.isfinite(torch.tensor(entry["loss"]))


# ── Population with real model weights ────────────────────────────────────────

class TestPopulationIntegration:

    def test_initialize_population_from_real_model(self, student):
        from genesis.core.population import Population
        from genesis.core.genetics import Genetics

        genetics = Genetics(mutation_rate=1.0, mutation_scale=0.01)
        pop = Population(size=3, genetics=genetics, elite_size=1)
        pop.initialize_from_model(student.model, perturbation_scale=0.01)

        assert len(pop) == 3
        for ind in pop:
            assert ind.state_dict is not None
            assert len(ind.state_dict) > 0

    def test_population_individuals_are_distinct(self, student):
        from genesis.core.population import Population
        from genesis.core.genetics import Genetics

        genetics = Genetics(mutation_rate=1.0, mutation_scale=0.05)
        pop = Population(size=3, genetics=genetics, elite_size=1)
        pop.initialize_from_model(student.model, perturbation_scale=0.05)

        # Pick a LoRA parameter that should differ between individuals
        first_key = next(
            k for k in pop[0].state_dict
            if "lora" in k.lower() and pop[0].state_dict[k].numel() > 1
        )
        vals = [pop[i].state_dict[first_key] for i in range(3)]
        # At least some should differ
        assert not (torch.allclose(vals[0], vals[1]) and torch.allclose(vals[1], vals[2]))

    def test_evolve_does_not_crash(self, student):
        from genesis.core.population import Population
        from genesis.core.genetics import Genetics

        genetics = Genetics(mutation_rate=0.5, mutation_scale=0.01)
        pop = Population(size=4, genetics=genetics, elite_size=1)
        pop.initialize_from_model(student.model, perturbation_scale=0.01)

        # Assign dummy fitnesses
        for i, ind in enumerate(pop):
            ind.fitness = float(i) / len(pop)

        pop.evolve()
        assert len(pop) == 4


# ── Perplexity fitness with real model ────────────────────────────────────────

class TestPerplexityFitnessIntegration:

    def test_perplexity_fitness_on_real_student(self, student):
        from genesis.core.fitness import PerplexityFitness

        evaluator = PerplexityFitness(
            dataloader=list(_batches(num=2)),
            device="cpu",
            max_samples=4,
        )

        result = evaluator.evaluate(student.model)
        assert 0 <= result.score <= 1
        assert result.metrics["perplexity"] > 1.0  # random model has high perplexity
        assert torch.isfinite(torch.tensor(result.score))
