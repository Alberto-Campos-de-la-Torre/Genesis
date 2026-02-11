# TTS Module API Reference

The TTS module provides components for evolving Text-to-Speech models.

## TTSChild

### `genesis.tts.tts_child.TTSChild`

TTS model child for evolutionary optimization.

```python
from genesis.tts import TTSChild, TTSConfig

config = TTSConfig(
    model_type="tacotron2",  # tacotron2, fastspeech2, vits
    style_dim=128,
    speaker_dim=256,
    num_speakers=1,
    sample_rate=22050,
    n_mel_channels=80,
    hop_length=256,
)

child = TTSChild(config=config)

# Load model weights
child.load_model("./model_checkpoint.pt", device="cuda")

# Synthesize speech
output = child.synthesize(
    text="Hello, world!",
    speaker_id=0,
    style_idx=None,  # Use average style
    device="cuda",
)
mel = output["mel_spectrogram"]

# Genetic operations
mutated_child = child.mutate(mutation_rate=0.1, mutation_scale=0.01)
crossed_child = child.crossover(other_child)
cloned_child = child.clone()

# Save/load
child.save("./tts_child.pt")
loaded_child = TTSChild.load("./tts_child.pt")
```

### TTSConfig

```python
@dataclass
class TTSConfig:
    model_type: str = "tacotron2"
    style_dim: int = 128
    speaker_dim: int = 256
    num_speakers: int = 1
    sample_rate: int = 22050
    n_mel_channels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    mutation_rate: float = 0.1
    mutation_scale: float = 0.01
```

## Style Evolution

### `genesis.tts.style_evolution.StyleEvolution`

Evolutionary optimization of TTS style tokens.

```python
from genesis.tts import StyleEvolution

evolution = StyleEvolution(
    style_dim=128,
    num_tokens=10,
    population_size=20,
    elite_size=2,
    mutation_rate=0.1,
    mutation_scale=0.05,
    crossover_rate=0.7,
)

# Initialize population
evolution.initialize_population(
    base_tokens=initial_tokens,  # Optional
    perturbation_scale=0.1,
)

# Evolution loop
for generation in range(100):
    # Evaluate fitness for each individual
    for idx in range(evolution.population_size):
        tokens = evolution.get_individual(idx)
        fitness = evaluate_style(tokens)
        evolution.set_fitness(idx, fitness)

    # Get best
    best_tokens, best_fitness = evolution.get_best()
    print(f"Gen {generation}: Best fitness = {best_fitness}")

    # Evolve
    evolution.evolve()

# Save/load state
evolution.save_state("./evolution_state.pt")
evolution.load_state("./evolution_state.pt")
```

### StyleToken

```python
@dataclass
class StyleToken:
    embedding: torch.Tensor
    name: str = ""
    fitness: float = 0.0
    metadata: dict = field(default_factory=dict)
```

### MultiStyleEvolution

Evolve multiple style aspects simultaneously.

```python
from genesis.tts.style_evolution import MultiStyleEvolution

multi_evolution = MultiStyleEvolution(
    style_configs={
        "prosody": {"dim": 64, "num_tokens": 5},
        "emotion": {"dim": 32, "num_tokens": 8},
        "speaker": {"dim": 128, "num_tokens": 4},
    },
    population_size=20,
)

multi_evolution.initialize_all()

# Get combined style from multiple evolutions
combined_style = multi_evolution.get_combined_style({
    "prosody": 0,
    "emotion": 5,
    "speaker": 2,
})
```

## MCD Fitness

### `genesis.tts.mcd_fitness`

Mel Cepstral Distortion based fitness evaluation.

#### `compute_mcd(reference_mel, synthesized_mel, reduction="mean")`

Compute MCD between reference and synthesized mel spectrograms.

```python
from genesis.tts.mcd_fitness import compute_mcd

mcd = compute_mcd(
    reference_mel,      # [batch, mel_dim, time]
    synthesized_mel,    # [batch, mel_dim, time]
    reduction="mean"    # "mean", "sum", "none"
)
```

### MCDFitness Class

```python
from genesis.tts import MCDFitness

fitness_evaluator = MCDFitness(
    reference_mels=list_of_reference_mels,
    target_mcd=5.0,               # Target MCD value
    weight_naturalness=0.5,
    weight_similarity=0.5,
    device="cuda",
)

# Add more references
fitness_evaluator.add_reference(new_mel)

# Evaluate single mel
result = fitness_evaluator.evaluate(synthesized_mel)
fitness = result["fitness"]
mcd = result["mcd"]
naturalness = result["naturalness_fitness"]
similarity = result["similarity_fitness"]

# Batch evaluation
results = fitness_evaluator.batch_evaluate(list_of_mels)

# Rank population
rankings = fitness_evaluator.rank_population(synthesized_mels)
# Returns: [(index, fitness), ...] sorted by fitness
```

### Additional Metrics

```python
from genesis.tts.mcd_fitness import compute_f0_rmse, compute_vuv_error

# F0 RMSE (fundamental frequency)
f0_rmse = compute_f0_rmse(reference_f0, synthesized_f0)

# Voiced/Unvoiced error rate
vuv_error = compute_vuv_error(reference_f0, synthesized_f0)
```
