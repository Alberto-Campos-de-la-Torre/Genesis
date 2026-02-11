# TTS Voice Evolution Tutorial

This tutorial demonstrates how to evolve TTS style tokens for voice customization.

## Overview

We'll evolve style tokens to:
1. Match a target speaker's voice characteristics
2. Optimize for naturalness and expressiveness
3. Create customized voice profiles

## Prerequisites

- GPU with at least 8GB VRAM
- Reference audio samples (WAV format)
- Pre-trained TTS model (optional)

## Step 1: Prepare Reference Audio

```python
from pathlib import Path
from genesis.data.preprocessing import AudioPreprocessor

# Configure audio processing
preprocessor = AudioPreprocessor(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
)

# Load reference audio files
reference_dir = Path("./data/reference_audio")
reference_mels = []

for audio_file in reference_dir.glob("*.wav"):
    features = preprocessor.process_audio_file(str(audio_file))
    reference_mels.append(features["mel_spectrogram"])

print(f"Loaded {len(reference_mels)} reference samples")
```

## Step 2: Configure TTS Evolution

```python
from genesis.tts import TTSChild, TTSConfig
from genesis.tts.style_evolution import StyleEvolution
from genesis.tts.mcd_fitness import MCDFitness

# TTS configuration
tts_config = TTSConfig(
    model_type="tacotron2",
    style_dim=128,
    speaker_dim=256,
    num_speakers=1,
    sample_rate=22050,
    n_mel_channels=80,
)

# Style evolution configuration
style_evolution = StyleEvolution(
    style_dim=128,
    num_tokens=10,
    population_size=30,
    elite_size=3,
    mutation_rate=0.15,
    mutation_scale=0.02,
    crossover_rate=0.7,
)
```

## Step 3: Create Fitness Evaluator

```python
# MCD-based fitness evaluation
fitness_evaluator = MCDFitness(
    reference_mels=reference_mels,
    target_mcd=5.0,  # Lower MCD = better match
    weight_naturalness=0.4,
    weight_similarity=0.6,
    device="cuda",
)
```

## Step 4: Initialize TTS Model

```python
# Create TTS child
tts_child = TTSChild(config=tts_config)

# Optional: Load pre-trained weights
# tts_child.load_model("./pretrained_tts.pt", device="cuda")

# Initialize style evolution population
style_evolution.initialize_population(
    base_tokens=None,  # Random initialization
    perturbation_scale=0.1,
)
```

## Step 5: Run Evolution

```python
import torch
from tqdm import tqdm

num_generations = 100
test_text = "The quick brown fox jumps over the lazy dog."

best_overall_fitness = 0.0
best_style_tokens = None

for generation in tqdm(range(num_generations)):
    # Evaluate each individual
    population = style_evolution.get_population()
    fitnesses = []

    for idx, style_tokens in enumerate(population):
        # Set style tokens
        tts_child.style_tokens = style_tokens

        # Synthesize
        with torch.no_grad():
            output = tts_child.synthesize(
                text=test_text,
                device="cuda",
            )
        synthesized_mel = output["mel_spectrogram"].squeeze(0)

        # Evaluate fitness
        result = fitness_evaluator.evaluate(synthesized_mel)
        fitnesses.append(result["fitness"])

    # Update fitnesses
    style_evolution.set_all_fitnesses(fitnesses)

    # Get best
    best_tokens, best_fitness = style_evolution.get_best()

    if best_fitness > best_overall_fitness:
        best_overall_fitness = best_fitness
        best_style_tokens = best_tokens.clone()
        print(f"\nNew best at gen {generation}: {best_fitness:.4f}")

    # Log progress
    if generation % 10 == 0:
        avg = style_evolution.average_fitness
        print(f"Gen {generation}: Best={best_fitness:.4f}, Avg={avg:.4f}")

    # Evolve
    style_evolution.evolve()

print(f"\nEvolution complete! Best fitness: {best_overall_fitness:.4f}")
```

## Step 6: Evaluate Results

```python
# Set best style tokens
tts_child.style_tokens = best_style_tokens

# Synthesize multiple test sentences
test_sentences = [
    "Hello, how are you today?",
    "The weather is beautiful.",
    "I'm excited to demonstrate this voice.",
]

results = []
for text in test_sentences:
    output = tts_child.synthesize(text=text, device="cuda")
    mel = output["mel_spectrogram"].squeeze(0)

    # Evaluate against references
    fitness_result = fitness_evaluator.evaluate(mel)
    results.append({
        "text": text,
        "mcd": fitness_result["mcd"],
        "naturalness": fitness_result["naturalness_fitness"],
        "similarity": fitness_result["similarity_fitness"],
    })

# Print results
for r in results:
    print(f"Text: {r['text'][:30]}...")
    print(f"  MCD: {r['mcd']:.2f}")
    print(f"  Naturalness: {r['naturalness']:.4f}")
    print(f"  Similarity: {r['similarity']:.4f}")
```

## Step 7: Save Results

```python
import json

# Save best style tokens
torch.save({
    "style_tokens": best_style_tokens,
    "fitness": best_overall_fitness,
    "config": tts_config.__dict__,
}, "./outputs/best_voice_style.pt")

# Save evolution state
style_evolution.save_state("./outputs/evolution_state.pt")

# Save evaluation results
with open("./outputs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Multi-Aspect Evolution

For more sophisticated voice control, evolve multiple style aspects:

```python
from genesis.tts.style_evolution import MultiStyleEvolution

multi_evolution = MultiStyleEvolution(
    style_configs={
        "prosody": {
            "dim": 64,
            "num_tokens": 5,
            "mutation_rate": 0.2,
        },
        "emotion": {
            "dim": 32,
            "num_tokens": 8,
            "mutation_rate": 0.15,
        },
        "timbre": {
            "dim": 128,
            "num_tokens": 4,
            "mutation_rate": 0.1,
        },
    },
    population_size=20,
)

multi_evolution.initialize_all()

# During evolution, get combined styles
for gen in range(50):
    for idx in range(20):
        combined_style = multi_evolution.get_combined_style({
            "prosody": idx,
            "emotion": idx,
            "timbre": idx,
        })
        # Use combined_style for synthesis...

    multi_evolution.evolve_all()
```

## Tips for Better Voice Quality

1. **More Reference Samples**: Use diverse samples of target speaker
2. **F0 Matching**: Add F0 RMSE to fitness function
3. **Duration Modeling**: Match speaking rate of target
4. **Vocoder Quality**: Use high-quality neural vocoder (HiFi-GAN)
5. **Longer Evolution**: TTS often needs 100+ generations

## Troubleshooting

### Unnatural Voice

- Increase `weight_naturalness`
- Add more diverse reference samples
- Check mel spectrogram normalization

### Poor Speaker Similarity

- Increase `weight_similarity`
- Use more reference samples from target speaker
- Ensure reference audio is clean and well-recorded

### Slow Progress

- Increase `mutation_rate` early, decrease later
- Try larger `population_size`
- Use multiple random restarts
