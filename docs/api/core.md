# Core Module API Reference

The core module provides evolutionary algorithm components.

## Genetics

### `genesis.core.genetics`

#### `slerp(t, v0, v1, epsilon=1e-8)`

Spherical Linear Interpolation between two tensors.

**Parameters:**
- `t` (float): Interpolation factor (0 = v0, 1 = v1)
- `v0` (Tensor): First tensor
- `v1` (Tensor): Second tensor
- `epsilon` (float): Small value to prevent division by zero

**Returns:** Interpolated tensor

**Example:**
```python
from genesis.core.genetics import slerp

child_weights = slerp(0.5, parent1_weights, parent2_weights)
```

#### `crossover(parent1_state, parent2_state, crossover_rate=0.7, method="slerp", slerp_ratio=0.5)`

Perform crossover between two parent state dictionaries.

**Parameters:**
- `parent1_state` (dict): First parent state dict
- `parent2_state` (dict): Second parent state dict
- `crossover_rate` (float): Probability of crossover
- `method` (str): "slerp", "uniform", or "single_point"
- `slerp_ratio` (float): Interpolation ratio for SLERP

**Returns:** Child state dictionary

#### `mutate(state_dict, mutation_rate=0.1, mutation_scale=0.01, mutation_prob_per_weight=0.1, method="gaussian")`

Apply mutation to a state dictionary.

**Parameters:**
- `state_dict` (dict): Model state dictionary
- `mutation_rate` (float): Overall mutation probability
- `mutation_scale` (float): Scale of Gaussian noise
- `mutation_prob_per_weight` (float): Per-weight mutation probability
- `method` (str): "gaussian", "uniform", or "adaptive"

**Returns:** Mutated state dictionary

#### `class Genetics`

Manager class for genetic operations.

```python
genetics = Genetics(
    crossover_rate=0.7,
    mutation_rate=0.1,
    mutation_scale=0.01,
    slerp_ratio=0.5,
    adaptive_mutation=True,
)

child_state = genetics.create_offspring(parent1, parent2)
genetics.step_generation()  # For adaptive mutation decay
```

## Population

### `genesis.core.population`

#### `class Individual`

Represents an individual in the population.

**Attributes:**
- `id` (str): Unique identifier
- `state_dict` (dict): Model weights
- `fitness` (float): Fitness score
- `generation` (int): Generation number
- `parent_ids` (list): IDs of parents

#### `class Population`

Manages a population of individuals.

```python
population = Population(
    size=20,
    genetics=genetics,
    elite_size=2,
)

# Initialize from model
population.initialize_from_model(model, perturbation_scale=0.01)

# Evaluate fitness
population.evaluate(fitness_fn)

# Evolve to next generation
population.evolve()

# Access individuals
best = population.best
avg_fitness = population.average_fitness
diversity = population.diversity
```

## Fitness

### `genesis.core.fitness`

#### `class FitnessEvaluator`

Abstract base class for fitness evaluators.

```python
class CustomFitness(FitnessEvaluator):
    def evaluate(self, model, state_dict=None) -> FitnessResult:
        if state_dict:
            model.load_state_dict(state_dict)
        score = compute_score(model)
        return FitnessResult(score=score)
```

#### Built-in Evaluators

- `PerplexityFitness`: For language models (lower perplexity = higher fitness)
- `AccuracyFitness`: For classification tasks
- `QAFitness`: For question-answering tasks
- `CompositeFitness`: Combine multiple evaluators

## Selection

### `genesis.core.selection`

#### `class SelectionStrategy`

Base class for selection strategies.

#### Built-in Strategies

- `ElitismSelection`: Preserve top individuals
- `TournamentSelection`: Tournament-based selection
- `RouletteWheelSelection`: Fitness-proportionate selection
- `RankSelection`: Rank-based selection
- `BoltzmannSelection`: Temperature-controlled selection

```python
from genesis.core.selection import TournamentSelection

selection = TournamentSelection(tournament_size=3)
selected = selection.select(population, num_to_select=10)
```
