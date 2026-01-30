# Kkachi Optimizer Guide

This guide covers all prompt optimization strategies available in kkachi, from simple few-shot selection to advanced multi-stage optimization.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Optimizer Types](#optimizer-types)
  - [Labeled Few-Shot](#labeled-few-shot)
  - [KNN Few-Shot](#knn-few-shot)
  - [COPRO](#copro-coordinate-prompt-optimization)
  - [MIPRO](#mipro-multi-instruction-prompt-optimization)
  - [SIMBA](#simba-self-improving-modular-boosting)
  - [Ensemble](#ensemble)
- [Choosing an Optimizer](#choosing-an-optimizer)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)

## Overview

Kkachi provides DSPy-style optimizers that automatically improve prompts by:
- **Selecting effective demonstrations** (few-shot examples)
- **Optimizing instruction text** (prompt engineering)
- **Combining multiple strategies** (ensemble methods)
- **Self-improving through feedback** (iterative refinement)

All optimizers work by evaluating candidate prompts against a training dataset using a user-defined metric.

## Quick Start

### Basic Optimization Workflow

```rust
use kkachi::recursive::{optimize::*, ApiLlm};

// 1. Create a dataset
let dataset = Dataset::new()
    .example("What is 2+2?", "4")
    .example("What is 3*5?", "15")
    .example("What is 10-7?", "3");

// 2. Define a metric (0.0-1.0)
let metric = |output: &str, expected: &str| {
    if output.trim() == expected { 1.0 } else { 0.0 }
};

// 3. Run optimizer
let llm = ApiLlm::from_env()?;
let result = Optimizer::new(&llm, "Answer math questions")
    .dataset(&dataset)
    .metric(metric)
    .strategy(Strategy::BootstrapFewShot { max_examples: 2 })
    .go();

println!("Optimized prompt: {}", result.prompt);
println!("Score: {:.2}", result.score);
```

## Optimizer Types

### Labeled Few-Shot

**Use Case**: You have labeled training data and want to select the best demonstrations.

**How It Works**: Evaluates all training examples, scores them, and selects the top-k as few-shot demonstrations.

```rust
use kkachi::optimizers::{LabeledFewShot, SelectionStrategy};

let optimizer = LabeledFewShot::builder()
    .max_examples(5)
    .selection_strategy(SelectionStrategy::BestN)
    .build();
```

**Selection Strategies**:
- `BestN`: Top-k scoring examples
- `First`: First k examples (preserves order)
- `Last`: Last k examples
- `Random`: Random k examples (with seed for reproducibility)

**Example**:
```rust
let dataset = Dataset::new()
    .labeled_example("Translate: Hello", "Hola", "greeting")
    .labeled_example("Translate: Goodbye", "Adiós", "farewell")
    .labeled_example("Translate: Thank you", "Gracias", "courtesy");

let result = LabeledFewShot::builder()
    .max_examples(2)
    .selection_strategy(SelectionStrategy::BestN)
    .build()
    .optimize(&llm, &dataset, metric);
```

**Best For**:
- Small datasets (< 100 examples)
- When you want explicit control over selection
- Fast iteration during development

### KNN Few-Shot

**Use Case**: Large datasets where you want context-aware example selection.

**How It Works**: Uses embeddings to find the k-nearest examples to each test input.

```rust
use kkachi::optimizers::{KNNFewShot, Embedder};

let embedder = Embedder::default(); // Uses local embedding model
let optimizer = KNNFewShot::builder()
    .k(3)
    .embedder(embedder)
    .build();
```

**Features**:
- **Semantic similarity**: Finds relevant examples based on meaning
- **Dynamic selection**: Different examples for each test input
- **Scalable**: Efficient even with 1000+ training examples

**Example**:
```rust
let large_dataset = Dataset::new()
    .example("Debug this Python code: ...", "Fixed code: ...")
    .example("Explain async/await in Rust", "Explanation: ...")
    // ... 1000 more examples

let optimizer = KNNFewShot::builder()
    .k(5)
    .embedder(Embedder::default())
    .build();

// For each test input, finds 5 most similar training examples
let result = optimizer.optimize(&llm, &large_dataset, metric);
```

**Best For**:
- Large datasets (> 100 examples)
- Diverse task types in one dataset
- When test inputs vary significantly

### COPRO (Coordinate Prompt Optimization)

**Use Case**: Optimize the instruction text itself, not just examples.

**How It Works**: Generates candidate instruction variations, evaluates them, and selects the best.

```rust
use kkachi::optimizers::{COPRO, COPROConfig};

let optimizer = COPRO::new(
    COPROConfig::new()
        .with_num_candidates(5)
        .with_breadth(3)
        .with_depth(2)
);
```

**Configuration**:
- `num_candidates`: How many instruction variations to try (default: 5)
- `breadth`: Variations per generation (default: 3)
- `depth`: Refinement iterations (default: 2)

**Example**:
```rust
let config = COPROConfig::new()
    .with_num_candidates(10)
    .with_breadth(5)
    .with_depth(3);

let result = COPRO::new(config).optimize(
    &llm,
    "Summarize the following text",  // Base instruction
    &dataset,
    metric
);

// result.prompt might be:
// "Provide a concise 2-3 sentence summary highlighting key points"
```

**Best For**:
- When instruction wording matters
- High-stakes applications (product descriptions, legal text)
- When you have budget for thorough optimization

### MIPRO (Multi-Instruction Prompt Optimization)

**Use Case**: Optimize both instructions AND few-shot examples simultaneously.

**How It Works**: Uses Tree-Parzen Estimator (TPE) to efficiently search the space of instructions + example combinations.

```rust
use kkachi::optimizers::{MIPRO, MIPROConfig};

let optimizer = MIPRO::new(
    MIPROConfig::new()
        .with_num_instructions(10)
        .with_num_examples(5)
        .with_num_trials(50)
);
```

**Configuration**:
- `num_instructions`: Candidate instructions to generate
- `num_examples`: Max few-shot examples to include
- `num_trials`: Total optimization trials (default: 50)

**Example**:
```rust
let config = MIPROConfig::new()
    .with_num_instructions(8)
    .with_num_examples(3)
    .with_num_trials(100);

let result = MIPRO::new(config).optimize(&llm, &dataset, metric);

println!("Best instruction: {}", result.instruction);
println!("Best examples: {:?}", result.examples);
println!("Score: {:.2} after {} trials", result.score, result.trials);
```

**Best For**:
- Complex tasks requiring both good instructions and examples
- When you have time for thorough search (50-100 trials)
- Production systems where quality matters

**Performance**: Uses TPE for smart sampling - typically finds 90% optimal solution in 50 trials.

### SIMBA (Self-Improving Modular Boosting)

**Use Case**: Iteratively improve prompts by analyzing failures.

**How It Works**: Runs optimization, analyzes failures, generates improvements, and repeats.

```rust
use kkachi::optimizers::{SIMBA, SIMBAConfig};

let optimizer = SIMBA::new(
    SIMBAConfig::new()
        .with_max_iterations(5)
        .with_improvement_threshold(0.1)
);
```

**Configuration**:
- `max_iterations`: How many improvement cycles (default: 5)
- `improvement_threshold`: Minimum score gain to continue (default: 0.05)
- `failure_analysis_depth`: How many failures to analyze (default: 10)

**Example**:
```rust
let config = SIMBAConfig::new()
    .with_max_iterations(10)
    .with_improvement_threshold(0.05)
    .with_failure_analysis_depth(20);

let result = SIMBA::new(config).optimize(&llm, &dataset, metric);

for (i, improvement) in result.improvements.iter().enumerate() {
    println!("Iteration {}: {:.2} -> {:.2}",
        i, improvement.old_score, improvement.new_score);
    println!("  Type: {:?}", improvement.kind);
}
```

**Improvement Types**:
- `InstructionRefinement`: Better instruction wording
- `ExampleAddition`: Added helpful demonstrations
- `ExampleReplacement`: Swapped in better examples
- `ConstraintAddition`: Added output format constraints

**Best For**:
- When you can afford multiple rounds of optimization
- Complex tasks with many failure modes
- When you want insight into what's being improved

### Ensemble

**Use Case**: Combine multiple optimizers for robust results.

**How It Works**: Runs multiple optimizers and combines their outputs using a strategy.

```rust
use kkachi::optimizers::{Ensemble, CombineStrategy};

let optimizer = Ensemble::builder()
    .add_optimizer(Box::new(LabeledFewShot::default()))
    .add_optimizer(Box::new(COPRO::default()))
    .strategy(CombineStrategy::Best)
    .build();
```

**Combine Strategies**:
- `Best`: Use the single best-performing optimizer's result
- `Union`: Combine examples from all optimizers
- `Intersection`: Only include examples that all optimizers agree on
- `Vote`: Weighted voting based on optimizer scores

**Example**:
```rust
let ensemble = Ensemble::builder()
    .add_optimizer(Box::new(LabeledFewShot::default()))
    .add_optimizer(Box::new(KNNFewShot::default()))
    .add_optimizer(Box::new(COPRO::default()))
    .strategy(CombineStrategy::Best)
    .build();

let result = ensemble.optimize(&llm, &dataset, metric);

println!("Best optimizer: {}", result.best_optimizer);
println!("Individual scores: {:?}", result.optimizer_scores);
```

**Best For**:
- Production systems requiring robustness
- When you're unsure which optimizer to use
- Hedging against optimizer-specific weaknesses

## Choosing an Optimizer

| Scenario | Recommended Optimizer | Why |
|----------|----------------------|-----|
| Small dataset (< 50 examples) | LabeledFewShot | Fast, simple, effective |
| Large dataset (> 200 examples) | KNNFewShot | Scales well, context-aware |
| Instruction wording matters | COPRO | Optimizes instruction text |
| Need both instructions + examples | MIPRO | Joint optimization |
| Complex task with failures | SIMBA | Iterative improvement |
| Production deployment | Ensemble | Robust, hedges risk |
| Tight budget | LabeledFewShot | Fewest LLM calls |
| Large budget | MIPRO + SIMBA | Thorough search + refinement |

### Decision Tree

```
Do you need to optimize instructions?
├─ Yes: COPRO or MIPRO
│   ├─ Also need examples? → MIPRO
│   └─ Just instructions? → COPRO
└─ No: Just examples
    ├─ Dataset size?
    │   ├─ < 100: LabeledFewShot
    │   └─ > 100: KNNFewShot
    └─ Need iterative improvement? → SIMBA
```

## Best Practices

### 1. Start Simple

Begin with `LabeledFewShot` to establish a baseline:
```rust
let baseline = LabeledFewShot::default()
    .optimize(&llm, &dataset, metric);
println!("Baseline score: {:.2}", baseline.score);
```

Then try advanced optimizers if needed.

### 2. Use Representative Datasets

Your training dataset should:
- Cover diverse inputs (not just edge cases)
- Include typical real-world examples
- Be large enough (20+ examples minimum)
- Have high-quality reference outputs

```rust
// Good: Diverse, realistic examples
let dataset = Dataset::new()
    .example("simple case", "expected")
    .example("edge case with unusual input", "expected")
    .example("typical real-world scenario", "expected");

// Bad: All edge cases
let dataset = Dataset::new()
    .example("weird edge case #1", "expected")
    .example("weird edge case #2", "expected");
```

### 3. Choose Appropriate Metrics

```rust
// Exact match (strict)
let exact = |output: &str, expected: &str| {
    if output.trim() == expected.trim() { 1.0 } else { 0.0 }
};

// Contains (lenient)
let contains = |output: &str, expected: &str| {
    if output.contains(expected) { 1.0 } else { 0.0 }
};

// Partial credit (graduated)
let partial = |output: &str, expected: &str| {
    let common = output.chars()
        .zip(expected.chars())
        .filter(|(a, b)| a == b)
        .count();
    common as f64 / expected.len() as f64
};
```

### 4. Budget Your LLM Calls

Different optimizers use different numbers of LLM calls:

| Optimizer | Approx. Calls | Formula |
|-----------|---------------|---------|
| LabeledFewShot | n | n = dataset size |
| KNNFewShot | n + k*m | n = training, m = test size, k = neighbors |
| COPRO | c*b*d*n | c = candidates, b = breadth, d = depth, n = examples |
| MIPRO | t*(i+e) | t = trials, i = instructions, e = examples |
| SIMBA | iter*n | iter = iterations, n = examples |
| Ensemble | sum(optimizers) | Sum of all component calls |

Budget accordingly for API costs.

### 5. Save Optimized Prompts

```rust
use std::fs;

let result = optimizer.optimize(&llm, &dataset, metric);

// Save for reuse
let serialized = serde_json::to_string(&result)?;
fs::write("optimized_prompt.json", serialized)?;

// Load later
let loaded: OptimizationResult = serde_json::from_str(&fs::read_to_string("optimized_prompt.json")?)?;
```

### 6. Validate on Hold-Out Set

```rust
// Split dataset
let (train, test) = dataset.split(0.8);

// Optimize on training set
let result = optimizer.optimize(&llm, &train, metric);

// Validate on test set
let test_score = evaluate(&llm, &result.prompt, &test, metric);
println!("Train: {:.2}, Test: {:.2}", result.score, test_score);
```

## Performance Considerations

### Memory Usage

All optimizers use zero-copy patterns:
- `ExampleSet` stores examples in a shared `Buffer`
- `ExampleMeta` uses fixed-size arrays (Copy semantics)
- No per-example allocations during optimization

For large datasets (1000+ examples):
```rust
use kkachi::buffer::Buffer;

// Efficient: All examples in one buffer
let buffer = Buffer::from_examples(&examples);
let example_set = ExampleSet::new(&buffer, &metadata);
```

### Parallel Execution

Optimizers that evaluate multiple candidates can be parallelized:

```rust
use rayon::prelude::*;

let candidates: Vec<_> = (0..num_candidates)
    .into_par_iter()
    .map(|i| evaluate_candidate(i))
    .collect();
```

Currently implemented for:
- COPRO (parallel candidate evaluation)
- MIPRO (parallel trial evaluation)
- Ensemble (parallel optimizer execution)

### Caching

Use LLM caching to avoid redundant calls:

```rust
use kkachi::recursive::CacheExt;

let cached_llm = llm.with_cache(1000);
let result = optimizer.optimize(&cached_llm, &dataset, metric);
```

This is especially valuable for:
- MIPRO (many repeated evaluations)
- SIMBA (iterative refinement)
- Ensemble (shared examples across optimizers)

## Advanced Patterns

### Custom Metrics

```rust
// Combine multiple criteria
let multi_metric = |output: &str, expected: &str| {
    let correctness = if output.contains(expected) { 0.5 } else { 0.0 };
    let length_score = if output.len() < 100 { 0.3 } else { 0.0 };
    let format_score = if output.starts_with("Answer:") { 0.2 } else { 0.0 };
    correctness + length_score + format_score
};
```

### Progressive Optimization

```rust
// Stage 1: Fast baseline with few examples
let baseline = LabeledFewShot::builder()
    .max_examples(2)
    .build()
    .optimize(&llm, &dataset, metric);

// Stage 2: Optimize instruction around those examples
let improved = COPRO::default().optimize(
    &llm,
    &baseline.prompt,
    &dataset,
    metric
);

// Stage 3: Refine through failure analysis
let final_result = SIMBA::default().optimize(
    &llm,
    &improved.prompt,
    &dataset,
    metric
);
```

### Domain-Specific Optimizers

Wrap optimizers for your domain:

```rust
pub struct CodeGenerationOptimizer {
    inner: MIPRO,
}

impl CodeGenerationOptimizer {
    pub fn new() -> Self {
        Self {
            inner: MIPRO::new(
                MIPROConfig::new()
                    .with_num_examples(3)  // Code examples are verbose
                    .with_num_instructions(15)  // Instructions matter
            )
        }
    }

    pub fn optimize_for_language(
        &self,
        llm: &impl Llm,
        language: &str,
        dataset: &Dataset,
    ) -> OptimizationResult {
        let base_instruction = format!(
            "Generate {language} code following best practices"
        );
        self.inner.optimize(llm, &base_instruction, dataset, |out, exp| {
            code_similarity(out, exp, language)
        })
    }
}
```

## Troubleshooting

### Low Scores

If optimization produces low scores:

1. **Check metric**: Is it too strict?
2. **Increase dataset size**: Need 20+ examples minimum
3. **Try different optimizer**: MIPRO often works when others don't
4. **Increase budget**: More trials/candidates = better results
5. **Validate examples**: Are reference outputs actually good?

### High Variance

If scores vary widely between runs:

1. **Use more examples**: Larger datasets are more stable
2. **Set random seed**: For reproducible results
3. **Use Ensemble**: Averages out variance
4. **Increase trials**: More exploration = more stable

### Slow Optimization

If optimization is too slow:

1. **Use caching**: Avoid redundant LLM calls
2. **Start with LabeledFewShot**: Fastest optimizer
3. **Reduce trials/candidates**: Trade quality for speed
4. **Use smaller dataset**: Subsample for initial testing

## References

- **DSPy Paper**: [Composing retrieval and language models](https://arxiv.org/abs/2310.03714)
- **COPRO**: [Automatic prompt optimization](https://arxiv.org/abs/2309.03409)
- **MIPRO**: [Multi-instruction prompt optimization](https://arxiv.org/abs/2406.11695)
- **Optimization Guide**: See `OPTIMIZATION_GUIDE.md` for Arc-based patterns
- **Examples**: See `examples/optimize.rs` for working code

## Next Steps

1. Start with `examples/optimize.rs` for a working example
2. Try `LabeledFewShot` on your dataset
3. Experiment with metrics until you get good scores
4. Graduate to advanced optimizers (MIPRO, SIMBA) as needed
5. Deploy with `Ensemble` for production robustness
