# Kkachi - High-Performance LM Optimization Library

[![Rust](https://img.shields.io/badge/rust-1.75%2B-blue.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-PolyForm%20NC-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/kkachi.svg)](https://crates.io/crates/kkachi)
[![PyPI](https://img.shields.io/pypi/v/kkachi.svg)](https://pypi.org/project/kkachi/)

**Kkachi** is a zero-copy, embeddable Rust library for optimizing language model prompts and programs. Inspired by DSPy, Kkachi provides production-grade performance with deep async/await integration, WASM support, Python bindings, and build-time optimization capabilities.

## Features

- **Zero-Copy Architecture**: Lifetime-bound types eliminate unnecessary allocations
- **High Performance**: Tokio for async I/O, Rayon for CPU-bound parallelism
- **Type-Safe**: Strong typing with compile-time guarantees
- **Embeddable**: Easy to integrate as a library in any Rust project
- **WASM Ready**: Run optimizations in browsers, edge workers, and serverless
- **Python Bindings**: PyO3-based bindings for Python users
- **Build-Time Optimization**: Compile prompts at build time for zero runtime cost
- **Smart Caching**: Memory and disk-based caching with zero-copy serialization
- **Multiple Optimizers**: BootstrapFewShot, MIPRO, COPRO, SIMBA, and more
- **Recursive Language Prompting**: Iterative refinement with RAG integration

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
kkachi = "0.1"
kkachi-client = "0.1"
kkachi-eval = "0.1"
```

### Python

```bash
pip install kkachi
```

## Quick Start

### Rust - Declarative API (Recommended)

```rust
use kkachi::declarative::{pipeline, Cli, CliPipeline};

#[tokio::main]
async fn main() -> Result<()> {
    // Compose your own validator - fully generic
    let rust_validator = CliPipeline::new()
        .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
        .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
        .stage("lint", Cli::new("cargo").args(["clippy"]).weight(0.3))
        .file_ext("rs");

    // Run pipeline with your validator
    let result = pipeline("question -> code")
        .validate(rust_validator)
        .refine(5, 0.9)
        .run("Write a URL parser", &llm)
        .await?;

    println!("Score: {:.2}", result.score);
    println!("Answer: {}", result.answer());
    Ok(())
}
```

### Generic CLI Validation

Any CLI tool can be used for validation - no hardcoded language support:

```rust
// Terraform validation
let terraform = CliPipeline::new()
    .stage("fmt", Cli::new("terraform").args(["fmt", "-check"]).weight(0.2))
    .stage("validate", Cli::new("terraform").args(["validate"]).required())
    .file_ext("tf");

// Kubernetes YAML validation
let k8s = CliPipeline::new()
    .stage("lint", Cli::new("kubeval").args(["--strict"]).required())
    .file_ext("yaml");

// Pulumi with GCP credentials
let pulumi = Cli::new("pulumi")
    .args(["preview", "--non-interactive"])
    .env_inherit("GOOGLE_APPLICATION_CREDENTIALS")
    .env("PULUMI_CONFIG_PASSPHRASE", "")
    .file_ext("yaml");
```

### Python

```python
from kkachi import Kkachi, CliPipeline, Cli

def generate(iteration: int, feedback: str | None) -> str:
    if feedback:
        return f"Improved code based on: {feedback}"
    return "def parse_url(url: str) -> dict: ..."

# Compose your own validator
validator = CliPipeline() \
    .stage("syntax", Cli("python").args(["-m", "py_compile"]).required()) \
    .stage("lint", Cli("ruff").args(["check"])) \
    .file_ext("py")

result = Kkachi.refine("question -> code") \
    .domain("python") \
    .validate(validator) \
    .max_iterations(5) \
    .run("Write a URL parser", generate)

print(f"Score: {result.score}")
print(f"Answer: {result.answer}")
```

## Architecture

```
kkachi/
├── crates/
│   ├── kkachi/              # Core library (zero-copy types, traits)
│   ├── kkachi-client/       # LM client abstraction (async, pooling)
│   ├── kkachi-cache/        # Caching layer (memory + disk)
│   ├── kkachi-eval/         # Evaluation framework (rayon-based)
│   ├── kkachi-refine/       # Build-time optimization
│   ├── kkachi-cli/          # CLI tool
│   ├── kkachi-wasm/         # WASM bindings
│   └── kkachi-python/       # Python bindings
```

## Core Modules

### DSPy-Style Modules

- **ChainOfThought**: Step-by-step reasoning
- **ReAct**: Reasoning + Action framework
- **ProgramOfThought**: Code execution
- **BestOfN**: N-way selection
- **MultiChain**: Multi-branch ensemble

### Kkachimizers

- **BootstrapFewShot**: Few-shot learning
- **COPRO**: Contrastive Prompt Optimization
- **MIPRO**: Mixture of In-Context Prompts
- **KNNFewShot**: KNN-based example selection
- **LabeledFewShot**: Label-based selection
- **SIMBA**: Self-Improvement optimizer
- **Ensemble**: Combine multiple strategies

### Recursive Language Prompting

- **Self-Refinement**: LLM critiques and improves its own output
- **Convergence Detection**: Automatic stopping when quality plateaus
- **Human-in-the-Loop**: Manual intervention points
- **RAG Integration**: Retrieval-augmented generation

## Performance

Kkachi is designed for production use with:

- **Zero allocations** in prediction hot paths
- **< 1μs overhead** for signature access via lifetimes
- **Linear scaling** with Rayon on multi-core systems
- **Minimal memory footprint**: < 100MB for 10k examples

## CLI Usage

```bash
# Install CLI
cargo install kkachi-cli

# Interactive REPL
kkachi

# Available commands in REPL
> help
> load examples.json
> module cot "question -> answer"
> execute "What is 2+2?"
> save results.json
```

## WASM Support

Kkachi runs in browsers and edge runtimes:

```typescript
import init, { optimize } from 'kkachi-wasm';

await init();

// Run optimization client-side
const result = await optimize({
    examples: [...],
    metric: (pred, gold) => pred === gold
});
```

## Development

```bash
# Build all crates
cargo build --workspace --release

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace

# Build WASM
cd crates/kkachi-wasm && wasm-pack build --target web

# Build Python wheels
cd crates/kkachi-python && maturin build --release
```

## Examples

See [`examples/`](examples/) directory for:

- `declarative_api.rs` - Fluent builder API
- `template_refinement.rs` - Template-based refinement
- `complete_dspy_pipeline.rs` - Full DSPy-style pipeline
- `repl_hitl_demo.rs` - Human-in-the-loop review demo

## License

Copyright (c) Lituus-io. All rights reserved.

Author: terekete <spicyzhug@gmail.com>

Licensed under PolyForm Noncommercial 1.0.0. See [LICENSE](LICENSE) for details.
