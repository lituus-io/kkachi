# Kkachi - Quick Start Guide

## Installation & Setup

### Prerequisites
- Rust 1.75+ (`rustup install stable`)
- OpenAI API key (for LM client examples)

### Build from Source

```bash
cd /Users/gatema/Desktop/git/lituus-io/kkachi

# Build entire workspace
cargo build --release

# Run tests (20 tests should pass)
cargo test --workspace

# Build specific components
cargo build -p kkachi --release
cargo build -p kkachi-cli --release
```

## Usage Examples

### 1. Basic Prediction

```rust
use kkachi::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a signature
    let sig = Signature::from_str("question -> answer")?;

    // Create example with demo
    let mut demo = Example::new();
    demo.insert_input("question", "What is 2+2?");
    demo.insert_output("answer", "4");

    // Note: You'd add an LM client here
    // let predict = Predict::new(sig).add_demo(demo).with_lm(lm);

    Ok(())
}
```

### 2. Use as Library

Add to your `Cargo.toml`:

```toml
[dependencies]
kkachi = { path = "/Users/gatema/Desktop/git/lituus-io/kkachi/crates/kkachi" }
kkachi-eval = { path = "/Users/gatema/Desktop/git/lituus-io/kkachi/crates/kkachi-eval" }
tokio = { version = "1", features = ["full"] }
```

### 3. Evaluation

```rust
use kkachi::*;
use kkachi_eval::*;
use std::sync::Arc;

// Create examples
let mut example = Example::new();
example.insert_output("answer", "42");

// Create prediction
let mut prediction = Prediction::new();
prediction.insert("answer", "42");

// Evaluate
let metric = metric::ExactMatch;
let result = metric.evaluate(&example, &prediction);

assert!(result.passed);
assert_eq!(result.score, 1.0);
```

### 4. Parallel Evaluation

```rust
use kkachi_eval::*;

let evaluator = ParallelEvaluator::new(Arc::new(metric::ExactMatch))
    .with_threads(8);

let result = evaluator.evaluate_predictions(&examples, &predictions)?;
println!("Accuracy: {:.2}%", result.accuracy() * 100.0);
```

### 5. Caching

```rust
use kkachi_cache::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Memory cache (LRU)
    let cache = MemoryCache::new(1000);

    let key = CacheKey::from_request("gpt-4", "test prompt", 0.0);
    cache.set(key.clone(), b"response".to_vec()).await?;

    let value = cache.get(&key).await;
    assert!(value.is_some());

    Ok(())
}
```

### 6. Build-Time Optimization

In your `build.rs`:

```rust
use kkachi_refine::PromptBuilder;

fn main() {
    PromptBuilder::new()
        .examples_from("training/examples.json")
        .output("src/generated/prompts.rs")
        .build()
        .unwrap();
}
```

### 7. CLI Usage

```bash
# Build CLI
cargo build --release --bin kkachi

# Use CLI
./target/release/kkachi refine --examples data.json --metric exact_match
./target/release/kkachi compile --input module.json --output optimized.bin
./target/release/kkachi eval --dataset test.json --parallel 16
```

## Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p kkachi
cargo test -p kkachi-eval

# With output
cargo test -- --nocapture

# Integration tests (when fully configured)
cargo test --test integration_signature
```

## Running Benchmarks

```bash
# Requires criterion
cargo bench --workspace
```

## Project Structure

```
kkachi/
├── crates/
│   ├── kkachi/           # Core library
│   ├── kkachi-client/    # LM client
│   ├── kkachi-cache/     # Caching
│   ├── kkachi-eval/      # Evaluation
│   ├── kkachi-refine/    # Build-time optimization
│   ├── kkachi-cli/       # CLI tool
│   └── kkachi-wasm/      # WASM bindings
├── tests/              # Integration tests
├── examples/           # Usage examples
├── benches/            # Performance benchmarks
└── *.md               # Documentation
```

## Common Tasks

### Create a Signature

```rust
// From string
let sig = Signature::from_str("input1, input2 -> output")?;

// Using builder
let sig = SignatureBuilder::new("Process data")
    .input(InputField::new("data", "Input data"))
    .output(OutputField::new("result", "Result"))
    .build();
```

### Work with Examples

```rust
let mut example = Example::new();
example.insert_input("question", "What is AI?");
example.insert_output("answer", "Artificial Intelligence");

// Clone for ownership
let owned = example.into_owned();
```

### Handle Errors

```rust
use kkachi::Result;

fn my_function() -> Result<()> {
    let sig = Signature::from_str("invalid format")
        .map_err(|e| {
            eprintln!("Signature error: {}", e);
            e
        })?;

    Ok(())
}
```

## Environment Setup

### For LM Client

```bash
export OPENAI_API_KEY="your-api-key"
```

### For Development

```bash
# Format code
cargo fmt

# Check for issues
cargo clippy -- -D warnings

# Build docs
cargo doc --open
```

## WASM Build

```bash
cd crates/kkachi-wasm

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
wasm-pack build --target web

# Build for node
wasm-pack build --target nodejs
```

## Troubleshooting

### Build Errors

```bash
# Clean and rebuild
cargo clean
cargo build --release
```

### Test Failures

```bash
# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run single test
cargo test test_name -- --exact
```

### Performance Issues

```bash
# Build with optimizations
cargo build --release

# Profile with flamegraph
cargo install flamegraph
cargo flamegraph --bin your_bin
```

## Next Steps

1. Read the [Architecture Guide](ARCHITECTURE.md)
2. Review [Test Report](TEST_REPORT.md)
3. Check [Implementation Status](IMPLEMENTATION_STATUS.md)
4. Explore examples in `examples/`
5. Run benchmarks with `cargo bench`

## Getting Help

- **Documentation**: See `*.md` files in root
- **Examples**: Check `examples/` directory
- **Tests**: Review test files for usage patterns
- **Source**: Read inline documentation in `src/`

## License

MIT OR Apache-2.0
