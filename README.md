# Kkachi

[![CI](https://github.com/lituus-io/kkachi/actions/workflows/ci.yml/badge.svg)](https://github.com/lituus-io/kkachi/actions/workflows/ci.yml)
[![Python Package](https://github.com/lituus-io/kkachi/actions/workflows/publish-python.yml/badge.svg)](https://github.com/lituus-io/kkachi/actions/workflows/publish-python.yml)
[![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/kkachi.svg)](https://crates.io/crates/kkachi)
[![PyPI](https://img.shields.io/pypi/v/kkachi.svg)](https://pypi.org/project/kkachi/)

High-performance LLM prompt optimization library with DSPy-style patterns for Rust and Python.

## Features

### Core Capabilities

- 🎯 **DSPy-Style Modules** - Chain of Thought, Best of N, Ensemble, Refinement, ReAct Agent
- 🔄 **Iterative Refinement** - Adaptive improvement with validation feedback
- 📊 **Multi-Candidate Selection** - Best-of-N with scoring and pool statistics
- 🤖 **LLM-as-Judge** - Semantic validation and scoring
- 🔍 **Prompt Optimization** - LabeledFewShot, KNN, COPRO, MIPRO, SIMBA, Ensemble

### Validation & Quality

- ✅ **Pattern Validators** - Regex, substring, length checks
- 🛠️ **CLI Validators** - External tool validation (rustfmt, clippy, etc.)
- 🔗 **Validator Composition** - Combine validators with `.and_()` and `.or_()`
- 📝 **Jinja2 Templates** - Dynamic prompt generation with context

### Memory & Storage

- 💾 **Persistent Memory** - Vector store with DuckDB backend
- 🔎 **Semantic Search** - RAG-style document retrieval
- ⚡ **High Performance** - Zero-copy optimizations with Arc
- 🏷️ **Tagging & Filtering** - Organize and search by tags

### Developer Experience

- 🚀 **Simple API** - Fluent builder pattern
- 📚 **Comprehensive Examples** - Rust and Python
- 🧪 **Well Tested** - 699 tests with 100% pass rate
- 🔧 **Production Ready** - Zero-copy, benchmarked performance

## Installation

### Python

```bash
pip install kkachi
```

### Rust

```toml
[dependencies]
kkachi = "0.4"
```

## Quick Start

### Rust

```rust
use kkachi::recursive::{refine, checks, ApiLlm};

fn main() -> anyhow::Result<()> {
    // Create LLM client (auto-detects API keys)
    let llm = ApiLlm::from_env()?;

    // Iterative refinement with validation
    let result = refine(
        &llm,
        "Write a Rust function that parses URLs"
    )
    .validate(
        checks()
            .require("fn ")
            .require("Result")
            .forbid(".unwrap()")
            .min_len(50)
    )
    .max_iter(5)
    .target(0.9)
    .go()?;

    println!("Score: {:.0}%", result.score * 100.0);
    println!("Output:\n{}", result.output);
    Ok(())
}
```

### Python

```python
from kkachi import refine, Checks, ApiLlm

# Create LLM client
llm = ApiLlm.from_env()

# Iterative refinement with validation
result = refine(llm, "Write a Python function that parses URLs") \
    .validate(
        Checks()
            .require("def ")
            .require("return")
            .forbid("eval(")
            .min_len(50)
    ) \
    .max_iter(5) \
    .target(0.9) \
    .go()

print(f"Score: {result.score * 100:.0f}%")
print(f"Output:\n{result.output}")
```

See [PYTHON_EXAMPLES.md](PYTHON_EXAMPLES.md) for more Python examples.

## Usage Examples

### Best-of-N Candidate Selection

Generate multiple candidates and select the best based on custom metrics:

```rust
use kkachi::recursive::{best_of, checks, ApiLlm};

let llm = ApiLlm::from_env()?;

let (result, pool) = best_of(&llm, "Write a haiku about Rust")
    .n(5)
    .metric(|output| {
        let lines: Vec<_> = output.trim().lines().collect();
        if lines.len() == 3 { 0.8 } else { 0.2 }
    })
    .validate(checks().min_len(10).forbid("```"))
    .go_with_pool();

println!("Best haiku (score={:.2}):\n{}", result.score, result.output);
println!("Pool stats: {} candidates, mean={:.2}", pool.count(), pool.mean());
```

### Memory & RAG

Store and retrieve documents with semantic search:

```rust
use kkachi::recursive::memory;

let mem = memory();

// Add documents
mem.add("Rust uses ownership for memory safety");
mem.add("Python uses reference counting");

// Search with semantic similarity
let results = mem.search("memory management", 2)?;
for recall in results {
    println!("Score: {:.2} - {}", recall.score, recall.content);
}

// Persist to disk
let mem = memory().persist("./my_knowledge.db")?;
```

### CLI Validators

Validate output with external tools:

```rust
use kkachi::recursive::{refine, cli, checks, ApiLlm};

let llm = ApiLlm::from_env()?;

let result = refine(&llm, "Write a Rust URL parser")
    .validate(
        cli("rustfmt")
            .arg("--check")
            .stdin()
            .and(checks().forbid(".unwrap()"))
    )
    .go()?;
```

### Jinja2 Templates

Dynamic prompt generation with context:

```rust
use kkachi::recursive::{JinjaTemplate, JinjaFormatter};

let template = JinjaTemplate::from_str("task", r#"
Task: {{ task }}

Context:
{% for item in context %}
- {{ item }}
{% endfor %}

Output format: {{ format }}
"#)?;

let formatter = JinjaFormatter::new()
    .template(template)
    .context([
        ("task", "Parse URL"),
        ("context", vec!["Use Result", "No unwrap()"]),
        ("format", "Rust code"),
    ]);
```

### Ensemble & Voting

Combine multiple strategies:

```rust
use kkachi::recursive::{ensemble, checks, ApiLlm};

let llm = ApiLlm::from_env()?;

let result = ensemble(&llm, "Explain ownership in Rust")
    .strategies(vec![
        "Explain with analogies",
        "Explain with code examples",
        "Explain step-by-step",
    ])
    .vote_threshold(2)
    .validate(checks().min_len(100))
    .go()?;
```

### Rate Limiting

Control API request rates:

```rust
use kkachi::recursive::{ApiLlm, RateLimitedLlm};

let llm = ApiLlm::from_env()?;
let limited = llm.rate_limit()
    .per_minute(20)
    .burst(5);

// Use like any other LLM
let result = limited.generate("Hello", "", None).await?;
```

## What's New in v0.4.0

- ⚡ **Zero-copy optimizations** - 10-50x faster cache hits and retrievals
- 📊 **Performance baselines** - Comprehensive benchmarking suite
- 🧪 **Extended test coverage** - 699 tests with 100% pass rate
- 📚 **Enhanced documentation** - Complete optimizer guide and examples
- ✅ **Production ready** - Zero compiler warnings, full CI coverage

## Repository Structure

```
kkachi/
├── crates/
│   ├── kkachi/          # Core Rust library
│   └── kkachi-python/   # Python bindings
├── examples/            # Rust examples
├── benches/             # Performance benchmarks
└── tests/               # Integration tests
```

## API Compatibility

### Supported LLM Providers

- **Anthropic Claude** - Claude 3/4 models
- **OpenAI** - GPT-4, GPT-3.5
- **Claude Code CLI** - Local CLI (no API key needed)
- **Custom Endpoints** - Any OpenAI-compatible API

### Environment Variables

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Or use Claude Code CLI (no key needed)
# Just have 'claude' in your PATH
```

## Performance

Optimized for production workloads:

- **Cache hits**: ~100ns (zero-copy with Arc)
- **Document retrieval**: 27ns (10-50x faster)
- **Search (k=5, n=100)**: ~7µs
- **Memory efficient**: Zero-copy architecture

See benchmarks: `cargo bench`

## Documentation

- **Rust API Docs**: [docs.rs/kkachi](https://docs.rs/kkachi)
- **Python Examples**: [PYTHON_EXAMPLES.md](PYTHON_EXAMPLES.md)
- **Rust Examples**: [examples/](examples/)
- **Python Package**: [crates/kkachi-python/README.md](crates/kkachi-python/README.md)

## Contributing

Contributions welcome! Please ensure:
- Tests pass: `cargo test`
- Format code: `cargo fmt`
- No warnings: `cargo clippy`

## License

PolyForm Noncommercial 1.0.0

See [LICENSE](LICENSE) for details.

## Links

- **GitHub**: https://github.com/lituus-io/kkachi
- **Crates.io**: https://crates.io/crates/kkachi
- **PyPI**: https://pypi.org/project/kkachi/
- **Issues**: https://github.com/lituus-io/kkachi/issues
