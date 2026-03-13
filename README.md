# Kkachi

[![CI](https://github.com/lituus-io/kkachi/actions/workflows/ci.yml/badge.svg)](https://github.com/lituus-io/kkachi/actions/workflows/ci.yml)
[![Security](https://github.com/lituus-io/kkachi/actions/workflows/security.yml/badge.svg)](https://github.com/lituus-io/kkachi/actions/workflows/security.yml)
[![Release](https://github.com/lituus-io/kkachi/actions/workflows/release.yml/badge.svg)](https://github.com/lituus-io/kkachi/actions/workflows/release.yml)
[![crates.io](https://img.shields.io/crates/v/kkachi.svg)](https://crates.io/crates/kkachi)
[![PyPI](https://img.shields.io/pypi/v/kkachi.svg)](https://pypi.org/project/kkachi/)

High-performance LLM prompt optimization library with composable pipelines.

## Features

- **Composable Pipelines** — Chain steps: refine, best_of, ensemble, reason, extract, map
- **Concurrent Execution** — Run multiple pipelines concurrently with shared LLM and rate limiting
- **Step Combinators** — then, race, par, retry, fallback, when
- **DSPy-Style Modules** — Chain of Thought, Best of N, Ensemble, Program of Thought, ReAct Agent
- **Jinja2 Templates** — Dynamic prompt generation
- **CLI Validators** — External tool validation with composition (.and_(), .or_())
- **Memory & RAG** — Persistent vector store with DuckDB
- **Pattern Validation** — Regex, substring, length checks
- **LLM-as-Judge** — Semantic validation
- **Multi-Objective Optimization** — Pareto-optimal prompt tuning
- **Skills & Defaults** — Reusable instruction injection and runtime substitution
- **Zero-Copy Core** — GATs over async/await, lifetimes over Arc, minimal cloning

## Python Installation

```bash
pip install kkachi
```

## Quick Start (Python)

```python
from kkachi import pipeline, concurrent, reason, Checks, ApiLlm

llm = ApiLlm.auto()

# Simple pipeline
result = pipeline(llm, "Write a URL parser in Rust") \
    .refine(Checks().has("fn parse"), max_iter=5, target=1.0) \
    .extract("rust") \
    .go()

# Concurrent pipelines
results = (
    concurrent(llm)
    .task("task_a", "Write tests", lambda p: p.refine(checks_a))
    .task("task_b", "Write docs", lambda p: p.refine(checks_b))
    .max_concurrency(2)
    .go()
)
```

## Quick Start (Rust)

```rust
use kkachi::prelude::*;

let llm = ApiLlm::auto()?;

// Composable pipeline
let result = pipeline(&llm, "Write a URL parser")
    .refine_with(checks, 5, 1.0)
    .extract("rust")
    .go();

// Concurrent execution
let results = ConcurrentRunner::new(&llm)
    .task("tests", |llm| Pipeline::new_owned(llm, prompt_a).refine_with(checks_a, 5, 1.0))
    .task("docs", |llm| Pipeline::new_owned(llm, prompt_b).refine_with(checks_b, 3, 1.0))
    .max_concurrency(2)
    .go();
```

## Repository Structure

- `crates/kkachi` — Core Rust library
- `crates/kkachi-python` — Python bindings (PyO3 + maturin)
- `examples/` — Rust and Python usage examples
- `benches/` — Benchmarks

## License

Dual-licensed: AGPL-3.0-or-later for open source use, commercial license available.
See [LICENSE](LICENSE) for details.
