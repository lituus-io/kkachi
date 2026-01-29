# Kkachi

[![CI](https://github.com/lituus-io/kkachi/actions/workflows/ci.yml/badge.svg)](https://github.com/lituus-io/kkachi/actions/workflows/ci.yml)
[![Python Package](https://github.com/lituus-io/kkachi/actions/workflows/publish-python.yml/badge.svg)](https://github.com/lituus-io/kkachi/actions/workflows/publish-python.yml)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/kkachi/)
[![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue)](LICENSE)

High-performance LLM prompt optimization library with iterative refinement, validation, and composable pipelines.

## Installation

```toml
[dependencies]
kkachi = { version = "0.1", features = ["api"] }
```

## Quick Start

```rust
use kkachi::recursive::prelude::*;

fn main() -> anyhow::Result<()> {
    let llm = ApiLlm::from_env()?; // uses ANTHROPIC_API_KEY or OPENAI_API_KEY

    let result = refine(&llm, "Write a URL parser in Rust")
        .validate(checks().require("fn ").require("Result").forbid(".unwrap()"))
        .max_iter(5)
        .go()?;

    println!("Score: {:.0}% after {} iterations", result.score * 100.0, result.iterations);
    Ok(())
}
```

## Features

### Iterative Refinement

Repeatedly generate and validate until constraints are met:

```rust
let result = refine(&llm, "Write a binary search in Rust")
    .validate(checks()
        .require("fn ")
        .require("-> Option<usize>")
        .forbid(".unwrap()")
        .min_len(100))
    .max_iter(5)
    .target(1.0)
    .go()?;
```

### Best of N

Generate N candidates and select the highest-scoring:

```rust
let (result, pool) = best_of(&llm, "Write a haiku about Rust")
    .n(5)
    .metric(|output| {
        let lines = output.trim().lines().count();
        if lines == 3 { 1.0 } else { 0.0 }
    })
    .validate(checks().min_len(10))
    .go_with_pool();

println!("Best: {} (score: {:.2})", result.output, result.score);
println!("Pool mean: {:.2}, std: {:.2}", pool.stats().mean, pool.stats().std_dev);
```

### Ensemble Voting

Multiple chains vote on the answer:

```rust
let (result, consensus) = ensemble(&llm, "What is the capital of Australia?")
    .n(7)
    .aggregate(Aggregate::MajorityVote)
    .go_with_consensus();

println!("Answer: {} (agreement: {:.0}%)", result.output, consensus.agreement_ratio() * 100.0);
```

### Chain of Thought

Step-by-step reasoning with automatic answer extraction:

```rust
// With answer marker - extracts the answer
let result = reason(&llm, "A farmer has 17 sheep. All but 9 die. How many are left?")
    .validate(checks().regex(r"\d+"))
    .max_iter(3)
    .go();

println!("Reasoning:\n{}", result.reasoning());
println!("Answer: {}", result.output);  // "9" (extracted)

// Multi-line content - preserves full response automatically
let result = reason(&llm, "Generate a YAML config template")
    .validate(checks().require("name:").require("config:").min_len(50))
    .go();

println!("Generated:\n{}", result.output);  // Full YAML preserved
```

### ReAct Agent

Tool-calling agent with reasoning loop:

```rust
let search = tool("search")
    .description("Search the web for information")
    .execute(|query| Ok(format!("Result for '{}': 14 million", query)));

let calc = tool("calculator")
    .description("Evaluate a math expression")
    .execute(|expr| Ok("7000000".to_string()));

let result = agent(&llm, "What is Tokyo's population divided by 2?")
    .tool(&search)
    .tool(&calc)
    .max_steps(10)
    .go();

println!("Answer: {}", result.output);
```

### Program of Thought

Generate and execute code:

```rust
let result = program(&llm, "Calculate the first 10 Fibonacci numbers")
    .executor(bash_executor())
    .validate(checks().regex(r"\d+"))
    .max_iter(3)
    .go();

println!("Output: {}", result.output);
```

### Pipeline with Fan-Out

Chain operations with parallel branches:

```rust
let result = pipeline(&llm, "Write a palindrome checker")
    .fan_out(
        vec![
            BranchBuilder::new("rust").refine(checks().require("fn ").require("bool")),
            BranchBuilder::new("python").refine(checks().require("def ").require("return")),
        ],
        MergeStrategy::BestScore,
    )
    .go();

println!("Best implementation:\n{}", result.output);
```

### Validators

Pattern-based, CLI, and semantic validators with weighted and batch variants:

```rust
// Single checks
let v = checks()
    .require("fn ")
    .forbid(".unwrap()")
    .regex(r"Result<.*>")
    .min_len(50)
    .max_len(500);

// Weighted checks (control scoring impact per check)
let v = checks()
    .require_weighted("fn ", 2.0)          // 2x weight for function signature
    .require_weighted("Result", 1.5)       // 1.5x weight for error handling
    .forbid_weighted("panic!", 3.0)        // 3x penalty for panic
    .regex_weighted(r"pub fn \w+", 1.0)    // standard weight for public API
    .min_len_weighted(100, 0.5)            // half weight for length
    .max_len_weighted(1000, 0.5);

// Batch checks (multiple patterns at once)
let v = checks()
    .require_all(["fn ", "->", "Result"])
    .forbid_all([".unwrap()", "panic!", "todo!"])
    .regex_all([r"fn \w+", r"-> \w+"]);

// Batch + weighted
let v = checks()
    .require_all_weighted(["fn ", "Result", "impl"], 2.0)
    .forbid_all_weighted([".unwrap()", "panic!"], 3.0)
    .regex_all_weighted([r"///", r"#\[doc"], 0.5);

// CLI validation (external tools)
let v = cli("rustfmt").args(&["--check"])
    .then("rustc").args(&["--emit=metadata", "-o", "/dev/null"]).required()
    .ext("rs");

// LLM-as-judge
let v = semantic(&llm)
    .criterion("Code is idiomatic Rust")
    .criterion("Error handling is complete")
    .threshold(0.8)
    .build();

// Compose validators
let strict = checks().require("fn ").and(cli("rustfmt").args(&["--check"]).ext("rs"));
```

### Composable LLM Wrappers

Stack caching, rate limiting, and retry:

```rust
let llm = ApiLlm::from_env()?
    .with_cache(100)                                        // LRU cache (innermost)
    .with_rate_limit_config(RateLimitConfig::new(5.0).with_burst(3))  // token bucket
    .with_retry(3);                                         // exponential backoff (outermost)
```

### Multi-Objective Optimization

Optimize across competing objectives with Pareto fronts:

```rust
let validator = multi_objective()
    .scalarize(Scalarization::Chebyshev)
    .objectives([
        (Objective::new("correctness").weight(2.0).target(0.9),
         checks().require("fn ").require("Result")),
        (Objective::new("brevity").weight(1.0).target(0.8),
         checks().max_len(300)),
        (Objective::new("safety").weight(1.5).target(0.9),
         checks().forbid(".unwrap()").forbid("panic!")),
    ]);

let result = refine_pareto_sync(&llm, "Write a file reader", &validator, 5);
println!("Pareto front: {} solutions", result.front.len());
```

### Memory / RAG

Store and retrieve examples for few-shot learning, with optional persistent storage:

```rust
use kkachi::recursive::prelude::*;

// In-memory (default)
let mut mem = memory();
mem.add("fn read_file(p: &str) -> io::Result<String> { fs::read_to_string(p) }");
mem.add("fn parse_json(s: &str) -> Result<Value, _> { serde_json::from_str(s) }");

// Persistent storage (requires 'storage' feature)
let mut mem = memory()
    .persist("./knowledge_base.db")?;

mem.add("fn config_reader() -> Result<Config> { /* ... */ }");

// Full CRUD operations
let doc_id = mem.add("Example code");
let content = mem.get(&doc_id);          // Read
mem.update(&doc_id, "Updated code");     // Update
mem.remove(&doc_id);                     // Delete

let result = refine(&llm, "Write a config file reader")
    .memory(&mut mem)
    .k(3)
    .validate(checks().require("fn ").require("Result"))
    .learn_above(0.8)
    .go()?;

// Data persists across program restarts
```

**Python**:
```python
from kkachi import Memory

# Persistent storage
mem = Memory().persist("./knowledge.db")

# CRUD operations
doc_id = mem.add("Example code here")
content = mem.get(doc_id)                 # Read
mem.update(doc_id, "Updated content")     # Update
mem.remove(doc_id)                        # Delete

# Search with semantic similarity
results = mem.search("config reader", k=3)
```

### LLM Optimization (Python)

Optimize API calls with caching, rate limiting, and retry:

```python
from kkachi import ApiLlm

# Create LLM client with full optimization stack
llm = (ApiLlm.from_env()
       .with_cache(100)          # Cache 100 responses (LRU)
       .with_rate_limit(10.0)    # Max 10 requests/second
       .with_retry(3))           # Retry up to 3 times

# Optimizations work transparently
response = llm.generate("Your prompt here")
```

**Individual Optimizations**:

```python
# Cache only - reduce API costs
llm = ApiLlm.from_env().with_cache(50)

# Rate limiting only - prevent 429 errors
llm = ApiLlm.anthropic(api_key, model).with_rate_limit(5.0)

# Retry only - handle transient failures
llm = ApiLlm.openai(api_key, model).with_retry(5)
```

**Benefits**:
- **`with_cache(capacity)`**: LRU caching for identical prompts → reduce API costs
- **`with_rate_limit(rps)`**: Token bucket rate limiting → prevent 429 errors
- **`with_retry(max_retries)`**: Exponential backoff retry → handle transient failures

**Recommended Patterns**:
- Development: `llm.with_cache(50)` (fast iteration)
- Testing: `llm.with_cache(50).with_retry(3)` (repeatable + resilient)
- Production: `llm.with_cache(100).with_rate_limit(10.0).with_retry(3)` (full optimization)

## API Reference

### Entry Points

| Function | Description |
|----------|-------------|
| `refine(llm, prompt)` | Iterative refinement with validation |
| `best_of(llm, prompt).n(N)` | Best of N candidate selection |
| `ensemble(llm, prompt).n(N)` | Multi-chain ensemble voting |
| `reason(llm, prompt)` | Chain of Thought reasoning |
| `agent(llm, goal)` | ReAct agent with tools |
| `program(llm, problem)` | Code generation + execution |
| `pipeline(llm, prompt)` | Composable pipeline with fan-out |
| `refine_pareto_sync(llm, prompt, validator, iter)` | Multi-objective Pareto optimization |

### Validators

| Function | Variants |
|----------|----------|
| `require(pat)` | `require_weighted(pat, w)`, `require_all(pats)`, `require_all_weighted(pats, w)` |
| `forbid(pat)` | `forbid_weighted(pat, w)`, `forbid_all(pats)`, `forbid_all_weighted(pats, w)` |
| `regex(pat)` | `regex_weighted(pat, w)`, `regex_all(pats)`, `regex_all_weighted(pats, w)` |
| `min_len(n)` | `min_len_weighted(n, w)` |
| `max_len(n)` | `max_len_weighted(n, w)` |
| `cli(cmd)` | `.args()`, `.ext()`, `.then()`, `.required()`, `.weight()` |
| `semantic(llm)` | `.criterion()`, `.threshold()`, `.build()` |
| `v1.and(v2)` | Both must pass |
| `v1.or(v2)` | At least one passes |

### LLM Wrappers

| Method | Description |
|--------|-------------|
| `.with_cache(capacity)` | LRU response cache |
| `.with_rate_limit(rps)` | Token-bucket rate limiting |
| `.with_retry(max_retries)` | Exponential backoff retry |

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key (fallback) |
| `KKACHI_MODEL` | Override model name |
| `KKACHI_BASE_URL` | Override API endpoint |

### Explicit Providers

```rust
let llm = ApiLlm::from_env()?;                      // auto-detect from env
let llm = ApiLlm::anthropic("sk-...", "model-id");   // Anthropic
let llm = ApiLlm::openai("sk-...", "gpt-4o");        // OpenAI
```

## Cargo Features

| Feature | Description |
|---------|-------------|
| `api` | Real LLM API client (Anthropic, OpenAI) |
| `native` | Full async runtime with tokio (default) |
| `storage` | DuckDB persistent storage for Memory |
| `tracing` | Instrumentation with the `tracing` crate |
| `tiktoken` | OpenAI tokenization |
| `huggingface` | HuggingFace tokenizers |
| `embeddings-onnx` | ONNX-based semantic embeddings |

## License

PolyForm Noncommercial 1.0.0
