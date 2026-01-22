# Kkachi Python

High-performance LLM prompt optimization library with Python bindings.

## Installation

```bash
pip install kkachi
```

## Quick Start

```python
from kkachi import Kkachi, ToolType, SimilarityWeights

# Define a generation function
def generate(iteration: int, feedback: str | None) -> str:
    if feedback:
        # Use feedback to improve the output
        return f"Improved code based on: {feedback}"
    return "def parse_url(url: str) -> dict: ..."

# Simple usage
result = Kkachi.refine("question -> code") \
    .domain("rust") \
    .critic_rust() \
    .max_iterations(5) \
    .run("Write a URL parser", generate)

print(f"Score: {result.score}")
print(f"Iterations: {result.iterations}")
print(f"Answer: {result.answer}")
```

## Features

### Declarative API

The `Kkachi.refine()` builder provides a fluent interface for configuring refinement pipelines:

```python
result = Kkachi.refine("requirement -> terraform_code") \
    .domain("terraform") \
    .storage("~/.kkachi/context.db") \
    .max_iterations(10) \
    .until_score(0.95) \
    .critic_terraform() \
    .semantic_cache(True) \
    .similarity_threshold(0.90) \
    .auto_condense(True) \
    .with_chain_of_thought() \
    .run("Create S3 bucket with encryption", generate)
```

### Built-in Critics

- `critic_rust()` - Rust (cargo check, test, clippy)
- `critic_python()` - Python (py_compile, pytest, ruff)
- `critic_terraform()` - Terraform (fmt, validate, plan)
- `critic_pulumi()` - Pulumi (preview, policy validate)
- `critic_kubernetes()` - Kubernetes (kubectl dry-run)
- `critic_heuristic(min_length, max_length)` - Simple heuristic checks

### Similarity Weights

Configure multi-signal similarity scoring:

```python
weights = SimilarityWeights(
    embedding=0.50,   # Semantic similarity
    keyword=0.25,     # TF-IDF keyword matching
    metadata=0.15,    # Tag overlap
    hierarchy=0.10,   # Category distance
)

result = Kkachi.refine("q -> a") \
    .similarity_weights(weights) \
    .run(question, generate)
```

### Vector Store

Use the in-memory vector store for few-shot examples:

```python
from kkachi import InMemoryVectorStore

store = InMemoryVectorStore(dimension=64)
store.add("doc1", "Example code for parsing URLs")
store.add("doc2", "Another example with error handling")

results = store.search("URL parsing", k=5)
for r in results:
    print(f"{r.id}: {r.score:.3f}")
```

## Development

Build from source:

```bash
cd crates/kkachi-python
maturin develop  # Development build
maturin build --release  # Release wheel
```

## License

MIT License

Copyright (c) 2025 lituus-io
Author: terekete <spicyzhug@gmail.com>
