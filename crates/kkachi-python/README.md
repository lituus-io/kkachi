# Kkachi Python

High-performance LLM prompt optimization library with Python bindings.

## Installation

```bash
pip install kkachi
```

## Quick Start

```python
from kkachi import Kkachi, Cli, CliPipeline

# Define a generation function
def generate(iteration: int, feedback: str | None) -> str:
    if feedback:
        # Use feedback to improve the output
        return f"Improved code based on: {feedback}"
    return "def parse_url(url: str) -> dict: ..."

# Compose your own validator
validator = CliPipeline() \
    .stage("syntax", Cli("python").args(["-m", "py_compile"]).required()) \
    .stage("lint", Cli("ruff").args(["check"])) \
    .file_ext("py")

# Run refinement with your validator
result = Kkachi.refine("question -> code") \
    .domain("python") \
    .validate(validator) \
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
# Compose a Terraform validator
terraform_validator = CliPipeline() \
    .stage("fmt", Cli("terraform").args(["fmt", "-check"]).weight(0.2)) \
    .stage("validate", Cli("terraform").args(["validate"]).required()) \
    .file_ext("tf")

result = Kkachi.refine("requirement -> terraform_code") \
    .domain("terraform") \
    .storage("~/.kkachi/context.db") \
    .max_iterations(10) \
    .until_score(0.95) \
    .validate(terraform_validator) \
    .semantic_cache(True) \
    .similarity_threshold(0.90) \
    .auto_condense(True) \
    .with_chain_of_thought() \
    .run("Create S3 bucket with encryption", generate)
```

### Generic CLI Validators

Compose your own validators using `Cli` and `CliPipeline`:

```python
# Rust validator
rust_validator = CliPipeline() \
    .stage("format", Cli("rustfmt").args(["--check"]).weight(0.1)) \
    .stage("compile", Cli("rustc").args(["--emit=metadata"]).required()) \
    .stage("lint", Cli("cargo").args(["clippy"]).weight(0.3)) \
    .file_ext("rs")

# Python validator
python_validator = CliPipeline() \
    .stage("syntax", Cli("python").args(["-m", "py_compile"]).required()) \
    .stage("lint", Cli("ruff").args(["check"])) \
    .stage("types", Cli("mypy").args(["--ignore-missing-imports"])) \
    .file_ext("py")

# Kubernetes YAML validator
k8s_validator = CliPipeline() \
    .stage("lint", Cli("kubeval").args(["--strict"]).required()) \
    .file_ext("yaml")
```

### Similarity Weights

Configure multi-signal similarity scoring:

```python
from kkachi import SimilarityWeights

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

PolyForm Noncommercial 1.0.0

Copyright (c) 2025 lituus-io
Author: terekete <spicyzhug@gmail.com>
