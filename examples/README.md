# Kkachi Examples

Examples demonstrating the kkachi recursive language prompting library.

## Prerequisites

- **Rust examples:** Rust toolchain (stable), plus an LLM API key
- **Python examples:** `pip install kkachi`, plus an LLM API key
- **Pipeline examples:** Additionally require `pulumi`, `pulumi-gcp`, and GCP credentials

Set one of these environment variables for LLM access:

```bash
export KKACHI_API_KEY="your-api-key"
# or
export OPENAI_API_KEY="sk-..."
```

## Learning Progression

Start with the simple examples and work your way up.

### 1. Basics (No LLM needed)

| Example | Language | Description |
|---------|----------|-------------|
| `validators.rs` | Rust | Validator composition — test validation rules locally |

```bash
cargo run --example validators
```

### 2. Simple LLM Examples

| Example | Language | Description |
|---------|----------|-------------|
| `refine.rs` / `python/refine.py` | Rust / Python | Iterative refinement with validation |
| `reason.rs` / `python/reason.py` | Rust / Python | Chain-of-Thought reasoning |
| `best_of.rs` / `python/best_of.py` | Rust / Python | Best-of-N candidate selection |
| `ensemble.rs` / `python/ensemble.py` | Rust / Python | Multi-chain ensemble voting |

```bash
# Rust
cargo run --example refine --features api
cargo run --example reason --features api
cargo run --example best_of --features api
cargo run --example ensemble --features api

# Python
python examples/python/refine.py
python examples/python/reason.py
python examples/python/best_of.py
python examples/python/ensemble.py
```

### 3. Intermediate

| Example | Language | Description |
|---------|----------|-------------|
| `rate_limit.rs` | Rust | Composable middleware (cache, rate limit, retry) |
| `fan_out.rs` | Rust | Parallel fan-out with merge strategies |
| `optimize.rs` | Rust | Prompt optimization via dataset evaluation |
| `pareto.rs` | Rust | Multi-objective Pareto optimization |
| `python/api_llm_basic.py` | Python | All LLM client initialization patterns |

```bash
cargo run --example rate_limit --features api
cargo run --example fan_out --features api
cargo run --example optimize --features api
cargo run --example pareto --features api
```

### 4. Production Pipelines

These require Pulumi, GCP credentials, and environment configuration.

| Example | Language | Description |
|---------|----------|-------------|
| `pulumi_table_pipeline.rs` | Rust | 7-step BigQuery table pipeline with RAG + CLI validation |
| `python/pulumi_cli_validator_test.py` | Python | CLI validator test suite with Pulumi |
| `python/pulumi_table_pipeline.py` | Python | 7-step BigQuery table pipeline (composable orchestration) |
| `python/pulumi_template_pipeline.py` | Python | BigQuery dataset template optimization |
| `python/pulumi_tree_pipeline.py` | Python | Tree-walk recursive template generation |

Pipeline examples use environment variables for configuration:

```bash
export SOURCE_FOLDER="/path/to/pulumi/templates"
export LLM_API_KEY_PATH="/path/to/api/key"
export GCP_CREDS="/path/to/gcp/credentials.json"
export KKACHI_RAG_DB="./template_knowledge.db"
```

## Directory Structure

```
examples/
├── README.md                          # This file
├── Cargo.toml                         # Rust examples workspace
├── refine.rs                          # Basic refinement
├── reason.rs                          # Chain-of-thought
├── best_of.rs                         # Best-of-N selection
├── ensemble.rs                        # Ensemble voting
├── validators.rs                      # Validator composition (no LLM)
├── fan_out.rs                         # Parallel fan-out
├── optimize.rs                        # Prompt optimization
├── pareto.rs                          # Multi-objective optimization
├── rate_limit.rs                      # Composable middleware
├── pulumi_table_pipeline.rs           # Production pipeline (Rust)
└── python/
    ├── refine.py                      # Basic refinement
    ├── reason.py                      # Chain-of-thought
    ├── best_of.py                     # Best-of-N selection
    ├── ensemble.py                    # Ensemble voting
    ├── api_llm_basic.py               # LLM client patterns
    ├── _pipeline_common.py            # Shared pipeline utilities
    ├── pulumi_cli_validator_test.py   # CLI validator tests
    ├── pulumi_table_pipeline.py       # BigQuery table pipeline
    ├── pulumi_template_pipeline.py    # Dataset template pipeline
    └── pulumi_tree_pipeline.py        # Tree-walk pipeline
```
