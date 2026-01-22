# Kkachi Architecture

## Overview

Kkachi is a high-performance Rust library for optimizing language model prompts and programs. It implements equivalent functionality to DSPy with significant performance and usability enhancements.

## Design Principles

1. **Zero-Copy First**: Extensive use of lifetimes (`'a`) to avoid allocations
2. **Type Safety**: Compile-time guarantees through Rust's type system
3. **Async by Default**: Tokio for I/O, Rayon for CPU parallelism
4. **Embeddable**: Library-first design for easy integration
5. **Multi-Runtime**: Native, WASM, and edge support

## Crate Structure

### Core Library (`kkachi`)

The foundational crate providing zero-copy primitives:

- **Signature System**: Lifetime-bound field definitions
- **Module Trait**: Core abstraction for executable programs
- **Prediction Types**: Zero-copy output structures
- **Example Types**: Training/evaluation data structures

**Key Types:**
```rust
pub struct Signature<'a> {
    instructions: Cow<'a, str>,
    input_fields: Vec<Field<'a>>,
    output_fields: Vec<Field<'a>>,
}

pub struct Field<'a> {
    name: Cow<'a, str>,
    desc: Cow<'a, str>,
    prefix: Cow<'a, str>,
    field_type: FieldType,
}
```

### LM Client (`kkachi-client`)

Async language model client with:

- **Provider Abstraction**: Unified interface for OpenAI, Anthropic, local models
- **Connection Pooling**: Semaphore-based concurrency control
- **Request/Response Types**: Zero-copy message structures
- **Retry Logic**: Exponential backoff

**Design:**
- Uses `reqwest` for HTTP with connection reuse
- Pooling via `tokio::sync::Semaphore` for rate limiting
- Provider trait allows custom backends

### Cache Layer (`kkachi-cache`)

Multi-tier caching system:

- **Memory Cache**: LRU cache with `lru` crate
- **Concurrent Cache**: Lock-free with `DashMap`
- **Disk Cache**: Bincode serialization to filesystem
- **Cache Keys**: Hash-based with model and temperature

**Performance:**
- Zero-copy reads from cache when possible
- Async I/O for disk operations
- Thread-safe access patterns

### Evaluation (`kkachi-eval`)

Parallel evaluation framework:

- **Metric Trait**: Extensible evaluation functions
- **Rayon Integration**: CPU-parallel evaluation
- **Built-in Metrics**: ExactMatch, F1Score
- **Result Aggregation**: Statistics and reporting

**Key Feature:**
- `ParallelEvaluator` uses Rayon's `par_iter()` for multi-threaded metric computation
- Fold-reduce pattern for efficient aggregation

### Refinement (`kkachi-refine`)

Build-time optimization:

- **Prompt Builder**: DSL for optimization configuration
- **Code Generation**: Const-based optimized prompts
- **Incremental Learning**: Update prompts from failures

**Usage in build.rs:**
```rust
PromptBuilder::new()
    .examples_from("training/*.json")
    .output("src/generated/optimized.rs")
    .build();
```

### CLI (`kkachi-cli`)

Command-line interface with:

- **Refine**: Interactive prompt optimization
- **Compile**: Save optimized modules
- **Eval**: Parallel evaluation on datasets
- **Serve**: HTTP/gRPC API server

**Built with:**
- `clap` for argument parsing
- Future: `ratatui` for TUI

### WASM (`kkachi-wasm`)

WebAssembly bindings for:

- **Edge Deployment**: Cloudflare Workers, Fastly
- **Browser Usage**: Client-side optimization
- **Sandboxed Execution**: Safe multi-tenancy

**Export:**
```rust
#[wasm_bindgen]
pub struct WasmOptimizer {
    // Zero-copy WASM interface
}
```

## Performance Optimizations

### Memory Management

1. **String Interning** (planned): Reuse common strings
2. **Cow<'a, str>**: Zero-copy when possible, owned when needed
3. **Bump Allocators** (planned): Fast allocation for short-lived data

### Concurrency

1. **Tokio**: Async runtime for all I/O operations
2. **Rayon**: Data parallelism for CPU-bound tasks (eval, optimization)
3. **DashMap**: Lock-free concurrent HashMap
4. **Semaphore**: Fine-grained concurrency control

### I/O

1. **Connection Pooling**: Reuse HTTP connections
2. **Batching**: Group LM requests when possible
3. **Streaming**: Process results as they arrive
4. **Zero-Copy**: Memory-mapped files for large datasets (planned)

## Comparison to DSPy

| Feature | DSPy (Python) | Kkachi (Rust) |
|---------|--------------|-------------|
| Type Safety | Runtime | Compile-time |
| Concurrency | GIL-limited | True parallelism |
| Memory | GC overhead | Zero-copy, explicit |
| Startup | ~1s (Python import) | ~10ms |
| WASM | Not supported | First-class |
| Deployment | Container required | Single binary |

## Data Flow

```
User Code
    ↓
Signature (zero-copy)
    ↓
Module::forward() (async)
    ↓
LM Client (pooled, cached)
    ↓
Provider (OpenAI, etc.)
    ↓
Prediction (zero-copy)
    ↓
Metric Evaluation (parallel)
    ↓
Results
```

## WASM Architecture

### Use Cases

1. **Edge Optimization**: Run close to users
2. **Privacy-Preserving**: Local computation
3. **Sandboxed Plugins**: Safe untrusted code
4. **Cross-Platform**: Browser + server with same code

### Implementation

- `wasm-bindgen` for JavaScript interop
- `tokio_wasi` for async in WASM
- Component Model for polyglot use
- IndexedDB for browser caching

## Future Enhancements

1. **String Interning**: Global intern pool for common strings
2. **SIMD**: Vectorized metric computation
3. **Distributed**: gRPC for multi-node optimization
4. **Native Models**: ONNX/HF without Python
5. **Hot Reload**: Watch mode for development

## Testing Strategy

1. **Unit Tests**: Per-module in `#[cfg(test)]`
2. **Integration Tests**: `tests/` directory
3. **Benchmarks**: Criterion.rs in `benches/`
4. **WASM Tests**: `wasm-bindgen-test`
5. **Property Tests**: Proptest for invariants

## Build Configuration

### Release Profile

```toml
[profile.release]
lto = "fat"              # Link-time optimization
codegen-units = 1        # Single codegen for better optimization
opt-level = 3            # Maximum optimization
strip = true             # Remove debug symbols
```

### Features

- `std` (default): Standard library support
- `wasm`: WebAssembly compatibility
- `disk`: Disk caching (requires filesystem)

## Deployment Targets

1. **Native**: Linux, macOS, Windows (x86_64, aarch64)
2. **WASM**: `wasm32-wasi`, `wasm32-unknown-unknown`
3. **Embedded**: `no_std` support (planned)

## Integration Guide

### As a Library

```toml
[dependencies]
kkachi = "0.1"
kkachi-client = "0.1"
tokio = { version = "1", features = ["full"] }
```

### In build.rs

```toml
[build-dependencies]
kkachi-refine = "0.1"
```

### WASM

```bash
wasm-pack build crates/kkachi-wasm --target web
```

## License

MIT OR Apache-2.0 (dual-licensed for maximum compatibility)
