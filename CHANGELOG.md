# Changelog

## v0.1.8 (2026-03-13)

- Added Python 3.13 wheel builds and CI testing
- Removed remaining branded references from comments and messages
- Added author and copyright attribution

## v0.1.7 (2026-03-12)

- Clean repo release: removed all branded references, fresh history

## v0.1.2 (2026-03-04)

### Examples

- **New Python examples** — Added simple `refine.py`, `reason.py`, `best_of.py`, `ensemble.py` matching the Rust examples
- **Shared pipeline utilities** — Extracted `_pipeline_common.py` to deduplicate ~150 lines across pipeline examples
- **Examples README** — Added `examples/README.md` with learning progression, prerequisites, and quick-start commands
- **Environment variable overrides** — All pipeline examples now use `os.environ.get()` with configurable defaults
- **Removed hardcoded paths** — Replaced user-specific paths with generic placeholders
- **Consistent error handling** — All Python examples have `try/except` with clear setup instructions
- **Removed branded references** — Replaced provider-specific references with generic LLM placeholders

## v0.1.0 (2026-03-02)

Initial release.

### Core Library

- **Composable Pipelines** — `pipeline()` builder with chainable steps: refine, best_of, ensemble, reason, extract, map
- **Concurrent Execution** — `ConcurrentRunner` for running multiple pipelines concurrently on a shared LLM with rate limiting and error isolation
- **Step Combinators** — GAT-based `Step` trait with `then`, `race`, `par`, `retry`, `fallback`, `when`, `map` combinators
- **Generation Engine** — Shared `GenerationEngine` for best_of and ensemble with parallel/sequential generation, diversity context, and transform output
- **DSPy-Style Modules** — Chain of Thought (`reason`), Best of N (`best_of`), Ensemble (`ensemble`), Program of Thought (`program`), ReAct Agent (`agent`)
- **Typed Signatures** — Stack-allocated, const-constructible signatures with `ValueKind` annotations and zero-copy parsing via `StrView`
- **TypedAdapter** — Automatic prompt formatting and response parsing based on field types
- **Validation System** — Pattern checks (regex, substring, length), CLI validators, LLM-as-Judge semantic validation, and composable `.and_()` / `.or_()` logic
- **Multi-Objective Optimization** — Pareto-optimal prompt tuning with scalarization strategies (weighted sum, Chebyshev, epsilon-constraint)
- **Evaluation Harness** — `Evaluate` builder with built-in metrics (ExactMatch, Contains, F1Token, FnMetric)
- **LM-as-Critic** — `Critic` trait for structured feedback: NoCritic (zero-cost), LlmCritic, FnCritic
- **Skills** — Persistent reusable instruction context with label-based organization and priority control
- **Runtime Defaults** — Regex-based substitution for placeholder replacement in LLM output
- **Memory & RAG** — Persistent vector store with DuckDB, auto-packaging into pip-installable wheels
- **Jinja2 Templates** — Dynamic prompt generation with full Jinja2 syntax support
- **API Client** — Multi-provider LLM client with streaming and rate limiting

### Architecture

- Zero-copy core with GATs over async/await, lifetimes over Arc, minimal cloning
- `ErasedLlm` trait + `BoxedLlmInner` eliminates Arc and Box::leak from LLM type erasure
- GAT-based `Tool` and `CodeExecutor` traits with `DynTool` / `DynCodeExecutor` object-safe bridges
- `GenerationEngine` deduplicates ~310 LOC between best_of and ensemble
- `IterationConfig` shares transform logic between reason and refine

### Python Bindings

- Full API parity: `pipeline()`, `concurrent()`, `reason()`, `best_of()`, `ensemble()`, `program()`, `agent()`
- `ApiLlm` — auto-detect, explicit provider constructors, rate limiting, streaming
- `Checks`, `CliValidator`, `SemanticJudge` — all validation types
- `Skill`, `Defaults` — instruction injection and runtime defaults
- `StepDef` — step combinator tree (then, race, par, retry, fallback)
- `Optimizer`, `Dataset`, `Evaluate`, `MultiObjective` — optimization and evaluation

### CI/CD

- GitHub Actions: fmt, clippy, test (Linux x86/ARM, macOS ARM, Windows), Python tests, cross-compile check
- Release workflow: wheel builds (5 platforms via maturin), GitHub Release with SHA256SUMS, PyPI publish
- Security: cargo-audit, cargo-deny, clippy security lints
