# Kkachi

High-performance LLM prompt optimization library with DSPy-style patterns.

## Features

- 🎯 **DSPy-Style Modules** - Chain of Thought, Best of N, Ensemble, Program of Thought, ReAct Agent
- 📝 **Jinja2 Templates** - Dynamic prompt generation (v0.3.0)
- ✅ **CLI Validators** - External tool validation with full integration (v0.3.0)
- 💾 **Memory & RAG** - Persistent vector store with DuckDB
- 🔧 **Pattern Validation** - Regex, substring, length checks
- 🤖 **LLM-as-Judge** - Semantic validation

## Python Installation

```bash
pip install kkachi
```

See `crates/kkachi-python/README.md` for detailed Python documentation.

## Quick Example

```python
from kkachi import reason, CliValidator, Checks

# Create validator
validator = (
    CliValidator("rustfmt")
    .args(["--check"])
    .and_(Checks().forbid(".unwrap()"))
)

# Generate with validation
result = reason(llm, "Write a URL parser").validate(validator).go()
```

## What's New in v0.3.0

- ✨ **JinjaTemplate & JinjaFormatter** - Dynamic Jinja2-based prompt generation
- ✨ **CliValidator DSPy Integration** - Works with all modules (reason, best_of, ensemble, program)
- ✨ **Validator Composition** - `.and_()` and `.or_()` for complex validation logic
- 📚 Comprehensive examples and documentation

## Repository Structure

- `crates/kkachi` - Core Rust library
- `crates/kkachi-python` - Python bindings
- `examples/` - Usage examples
- `tests/` - Test suite

## License

PolyForm Noncommercial 1.0.0
