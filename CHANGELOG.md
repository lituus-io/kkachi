# Changelog

## v0.6.0 (2026-02-26)

### New Features

- **Typed Signature System** - Stack-allocated, const-constructible signatures with `ValueKind` annotations (Int, Float, Str, Bool) and zero-copy parsing via `StrView`
- **TypedAdapter** - Automatic prompt formatting and response parsing based on field types
- **Compiled Program Export** - Serialize optimization results (tuned instructions, demo indices, scores) for checkpointing and reuse
- **Composable Module System** - `ComposableModule` trait with zero-cost async via GATs, `ModuleState` snapshots, and `kkachi_module!` macro for recursive state save/load
- **Evaluation Harness** - `Evaluate` builder with `Metric` trait and built-in implementations (ExactMatch, Contains, F1Token, FnMetric)
- **LM-as-Critic** - `Critic` trait for structured feedback loops: NoCritic (zero-cost), LlmCritic, FnCritic
- **Skills** - Persistent reusable instruction context injected into LLM prompts with label-based organization and priority control
- **Runtime Defaults** - Regex-based substitution system for placeholder replacement in LLM output with per-entry metadata
- **Multimodal Input** - ContentType enum supporting Text, PNG, and audio with zero-copy `Cow` types and SmallVec inline storage
- **State Serialization** - Generic state system with StateMap (BTreeMap) for deterministic serialization and session restoration
- **DuckDB RAG Auto-Packaging** - Package persistent knowledge bases into PEP 427 pip-installable wheels

### Python Bindings

- **PySkill** - `instruct()`, `instruct_at()` with priority control for skill injection
- **PyDefaults** - `set()`, `set_with_note()`, `from_env()` for runtime default management

### Tests

- `integration_skill.rs` - Skill injection with reason(), combined skill+defaults workflows
- `integration_packager.rs` - Memory persistence, wheel generation, metadata validation
- `test_skill.py` - Skill creation, rendering, integration with reason/best_of/ensemble/agent
- `test_cli_standalone.py` - Standalone CliValidator validation with ScoreResult
- `test_cli_fuzz.py` - Property-based testing with hypothesis for validate() determinism
- `test_memory_package.py` - Wheel generation, ZIP validation, pip installation verification

### CI/CD

- Publish workflow now uploads to PyPI via twine (requires `PYPI_PACKAGE` secret)
- Simplified build matrix: single Python 3.12 per OS with manylinux support

### Breaking Changes

None - all changes are backwards compatible.

## v0.3.0 (2026-01-29)

### New Features

- ✨ **JinjaTemplate and JinjaFormatter** - Dynamic prompt generation with Jinja2 templates
  - Load templates from files or strings
  - Full Jinja2 syntax support (variables, loops, conditionals, filters)
  - JinjaFormatter for refinement loops with task/feedback/iteration variables

- ✨ **CliValidator DSPy Integration** - External tool validation now works with all DSPy modules
  - Fixed composition system to recognize CliValidator
  - Works with `reason()`, `best_of()`, `ensemble()`, `program()`
  - Full support for validator composition

- ✨ **Validator Composition** - Combine validators with AND/OR logic
  - `.and_()` method for AND semantics (both must pass)
  - `.or_()` method for OR semantics (at least one must pass)
  - Chain multiple validators: `cli.and_(checks).or_(semantic)`

- ✨ **CliValidator Output Capture** - Capture stdout/stderr for feedback loops
  - `.capture()` method enables output capture
  - `.get_captures()` retrieves CliCapture objects
  - Use captured output in refinement feedback

### Implementation Details

- Added `Clone` derive to `JinjaTemplate` and `JinjaFormatter`
- Extended `ValidatorNode` enum with `Cli` variant
- Updated `extract_validator_node()` to recognize `PyCliValidator`
- Implemented composition methods in `PyCliValidator`
- Registered Jinja types in Python module
- Added comprehensive type stubs for IDE support

### Examples

- `examples/combined_workflow.py` - Memory + Jinja + CliValidator integration (270 lines)
- `examples/cli_validator_dspy.py` - CliValidator with all DSPy modules (310 lines)

### Tests

- `tests/test_cli_validator.py` - CliValidator composition tests (230 lines)
- `tests/test_integration.py` - Full integration tests (260 lines)

### Documentation

- Comprehensive API reference in README
- CHANGELOG with full release notes
- Type stubs for all new features

### Breaking Changes

None - all changes are backwards compatible.

## v0.2.8

- Memory persistence with DuckDB
- CRUD operations for memory entries
- Windows support improvements

## v0.2.7

- Initial DSPy-style module implementations
- Pattern-based and semantic validation
- ReAct agent support
