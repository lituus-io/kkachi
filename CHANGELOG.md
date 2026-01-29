# Changelog

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
