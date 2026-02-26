#!/usr/bin/env python3
# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Tests for CliValidator integration with DSPy-style modules."""

import pytest
from kkachi import (
    CliValidator,
    Checks,
    Validator,
    reason,
    best_of,
    ensemble,
    program,
    Executor,
)


def mock_llm(prompt, feedback=None):
    """Simple mock LLM for testing."""
    if feedback and "fix" in feedback.lower():
        return "fn fixed() -> i32 { 42 }"
    return "fn test() -> i32 { 42 }"


def code_llm(prompt, feedback=None):
    """Mock LLM for code generation."""
    return "def test():\n    return 42\n\nresult = test()\nprint(result)"


# =============================================================================
# Test 1: CliValidator with reason()
# =============================================================================

def test_cli_validator_with_reason():
    """CliValidator should work with reason()."""
    validator = CliValidator("echo").args(["test"]).weight(0.5)

    result = reason(mock_llm, "Write a function").validate(validator).max_iter(1).go()

    assert result.output
    assert result.success()
    assert result.score >= 0.0


def test_cli_validator_composed_with_checks_in_reason():
    """CliValidator.and_(Checks) should work with reason()."""
    cli = CliValidator("echo").args(["validate"]).weight(0.3)
    checks = Checks().require("fn").min_len(10)
    combined = cli.and_(checks)

    result = reason(mock_llm, "Write a function").validate(combined).max_iter(2).go()

    assert result.output
    assert "fn" in result.output
    assert len(result.output) >= 10


def test_cli_validator_or_composition_in_reason():
    """CliValidator.or_(Checks) should work with reason()."""
    cli = CliValidator("echo").args(["test"]).weight(0.5)
    checks = Checks().require("function")  # Won't match
    combined = cli.or_(checks)

    result = reason(mock_llm, "Write code").validate(combined).max_iter(1).go()

    # Should pass because of OR semantics (cli passes even if checks fails)
    assert result.output


# =============================================================================
# Test 2: CliValidator with best_of()
# =============================================================================

def test_cli_validator_with_best_of():
    """CliValidator should work with best_of()."""
    validator = CliValidator("echo").args(["validate"]).weight(0.4)

    result = best_of(mock_llm, "Write a function", 2).validate(validator).go()

    assert result.output
    assert result.success()
    assert result.candidates_generated == 2


def test_cli_validator_with_best_of_pool():
    """CliValidator should work with best_of() pool."""
    cli = CliValidator("echo").args(["test"]).weight(0.3)
    checks = Checks().require("fn")
    combined = cli.and_(checks)

    result, pool = (
        best_of(mock_llm, "Write a function", 3).validate(combined).go_with_pool()
    )

    assert result.output
    assert len(pool) > 0
    assert pool.stats().count == 3


# =============================================================================
# Test 3: CliValidator with ensemble()
# =============================================================================

def test_cli_validator_with_ensemble():
    """CliValidator should work with ensemble()."""
    validator = CliValidator("echo").args(["test"]).weight(0.5)

    result = ensemble(mock_llm, "Write a function", 2).validate(validator).go()

    assert result.output
    assert result.success()
    assert result.chains_generated == 2


def test_cli_validator_with_ensemble_consensus():
    """CliValidator should work with ensemble() consensus pool."""
    cli = CliValidator("echo").args(["validate"]).weight(0.3)
    checks = Checks().require("fn")
    combined = cli.or_(checks)

    result, consensus = (
        ensemble(mock_llm, "Write code", 3)
        .validate(combined)
        .aggregate("majority_vote")
        .go_with_consensus()
    )

    assert result.output
    assert len(consensus) == 3
    assert 0.0 <= consensus.agreement_ratio() <= 1.0


# =============================================================================
# Test 4: CliValidator with program()
# =============================================================================

def test_cli_validator_with_program():
    """CliValidator should work with program()."""
    validator = CliValidator("echo").args(["python"]).weight(0.4)

    result = (
        program(code_llm, "Calculate result")
        .validate(validator)
        .executor(Executor.python())
        .max_iter(1)
        .language("python")
        .go()
    )

    assert result.code
    assert result.success or result.attempts > 0


def test_cli_validator_composed_with_program():
    """CliValidator composition should work with program()."""
    cli = CliValidator("echo").args(["test"]).weight(0.3)
    checks = Checks().require("def").min_len(20)
    combined = cli.and_(checks)

    result = (
        program(code_llm, "Write code")
        .validate(combined)
        .executor(Executor.python())
        .max_iter(1)
        .go()
    )

    assert result.code
    assert "def" in result.code


# =============================================================================
# Test 5: Validator Composition
# =============================================================================

def test_cli_and_composition():
    """CliValidator.and_() should create composed validator."""
    cli = CliValidator("echo").args(["test"]).weight(0.5)
    checks = Checks().require("fn")

    combined = cli.and_(checks)

    assert isinstance(combined, Validator)
    # Test that it works in practice
    result = reason(mock_llm, "Write code").validate(combined).max_iter(1).go()
    assert result.output


def test_cli_or_composition():
    """CliValidator.or_() should create composed validator."""
    cli = CliValidator("echo").args(["test"]).weight(0.5)
    checks = Checks().require("impossible_pattern")

    combined = cli.or_(checks)

    assert isinstance(combined, Validator)
    # Should pass because cli passes (OR semantics)
    result = reason(mock_llm, "Write code").validate(combined).max_iter(1).go()
    assert result.output


def test_cli_chained_composition():
    """Multiple CliValidator composition should work."""
    cli1 = CliValidator("echo").args(["step1"]).weight(0.3)
    cli2 = CliValidator("echo").args(["step2"]).weight(0.3)
    checks = Checks().require("fn")

    # (cli1 AND cli2) OR checks
    combined = cli1.and_(cli2).or_(checks)

    assert isinstance(combined, Validator)
    result = reason(mock_llm, "Write code").validate(combined).max_iter(1).go()
    assert result.output


# =============================================================================
# Test 6: CliValidator Capture
# =============================================================================

def test_cli_validator_capture():
    """CliValidator.capture() should enable output capture."""
    validator = CliValidator("echo").args(["test output"]).capture()

    result = reason(mock_llm, "Write code").validate(validator).max_iter(1).go()

    assert result.output

    # Get captured outputs
    captures = validator.get_captures()
    assert len(captures) > 0
    assert captures[0].stdout or captures[0].stderr


def test_cli_validator_capture_with_composition():
    """Capture should work with composed validators."""
    cli = CliValidator("echo").args(["validation"]).capture()
    checks = Checks().require("fn")
    combined = cli.and_(checks)

    result = reason(mock_llm, "Write code").validate(combined).max_iter(2).go()

    assert result.output

    # Check captures
    captures = cli.get_captures()
    assert len(captures) >= 1


# =============================================================================
# Test 7: Error Handling
# =============================================================================

def test_cli_validator_with_invalid_command():
    """CliValidator with non-existent command should handle gracefully."""
    validator = CliValidator("nonexistent_command_12345").weight(0.5)

    # Should not crash, may return low score
    result = reason(mock_llm, "Write code").validate(validator).max_iter(1).go()

    assert result.output  # Should still produce output


def test_cli_validator_required_fail():
    """CliValidator with required() should affect validation."""
    validator = CliValidator("false").required()  # Always fails

    result = reason(mock_llm, "Write code").validate(validator).max_iter(2).go()

    # Should still complete but score may be affected
    assert result.output


# =============================================================================
# Test 8: CliValidator with Checks Shortcuts
# =============================================================================

def test_cli_with_reason_shortcuts():
    """CliValidator should work with reason() shortcut methods."""
    validator = CliValidator("echo").args(["test"]).weight(0.4)

    result = (
        reason(mock_llm, "Write code")
        .validate(validator)
        .require("fn")  # Shortcut
        .forbid("panic")  # Shortcut
        .max_iter(1)
        .go()
    )

    assert result.output
    assert "fn" in result.output


def test_cli_with_best_of_shortcuts():
    """CliValidator should work with best_of() shortcut methods."""
    validator = CliValidator("echo").args(["test"]).weight(0.3)

    result = (
        best_of(mock_llm, "Write code", 2)
        .validate(validator)
        .require("fn")
        .go()
    )

    assert result.output
    assert "fn" in result.output


# =============================================================================
# Test 9: Real-world Scenarios
# =============================================================================

def test_cli_validator_rust_formatting_simulation():
    """Simulate Rust code validation pipeline."""
    # Simulated rustfmt check
    format_check = CliValidator("echo").args(["format ok"]).weight(0.2)

    # Simulated rustc check
    compile_check = CliValidator("echo").args(["compilation ok"]).required()

    # Pattern checks
    pattern_check = Checks().require("fn").forbid(".unwrap()").min_len(20)

    # Compose: (format AND compile) AND patterns
    validator = format_check.and_(compile_check).and_(pattern_check)

    result = reason(mock_llm, "Write Rust function").validate(validator).max_iter(1).go()

    assert result.output
    assert "fn" in result.output


def test_cli_validator_python_linting_simulation():
    """Simulate Python code validation pipeline."""
    # Simulated pylint
    lint_check = CliValidator("echo").args(["10/10"]).weight(0.4)

    # Pattern checks
    pattern_check = Checks().require("def").forbid("import os")

    # Compose with OR (pass if either passes)
    validator = lint_check.or_(pattern_check)

    result = reason(mock_llm, "Write Python function").validate(validator).max_iter(1).go()

    assert result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
