#!/usr/bin/env python3
# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
CliValidator with DSPy-style Modules

Demonstrates CliValidator working with all DSPy-style builders:
1. reason() - Chain of Thought reasoning
2. best_of() - Best of N candidate generation
3. ensemble() - Multi-chain consensus
4. program() - Program of Thought (code generation)

This example shows that CliValidator now properly integrates with
the validator composition system and works across all module types.
"""

from kkachi import (
    CliValidator,
    Checks,
    reason,
    best_of,
    ensemble,
    program,
    Executor,
)
from typing import Optional
import shutil


# =============================================================================
# Mock LLM
# =============================================================================

def create_simple_llm():
    """Create a simple mock LLM for testing."""

    responses = [
        'fn add(a: i32, b: i32) -> i32 { a + b }',
        'fn add(x: i32, y: i32) -> i32 { x + y }',
        'fn add(left: i32, right: i32) -> i32 { left + right }',
    ]
    counter = [0]

    def mock_llm(prompt: str, feedback: Optional[str] = None) -> str:
        idx = counter[0] % len(responses)
        counter[0] += 1
        return responses[idx]

    return mock_llm


# =============================================================================
# Example 1: reason() with CliValidator
# =============================================================================

def example_reason_with_cli():
    """Demonstrate CliValidator working with reason()."""

    print("=" * 60)
    print("Example 1: reason() + CliValidator")
    print("=" * 60)

    llm = create_simple_llm()

    # Create CliValidator
    has_rustfmt = shutil.which("rustfmt") is not None

    if has_rustfmt:
        print("Using real rustfmt validation")
        validator = (
            CliValidator("rustfmt")
            .args(["--check", "--edition", "2021"])
            .weight(0.3)
            .capture()
            .ext("rs")
        )
    else:
        print("Using mock validation (rustfmt not found)")
        validator = CliValidator("echo").args(["mock"]).weight(0.3)

    # Compose with Checks
    combined = validator.and_(
        Checks()
        .require("fn")
        .require("i32")
        .min_len(10)
    )

    print("\nRunning reason() with CliValidator...")
    result = (
        reason(llm, "Write an add function in Rust")
        .validate(combined)
        .max_iter(3)
        .go()
    )

    print(f"Result: {result.output}")
    print(f"Score: {result.score:.2f}")
    print(f"Iterations: {result.iterations}")
    print(f"Success: {result.success()}")


# =============================================================================
# Example 2: best_of() with CliValidator
# =============================================================================

def example_best_of_with_cli():
    """Demonstrate CliValidator working with best_of()."""

    print("\n" + "=" * 60)
    print("Example 2: best_of() + CliValidator")
    print("=" * 60)

    llm = create_simple_llm()

    # Create CliValidator for Rust code
    has_rustc = shutil.which("rustc") is not None

    if has_rustc:
        print("Using real rustc validation")
        validator = (
            CliValidator("rustc")
            .args(["--crate-type", "lib", "-o", "/dev/null"])
            .required()
            .capture()
            .ext("rs")
        )
    else:
        print("Using mock validation (rustc not found)")
        validator = CliValidator("echo").args(["compile"]).weight(0.5)

    # Compose with semantic checks
    combined = validator.or_(
        Checks()
        .require("fn")
        .forbid("panic")
    )

    print("\nRunning best_of() with CliValidator...")
    result, pool = (
        best_of(llm, "Write a multiply function", 3)
        .validate(combined)
        .go_with_pool()
    )

    print(f"Best result: {result.output}")
    print(f"Score: {result.score:.2f}")
    print(f"Candidates generated: {result.candidates_generated}")
    print(f"Pool stats: {pool.stats()}")


# =============================================================================
# Example 3: ensemble() with CliValidator
# =============================================================================

def example_ensemble_with_cli():
    """Demonstrate CliValidator working with ensemble()."""

    print("\n" + "=" * 60)
    print("Example 3: ensemble() + CliValidator")
    print("=" * 60)

    llm = create_simple_llm()

    # Create simple CliValidator
    validator = CliValidator("echo").args(["valid"]).weight(0.2)

    # Compose with pattern checks
    combined = validator.and_(
        Checks()
        .require("fn")
        .min_len(20)
    )

    print("\nRunning ensemble() with CliValidator...")
    result, consensus = (
        ensemble(llm, "Write a subtract function", 3)
        .validate(combined)
        .aggregate("majority_vote")
        .go_with_consensus()
    )

    print(f"Consensus result: {result.output}")
    print(f"Chains: {result.chains_generated}")
    print(f"Agreement ratio: {consensus.agreement_ratio():.2f}")
    print(f"Unanimous: {consensus.has_unanimous_agreement()}")


# =============================================================================
# Example 4: program() with CliValidator
# =============================================================================

def example_program_with_cli():
    """Demonstrate CliValidator working with program()."""

    print("\n" + "=" * 60)
    print("Example 4: program() + CliValidator")
    print("=" * 60)

    def code_llm(prompt: str, feedback: Optional[str] = None) -> str:
        """LLM that generates Python code."""
        return '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(result)
'''

    # Create CliValidator for Python
    has_python = shutil.which("python3") is not None

    if has_python:
        print("Using real python validation")
        validator = (
            CliValidator("python3")
            .args(["-m", "py_compile"])
            .weight(0.4)
            .capture()
            .ext("py")
        )
    else:
        print("Using mock validation")
        validator = CliValidator("echo").args(["python"]).weight(0.4)

    # Compose validators
    combined = validator.and_(
        Checks()
        .require("def")
        .forbid("import os")  # Security check
        .min_len(30)
    )

    print("\nRunning program() with CliValidator...")
    result = (
        program(code_llm, "Calculate Fibonacci(10)")
        .validate(combined)
        .executor(Executor.python())
        .max_iter(2)
        .language("python")
        .go()
    )

    print(f"Success: {result.success}")
    print(f"Attempts: {result.attempts}")
    print(f"Code:\n{result.code}")
    print(f"Output: {result.output}")


# =============================================================================
# Example 5: Validator Composition
# =============================================================================

def example_validator_composition():
    """Demonstrate CliValidator composition with and_() and or_()."""

    print("\n" + "=" * 60)
    print("Example 5: Validator Composition")
    print("=" * 60)

    llm = create_simple_llm()

    # Create multiple validators
    cli1 = CliValidator("echo").args(["format"]).weight(0.3)
    cli2 = CliValidator("echo").args(["compile"]).weight(0.4)
    checks = Checks().require("fn").min_len(15)

    # Compose using and_() and or_()
    # (cli1 AND cli2) OR checks
    composed = cli1.and_(cli2).or_(checks)

    print("\nComposed validator: (cli1 AND cli2) OR checks")
    print(f"Validator: {composed}")

    print("\nRunning reason() with composed validator...")
    result = (
        reason(llm, "Write a divide function")
        .validate(composed)
        .max_iter(2)
        .go()
    )

    print(f"Result: {result.output}")
    print(f"Score: {result.score:.2f}")


# =============================================================================
# Example 6: Capture and Feedback
# =============================================================================

def example_capture_feedback():
    """Demonstrate CliValidator output capture for feedback."""

    print("\n" + "=" * 60)
    print("Example 6: Output Capture and Feedback")
    print("=" * 60)

    llm = create_simple_llm()

    # Create validator with capture enabled
    validator = (
        CliValidator("echo")
        .args(["Formatting issue detected"])
        .weight(0.5)
        .capture()  # Enable output capture
    )

    print("\nRunning with capture enabled...")
    result = (
        reason(llm, "Write a max function")
        .validate(validator)
        .max_iter(2)
        .go()
    )

    print(f"Result: {result.output}")
    print(f"Score: {result.score:.2f}")

    # Access captured output
    captures = validator.get_captures()
    print(f"\nCaptured {len(captures)} outputs:")
    for i, capture in enumerate(captures):
        print(f"  [{i+1}] Exit code: {capture.exit_code}")
        print(f"      Stdout: {capture.stdout[:50]}...")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""

    print("\n")
    print("#" * 60)
    print("# CliValidator with DSPy-style Modules")
    print("#" * 60)
    print()

    try:
        example_reason_with_cli()
        example_best_of_with_cli()
        example_ensemble_with_cli()
        example_program_with_cli()
        example_validator_composition()
        example_capture_feedback()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
