#!/usr/bin/env python3
# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Complete workflow: Memory + JinjaFormatter + CliValidator

Demonstrates generating validated Rust code using:
1. Memory - Store best practices and patterns (with persistence)
2. JinjaFormatter - Dynamic prompt generation with Jinja2 templates
3. CliValidator - Real rustfmt/rustc validation

This example shows how all three features work together to create
a production-ready code generation pipeline with:
- Knowledge base for context
- Dynamic prompts that adapt to feedback
- Real tool validation for quality assurance
"""

from kkachi import (
    Memory,
    JinjaTemplate,
    JinjaFormatter,
    reason,
    CliValidator,
    Checks,
)
from typing import Optional
import tempfile
import os


# =============================================================================
# Step 1: Build Knowledge Base with Memory
# =============================================================================

def setup_memory(db_path: str) -> Memory:
    """Create a persistent memory store with Rust best practices."""
    print("Setting up memory with Rust best practices...")

    mem = Memory().persist(db_path)

    # Add best practices for Rust code generation
    patterns = [
        "Use Result<T, E> for error handling instead of panic",
        "Avoid .unwrap() in production code - use proper error handling",
        "Prefer &str over String in function signatures for efficiency",
        "Use match statements for exhaustive error handling",
        "Implement proper error types using thiserror or similar",
        "Always handle edge cases like empty strings or None values",
        "Use descriptive error messages in Result::Err",
        "Prefer iterators over explicit loops when possible",
    ]

    for pattern in patterns:
        mem.add(pattern)

    print(f"Added {len(patterns)} patterns to memory")
    return mem


# =============================================================================
# Step 2: Create JinjaFormatter with Memory Integration
# =============================================================================

def create_formatter() -> JinjaFormatter:
    """Create a JinjaFormatter that incorporates feedback and patterns."""
    print("Creating JinjaFormatter with dynamic prompt template...")

    template = JinjaTemplate.from_str("code_gen", '''
# Task
{{ task }}

# Best Practices
Apply these Rust best practices:
{% for pattern in patterns %}
- {{ pattern }}
{% endfor %}

{% if feedback %}
# Issues from Previous Attempt
The previous code had the following issues:
{{ feedback }}

Please fix these issues in your new implementation.
{% endif %}

# Output Format
Provide ONLY the Rust code in a fenced code block:
```rust
// your code here
```

# Requirements
- Use proper error handling with Result<T, E>
- No .unwrap() calls
- Handle all edge cases
- Include descriptive error messages
''')

    return JinjaFormatter(template)


# =============================================================================
# Step 3: Create CliValidator Pipeline
# =============================================================================

def create_validator() -> CliValidator:
    """Create a CliValidator for Rust code validation.

    This validator:
    1. Checks formatting with rustfmt (weight: 0.2)
    2. Validates compilation with rustc (required)
    3. Captures output for feedback
    """
    print("Creating CliValidator pipeline...")

    # Check if rustfmt and rustc are available
    import shutil
    has_rust = shutil.which("rustfmt") and shutil.which("rustc")

    if has_rust:
        print("Found Rust tools - using real validation")
        # Real Rust validation pipeline
        validator = (
            CliValidator("rustfmt")
            .args(["--check", "--edition", "2021"])
            .weight(0.2)
            .capture()  # Capture output for feedback
            .then("rustc")
            .args(["--crate-type", "lib", "-o", "/dev/null"])
            .required()
            .capture()
            .ext("rs")
        )
    else:
        print("Rust tools not found - using mock validation")
        # Fallback to echo for demonstration
        validator = (
            CliValidator("echo")
            .args(["validation"])
            .weight(0.5)
            .capture()
        )

    return validator


# =============================================================================
# Step 4: Mock LLM (Replace with Real ApiLlm)
# =============================================================================

def create_mock_llm(iteration: int = 0):
    """Create a mock LLM that simulates responses.

    In production, replace this with:
        from kkachi import ApiLlm
        llm = ApiLlm("anthropic", api_key="your-key")
        llm_fn = lambda prompt, feedback: llm.call(prompt, feedback)
    """

    def mock_llm(prompt: str, feedback: Optional[str] = None) -> str:
        """Mock LLM that improves based on feedback."""

        # First attempt - has .unwrap() issue
        if feedback is None or iteration == 0:
            return '''```rust
fn parse_url(url: &str) -> String {
    // Bad: uses .unwrap()
    url.to_string().unwrap()
}
```'''

        # Second attempt - better error handling
        if ".unwrap()" in feedback.lower() or "error handling" in feedback.lower():
            return '''```rust
fn parse_url(url: &str) -> Result<String, String> {
    if url.is_empty() {
        return Err("URL cannot be empty".to_string());
    }

    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("URL must start with http:// or https://".to_string());
    }

    Ok(url.to_string())
}
```'''

        # Default response
        return '''```rust
fn parse_url(url: &str) -> Result<String, String> {
    match url {
        "" => Err("Empty URL".to_string()),
        s if s.starts_with("http") => Ok(s.to_string()),
        _ => Err("Invalid URL".to_string()),
    }
}
```'''

    return mock_llm


# =============================================================================
# Step 5: Integrated Workflow
# =============================================================================

def generate_code(
    task: str,
    mem: Memory,
    formatter: JinjaFormatter,
    validator: CliValidator,
) -> str:
    """Generate and validate Rust code using the complete pipeline."""

    print(f"\nGenerating code for task: {task}")
    print("=" * 60)

    # Get relevant patterns from memory
    print("Retrieving relevant patterns from memory...")
    search_results = mem.search(task, k=3)
    pattern_list = [result.content for result in search_results]
    print(f"Found {len(pattern_list)} relevant patterns")

    # Create mock LLM
    llm = create_mock_llm()

    # Compose validators for comprehensive checking
    print("Composing validators (CliValidator + Checks)...")
    combined_validator = validator.and_(
        Checks()
        .forbid(".unwrap()")
        .require("Result")
        .require("Err")
        .min_len(50)
    )

    print("Running refinement loop with reason()...")
    print("-" * 60)

    # Generate and validate using reason() with all components
    result = (
        reason(llm, task)
        .validate(combined_validator)
        .max_iter(5)
        .target(0.9)
        .go()
    )

    print("-" * 60)
    print(f"Refinement complete!")
    print(f"  Iterations: {result.iterations}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Success: {result.success()}")

    if result.reasoning:
        print(f"  Reasoning: {result.reasoning[:100]}...")

    print("\nGenerated Code:")
    print("=" * 60)
    print(result.output)
    print("=" * 60)

    return result.output


# =============================================================================
# Step 6: Memory Updates - Learn from Results
# =============================================================================

def update_memory(mem: Memory, code: str, task: str):
    """Update memory with successful patterns from generated code."""
    print("\nUpdating memory with learned patterns...")

    # Extract patterns from successful code
    if "Result<" in code and "Err(" in code:
        mem.add(f"Pattern from '{task}': {code[:100]}...")
        print("Added successful pattern to memory")


# =============================================================================
# Main Example
# =============================================================================

def main():
    """Run the complete workflow example."""

    print("=" * 60)
    print("Combined Workflow: Memory + JinjaFormatter + CliValidator")
    print("=" * 60)
    print()

    # Setup with temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "rust_patterns.db")

        # Initialize components
        mem = setup_memory(db_path)
        formatter = create_formatter()
        validator = create_validator()

        print()
        print("All components initialized!")
        print()

        # Example 1: URL Parser
        code1 = generate_code(
            "Write a URL parser function that validates URLs",
            mem,
            formatter,
            validator,
        )
        update_memory(mem, code1, "URL parser")

        print("\n" + "=" * 60)
        print("Example complete!")
        print("=" * 60)

        # Show memory statistics
        print("\nMemory Statistics:")
        print(f"  Total patterns stored: {len(mem.list())}")

        # Test memory persistence
        print("\nTesting memory persistence...")
        mem2 = Memory().persist(db_path)
        patterns = mem2.list()
        print(f"  Patterns persisted: {len(patterns)}")
        print("  Memory successfully persisted to disk!")


if __name__ == "__main__":
    main()
