// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! CLI-Driven Recursive Code Generation Pipeline
//!
//! This example demonstrates how CLI tool output (compilers, linters, validators)
//! drives recursive prompt refinement. The pipeline:
//!
//! 1. Generates code from a prompt
//! 2. Runs CLI tools (compiler, linter) on the generated code
//! 3. Captures errors and uses them as feedback to refine the prompt
//! 4. Iterates until the code passes all checks or max iterations reached
//!
//! Run with: cargo run --example cli_driven_pipeline

use kkachi::diff::{DiffRenderer, TextDiff};
use kkachi::recursive::{
    cli, refine, IterativeMockLlm, Validate, Score, CliExecutor,
};

use std::time::Duration;

// ============================================================================
// Mock Code Generator
// ============================================================================

/// Creates a mock LLM that generates progressively better URL parser code.
fn create_url_parser_llm() -> impl kkachi::recursive::Llm {
    let responses = [
        // Iteration 0: Initial attempt - has syntax errors
        r#"fn parse_url(s: &str) -> String {
    // Parse URL and extract hostname
    let parts = s.split("://")
    let host = parts[1].split("/")[0]
    host
}"#,

        // Iteration 1: Fixes syntax errors
        r#"fn parse_url(s: &str) -> String {
    // Parse URL and extract hostname
    let parts: Vec<&str> = s.split("://").collect();
    let host = parts[1].split("/").next().unwrap();
    host.to_string()
}"#,

        // Iteration 2: Adds error handling
        r#"/// Parses a URL and extracts the hostname.
///
/// # Errors
/// Returns an error if the URL is malformed.
fn parse_url(s: &str) -> Result<String, &'static str> {
    let without_scheme = s.split("://")
        .nth(1)
        .ok_or("Missing URL scheme")?;

    let host = without_scheme
        .split('/')
        .next()
        .ok_or("Missing hostname")?;

    Ok(host.to_string())
}"#,

        // Iteration 3: Final polished version
        r#"//! URL parsing utilities.

/// Represents a parsed URL.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedUrl {
    pub scheme: String,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
}

impl ParsedUrl {
    /// Parses a URL string into its components.
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let (scheme, rest) = s.split_once("://")
            .ok_or(ParseError::MissingScheme)?;

        let (authority, path) = rest.split_once('/')
            .map(|(a, p)| (a, format!("/{}", p)))
            .unwrap_or((rest, "/".to_string()));

        let (host, port) = if let Some((h, p)) = authority.split_once(':') {
            (h.to_string(), Some(p.parse().map_err(|_| ParseError::InvalidPort)?))
        } else {
            (authority.to_string(), None)
        };

        Ok(Self { scheme: scheme.to_string(), host, port, path })
    }
}

/// Errors that can occur during URL parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingScheme,
    InvalidPort,
}"#,
    ];

    IterativeMockLlm::new(move |iter, _prompt, _feedback| {
        let idx = (iter as usize).min(responses.len() - 1);
        responses[idx].to_string()
    })
}

// ============================================================================
// Mock Rust Compiler Validator
// ============================================================================

/// A mock validator that simulates Rust compiler behavior.
/// In production, use cli("rustc") with actual CLI tools.
struct MockRustCompilerValidator {
    error_patterns: Vec<(&'static str, &'static str)>,
}

impl MockRustCompilerValidator {
    fn new() -> Self {
        Self {
            error_patterns: vec![
                ("let parts = s.split", "error[E0658]: expected `;`"),
                ("let host = parts[1]", "error[E0658]: expected `;`"),
                (".unwrap()", "warning: use of `unwrap` in production code"),
                ("fn parse_url", "warning: missing documentation for a function"),
            ],
        }
    }
}

impl Validate for MockRustCompilerValidator {
    fn validate(&self, text: &str) -> Score<'static> {
        let mut errors = Vec::new();

        for (pattern, error) in &self.error_patterns {
            if text.contains(pattern) {
                errors.push(*error);
            }
        }

        // Check for missing Result type when error handling is needed
        if text.contains("read_to_string") && !text.contains("Result<") {
            errors.push("error[E0308]: mismatched types - expected String, found Result");
        }

        if errors.is_empty() {
            Score::pass()
        } else {
            let error_count = errors.iter().filter(|e| e.contains("error[")).count();
            let warning_count = errors.iter().filter(|e| e.contains("warning:")).count();

            let score = if error_count > 0 {
                0.0
            } else if warning_count > 0 {
                0.5 + (0.5 / (warning_count as f64 + 1.0))
            } else {
                0.8
            };

            Score::with_feedback(
                score,
                format!(
                    "Compilation issues:\n{}\n\nErrors: {}, Warnings: {}",
                    errors.join("\n"),
                    error_count,
                    warning_count
                ),
            )
        }
    }

    fn name(&self) -> &'static str {
        "mock_rust_compiler"
    }
}

// ============================================================================
// CLI-Driven Pipeline Demo
// ============================================================================

fn demo_cli_driven_pipeline() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║           CLI-Driven Code Generation Pipeline                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let prompt = "Write a robust URL parser in Rust with proper error handling";
    println!("Prompt: {}\n", prompt);

    // Create the mock LLM and validator
    let llm = create_url_parser_llm();
    let validator = MockRustCompilerValidator::new();

    // Run refinement with progress tracking
    let diff_renderer = DiffRenderer::new();

    let result = refine(&llm, prompt)
        .validate(validator)
        .max_iter(5)
        .target(1.0)
        .on_iter(move |iter, score| {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Iteration {}", iter);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            println!("  Score: {:.2}", score);
            if score >= 1.0 {
                println!("  ✓ All checks passed!");
            }
            println!();
        })
        .go_full()
        .expect("Refinement failed");

    // Show diff between first and last iteration if we have history
    if result.history.len() >= 2 {
        let first = &result.history[0].output;
        let last = &result.output;

        println!("\n→ Changes from first to final iteration:");
        let diff = TextDiff::new(first, last);
        if diff.has_changes() {
            let stats = diff.stats();
            println!("  +{} lines, -{} lines", stats.lines_added, stats.lines_removed);
            let rendered = diff_renderer.render_text(&diff);
            for line in rendered.lines().take(15) {
                println!("    {}", line);
            }
            if rendered.lines().count() > 15 {
                println!("    ... ({} more lines)", rendered.lines().count() - 15);
            }
        }
    }

    // Show final result
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                        FINAL RESULT");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("Iterations: {}", result.iterations);
    println!("Final score: {:.2}", result.score);
    println!("\nGenerated code:\n");
    println!("```rust");
    for line in result.output.lines().take(20) {
        println!("{}", line);
    }
    if result.output.lines().count() > 20 {
        println!("... ({} more lines)", result.output.lines().count() - 20);
    }
    println!("```");
}

// ============================================================================
// CLI API Demo
// ============================================================================

fn demo_new_cli_api() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              New CLI API (cli() builder)                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Create a Rust validator pipeline
    let rust_validator = cli("rustfmt").arg("--check").weight(0.1)
        .then("rustc").args(&["--edition", "2021", "--emit=metadata"]).weight(0.5).required()
        .then("cargo").args(&["clippy", "--", "-D", "warnings"]).weight(0.4)
        .ext("rs");

    println!("Created Rust validator pipeline:");
    println!("  • format  (rustfmt --check)            weight: 0.1");
    println!("  • compile (rustc --emit=metadata)      weight: 0.5, required");
    println!("  • clippy  (cargo clippy -D warnings)   weight: 0.4");
    println!();

    // Create a Python validator pipeline
    let python_validator = cli("python").args(&["-m", "py_compile"]).required()
        .then("ruff").arg("check")
        .then("mypy").arg("--ignore-missing-imports")
        .ext("py");

    println!("Created Python validator pipeline:");
    println!("  • syntax (python -m py_compile)        required");
    println!("  • lint   (ruff check)");
    println!("  • types  (mypy --ignore-missing-imports)");
    println!();

    // Create a Terraform validator pipeline
    let terraform_validator = cli("terraform").args(&["fmt", "-check"]).weight(0.2)
        .then("terraform").arg("validate").required()
        .then("terraform").args(&["plan", "-no-color"]).weight(0.3)
        .ext("tf");

    println!("Created Terraform validator pipeline:");
    println!("  • fmt      (terraform fmt -check)      weight: 0.2");
    println!("  • validate (terraform validate)        required");
    println!("  • plan     (terraform plan -no-color)  weight: 0.3");
    println!();

    // Using with refine()
    println!("Using with refine():\n");
    println!("```rust");
    println!("let llm = MyLlmClient::new();");
    println!("let validator = cli(\"rustfmt\").arg(\"--check\")");
    println!("    .then(\"rustc\").args(&[\"--emit=metadata\"]).required()");
    println!("    .ext(\"rs\");");
    println!();
    println!("let result = refine(&llm, \"Write a parser\")");
    println!("    .validate(validator)");
    println!("    .max_iter(5)");
    println!("    .go();");
    println!("```");

    // Prevent unused variable warnings
    let _ = (rust_validator, python_validator, terraform_validator);
}

/// Shows how the CliExecutor works for running arbitrary commands.
fn demo_cli_executor() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║               CLI Executor Demonstration                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let executor = CliExecutor::new()
        .with_timeout(Duration::from_secs(30));

    // Example 1: Simple command
    println!("Example 1: Running 'echo hello'");
    match executor.execute("echo", &["hello", "from", "kkachi"]) {
        Ok(result) => {
            println!("  Success: {}", result.success);
            println!("  stdout: {}", result.stdout.trim());
            println!("  Duration: {}ms", result.duration_ms);
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Example 2: Command with stdin
    println!("\nExample 2: Piping to 'wc -w' (word count)");
    match executor.execute_with_stdin("wc", &["-w"], "one two three four five") {
        Ok(result) => {
            println!("  Word count: {}", result.stdout.trim());
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Example 3: Demonstrating failure handling
    println!("\nExample 3: Handling command failure");
    match executor.execute("false", &[]) {
        Ok(result) => {
            println!("  Success: {}", result.success);
            println!("  Exit code: {:?}", result.exit_code);
        }
        Err(e) => println!("  Error: {}", e),
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("    CLI-Driven Recursive Code Generation - Kkachi Example");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Demonstrate CLI executor
    demo_cli_executor();

    // Demonstrate new CLI API
    demo_new_cli_api();

    // Run the main pipeline
    demo_cli_driven_pipeline();

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                          SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("This example demonstrated:");
    println!("  1. CLI-based code validation (compiler/linter integration)");
    println!("  2. Error capture and feedback loop");
    println!("  3. Iterative refinement using compiler errors as context");
    println!("  4. Diff visualization between iterations");
    println!();
    println!("Key components used:");
    println!("  • cli()        - CLI validator builder");
    println!("  • refine()     - Main refinement entry point");
    println!("  • CliExecutor  - Runs CLI commands with timeout");
    println!("  • TextDiff     - Shows changes between iterations");
    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!("                      Demo Complete!");
    println!("════════════════════════════════════════════════════════════════════\n");
}
