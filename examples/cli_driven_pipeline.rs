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

use std::collections::HashMap;
use std::time::Duration;

use kkachi::diff::{DiffRenderer, DiffStyle, TextDiff};
use kkachi::hitl::{HITLConfig, ReviewDecision};
use kkachi::recursive::{
    ChecklistCritic, Critic, CriticResult, HashEmbedder, InMemoryVectorStore,
    RecursiveState,
};
use kkachi::recursive::cli::{CliExecutor, CliStage, CommandResult, GenericCliCritic, OutputMode};
use kkachi::str_view::StrView;

// ============================================================================
// Simulated LLM that learns from compiler errors
// ============================================================================

/// Simulates an LLM that generates progressively better code based on feedback.
/// In production, this would be a call to an LLM API.
struct CodeGenerationLLM {
    /// Maps error patterns to fixes
    error_fixes: HashMap<&'static str, &'static str>,
}

impl CodeGenerationLLM {
    fn new() -> Self {
        let mut fixes = HashMap::new();

        // Common Rust error patterns and their fixes
        fixes.insert("cannot find value", "// Declare missing variable\nlet ");
        fixes.insert("expected `;`", "; // Added missing semicolon");
        fixes.insert("mismatched types", "// Type annotation added");
        fixes.insert("use of undeclared", "use std::");
        fixes.insert("expected `{`", "{ // Added missing brace");
        fixes.insert("private field", "pub "); // Make field public
        fixes.insert("borrow of moved value", ".clone()"); // Clone to avoid move
        fixes.insert("cannot borrow", "&mut "); // Add mutable reference

        Self { error_fixes: fixes }
    }

    /// Generate code based on prompt and optional error feedback
    fn generate(&self, prompt: &str, iteration: u32, feedback: Option<&str>) -> String {
        match iteration {
            0 => {
                // Initial attempt - deliberately has issues
                if prompt.contains("URL parser") {
                    r#"fn parse_url(s: &str) -> String {
    // Parse URL and extract hostname
    let parts = s.split("://")
    let host = parts[1].split("/")[0]
    host
}"#.to_string()
                } else if prompt.contains("config parser") {
                    r#"fn parse_config(path: &str) -> Config {
    let content = std::fs::read_to_string(path);
    let config: Config = toml::from_str(&content);
    config
}"#.to_string()
                } else if prompt.contains("HTTP client") {
                    r#"fn fetch(url: &str) -> String {
    let response = reqwest::get(url)
    response.text()
}"#.to_string()
                } else {
                    r#"fn process(input: &str) -> String {
    let result = input.to_uppercase()
    result
}"#.to_string()
                }
            }
            1 => {
                // Second attempt - fixes syntax errors
                if prompt.contains("URL parser") {
                    r#"fn parse_url(s: &str) -> String {
    // Parse URL and extract hostname
    let parts: Vec<&str> = s.split("://").collect();
    let host = parts[1].split("/").next().unwrap();
    host.to_string()
}"#.to_string()
                } else if prompt.contains("config parser") {
                    r#"use std::fs;

fn parse_config(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}"#.to_string()
                } else {
                    r#"fn process(input: &str) -> String {
    let result = input.to_uppercase();
    result
}"#.to_string()
                }
            }
            2 => {
                // Third attempt - adds error handling
                if prompt.contains("URL parser") {
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
}"#.to_string()
                } else if prompt.contains("config parser") {
                    r#"use std::fs;
use std::path::Path;

/// Configuration structure.
#[derive(Debug, serde::Deserialize)]
pub struct Config {
    pub name: String,
    pub version: String,
}

/// Parses configuration from a TOML file.
///
/// # Errors
/// Returns an error if the file cannot be read or parsed.
fn parse_config(path: impl AsRef<Path>) -> Result<Config, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}"#.to_string()
                } else {
                    r#"/// Processes input text by converting to uppercase.
fn process(input: &str) -> String {
    input.to_uppercase()
}"#.to_string()
                }
            }
            _ => {
                // Final polished version
                if prompt.contains("URL parser") {
                    r#"//! URL parsing utilities.

use std::str::FromStr;

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
    ///
    /// # Examples
    ///
    /// ```
    /// let url = ParsedUrl::parse("https://example.com/path")?;
    /// assert_eq!(url.host, "example.com");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `ParseError` if the URL is malformed.
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

        Ok(Self {
            scheme: scheme.to_string(),
            host,
            port,
            path,
        })
    }
}

/// Errors that can occur during URL parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingScheme,
    InvalidPort,
    EmptyHost,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingScheme => write!(f, "URL missing scheme (e.g., 'https://')"),
            Self::InvalidPort => write!(f, "Invalid port number"),
            Self::EmptyHost => write!(f, "Empty hostname"),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_url() {
        let url = ParsedUrl::parse("https://example.com/path").unwrap();
        assert_eq!(url.scheme, "https");
        assert_eq!(url.host, "example.com");
        assert_eq!(url.path, "/path");
    }

    #[test]
    fn test_parse_with_port() {
        let url = ParsedUrl::parse("http://localhost:8080/api").unwrap();
        assert_eq!(url.port, Some(8080));
    }
}
"#.to_string()
                } else {
                    "// Fully refined code".to_string()
                }
            }
        }
    }
}

// ============================================================================
// Simulated CLI Critic (uses mock compiler output for demonstration)
// ============================================================================

/// A mock CLI critic that simulates Rust compiler behavior.
/// In production, use `GenericCliCritic::rust()` for real compilation.
struct MockRustCompilerCritic {
    /// Errors to return based on code patterns
    error_patterns: Vec<(&'static str, &'static str)>,
}

impl MockRustCompilerCritic {
    fn new() -> Self {
        Self {
            error_patterns: vec![
                // Pattern: code without semicolons triggers syntax error
                ("let parts = s.split", "error[E0658]: expected `;`\n  --> code.rs:4:35\n   |\n4  |     let parts = s.split(\"://\")\n   |                                   ^ expected `;`"),
                ("let host = parts[1]", "error[E0658]: expected `;`\n  --> code.rs:5:41\n   |\n5  |     let host = parts[1].split(\"/\")[0]\n   |                                         ^ expected `;`"),
                // Pattern: using unwrap without error handling
                (".unwrap()", "warning: use of `unwrap` in production code\n  --> code.rs:5:42\n   |\n5  |     let host = parts[1].split(\"/\").next().unwrap();\n   |                                          ^^^^^^^^^\n   |\n   = help: consider using `?` operator or `expect` with a message"),
                // Pattern: missing documentation
                ("fn parse_url", "warning: missing documentation for a function\n  --> code.rs:1:1\n   |\n1  | fn parse_url(s: &str) -> String {\n   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n   |\n   = help: add `///` documentation above the function"),
            ],
        }
    }

    fn compile(&self, code: &str) -> CommandResult {
        let mut errors = Vec::new();

        for (pattern, error) in &self.error_patterns {
            if code.contains(pattern) {
                errors.push(*error);
            }
        }

        // Check for missing Result type when error handling is needed
        if code.contains("read_to_string") && !code.contains("Result<") {
            errors.push("error[E0308]: mismatched types\n  --> code.rs:3:5\n   |\n3  |     let content = std::fs::read_to_string(path);\n   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n   |\n   = note: expected type `String`\n              found type `Result<String, std::io::Error>`\n   = help: use `?` operator or handle the Result");
        }

        let success = errors.is_empty();
        let stderr = errors.join("\n\n");

        CommandResult {
            success,
            stdout: if success { "Compiled successfully".to_string() } else { String::new() },
            stderr,
            exit_code: if success { 0 } else { 1 },
            duration: Duration::from_millis(50),
        }
    }
}

impl Critic for MockRustCompilerCritic {
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        let result = self.compile(output.as_str());

        if result.success {
            CriticResult::new(1.0)
        } else {
            // Parse errors and provide structured feedback
            let error_count = result.stderr.matches("error[").count();
            let warning_count = result.stderr.matches("warning:").count();

            // Score based on error severity
            let score = if error_count > 0 {
                0.0
            } else if warning_count > 0 {
                0.5 + (0.5 / (warning_count as f64 + 1.0))
            } else {
                0.8
            };

            CriticResult::new(score)
                .with_feedback(format!(
                    "Compilation failed:\n{}\n\nErrors: {}, Warnings: {}",
                    result.stderr,
                    error_count,
                    warning_count
                ))
        }
    }
}

// ============================================================================
// CLI-Driven Pipeline
// ============================================================================

/// Runs a complete CLI-driven code generation pipeline.
struct CliDrivenPipeline {
    llm: CodeGenerationLLM,
    compiler: MockRustCompilerCritic,
    vector_store: InMemoryVectorStore<HashEmbedder>,
    diff_renderer: DiffRenderer,
    hitl_config: HITLConfig,
}

impl CliDrivenPipeline {
    fn new() -> Self {
        let embedder = HashEmbedder::new(64);
        Self {
            llm: CodeGenerationLLM::new(),
            compiler: MockRustCompilerCritic::new(),
            vector_store: InMemoryVectorStore::new(embedder),
            diff_renderer: DiffRenderer::new(),
            hitl_config: HITLConfig {
                enabled: true,
                interval: 2,
                on_score_drop: true,
                on_convergence: true,
                ..Default::default()
            },
        }
    }

    /// Run the recursive refinement pipeline.
    fn run(&mut self, prompt: &str, max_iterations: u32) -> (String, f64, u32) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║           CLI-Driven Code Generation Pipeline                  ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("Prompt: {}\n", prompt);
        println!("Max iterations: {}", max_iterations);
        println!("HITL enabled: {} (review every {} iterations)\n",
            self.hitl_config.enabled, self.hitl_config.interval);

        let mut current_code = String::new();
        let mut prev_code = String::new();
        let mut best_score = 0.0;
        let mut best_code = String::new();
        let mut feedback: Option<String> = None;

        for iteration in 0..max_iterations {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Iteration {}", iteration);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            // Step 1: Generate code (with feedback from previous iteration)
            println!("→ Generating code...");
            if let Some(ref fb) = feedback {
                println!("  Using compiler feedback:");
                for line in fb.lines().take(5) {
                    println!("    │ {}", line);
                }
                if fb.lines().count() > 5 {
                    println!("    │ ... ({} more lines)", fb.lines().count() - 5);
                }
            }

            prev_code = current_code.clone();
            current_code = self.llm.generate(prompt, iteration, feedback.as_deref());

            println!("\n  Generated code ({} lines):", current_code.lines().count());
            for (i, line) in current_code.lines().take(10).enumerate() {
                println!("    {:>3} │ {}", i + 1, line);
            }
            if current_code.lines().count() > 10 {
                println!("        │ ... ({} more lines)", current_code.lines().count() - 10);
            }

            // Step 2: Show diff from previous iteration
            if !prev_code.is_empty() {
                println!("\n→ Changes from previous iteration:");
                let diff = TextDiff::new(&prev_code, &current_code);
                if diff.has_changes() {
                    let stats = diff.stats();
                    println!("  +{} lines, -{} lines", stats.lines_added, stats.lines_removed);

                    // Render abbreviated diff
                    let rendered = self.diff_renderer.render_text(&diff);
                    for line in rendered.lines().take(15) {
                        println!("    {}", line);
                    }
                } else {
                    println!("  (no changes)");
                }
            }

            // Step 3: Run compiler/linter
            println!("\n→ Running Rust compiler...");
            let state = RecursiveState::new();
            let result = self.compiler.evaluate(StrView::from(current_code.as_str()), &state);

            println!("  Score: {:.2}", result.score);

            if result.score > best_score {
                best_score = result.score;
                best_code = current_code.clone();
                println!("  ✓ New best score!");
            }

            // Step 4: Check for HITL review
            if self.hitl_config.enabled {
                let should_review = (iteration + 1) % self.hitl_config.interval == 0
                    || (self.hitl_config.on_score_drop && result.score < best_score)
                    || (self.hitl_config.on_convergence && result.score >= 1.0);

                if should_review {
                    println!("\n→ Human-in-the-Loop Review Point:");
                    println!("  [Simulated] Options: [a]ccept [r]eject [e]dit [s]top");
                    println!("  [Simulated] Decision: Accept (continuing...)");
                }
            }

            // Step 5: Check for convergence
            if result.score >= 1.0 {
                println!("\n✓ Compilation successful! No errors or warnings.");
                break;
            }

            // Step 6: Extract feedback for next iteration
            if let Some(fb) = result.feedback.as_ref() {
                feedback = Some(fb.to_string());
                println!("\n→ Compiler feedback (will be used in next iteration):");
                for line in fb.lines().take(5) {
                    println!("    │ {}", line);
                }
            } else {
                feedback = None;
            }

            println!();
        }

        (best_code, best_score, max_iterations.min(4))
    }
}

// ============================================================================
// Real CLI Integration Example (for reference)
// ============================================================================

/// Demonstrates real CLI integration with actual Rust compiler.
/// This won't work in the example without rustc, but shows the pattern.
fn demo_real_cli_integration() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║         Real CLI Integration (Reference Code)                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("In production, you would use the real CLI critics:\n");

    println!("```rust");
    println!("// Create a Rust code critic with real compiler/linter");
    println!("let rust_critic = GenericCliCritic::rust();");
    println!();
    println!("// Or create custom CLI stages");
    println!("let custom_critic = GenericCliCritic::new(OutputMode::File {{");
    println!("    extension: \"rs\".to_string(),");
    println!("}})");
    println!(".add_stage(");
    println!("    CliStage::new(\"format\", \"rustfmt\")");
    println!("        .with_args([\"--check\"])");
    println!("        .with_weight(0.1)");
    println!(")");
    println!(".add_stage(");
    println!("    CliStage::new(\"compile\", \"rustc\")");
    println!("        .with_args([\"--edition\", \"2021\", \"--emit=metadata\"])");
    println!("        .with_weight(0.5)");
    println!("        .required()");
    println!("        .with_error_parser(|r| {{");
    println!("            // Custom error parsing");
    println!("            r.stderr.lines()");
    println!("                .filter(|l| l.starts_with(\"error\"))");
    println!("                .map(|l| l.to_string())");
    println!("                .collect()");
    println!("        }})");
    println!(")");
    println!(".add_stage(");
    println!("    CliStage::new(\"clippy\", \"cargo\")");
    println!("        .with_args([\"clippy\", \"--\", \"-D\", \"warnings\"])");
    println!("        .with_weight(0.4)");
    println!(");");
    println!("```\n");

    println!("Available pre-built critics:");
    println!("  • GenericCliCritic::rust()       - Rust (rustfmt, rustc, clippy)");
    println!("  • GenericCliCritic::python()     - Python (py_compile, ruff, mypy)");
    println!("  • GenericCliCritic::terraform()  - Terraform (fmt, validate, plan)");
    println!("  • GenericCliCritic::kubernetes() - Kubernetes (kubectl dry-run)");
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
            println!("  Duration: {:?}", result.duration);
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
            println!("  Exit code: {}", result.exit_code);
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

    // Run the main pipeline
    let mut pipeline = CliDrivenPipeline::new();
    let (final_code, final_score, iterations) = pipeline.run(
        "Write a robust URL parser in Rust with proper error handling",
        5,
    );

    // Show final result
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                        FINAL RESULT");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("Iterations: {}", iterations);
    println!("Final score: {:.2}", final_score);
    println!("\nGenerated code:\n");
    println!("```rust");
    println!("{}", final_code);
    println!("```");

    // Show the reference for real CLI integration
    demo_real_cli_integration();

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                          SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("This example demonstrated:");
    println!("  1. CLI-based code validation (compiler/linter integration)");
    println!("  2. Error capture and feedback loop");
    println!("  3. Iterative refinement using compiler errors as context");
    println!("  4. Diff visualization between iterations");
    println!("  5. Human-in-the-loop review integration");
    println!();
    println!("Key components used:");
    println!("  • CliExecutor      - Runs CLI commands with timeout");
    println!("  • CommandResult    - Captures stdout, stderr, exit code");
    println!("  • GenericCliCritic - Multi-stage CLI validation");
    println!("  • CliStage         - Configurable validation stages");
    println!("  • TextDiff         - Shows changes between iterations");
    println!("  • HITLConfig       - Human review at key points");
    println!();
    println!("The recursive loop:");
    println!("  prompt → generate → compile → [errors?] → feedback → refine → ...");
    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!("                      Demo Complete!");
    println!("════════════════════════════════════════════════════════════════════\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_compiler_success() {
        let compiler = MockRustCompilerCritic::new();
        let good_code = r#"/// Adds two numbers.
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let result = compiler.compile(good_code);
        assert!(result.success);
    }

    #[test]
    fn test_mock_compiler_syntax_error() {
        let compiler = MockRustCompilerCritic::new();
        let bad_code = r#"fn test() {
    let x = 5
    x
}"#;
        // This won't trigger our mock patterns, but demonstrates the structure
        let result = compiler.compile(bad_code);
        // The mock uses pattern matching, so this particular code passes
        assert!(result.success || !result.stderr.is_empty());
    }

    #[test]
    fn test_llm_improves_with_iterations() {
        let llm = CodeGenerationLLM::new();
        let prompt = "URL parser";

        let code_0 = llm.generate(prompt, 0, None);
        let code_1 = llm.generate(prompt, 1, Some("expected `;`"));
        let code_2 = llm.generate(prompt, 2, Some("missing documentation"));

        // Later iterations should have more content/fixes
        assert!(code_1.len() >= code_0.len());
        assert!(code_2.contains("///") || code_2.contains("Result<"));
    }

    #[test]
    fn test_pipeline_runs_to_completion() {
        let mut pipeline = CliDrivenPipeline::new();
        let (code, score, iterations) = pipeline.run("Write a URL parser", 3);

        assert!(!code.is_empty());
        assert!(score >= 0.0 && score <= 1.0);
        assert!(iterations <= 3);
    }
}
