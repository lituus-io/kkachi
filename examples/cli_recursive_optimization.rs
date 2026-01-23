// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! CLI-Driven Recursive Prompt Optimization Example
//!
//! This example demonstrates how to use CLI tool output as feedback
//! for recursive prompt optimization. The pattern:
//!
//! 1. Generate code from prompt
//! 2. Run CLI tool (compiler, linter, tests) on generated code
//! 3. Parse CLI output for errors/warnings
//! 4. Feed errors back into prompt for next iteration
//! 5. Repeat until success or max iterations
//!
//! This is similar to how DSPy's "Assertions" work but using real CLI tools.

#[cfg(feature = "storage")]
use kkachi::recursive::memory;
use kkachi::{
    diff::{DiffRenderer, DiffStyle, TextDiff},
    recursive::cli::{CliCapture, CliExecutor},
};
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   CLI-Driven Recursive Prompt Optimization");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Run all examples
    example_1_rust_compiler_feedback()?;
    example_2_python_linter_feedback()?;
    example_3_multi_stage_cli_pipeline()?;
    example_4_duckdb_rag_with_cli()?;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   All Examples Complete!");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}

/// Example 1: Rust Compiler as Critic
///
/// Uses `rustc` to validate generated Rust code and feeds compiler
/// errors back into the prompt for iterative improvement.
#[allow(clippy::useless_vec)]
fn example_1_rust_compiler_feedback() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 1: Rust Compiler Feedback Loop                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Create a CLI executor for running rustc
    let executor = CliExecutor::new().with_timeout(std::time::Duration::from_secs(30));

    // Simulated LLM responses that progressively improve
    let responses = vec![
        // Iteration 0: Missing imports, type errors
        r#"fn parse_config(path: &str) -> Config {
    let content = fs::read_to_string(path);
    toml::from_str(&content)
}"#,
        // Iteration 1: Added imports, still has errors
        r#"use std::fs;

fn parse_config(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    toml::from_str(&content)
}"#,
        // Iteration 2: Fixed return type
        r#"use std::fs;
use std::error::Error;

fn parse_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}"#,
        // Iteration 3: Complete valid code
        r#"use std::fs;
use std::error::Error;
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    name: String,
    port: u16,
}

fn parse_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

fn main() {}"#,
    ];

    let mut iteration = 0;
    let mut current_code = responses[0].to_string();
    let mut prev_code = String::new();
    let diff_renderer = DiffRenderer::new().with_style(DiffStyle::Unified);

    println!("Task: Generate a Rust function to parse TOML config files\n");

    loop {
        println!("─── Iteration {} ───", iteration);

        // Show diff from previous iteration
        if !prev_code.is_empty() {
            let diff = TextDiff::new(&prev_code, &current_code);
            println!("\nChanges from previous iteration:");
            println!("{}", diff_renderer.render_text(&diff));
        }

        // Write code to temp file for compilation
        let temp_path = "/tmp/kkachi_test.rs";
        fs::write(temp_path, &current_code)?;

        // Run rustc to check the code
        let result = executor.execute(
            "rustc",
            &[
                "--edition=2021",
                "--emit=metadata",
                "-o",
                "/tmp/kkachi_test",
                temp_path,
            ],
        )?;

        println!("\nCompiler output:");
        if result.success {
            println!("  ✅ Compilation successful!");
            break;
        } else {
            // Parse errors from compiler output
            let errors = parse_rust_errors(&result);
            println!("  ❌ {} error(s) found:", errors.len());
            for (i, err) in errors.iter().enumerate() {
                println!("    {}. {}", i + 1, err);
            }

            // Calculate score based on error count
            let score = if errors.is_empty() {
                1.0
            } else {
                1.0 / (errors.len() as f64 + 1.0)
            };
            println!("\n  Score: {:.0}%", score * 100.0);
        }

        // Move to next iteration
        iteration += 1;
        if iteration >= responses.len() {
            println!("\n  Max iterations reached");
            break;
        }

        prev_code = current_code.clone();
        current_code = responses[iteration].to_string();
        println!();
    }

    println!("\n═══ Final Code ═══");
    println!("{}", current_code);
    println!();

    Ok(())
}

/// Parse Rust compiler errors into structured feedback
fn parse_rust_errors(result: &CliCapture) -> Vec<String> {
    let mut errors = Vec::new();
    let output = format!("{}\n{}", result.stdout, result.stderr);

    for line in output.lines() {
        if line.contains("error[E") || line.contains("error:") {
            // Extract the error message
            if let Some(msg) = line.split("error").nth(1) {
                errors.push(format!("Error{}", msg.trim()));
            }
        }
    }

    if errors.is_empty() && !result.success {
        errors.push("Compilation failed with unknown error".to_string());
    }

    errors
}

/// Example 2: Python Linter Feedback
///
/// Uses Python's built-in syntax checker and pylint for code quality feedback.
#[allow(clippy::useless_vec)]
fn example_2_python_linter_feedback() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 2: Python Linter Feedback Loop                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let executor = CliExecutor::new();

    // Simulated Python code that improves over iterations
    let responses = vec![
        // Iteration 0: Syntax error, missing colon
        r#"def fetch_data(url)
    response = requests.get(url)
    return response.json()"#,
        // Iteration 1: Fixed syntax, missing import
        r#"def fetch_data(url):
    response = requests.get(url)
    return response.json()"#,
        // Iteration 2: Added import, no error handling
        r#"import requests

def fetch_data(url):
    response = requests.get(url)
    return response.json()"#,
        // Iteration 3: Complete with error handling
        r#"import requests
from typing import Any, Optional

def fetch_data(url: str) -> Optional[Any]:
    """Fetch JSON data from a URL.

    Args:
        url: The URL to fetch data from.

    Returns:
        The JSON response data, or None if the request fails.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None"#,
    ];

    let mut iteration = 0;
    let mut current_code = responses[0].to_string();

    println!("Task: Generate a Python function to fetch JSON from a URL\n");

    loop {
        println!("─── Iteration {} ───", iteration);
        println!("\nGenerated code:\n{}", current_code);

        // Write to temp file
        let temp_path = "/tmp/kkachi_test.py";
        fs::write(temp_path, &current_code)?;

        // Check Python syntax
        let syntax_result = executor.execute("python3", &["-m", "py_compile", temp_path])?;

        if !syntax_result.success {
            println!("\n  ❌ Syntax error:");
            println!(
                "    {}",
                syntax_result
                    .stderr
                    .lines()
                    .next()
                    .unwrap_or("Unknown error")
            );
        } else {
            println!("\n  ✅ Syntax valid");

            // Run basic static analysis with Python's ast module
            let ast_check = executor.execute_with_stdin(
                "python3",
                &[
                    "-c",
                    "import ast; ast.parse(open('/tmp/kkachi_test.py').read())",
                ],
                "",
            )?;

            if ast_check.success {
                // Check for common issues
                let issues = check_python_quality(&current_code);
                if issues.is_empty() {
                    println!("  ✅ Code quality checks passed!");
                    break;
                } else {
                    println!("  ⚠️  Quality issues found:");
                    for issue in &issues {
                        println!("    - {}", issue);
                    }
                }
            }
        }

        iteration += 1;
        if iteration >= responses.len() {
            println!("\n  Max iterations reached");
            break;
        }

        current_code = responses[iteration].to_string();
        println!();
    }

    println!("\n═══ Final Code ═══");
    println!("{}", current_code);
    println!();

    Ok(())
}

/// Check Python code quality (simulated linter)
fn check_python_quality(code: &str) -> Vec<String> {
    let mut issues = Vec::new();

    if !code.contains("import") && code.contains("requests") {
        issues.push("Missing import for 'requests'".to_string());
    }
    if !code.contains("try:") && code.contains("requests.get") {
        issues.push("No error handling for network requests".to_string());
    }
    if !code.contains("\"\"\"") && !code.contains("'''") {
        issues.push("Missing docstring".to_string());
    }
    if !code.contains("->") && code.contains("def ") {
        issues.push("Missing return type annotation".to_string());
    }

    issues
}

/// Example 3: Multi-Stage CLI Pipeline
///
/// Demonstrates a pipeline with multiple CLI tools:
/// 1. Format check (rustfmt)
/// 2. Lint check (clippy)
/// 3. Compile check (rustc)
/// 4. Test execution (cargo test)
fn example_3_multi_stage_cli_pipeline() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 3: Multi-Stage CLI Pipeline                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Define pipeline stages
    let stages = vec![
        ("Format", "rustfmt", vec!["--check"]),
        ("Syntax", "rustc", vec!["--edition=2021", "--emit=metadata"]),
    ];

    // Code that passes all stages
    let code = r#"use std::collections::HashMap;

/// A simple key-value cache with expiration.
pub struct Cache<T> {
    data: HashMap<String, T>,
    capacity: usize,
}

impl<T: Clone> Cache<T> {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Get a value from the cache.
    pub fn get(&self, key: &str) -> Option<T> {
        self.data.get(key).cloned()
    }

    /// Insert a value into the cache.
    pub fn insert(&mut self, key: String, value: T) {
        if self.data.len() >= self.capacity {
            // Simple eviction: remove first key
            if let Some(first_key) = self.data.keys().next().cloned() {
                self.data.remove(&first_key);
            }
        }
        self.data.insert(key, value);
    }
}

fn main() {}
"#;

    let temp_path = "/tmp/kkachi_pipeline.rs";
    fs::write(temp_path, code)?;

    println!("Running multi-stage validation pipeline:\n");

    let executor = CliExecutor::new();
    let mut all_passed = true;

    for (stage_name, cmd, base_args) in &stages {
        print!("  Stage: {} ({})... ", stage_name, cmd);

        let mut args: Vec<&str> = base_args.to_vec();

        // Add output path for rustc
        if *cmd == "rustc" {
            args.push("-o");
            args.push("/tmp/kkachi_pipeline");
        }
        args.push(temp_path);

        let result = executor.execute(cmd, &args)?;

        if result.success {
            println!("✅ Passed");
        } else {
            println!("❌ Failed");
            if !result.stderr.is_empty() {
                for line in result.stderr.lines().take(3) {
                    println!("      {}", line);
                }
            }
            all_passed = false;
        }
    }

    println!();
    if all_passed {
        println!("  🎉 All stages passed!");
    } else {
        println!("  ⚠️  Some stages failed - would trigger re-generation");
    }

    println!();
    Ok(())
}

/// Example 4: Persistent RAG with CLI Validation
///
/// Combines persistent memory storage for few-shot learning with CLI validation.
/// Successful code snippets are stored in DuckDB for future retrieval.
#[cfg(feature = "storage")]
fn example_4_duckdb_rag_with_cli() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 4: Persistent RAG + CLI Validation Pipeline           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Create persistent memory store (uses DuckDB under the hood)
    let mut mem = memory().persist("/tmp/kkachi_cli_rag.db")?;

    // Seed with known-good examples if empty
    if mem.is_empty() {
        let examples = vec![
            ("rust:file_read", "use std::fs; fn read_file(path: &str) -> std::io::Result<String> { fs::read_to_string(path) }"),
            ("rust:json_parse", "use serde_json::Value; fn parse_json(s: &str) -> serde_json::Result<Value> { serde_json::from_str(s) }"),
            ("rust:http_get", "use reqwest; async fn fetch(url: &str) -> reqwest::Result<String> { reqwest::get(url).await?.text().await }"),
        ];

        println!("Step 1: Seeding memory with {} examples...", examples.len());
        for (tag, content) in &examples {
            mem.add_tagged(tag, content);
        }
    } else {
        println!("Step 1: Memory already has {} examples", mem.len());
    }

    // Query for similar examples
    let query = "parse yaml configuration file";
    println!("\nStep 2: Querying for: \"{}\"", query);

    let results = mem.search(query, 3);
    println!("  Found {} similar examples:", results.len());
    for (i, result) in results.iter().enumerate() {
        let tag = result.tag.as_deref().unwrap_or("none");
        println!("    {}. {} (score: {:.3})", i + 1, tag, result.score);
    }

    // Generate code based on retrieved context
    println!("\nStep 3: Generating code with RAG context...");
    let generated_code = r#"use std::fs;
use serde::Deserialize;
use serde_yaml;

#[derive(Deserialize)]
struct Config {
    name: String,
    version: String,
}

fn parse_yaml(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let config: Config = serde_yaml::from_str(&content)?;
    Ok(config)
}

fn main() {}"#;

    println!("  Generated {} bytes of code", generated_code.len());

    // Validate with CLI
    println!("\nStep 4: Validating with CLI...");
    let temp_path = "/tmp/kkachi_rag_test.rs";
    fs::write(temp_path, generated_code)?;

    let executor = CliExecutor::new();
    let result = executor.execute(
        "rustc",
        &[
            "--edition=2021",
            "--emit=metadata",
            "-o",
            "/tmp/kkachi_rag_test",
            temp_path,
        ],
    )?;

    if result.success {
        println!("  ✅ Validation passed!");

        // Store successful result back to memory
        println!("\nStep 5: Storing successful code in memory...");
        let new_tag = format!("rust:yaml_parse_{:x}", hash_code(generated_code));
        mem.add_tagged(&new_tag, generated_code);
        println!("  Stored with tag: {}", new_tag);
        println!("  Total examples in memory: {}", mem.len());
    } else {
        println!("  ❌ Validation failed - would trigger re-generation");
        println!(
            "  Error: {}",
            result.stderr.lines().next().unwrap_or("Unknown")
        );
    }

    println!();
    Ok(())
}

#[cfg(not(feature = "storage"))]
fn example_4_duckdb_rag_with_cli() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 4: DuckDB RAG + CLI Validation Pipeline               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    println!("  ⚠️  This example requires the 'storage' feature.");
    println!("  Run with: cargo run --example cli_recursive_optimization --features storage\n");
    Ok(())
}

/// Simple hash function for generating IDs
#[cfg(feature = "storage")]
fn hash_code(code: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    code.hash(&mut hasher);
    hasher.finish()
}
