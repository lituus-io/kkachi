// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Template + RAG + CLI Recursive Pipeline with Markdown Output
//!
//! This comprehensive example demonstrates:
//! 1. Template files for structured prompt optimization
//! 2. Local RAG (InMemoryVectorStore) for context retrieval
//! 3. CLI-based validation (simulated compiler/linter errors)
//! 4. Iterative refinement capturing errors to improve context
//! 5. Markdown output file generation based on the template
//!
//! Run with: cargo run --example template_rag_cli_pipeline

use std::collections::HashMap;
use std::fs;

use kkachi::error::Result;
use kkachi::recursive::{
    Template,
    HashEmbedder, InMemoryVectorStore, VectorStore,
    Critic, CriticResult,
    StandaloneRunner, RecursiveConfig, RecursiveState, RefinementResult,
};
use kkachi::str_view::StrView;

// ============================================================================
// Template Definition
// ============================================================================

const CODE_GENERATION_TEMPLATE: &str = r#"---
name: rust_code_generator
version: "1.0"
signature: "requirement -> code"
format:
  type: json
  schema:
    type: object
    required:
      - code
      - explanation
options:
  strict: false
  include_in_prompt: true
---
You are an expert Rust programmer. Generate production-quality code.

## Output Format

Return JSON with:
- code: The Rust source code
- explanation: Brief explanation

## Requirements

1. Use Result for fallible operations
2. Avoid .unwrap()
3. Include documentation
"#;

// ============================================================================
// RAG Knowledge Base
// ============================================================================

const RAG_EXAMPLES: &[(&str, &str)] = &[
    ("rust:error_handling", "Pattern: Error Handling - Use Result<T, E> for operations that can fail. Use ? operator for propagation."),
    ("rust:testing", "Pattern: Unit Testing - Use #[cfg(test)] mod tests { use super::*; #[test] fn test_name() { } }"),
    ("rust:documentation", "Pattern: Documentation - Use /// for doc comments with # Examples and # Errors sections."),
    ("rust:cli_parsing", "Pattern: CLI Parsing - Use clap with #[derive(Parser)] for argument handling."),
];

// ============================================================================
// CLI Validator
// ============================================================================

struct CliValidator {
    error_patterns: Vec<(&'static str, &'static str)>,
}

impl CliValidator {
    fn new() -> Self {
        Self {
            error_patterns: vec![
                (".unwrap()", "error: Use of .unwrap() - consider using ? operator"),
                ("panic!", "warning: Direct panic! usage"),
                ("unsafe {", "warning: unsafe block detected"),
            ],
        }
    }

    fn validate(&self, code: &str) -> Vec<String> {
        let mut issues = Vec::new();
        for (pattern, message) in &self.error_patterns {
            if code.contains(pattern) {
                issues.push(message.to_string());
            }
        }
        if code.contains("pub fn") && !code.contains("///") {
            issues.push("warning: Public function missing documentation".to_string());
        }
        if (code.contains("fs::") || code.contains("File::")) && !code.contains("Result") {
            issues.push("error: File operations should return Result".to_string());
        }
        issues
    }
}

// ============================================================================
// CLI-aware Critic
// ============================================================================

struct CliTemplateCritic<'t> {
    template: &'t Template<'t>,
    validator: CliValidator,
}

impl<'t> CliTemplateCritic<'t> {
    fn new(template: &'t Template<'t>) -> Self {
        Self {
            template,
            validator: CliValidator::new(),
        }
    }
}

impl<'t> Critic for CliTemplateCritic<'t> {
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        let text = output.as_str();
        let mut total_score = 0.0;
        let mut issues = Vec::new();
        let mut breakdown = Vec::new();

        // Format validation (0.2)
        let format_score = match self.template.validate_output(text) {
            Ok(()) => 1.0,
            Err(e) => { issues.push(format!("Format: {}", e)); 0.0 }
        };
        breakdown.push(("format".to_string(), format_score));
        total_score += format_score * 0.2;

        // Extract code
        let code = extract_code_from_json(text);

        // CLI validation (0.4)
        let cli_issues = self.validator.validate(&code);
        let cli_score = if cli_issues.is_empty() {
            1.0
        } else {
            let penalty = (cli_issues.len() as f64 * 0.2).min(1.0);
            issues.extend(cli_issues.iter().map(|s| format!("CLI: {}", s)));
            1.0 - penalty
        };
        breakdown.push(("cli_validation".to_string(), cli_score));
        total_score += cli_score * 0.4;

        // Heuristic checks (0.4) - inline evaluation to avoid lifetime issues
        let mut heuristic_score: f64 = 1.0;
        let mut heuristic_issues = Vec::new();

        if code.len() < 50 {
            heuristic_score -= 0.33;
            heuristic_issues.push("Output too short (min 50 chars)");
        }
        if !code.contains("fn ") {
            heuristic_score -= 0.33;
            heuristic_issues.push("Missing function definition");
        }
        if code.contains(".unwrap()") {
            heuristic_score -= 0.34;
            heuristic_issues.push("Contains .unwrap()");
        }

        heuristic_score = heuristic_score.max(0.0);
        breakdown.push(("heuristics".to_string(), heuristic_score));
        if !heuristic_issues.is_empty() {
            issues.push(format!("Heuristic: {}", heuristic_issues.join(", ")));
        }
        total_score += heuristic_score * 0.4;

        let mut result = CriticResult::new(total_score).with_breakdown(breakdown);
        if !issues.is_empty() {
            result = result.with_feedback(issues.join("\n"));
        }
        result
    }
}

fn extract_code_from_json(json_str: &str) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
        if let Some(code) = value.get("code").and_then(|v| v.as_str()) {
            return code.to_string();
        }
    }
    json_str.to_string()
}

// ============================================================================
// Markdown Reporter
// ============================================================================

struct MarkdownReporter {
    iterations: Vec<IterationLog>,
}

struct IterationLog {
    iteration: u32,
    score: f64,
    feedback: Option<String>,
    output_preview: String,
}

impl MarkdownReporter {
    fn new() -> Self {
        Self { iterations: Vec::new() }
    }

    fn log_iteration(&mut self, iteration: u32, score: f64, feedback: Option<&str>, output: &str) {
        self.iterations.push(IterationLog {
            iteration,
            score,
            feedback: feedback.map(|s| s.to_string()),
            output_preview: output.chars().take(200).collect(),
        });
    }

    fn generate_report(
        &self,
        template: &Template<'_>,
        question: &str,
        result: &RefinementResult,
        rag_examples_used: &[String],
    ) -> String {
        let mut md = String::new();

        md.push_str("# Code Generation Report\n\n");
        md.push_str(&format!("**Template:** {}\n", template.name));
        md.push_str(&format!("**Version:** {}\n\n", template.version));

        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Final Score | {:.2} |\n", result.score));
        md.push_str(&format!("| Iterations | {} |\n", result.iterations));
        md.push_str(&format!("| Context ID | {} |\n\n", result.context_id));

        md.push_str("## Input\n\n");
        md.push_str(&format!("> {}\n\n", question));

        md.push_str("## RAG Context Retrieved\n\n");
        if rag_examples_used.is_empty() {
            md.push_str("_No similar examples found._\n\n");
        } else {
            for (i, ex) in rag_examples_used.iter().enumerate() {
                md.push_str(&format!("### Example {}\n\n```\n{}\n```\n\n", i + 1, truncate(ex, 200)));
            }
        }

        md.push_str("## Iteration History\n\n");
        for log in &self.iterations {
            md.push_str(&format!("### Iteration {}\n\n", log.iteration));
            md.push_str(&format!("**Score:** {:.2}\n\n", log.score));
            if let Some(ref fb) = log.feedback {
                md.push_str(&format!("**Feedback:**\n```\n{}\n```\n\n", fb));
            }
            md.push_str(&format!("**Output:**\n```\n{}\n```\n\n", log.output_preview));
        }

        if !result.error_corrections.is_empty() {
            md.push_str("## Error Corrections\n\n");
            for (i, (error, _)) in result.error_corrections.iter().enumerate() {
                md.push_str(&format!("{}. {}\n", i + 1, error));
            }
            md.push_str("\n");
        }

        md.push_str("## Final Output\n\n");
        md.push_str("```json\n");
        md.push_str(&result.answer);
        md.push_str("\n```\n\n");

        let code = extract_code_from_json(&result.answer);
        if !code.is_empty() && code != result.answer {
            md.push_str("### Generated Code\n\n```rust\n");
            md.push_str(&code);
            md.push_str("\n```\n\n");
        }

        md.push_str("---\n*Generated by Kkachi Template + RAG + CLI Pipeline*\n");
        md
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len { s.to_string() } else { format!("{}...", &s[..max_len]) }
}

// ============================================================================
// Mock LLM
// ============================================================================

struct MockCodeLLM {
    responses: HashMap<u32, String>,
}

impl MockCodeLLM {
    fn new(question: &str) -> Self {
        let mut responses = HashMap::new();

        if question.contains("CLI") || question.contains("argument") {
            responses.insert(0, r#"{"code": "fn parse_args() { let args = std::env::args().collect::<Vec<_>>(); args[1].clone().unwrap(); }", "explanation": "Basic parsing"}"#.to_string());
            responses.insert(1, r#"{"code": "use std::env;\n\nfn parse_args() -> Result<String, &'static str> {\n    env::args().nth(1).ok_or(\"Missing arg\")\n}", "explanation": "Added Result"}"#.to_string());
            responses.insert(2, r#"{"code": "use std::env;\n\n/// Parses CLI arguments.\n///\n/// # Errors\n/// Returns error if args missing.\npub fn parse_args() -> Result<Args, ArgError> {\n    let args: Vec<String> = env::args().collect();\n    let input = args.get(1).ok_or(ArgError::Missing)?.clone();\n    Ok(Args { input })\n}\n\npub struct Args { pub input: String }\npub enum ArgError { Missing }", "explanation": "Production ready"}"#.to_string());
        } else {
            responses.insert(0, r#"{"code": "fn parse_url(s: &str) -> String { s.split(\"://\").nth(1).unwrap().to_string() }", "explanation": "Basic"}"#.to_string());
            responses.insert(1, r#"{"code": "fn parse_url(s: &str) -> Result<String, &'static str> { s.split(\"://\").nth(1).map(|s| s.to_string()).ok_or(\"Invalid\") }", "explanation": "With Result"}"#.to_string());
            responses.insert(2, r#"{"code": "/// Parses URL host.\n///\n/// # Errors\n/// Returns error if malformed.\npub fn parse_url(s: &str) -> Result<String, ParseError> {\n    let rest = s.split(\"://\").nth(1).ok_or(ParseError::NoScheme)?;\n    let host = rest.split('/').next().ok_or(ParseError::NoHost)?;\n    Ok(host.to_string())\n}\n\npub enum ParseError { NoScheme, NoHost }", "explanation": "Production ready"}"#.to_string());
        }
        Self { responses }
    }

    fn generate(&self, iteration: u32, _context: &str, _feedback: Option<&str>) -> String {
        self.responses.get(&iteration)
            .or_else(|| self.responses.get(&2))
            .cloned()
            .unwrap_or_else(|| r#"{"code": "", "explanation": ""}"#.to_string())
    }
}

// ============================================================================
// Main Pipeline
// ============================================================================

fn run_pipeline() -> Result<()> {
    println!("================================================================");
    println!("   Template + RAG + CLI Recursive Pipeline");
    println!("================================================================\n");

    // Step 1: Parse Template
    println!("[Step 1] Loading template...");
    let template = Template::from_str(CODE_GENERATION_TEMPLATE)?;
    println!("  Template: {}", template.name);
    println!("  Format: {:?}\n", template.format.format_type);

    // Step 2: Initialize RAG
    println!("[Step 2] Initializing RAG knowledge base...");
    let embedder = HashEmbedder::new(64);
    let mut rag_store = InMemoryVectorStore::new(embedder);
    for (id, content) in RAG_EXAMPLES {
        rag_store.add(*id, *content);
    }
    println!("  Loaded {} patterns\n", RAG_EXAMPLES.len());

    // Step 3: Define Task
    let question = "Write a CLI argument parser that handles input/output paths and verbose flag";
    println!("[Step 3] Task: {}\n", question);

    // Step 4: Retrieve RAG Context
    println!("[Step 4] Retrieving RAG context...");
    let similar = rag_store.search_text(question, 3);
    let rag_examples_used: Vec<String> = similar.iter()
        .map(|r| format!("[{}] {}", r.id, r.content))
        .collect();
    for (i, r) in similar.iter().enumerate() {
        println!("  {}. {} (score: {:.3})", i + 1, r.id, r.score);
    }
    println!();

    let rag_context: String = similar.iter()
        .map(|r| r.content.clone())
        .collect::<Vec<_>>()
        .join("\n---\n");

    // Step 5: Set up Critic
    println!("[Step 5] Setting up CLI-aware critic...");
    let critic = CliTemplateCritic::new(&template);
    let config = RecursiveConfig {
        max_iterations: 5,
        score_threshold: 0.9,
        ..Default::default()
    };
    println!("  Max iterations: {}\n", config.max_iterations);

    // Step 6: Run Refinement
    println!("[Step 6] Running recursive refinement...\n");
    let mut reporter = MarkdownReporter::new();
    let runner = StandaloneRunner::with_config(&critic, "rust_code", config);
    let mock_llm = MockCodeLLM::new(question);

    let result = runner.refine(question, |iteration, feedback| {
        println!("  --- Iteration {} ---", iteration);

        let prompt = template.assemble_prompt(question, iteration, feedback);
        let full_context = format!("{}\n\nKnowledge:\n{}", prompt, rag_context);
        let response = mock_llm.generate(iteration, &full_context, feedback);

        let code = extract_code_from_json(&response);
        let cli_issues = CliValidator::new().validate(&code);
        println!("  Output: {} chars", response.len());
        if cli_issues.is_empty() {
            println!("  CLI: OK");
        } else {
            println!("  CLI: {} issues", cli_issues.len());
        }

        reporter.log_iteration(iteration, 0.0, feedback, &response);
        println!();
        Ok(response)
    })?;

    // Step 7: Generate Report
    println!("[Step 7] Generating markdown report...");
    let report = reporter.generate_report(&template, question, &result, &rag_examples_used);
    fs::create_dir_all("output").ok();
    let output_path = "output/code_generation_report.md";
    fs::write(output_path, &report)?;
    println!("  Written to: {}\n", output_path);

    // Results
    println!("================================================================");
    println!("   Results");
    println!("================================================================\n");
    println!("  Final Score: {:.2}", result.score);
    println!("  Iterations: {}", result.iterations);
    println!("  Context ID: {}", result.context_id);

    println!("\n  Generated Code Preview:");
    println!("  -----------------------------------------");
    let code = extract_code_from_json(&result.answer);
    for line in code.lines().take(10) {
        println!("  {}", line);
    }
    if code.lines().count() > 10 {
        println!("  ...");
    }

    println!("\nPipeline complete! Check {} for full report.", output_path);
    Ok(())
}

fn main() {
    if let Err(e) = run_pipeline() {
        eprintln!("Pipeline error: {}", e);
        std::process::exit(1);
    }
}
