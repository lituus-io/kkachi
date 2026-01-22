// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Persistent Template Pipeline with DuckDB Storage
//!
//! This advanced example demonstrates:
//! 1. Loading templates from external files
//! 2. DuckDB-backed vector store for persistent RAG
//! 3. Training mode - storing successful refinements
//! 4. Multi-stage validation pipeline
//! 5. Comprehensive markdown reporting
//!
//! Run with: cargo run --example template_persistent_pipeline --features storage

#[cfg(feature = "storage")]
mod pipeline {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;

    use kkachi::error::Result;
    use kkachi::recursive::{
        // Template
        Template, FormatType,
        // Storage
        DuckDBVectorStore, VectorStore, MutableVectorStore, HashEmbedder,
        TrainingExample,
        // Critics and runners
        Critic, CriticResult,
        StandaloneRunner, RecursiveConfig, RecursiveState, RefinementResult,
    };
    use kkachi::str_view::StrView;

    // ========================================================================
    // Configuration
    // ========================================================================

    const DB_PATH: &str = "kkachi_knowledge.db";
    const TEMPLATE_DIR: &str = "examples/templates";
    const OUTPUT_DIR: &str = "examples/output";
    const MIN_SCORE_TO_STORE: f64 = 0.85;

    // ========================================================================
    // Multi-Stage Critic
    // ========================================================================

    /// A multi-stage critic that runs validation in phases
    struct MultiStageCritic<'t> {
        template: &'t Template<'t>,
        stages: Vec<ValidationStage>,
    }

    struct ValidationStage {
        name: &'static str,
        weight: f64,
        required: bool,
        validator: Box<dyn Fn(&str) -> (f64, Vec<String>) + Send + Sync>,
    }

    impl<'t> MultiStageCritic<'t> {
        fn new(template: &'t Template<'t>) -> Self {
            let stages = vec![
                // Stage 1: Format validation (required)
                ValidationStage {
                    name: "format",
                    weight: 0.15,
                    required: true,
                    validator: Box::new(|output| {
                        if let Ok(_) = serde_json::from_str::<serde_json::Value>(output) {
                            (1.0, vec![])
                        } else {
                            (0.0, vec!["Invalid JSON format".to_string()])
                        }
                    }),
                },
                // Stage 2: Required fields check
                ValidationStage {
                    name: "required_fields",
                    weight: 0.20,
                    required: true,
                    validator: Box::new(|output| {
                        let mut score: f64 = 1.0;
                        let mut issues = vec![];

                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(output) {
                            if v.get("code").is_none() {
                                score -= 0.5;
                                issues.push("Missing 'code' field".to_string());
                            }
                            if v.get("explanation").is_none() {
                                score -= 0.5;
                                issues.push("Missing 'explanation' field".to_string());
                            }
                        }
                        (score.max(0.0), issues)
                    }),
                },
                // Stage 3: Code quality checks
                ValidationStage {
                    name: "code_quality",
                    weight: 0.35,
                    required: false,
                    validator: Box::new(|output| {
                        let code = extract_code(output);
                        let mut score: f64 = 1.0;
                        let mut issues = vec![];

                        if code.contains(".unwrap()") {
                            score -= 0.25;
                            issues.push("Uses .unwrap() - prefer ? operator".to_string());
                        }
                        if code.contains("panic!") {
                            score -= 0.15;
                            issues.push("Uses panic! - prefer Result".to_string());
                        }
                        if code.contains("pub fn") && !code.contains("///") {
                            score -= 0.20;
                            issues.push("Public function missing documentation".to_string());
                        }
                        if !code.contains("fn ") {
                            score -= 0.40;
                            issues.push("No function definition found".to_string());
                        }
                        (score.max(0.0), issues)
                    }),
                },
                // Stage 4: Best practices
                ValidationStage {
                    name: "best_practices",
                    weight: 0.30,
                    required: false,
                    validator: Box::new(|output| {
                        let code = extract_code(output);
                        let mut score: f64 = 1.0;
                        let mut issues = vec![];

                        // Check for Result/Option usage
                        if (code.contains("fs::") || code.contains("File::"))
                            && !code.contains("Result") {
                            score -= 0.30;
                            issues.push("I/O operations should return Result".to_string());
                        }

                        // Check for type annotations on public items
                        if code.contains("pub fn") && !code.contains("->") {
                            score -= 0.20;
                            issues.push("Public function missing return type".to_string());
                        }

                        // Positive: good patterns
                        if code.contains("impl") {
                            score += 0.1;
                        }
                        if code.contains("#[derive") {
                            score += 0.1;
                        }

                        (score.clamp(0.0, 1.0), issues)
                    }),
                },
            ];

            Self { template, stages }
        }
    }

    impl<'t> Critic for MultiStageCritic<'t> {
        fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
            let text = output.as_str();
            let mut total_score = 0.0;
            let mut all_issues = Vec::new();
            let mut breakdown = Vec::new();

            for stage in &self.stages {
                let (score, issues) = (stage.validator)(text);

                breakdown.push((stage.name.to_string(), score));
                total_score += score * stage.weight;

                if !issues.is_empty() {
                    for issue in issues {
                        all_issues.push(format!("[{}] {}", stage.name, issue));
                    }
                }

                // Stop on required stage failure
                if stage.required && score < 0.5 {
                    break;
                }
            }

            let mut result = CriticResult::new(total_score).with_breakdown(breakdown);
            if !all_issues.is_empty() {
                result = result.with_feedback(all_issues.join("\n"));
            }
            result
        }
    }

    fn extract_code(json_str: &str) -> String {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
                return code.to_string();
            }
        }
        json_str.to_string()
    }

    // ========================================================================
    // Mock LLM with Learning
    // ========================================================================

    struct AdaptiveMockLLM {
        base_responses: HashMap<u32, String>,
        learned_patterns: Vec<String>,
    }

    impl AdaptiveMockLLM {
        fn new() -> Self {
            let mut responses = HashMap::new();

            // Progressive improvement responses
            responses.insert(0, r#"{"code": "fn process(s: &str) -> String { s.to_uppercase().unwrap() }", "explanation": "Basic"}"#.to_string());
            responses.insert(1, r#"{"code": "fn process(s: &str) -> String { s.to_uppercase() }", "explanation": "Removed unwrap"}"#.to_string());
            responses.insert(2, r#"{"code": "/// Processes input string.\npub fn process(s: &str) -> Result<String, ProcessError> {\n    Ok(s.to_uppercase())\n}\n\npub enum ProcessError { InvalidInput }", "explanation": "Added docs and error handling"}"#.to_string());
            responses.insert(3, r#"{"code": "/// Processes and transforms the input string.\n///\n/// # Examples\n///\n/// ```\n/// let result = process(\"hello\")?;\n/// assert_eq!(result, \"HELLO\");\n/// ```\n///\n/// # Errors\n///\n/// Returns `ProcessError::InvalidInput` if the input is empty.\npub fn process(s: &str) -> Result<String, ProcessError> {\n    if s.is_empty() {\n        return Err(ProcessError::InvalidInput);\n    }\n    Ok(s.to_uppercase())\n}\n\n#[derive(Debug)]\npub enum ProcessError {\n    InvalidInput,\n}", "explanation": "Production-ready with full documentation and error handling"}"#.to_string());

            Self {
                base_responses: responses,
                learned_patterns: Vec::new(),
            }
        }

        fn add_learned_pattern(&mut self, pattern: &str) {
            self.learned_patterns.push(pattern.to_string());
        }

        fn generate(&self, iteration: u32, _context: &str, feedback: Option<&str>) -> String {
            // In production, this would incorporate feedback into the prompt
            if let Some(_fb) = feedback {
                // Could adjust response based on specific feedback
            }

            self.base_responses
                .get(&iteration)
                .or_else(|| self.base_responses.get(&3))
                .cloned()
                .unwrap_or_else(|| r#"{"code": "", "explanation": ""}"#.to_string())
        }
    }

    // ========================================================================
    // Report Generator
    // ========================================================================

    struct ReportGenerator {
        iterations: Vec<IterationRecord>,
    }

    struct IterationRecord {
        number: u32,
        score: f64,
        feedback: Option<String>,
        output: String,
        breakdown: Vec<(String, f64)>,
    }

    impl ReportGenerator {
        fn new() -> Self {
            Self { iterations: Vec::new() }
        }

        fn record(&mut self, number: u32, score: f64, feedback: Option<&str>, output: &str, breakdown: Vec<(String, f64)>) {
            self.iterations.push(IterationRecord {
                number,
                score,
                feedback: feedback.map(String::from),
                output: output.to_string(),
                breakdown,
            });
        }

        fn generate(&self, template: &Template<'_>, task: &str, result: &RefinementResult, rag_context: &[String]) -> String {
            let mut md = String::new();

            // Title
            md.push_str("# Kkachi Code Generation Report\n\n");

            // Metadata
            md.push_str("## Metadata\n\n");
            md.push_str(&format!("- **Template**: {} v{}\n", template.name, template.version));
            md.push_str(&format!("- **Signature**: {}\n", template.signature));
            md.push_str(&format!("- **Format**: {:?}\n", template.format.format_type));
            md.push_str(&format!("- **Strict Mode**: {}\n\n", template.options.strict));

            // Summary
            md.push_str("## Summary\n\n");
            md.push_str("| Metric | Value |\n");
            md.push_str("|--------|-------|\n");
            md.push_str(&format!("| Final Score | **{:.1}%** |\n", result.score * 100.0));
            md.push_str(&format!("| Iterations | {} |\n", result.iterations));
            md.push_str(&format!("| Context ID | `{}` |\n", result.context_id));
            md.push_str(&format!("| Cached | {} |\n\n", if result.from_cache { "Yes" } else { "No" }));

            // Task
            md.push_str("## Task\n\n");
            md.push_str(&format!("> {}\n\n", task));

            // RAG Context
            md.push_str("## Retrieved Context (RAG)\n\n");
            if rag_context.is_empty() {
                md.push_str("_No relevant context found in knowledge base._\n\n");
            } else {
                for (i, ctx) in rag_context.iter().enumerate() {
                    md.push_str(&format!("<details>\n<summary>Context {} (click to expand)</summary>\n\n", i + 1));
                    md.push_str("```\n");
                    md.push_str(&ctx.chars().take(500).collect::<String>());
                    if ctx.len() > 500 { md.push_str("\n..."); }
                    md.push_str("\n```\n</details>\n\n");
                }
            }

            // Iteration History
            md.push_str("## Iteration History\n\n");
            for record in &self.iterations {
                md.push_str(&format!("### Iteration {}\n\n", record.number));

                // Score breakdown
                md.push_str("**Score Breakdown:**\n\n");
                md.push_str("| Stage | Score |\n");
                md.push_str("|-------|-------|\n");
                for (stage, score) in &record.breakdown {
                    let bar = "█".repeat((score * 10.0) as usize);
                    let empty = "░".repeat(10 - (score * 10.0) as usize);
                    md.push_str(&format!("| {} | {}{} {:.0}% |\n", stage, bar, empty, score * 100.0));
                }
                md.push_str(&format!("| **Total** | **{:.1}%** |\n\n", record.score * 100.0));

                // Feedback
                if let Some(ref fb) = record.feedback {
                    md.push_str("**Feedback:**\n\n```\n");
                    md.push_str(fb);
                    md.push_str("\n```\n\n");
                }

                // Output preview
                md.push_str("**Output:**\n\n```json\n");
                md.push_str(&record.output.chars().take(300).collect::<String>());
                if record.output.len() > 300 { md.push_str("\n..."); }
                md.push_str("\n```\n\n");

                md.push_str("---\n\n");
            }

            // Error Corrections
            if !result.error_corrections.is_empty() {
                md.push_str("## Error Corrections Applied\n\n");
                for (i, (error, fix)) in result.error_corrections.iter().enumerate() {
                    md.push_str(&format!("{}. **Issue**: {}\n", i + 1, error));
                    if !fix.is_empty() {
                        md.push_str(&format!("   **Resolution**: {}\n", fix));
                    }
                    md.push_str("\n");
                }
            }

            // Final Output
            md.push_str("## Final Output\n\n");
            md.push_str("```json\n");
            md.push_str(&result.answer);
            md.push_str("\n```\n\n");

            // Extracted Code
            let code = extract_code(&result.answer);
            if !code.is_empty() && code != result.answer {
                md.push_str("### Generated Code\n\n");
                md.push_str("```rust\n");
                md.push_str(&code);
                md.push_str("\n```\n\n");
            }

            // Footer
            md.push_str("---\n\n");
            md.push_str("*Generated by [Kkachi](https://github.com/lituus-io/kkachi) - Production-grade LLM prompt optimization*\n");

            md
        }
    }

    // ========================================================================
    // Main Pipeline
    // ========================================================================

    pub fn run() -> Result<()> {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║  Kkachi Persistent Template Pipeline                             ║");
        println!("║  Template + DuckDB RAG + Multi-Stage Validation                ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        // ====================================================================
        // Step 1: Load Template from File
        // ====================================================================
        println!("▶ Step 1: Loading template from file...");

        let template_path = format!("{}/rust_code_generator.md", TEMPLATE_DIR);
        let template = if Path::new(&template_path).exists() {
            println!("  Loading: {}", template_path);
            Template::from_file(&template_path)?
        } else {
            println!("  Template file not found, using inline template");
            Template::from_str(include_str!("templates/rust_code_generator.md"))
                .unwrap_or_else(|_| create_default_template())
        };

        println!("  ✓ Template: {} v{}", template.name, template.version);
        println!("  ✓ Format: {:?}", template.format.format_type);
        println!("  ✓ Examples: {}\n", template.examples.len());

        // ====================================================================
        // Step 2: Initialize DuckDB Vector Store
        // ====================================================================
        println!("▶ Step 2: Initializing DuckDB knowledge base...");

        let embedder = HashEmbedder::new(64);
        let mut vector_store = DuckDBVectorStore::open(DB_PATH, embedder.clone())?;

        // Seed with initial knowledge if empty
        if vector_store.len() == 0 {
            println!("  Seeding knowledge base...");
            seed_knowledge_base(&mut vector_store)?;
        }

        println!("  ✓ Database: {}", DB_PATH);
        println!("  ✓ Documents: {}\n", vector_store.len());

        // ====================================================================
        // Step 3: Define Task
        // ====================================================================
        let task = "Write a configuration file parser that reads TOML and validates required fields";

        println!("▶ Step 3: Task Definition");
        println!("  {}\n", task);

        // ====================================================================
        // Step 4: Retrieve RAG Context
        // ====================================================================
        println!("▶ Step 4: Retrieving relevant context...");

        let k = 3;
        let similar = vector_store.search_text(task, k);
        let rag_context: Vec<String> = similar.iter()
            .map(|r| format!("[{}] (score: {:.3})\n{}", r.id, r.score, r.content))
            .collect();

        for (i, result) in similar.iter().enumerate() {
            println!("  {}. {} (similarity: {:.1}%)", i + 1, result.id, result.score * 100.0);
        }
        println!();

        // ====================================================================
        // Step 5: Configure Multi-Stage Critic
        // ====================================================================
        println!("▶ Step 5: Setting up multi-stage validation...");

        let critic = MultiStageCritic::new(&template);
        let config = RecursiveConfig {
            max_iterations: 5,
            score_threshold: 0.90,
            include_history: true,
            ..Default::default()
        };

        println!("  ✓ Stages: format → required_fields → code_quality → best_practices");
        println!("  ✓ Max iterations: {}", config.max_iterations);
        println!("  ✓ Target score: {:.0}%\n", config.score_threshold * 100.0);

        // ====================================================================
        // Step 6: Run Recursive Refinement
        // ====================================================================
        println!("▶ Step 6: Running recursive refinement...\n");

        let mut reporter = ReportGenerator::new();
        let runner = StandaloneRunner::with_config(&critic, "rust_config", config);
        let mut llm = AdaptiveMockLLM::new();

        // Add learned patterns from RAG
        for ctx in &rag_context {
            llm.add_learned_pattern(ctx);
        }

        let result = runner.refine(task, |iteration, feedback| {
            println!("  ┌─ Iteration {} ─────────────────────────────────", iteration);

            // Build context
            let rag_str = similar.iter()
                .map(|r| r.content.clone())
                .collect::<Vec<_>>()
                .join("\n---\n");

            let prompt = template.assemble_prompt(task, iteration, feedback);
            let context = format!("{}\n\n## Knowledge Base:\n{}", prompt, rag_str);

            // Generate
            let response = llm.generate(iteration, &context, feedback);

            // Evaluate (for logging)
            let eval_result = critic.evaluate(StrView::new(&response), &RecursiveState::new());

            println!("  │ Score: {:.1}%", eval_result.score * 100.0);
            if let Some(ref fb) = eval_result.feedback {
                let lines: Vec<&str> = fb.lines().take(3).collect();
                for line in &lines {
                    println!("  │ → {}", line);
                }
                if fb.lines().count() > 3 {
                    println!("  │ → ...");
                }
            }
            println!("  └────────────────────────────────────────────────\n");

            // Record for report
            let breakdown = eval_result.breakdown.unwrap_or_default();
            reporter.record(iteration, eval_result.score, feedback, &response, breakdown);

            Ok(response)
        })?;

        // ====================================================================
        // Step 7: Store Successful Result
        // ====================================================================
        println!("▶ Step 7: Processing result...");

        if result.score >= MIN_SCORE_TO_STORE {
            println!("  Score {:.1}% >= {:.0}% threshold, storing to knowledge base...",
                     result.score * 100.0, MIN_SCORE_TO_STORE * 100.0);

            let example = TrainingExample {
                id: format!("learned:{}", result.context_id),
                question: task.to_string(),
                answer: result.answer.clone(),
                score: result.score,
                iterations: result.iterations,
                domain: "rust".to_string(),
                error_corrections: result.error_corrections.clone(),
            };

            vector_store.add(&example.id, &example.as_learning_example())?;
            println!("  ✓ Stored as: {}", example.id);
            println!("  ✓ Knowledge base now has {} documents\n", vector_store.len());
        } else {
            println!("  Score {:.1}% < {:.0}% threshold, not storing\n",
                     result.score * 100.0, MIN_SCORE_TO_STORE * 100.0);
        }

        // ====================================================================
        // Step 8: Generate Report
        // ====================================================================
        println!("▶ Step 8: Generating report...");

        fs::create_dir_all(OUTPUT_DIR).ok();
        let report = reporter.generate(&template, task, &result, &rag_context);
        let report_path = format!("{}/persistent_pipeline_report.md", OUTPUT_DIR);
        fs::write(&report_path, &report)?;

        println!("  ✓ Report: {}\n", report_path);

        // ====================================================================
        // Summary
        // ====================================================================
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║  Results                                                       ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("  Final Score:  {:.1}%", result.score * 100.0);
        println!("  Iterations:   {}", result.iterations);
        println!("  Context ID:   {}", result.context_id);
        println!("  Report:       {}", report_path);

        println!("\n  Generated Code:");
        println!("  ─────────────────────────────────────────────────");
        let code = extract_code(&result.answer);
        for line in code.lines().take(12) {
            println!("  {}", line);
        }
        if code.lines().count() > 12 {
            println!("  ...");
        }

        println!("\n✓ Pipeline complete!\n");

        Ok(())
    }

    fn create_default_template() -> Template<'static> {
        Template::new("default")
            .with_format(FormatType::Json)
            .with_system_prompt("Generate Rust code with proper error handling.")
    }

    fn seed_knowledge_base(store: &mut DuckDBVectorStore<HashEmbedder>) -> Result<()> {
        let seeds = vec![
            ("rust:error_handling", "Pattern: Error Handling\n\nUse Result<T, E> for fallible operations:\n```rust\nfn read_file(path: &str) -> Result<String, io::Error> {\n    fs::read_to_string(path)\n}\n```\n\nUse ? for propagation, avoid .unwrap() in production code."),
            ("rust:config_parsing", "Pattern: Configuration Parsing\n\nUse serde for deserialization:\n```rust\n#[derive(Deserialize)]\nstruct Config {\n    name: String,\n    port: u16,\n}\n\nfn load_config(path: &str) -> Result<Config, ConfigError> {\n    let content = fs::read_to_string(path)?;\n    toml::from_str(&content).map_err(ConfigError::Parse)\n}\n```"),
            ("rust:documentation", "Pattern: Documentation\n\nUse doc comments for public items:\n```rust\n/// Parses configuration from a file.\n///\n/// # Examples\n///\n/// ```\n/// let config = parse_config(\"config.toml\")?;\n/// ```\n///\n/// # Errors\n///\n/// Returns error if file doesn't exist or is malformed.\npub fn parse_config(path: &str) -> Result<Config, Error> { ... }\n```"),
            ("rust:validation", "Pattern: Input Validation\n\nValidate inputs at boundaries:\n```rust\nimpl Config {\n    fn validate(&self) -> Result<(), ValidationError> {\n        if self.port == 0 {\n            return Err(ValidationError::InvalidPort);\n        }\n        Ok(())\n    }\n}\n```"),
        ];

        for (id, content) in seeds {
            store.add(id.to_string(), content.to_string());
        }

        Ok(())
    }
}

#[cfg(feature = "storage")]
fn main() {
    if let Err(e) = pipeline::run() {
        eprintln!("Pipeline error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(not(feature = "storage"))]
fn main() {
    println!("This example requires the 'storage' feature.");
    println!("Run with: cargo run --example template_persistent_pipeline --features storage");
}
