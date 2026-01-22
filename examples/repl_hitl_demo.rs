// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! REPL and Human-in-the-Loop Demonstration
//!
//! This example demonstrates the interactive REPL capabilities and HITL workflow:
//!
//! 1. **Session State Management** - Track module configuration, history, demos
//! 2. **Vector Store Integration** - Store and search examples in REPL
//! 3. **Pipeline Execution** - Build and run multi-stage pipelines
//! 4. **HITL Review Flow** - Interactive review with different decisions
//! 5. **Diff Visualization** - See changes between iterations
//!
//! Run with:
//! ```bash
//! cargo run --example repl_hitl_demo
//! ```

use kkachi::diff::{DiffRenderer, DiffStyle, TextDiff};
use kkachi::error::Result;
use kkachi::hitl::{HITLConfig, ReviewContext, ReviewDecision, ReviewTrigger};
use kkachi::recursive::{
    ChecklistCritic, Critic, CriticResult, HashEmbedder, HeuristicCritic,
    InMemoryVectorStore, RecursiveConfig, RecursiveState, StandaloneRunner, VectorStore,
};
use kkachi::StrView;
use std::time::Duration;

// ============================================================================
// REPL Session State Simulation
// ============================================================================

/// Simulates REPL session state
struct ReplSession {
    signature: String,
    instruction: String,
    demos: Vec<(String, String)>,
    model: String,
    temperature: f32,
    vector_store: InMemoryVectorStore<HashEmbedder>,
    history: Vec<String>,
}

impl ReplSession {
    fn new() -> Self {
        let embedder = HashEmbedder::new(64);
        Self {
            signature: String::new(),
            instruction: String::new(),
            demos: Vec::new(),
            model: "mock-llm".to_string(),
            temperature: 0.7,
            vector_store: InMemoryVectorStore::new(embedder),
            history: Vec::new(),
        }
    }

    fn set_signature(&mut self, sig: &str) {
        self.signature = sig.to_string();
        self.history.push(format!("signature {}", sig));
        println!("  ✓ Signature set: {}", sig);
    }

    fn set_instruction(&mut self, inst: &str) {
        self.instruction = inst.to_string();
        self.history.push(format!("instruction \"{}\"", inst));
        println!("  ✓ Instruction set ({} chars)", inst.len());
    }

    fn add_demo(&mut self, input: &str, output: &str) {
        self.demos.push((input.to_string(), output.to_string()));
        self.history.push(format!("demo add \"{}\" -> \"{}\"", input, output));
        println!("  ✓ Demo added ({} total)", self.demos.len());
    }

    fn store_add(&mut self, id: &str, content: &str) {
        self.vector_store.add(id, content);
        self.history.push(format!("store add {} \"{}\"", id, content));
        println!("  ✓ Added to store: {} ({} total)", id, self.vector_store.len());
    }

    fn store_search(&self, query: &str, k: usize) -> Vec<String> {
        let results = self.vector_store.search_text(query, k);
        results.iter().map(|r| r.id.clone()).collect()
    }

    fn show(&self) {
        println!("\n  ┌─────────────────────────────────────────────────────────────┐");
        println!("  │ Session State                                                │");
        println!("  ├─────────────────────────────────────────────────────────────┤");
        println!("  │ Signature: {}{}│",
            self.signature,
            " ".repeat(47 - self.signature.len().min(47)));
        println!("  │ Model: {} (temp: {:.1}){}│",
            self.model, self.temperature,
            " ".repeat(32));
        println!("  │ Demos: {}{}│",
            self.demos.len(),
            " ".repeat(52));
        println!("  │ Store: {} documents{}│",
            self.vector_store.len(),
            " ".repeat(44));
        println!("  └─────────────────────────────────────────────────────────────┘");
    }
}

// ============================================================================
// HITL Review Simulation
// ============================================================================

/// Simulates different HITL review scenarios
fn simulate_hitl_scenarios() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              HITL Review Scenarios                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Scenario 1: Accept and continue
    println!("Scenario 1: Accept and Continue");
    println!("─────────────────────────────────────────");
    let decision1 = ReviewDecision::Accept;
    println!("  Decision: {:?}", decision1);
    println!("  Action: Continue to next iteration\n");

    // Scenario 2: Reject and retry
    println!("Scenario 2: Reject and Retry");
    println!("─────────────────────────────────────────");
    let decision2 = ReviewDecision::Reject;
    println!("  Decision: {:?}", decision2);
    println!("  Action: Try alternative approach\n");

    // Scenario 3: Edit with guidance
    println!("Scenario 3: Edit with Guidance");
    println!("─────────────────────────────────────────");
    let decision3 = ReviewDecision::Edit {
        instruction: Some("Add more error handling".to_string()),
        output: None,
        guidance: Some("Focus on edge cases".to_string()),
    };
    println!("  Decision: Edit");
    if let ReviewDecision::Edit { instruction, output, guidance } = &decision3 {
        println!("  - Instruction: {:?}", instruction);
        println!("  - Output: {:?}", output);
        println!("  - Guidance: {:?}", guidance);
    }
    println!("  Action: Apply edits and regenerate\n");

    // Scenario 4: Stop early
    println!("Scenario 4: Accept Final (Stop Early)");
    println!("─────────────────────────────────────────");
    let decision4 = ReviewDecision::AcceptFinal;
    println!("  Decision: {:?}", decision4);
    println!("  Action: Stop iteration and accept current result\n");

    // Scenario 5: Rollback
    println!("Scenario 5: Rollback to Previous");
    println!("─────────────────────────────────────────");
    let decision5 = ReviewDecision::Rollback { to_iteration: 2 };
    println!("  Decision: {:?}", decision5);
    println!("  Action: Restore state from iteration 2\n");

    // Scenario 6: Skip reviews
    println!("Scenario 6: Skip Next Reviews");
    println!("─────────────────────────────────────────");
    let decision6 = ReviewDecision::SkipNext { count: 3 };
    println!("  Decision: {:?}", decision6);
    println!("  Action: Skip next 3 iterations without review\n");
}

// ============================================================================
// Pipeline Demonstration
// ============================================================================

fn demonstrate_pipeline() -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              Multi-Stage Pipeline Demo                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Stage 1: Retrieve context from vector store
    println!("Stage 1: Context Retrieval");
    println!("─────────────────────────────────────────");

    let embedder = HashEmbedder::new(64);
    let mut store = InMemoryVectorStore::new(embedder);

    // Seed with examples
    store.add("ex1", "Q: Parse JSON? A: use serde_json::from_str");
    store.add("ex2", "Q: Read file? A: std::fs::read_to_string");
    store.add("ex3", "Q: HTTP request? A: reqwest::get(url).await");
    store.add("ex4", "Q: Parse TOML? A: toml::from_str(&content)");

    let query = "How to parse configuration?";
    println!("  Query: \"{}\"", query);
    let results = store.search_text(query, 2);
    println!("  Found {} similar examples:", results.len());
    for r in &results {
        println!("    - {} (score: {:.3})", r.id, r.score);
    }

    // Stage 2: Generate with critic
    println!("\nStage 2: Code Generation");
    println!("─────────────────────────────────────────");

    let critic = ChecklistCritic::new()
        .add_check("has_fn", |s| s.contains("fn "), 0.3, "Missing function")
        .add_check("has_result", |s| s.contains("Result"), 0.3, "No error handling")
        .add_check("has_docs", |s| s.contains("///"), 0.4, "Missing docs");

    // Simulate iterations
    let outputs = vec![
        "fn parse() {}",
        "fn parse() -> Result<Config, Error> {}",
        "/// Parses config\nfn parse() -> Result<Config, Error> {}",
    ];

    for (i, output) in outputs.iter().enumerate() {
        let state = RecursiveState::new();
        let result = critic.evaluate(StrView::new(output), &state);
        println!("  Iteration {}: Score {:.2}", i, result.score);
        if result.score >= 1.0 {
            println!("  ✓ Converged!");
            break;
        }
    }

    // Stage 3: Diff visualization
    println!("\nStage 3: Diff Visualization");
    println!("─────────────────────────────────────────");

    let renderer = DiffRenderer::new()
        .with_style(DiffStyle::Unified)
        .with_context(1);

    for i in 1..outputs.len() {
        let diff = TextDiff::new(outputs[i-1], outputs[i]);
        if diff.has_changes() {
            let stats = diff.stats();
            println!("\n  Iteration {} → {}:", i-1, i);
            println!("    +{} lines, -{} lines", stats.lines_added, stats.lines_removed);

            // Show diff
            let rendered = renderer.render_text(&diff);
            for line in rendered.lines().take(10) {
                println!("    {}", line);
            }
        }
    }

    Ok(())
}

// ============================================================================
// HITL Configuration Demo
// ============================================================================

fn demonstrate_hitl_configs() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              HITL Configuration Options                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Config 1: Review every iteration
    let config1 = HITLConfig::every_iteration();
    println!("1. Every Iteration:");
    println!("   - Enabled: {}", config1.enabled);
    println!("   - Interval: {}", config1.interval);
    println!("   - Use case: Development/debugging\n");

    // Config 2: Review every N iterations
    let config2 = HITLConfig::every(5);
    println!("2. Every 5 Iterations:");
    println!("   - Enabled: {}", config2.enabled);
    println!("   - Interval: {}", config2.interval);
    println!("   - Use case: Periodic checkpoint\n");

    // Config 3: Review on completion only
    let config3 = HITLConfig::on_completion();
    println!("3. On Completion Only:");
    println!("   - Enabled: {}", config3.enabled);
    println!("   - On convergence: {}", config3.on_convergence);
    println!("   - Use case: Final validation\n");

    // Config 4: Custom configuration
    let config4 = HITLConfig {
        enabled: true,
        interval: 3,
        on_score_drop: true,
        on_convergence: true,
        on_first: true,
        on_keywords: vec!["error".to_string(), "fail".to_string()],
        timeout: Some(Duration::from_secs(300)),
        show_diff: true,
        diff_style: DiffStyle::Unified,
        auto_accept_timeout: Some(Duration::from_secs(60)),
        skip_above_score: Some(0.95),
    };
    println!("4. Custom Configuration:");
    println!("   - Interval: {}", config4.interval);
    println!("   - On score drop: {}", config4.on_score_drop);
    println!("   - On first: {}", config4.on_first);
    println!("   - Keywords: {:?}", config4.on_keywords);
    println!("   - Skip above score: {:?}", config4.skip_above_score);
    println!("   - Use case: Production with guardrails\n");
}

// ============================================================================
// REPL Command Simulation
// ============================================================================

fn simulate_repl_session() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              REPL Session Simulation                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let mut session = ReplSession::new();

    // Simulate REPL commands
    println!("kkachi> signature \"question -> answer\"");
    session.set_signature("question -> answer");

    println!("\nkkachi> instruction \"You are a Rust code assistant.\"");
    session.set_instruction("You are a Rust code assistant.");

    println!("\nkkachi> demo add \"Parse JSON\" -> \"use serde_json::from_str\"");
    session.add_demo("Parse JSON", "use serde_json::from_str");

    println!("\nkkachi> demo add \"Read file\" -> \"std::fs::read_to_string\"");
    session.add_demo("Read file", "std::fs::read_to_string");

    println!("\nkkachi> store add rust:parsing \"How to parse data formats in Rust\"");
    session.store_add("rust:parsing", "How to parse data formats in Rust");

    println!("\nkkachi> store add rust:io \"File I/O operations in Rust\"");
    session.store_add("rust:io", "File I/O operations in Rust");

    println!("\nkkachi> store add rust:error \"Error handling with Result and Option\"");
    session.store_add("rust:error", "Error handling with Result and Option");

    println!("\nkkachi> store search \"parse config\" 2");
    let results = session.store_search("parse config", 2);
    println!("  Found {} results:", results.len());
    for id in &results {
        println!("    - {}", id);
    }

    println!("\nkkachi> show");
    session.show();

    println!("\nkkachi> history");
    println!("  Command History:");
    for (i, cmd) in session.history.iter().enumerate() {
        println!("    {}. {}", i + 1, cmd);
    }
}

// ============================================================================
// Diff Visualization Demo
// ============================================================================

fn demonstrate_diff_visualization() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              Diff Visualization Styles                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let old = r#"fn parse(s: &str) -> Config {
    toml::from_str(s).unwrap()
}"#;

    let new = r#"/// Parse TOML configuration.
///
/// # Errors
/// Returns error if parsing fails.
fn parse(s: &str) -> Result<Config, Error> {
    toml::from_str(s)?
}"#;

    let diff = TextDiff::new(old, new);
    let stats = diff.stats();

    println!("Change Statistics:");
    println!("  Lines added:   {}", stats.lines_added);
    println!("  Lines removed: {}", stats.lines_removed);
    println!("  Lines changed: {}", stats.lines_changed);
    println!("  Total changes: {}", stats.total_changes);

    // Unified style
    println!("\n--- Unified Style ---");
    let renderer = DiffRenderer::new()
        .with_style(DiffStyle::Unified)
        .with_context(2);
    let output = renderer.render_text(&diff);
    for line in output.lines() {
        println!("{}", line);
    }

    // Compact style
    println!("\n--- Compact Style ---");
    let renderer = DiffRenderer::new()
        .with_style(DiffStyle::Compact);
    let output = renderer.render_text(&diff);
    for line in output.lines() {
        println!("{}", line);
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    println!("════════════════════════════════════════════════════════════════════");
    println!("          OPTI: REPL and HITL Demonstration");
    println!("════════════════════════════════════════════════════════════════════");

    // Part 1: REPL Session Simulation
    simulate_repl_session();

    // Part 2: HITL Configurations
    demonstrate_hitl_configs();

    // Part 3: HITL Review Scenarios
    simulate_hitl_scenarios();

    // Part 4: Pipeline Demonstration
    demonstrate_pipeline()?;

    // Part 5: Diff Visualization
    demonstrate_diff_visualization();

    println!("\n════════════════════════════════════════════════════════════════════");
    println!("                     Demo Complete!");
    println!("════════════════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_session_state() {
        let mut session = ReplSession::new();

        session.set_signature("q -> a");
        assert_eq!(session.signature, "q -> a");

        session.add_demo("input", "output");
        assert_eq!(session.demos.len(), 1);

        session.store_add("test", "content");
        assert_eq!(session.vector_store.len(), 1);
    }

    #[test]
    fn test_hitl_decisions() {
        let accept = ReviewDecision::Accept;
        let reject = ReviewDecision::Reject;
        let edit = ReviewDecision::Edit {
            instruction: None,
            output: Some("modified".to_string()),
            guidance: None,
        };
        let stop = ReviewDecision::Stop;
        let final_accept = ReviewDecision::AcceptFinal;
        let rollback = ReviewDecision::Rollback { to_iteration: 1 };
        let skip = ReviewDecision::SkipNext { count: 2 };

        // Verify all variants can be created
        assert!(matches!(accept, ReviewDecision::Accept));
        assert!(matches!(reject, ReviewDecision::Reject));
        assert!(matches!(edit, ReviewDecision::Edit { .. }));
        assert!(matches!(stop, ReviewDecision::Stop));
        assert!(matches!(final_accept, ReviewDecision::AcceptFinal));
        assert!(matches!(rollback, ReviewDecision::Rollback { .. }));
        assert!(matches!(skip, ReviewDecision::SkipNext { .. }));
    }

    #[test]
    fn test_hitl_configs() {
        let every = HITLConfig::every_iteration();
        assert!(every.enabled);
        assert_eq!(every.interval, 1);

        let every5 = HITLConfig::every(5);
        assert_eq!(every5.interval, 5);

        let completion = HITLConfig::on_completion();
        assert!(completion.on_convergence);

        let disabled = HITLConfig::disabled();
        assert!(!disabled.enabled);
    }

    #[test]
    fn test_vector_store_search() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        store.add("doc1", "Rust programming language");
        store.add("doc2", "Python scripting");
        store.add("doc3", "Rust memory safety");

        let results = store.search_text("Rust", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_critic_evaluation() {
        let critic = ChecklistCritic::new()
            .add_check("has_fn", |s| s.contains("fn"), 0.5, "No fn")
            .add_check("has_result", |s| s.contains("Result"), 0.5, "No Result");

        let state = RecursiveState::new();

        // Partial pass
        let result = critic.evaluate(StrView::new("fn foo()"), &state);
        assert!((result.score - 0.5).abs() < 0.01);

        // Full pass
        let result = critic.evaluate(StrView::new("fn foo() -> Result"), &state);
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn test_diff_visualization() {
        let old = "line1\nline2";
        let new = "line1\nmodified\nline3";

        let diff = TextDiff::new(old, new);
        assert!(diff.has_changes());

        let stats = diff.stats();
        assert!(stats.lines_added >= 1);
    }
}
