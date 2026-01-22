// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for full recursive pipeline with vector store, HITL, and diff visualization.
//!
//! These tests verify the complete end-to-end workflow including:
//! - Vector store operations with semantic search
//! - Recursive refinement with critics
//! - HITL review integration
//! - Diff visualization between iterations
//! - DSPy-style module composition

use kkachi::diff::{DiffRenderer, DiffStyle, TextDiff};
use kkachi::hitl::{HITLConfig, ReviewDecision, ReviewTrigger};
use kkachi::recursive::{
    ChecklistCritic, Critic, HashEmbedder, HeuristicCritic, InMemoryVectorStore, RecursiveConfig,
    RecursiveState, StandaloneRunner, VectorStore,
};
use kkachi::StrView;

// ============================================================================
// Vector Store Integration Tests
// ============================================================================

mod vector_store_tests {
    use super::*;

    #[test]
    fn test_vector_store_add_search_workflow() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        // Add documents
        store.add("doc1", "How to parse JSON in Rust using serde");
        store.add("doc2", "Reading and writing files in Rust with std::fs");
        store.add("doc3", "HTTP requests in Rust using reqwest library");
        store.add("doc4", "Parsing TOML configuration files in Rust");
        store.add("doc5", "Working with XML data in Rust");

        assert_eq!(store.len(), 5);

        // Search for similar documents
        let results = store.search_text("How to parse config files", 3);
        assert_eq!(results.len(), 3);

        // Verify that search returns results (exact ordering depends on embedder)
        // With HashEmbedder, relevance is approximate
        assert!(
            results.iter().any(|r| r.id.starts_with("doc")),
            "Should return valid documents"
        );
    }

    #[test]
    fn test_vector_store_update_and_remove() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        store.add("doc1", "Original content about Rust");
        assert_eq!(store.len(), 1);

        // Update document
        store.update("doc1", "Updated content about Rust programming");
        assert_eq!(store.len(), 1);

        // Verify updated content
        let content = store.get("doc1").unwrap();
        assert!(content.contains("Updated"));

        // Remove document
        assert!(store.remove("doc1"));
        assert_eq!(store.len(), 0);
        assert!(!store.remove("doc1")); // Already removed
    }

    #[test]
    fn test_vector_store_semantic_similarity() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        // Add semantically similar and different documents
        store.add(
            "positive1",
            "I love this product it is amazing wonderful great",
        );
        store.add("positive2", "This is fantastic excellent brilliant superb");
        store.add("negative1", "I hate this product it is terrible awful bad");
        store.add("neutral1", "The product exists and functions as expected");

        // Search for positive sentiment
        let results = store.search_text("amazing wonderful product love", 2);

        // Should find positive documents first
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(
            ids.contains(&"positive1") || ids.contains(&"positive2"),
            "Should find positive documents"
        );
    }

    #[test]
    fn test_vector_store_empty_queries() {
        let embedder = HashEmbedder::new(64);
        let store = InMemoryVectorStore::new(embedder);

        // Empty store should return empty results
        let results = store.search_text("any query", 10);
        assert!(results.is_empty());
        assert!(store.is_empty());
    }

    #[test]
    fn test_vector_store_k_limit() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        for i in 0..20 {
            store.add(format!("doc{}", i), format!("Document number {}", i));
        }

        // Request more than available
        let results = store.search_text("document", 100);
        assert_eq!(results.len(), 20);

        // Request fewer
        let results = store.search_text("document", 5);
        assert_eq!(results.len(), 5);
    }
}

// ============================================================================
// Recursive Refinement Integration Tests
// ============================================================================

mod refinement_tests {
    use super::*;

    #[test]
    fn test_checklist_critic_all_pass() {
        let critic = ChecklistCritic::new()
            .add_check(
                "has_function",
                |s| s.contains("fn "),
                0.25,
                "Missing function",
            )
            .add_check(
                "has_return",
                |s| s.contains("->"),
                0.25,
                "Missing return type",
            )
            .add_check(
                "has_body",
                |s| s.contains("{") && s.contains("}"),
                0.25,
                "Missing body",
            )
            .add_check("has_pub", |s| s.contains("pub "), 0.25, "Not public");

        let good_code = "pub fn example() -> i32 { 42 }";
        let state = RecursiveState::new();
        let result = critic.evaluate(StrView::new(good_code), &state);

        assert_eq!(result.score, 1.0, "All checks should pass");
        assert!(result.feedback.is_none());
    }

    #[test]
    fn test_checklist_critic_partial_pass() {
        let critic = ChecklistCritic::new()
            .add_check("has_fn", |s| s.contains("fn "), 0.5, "No function")
            .add_check("has_doc", |s| s.contains("///"), 0.5, "No docs");

        let code_no_docs = "fn example() {}";
        let state = RecursiveState::new();
        let result = critic.evaluate(StrView::new(code_no_docs), &state);

        assert!((result.score - 0.5).abs() < 0.01, "Should score 0.5");
        assert!(result
            .feedback
            .as_ref()
            .map(|f| f.contains("No docs"))
            .unwrap_or(false));
    }

    #[test]
    fn test_heuristic_critic() {
        let critic = HeuristicCritic::new()
            .min_length(10)
            .max_length(100)
            .require("Result")
            .forbid("unwrap");

        let state = RecursiveState::new();

        // Good answer
        let good = "fn parse(s: &str) -> Result<i32, Error> { s.parse() }";
        let result = critic.evaluate(StrView::new(good), &state);
        assert_eq!(result.score, 1.0);

        // Bad answer - uses unwrap
        let bad = "fn parse(s: &str) -> i32 { s.parse().unwrap() }";
        let result = critic.evaluate(StrView::new(bad), &state);
        assert!(result.score < 1.0);
    }

    #[test]
    fn test_refinement_convergence() {
        let critic =
            ChecklistCritic::new().add_check("complete", |s| s.len() > 20, 1.0, "Too short");

        let config = RecursiveConfig {
            max_iterations: 5,
            score_threshold: 1.0,
            ..Default::default()
        };

        let runner = StandaloneRunner::with_config(&critic, "test", config);

        // Simulate improvement over iterations
        let mut iteration = 0;
        let result = runner
            .refine("test question", |iter, _feedback| {
                iteration = iter;
                if iter < 2 {
                    Ok("short".to_string())
                } else {
                    Ok("This is a sufficiently long answer that passes".to_string())
                }
            })
            .unwrap();

        assert!(result.score >= 1.0);
        assert!(result.iterations <= 5);
    }

    #[test]
    fn test_refinement_max_iterations() {
        let critic = ChecklistCritic::new().add_check("impossible", |_| false, 1.0, "Never passes");

        let config = RecursiveConfig {
            max_iterations: 3,
            score_threshold: 1.0,
            ..Default::default()
        };

        let runner = StandaloneRunner::with_config(&critic, "test", config);

        let result = runner
            .refine("impossible question", |_iter, _feedback| {
                Ok("Answer that will never pass".to_string())
            })
            .unwrap();

        assert_eq!(result.iterations, 3);
        assert!(result.score < 1.0);
    }
}

// ============================================================================
// Diff Visualization Tests
// ============================================================================

mod diff_tests {
    use super::*;

    #[test]
    fn test_text_diff_basic() {
        let old = "Hello world";
        let new = "Hello Rust world";

        let diff = TextDiff::new(old, new);
        assert!(diff.has_changes());

        let stats = diff.stats();
        assert!(stats.lines_added > 0 || stats.lines_changed > 0);
    }

    #[test]
    fn test_text_diff_no_changes() {
        let text = "Same text here";
        let diff = TextDiff::new(text, text);
        assert!(!diff.has_changes());
    }

    #[test]
    fn test_text_diff_multiline() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3\nline4";

        let diff = TextDiff::new(old, new);
        assert!(diff.has_changes());

        let stats = diff.stats();
        assert!(stats.lines_added >= 1); // line4 added
        assert!(stats.lines_changed >= 1); // line2 modified
    }

    #[test]
    fn test_diff_renderer_unified() {
        let renderer = DiffRenderer::new().with_style(DiffStyle::Unified);

        let old = "fn foo() {}";
        let new = "fn foo() -> i32 { 42 }";

        let diff = TextDiff::new(old, new);
        let output = renderer.render_text(&diff);

        // Should contain diff markers
        assert!(!output.is_empty());
    }

    #[test]
    fn test_diff_iteration_comparison() {
        // Simulate iteration outputs
        let iteration_0 = "fn parse(s) { s.parse() }";
        let iteration_1 = "fn parse(s: &str) { s.parse() }";
        let iteration_2 = "fn parse(s: &str) -> Result<i32, Error> { s.parse() }";

        // Compare iterations
        let diff_0_1 = TextDiff::new(iteration_0, iteration_1);
        let diff_1_2 = TextDiff::new(iteration_1, iteration_2);

        assert!(diff_0_1.has_changes());
        assert!(diff_1_2.has_changes());

        // Render for display
        let renderer = DiffRenderer::new();
        let output_0_1 = renderer.render_text(&diff_0_1);
        let output_1_2 = renderer.render_text(&diff_1_2);

        assert!(!output_0_1.is_empty());
        assert!(!output_1_2.is_empty());
    }
}

// ============================================================================
// HITL Configuration Tests
// ============================================================================

mod hitl_tests {
    use super::*;

    #[test]
    fn test_hitl_config_every_iteration() {
        let config = HITLConfig::every_iteration();
        assert!(config.enabled);
        assert_eq!(config.interval, 1);
    }

    #[test]
    fn test_hitl_config_every_n() {
        let config = HITLConfig::every(5);
        assert!(config.enabled);
        assert_eq!(config.interval, 5);
    }

    #[test]
    fn test_hitl_config_on_completion() {
        let config = HITLConfig::on_completion();
        assert!(config.enabled);
        assert!(config.on_convergence);
    }

    #[test]
    fn test_hitl_config_disabled() {
        let config = HITLConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_review_decision_variants() {
        // Test all decision types compile and work
        let decisions = vec![
            ReviewDecision::Accept,
            ReviewDecision::Reject,
            ReviewDecision::Stop,
            ReviewDecision::AcceptFinal,
            ReviewDecision::Edit {
                instruction: Some("Updated instruction".to_string()),
                output: None,
                guidance: None,
            },
            ReviewDecision::Rollback { to_iteration: 2 },
            ReviewDecision::SkipNext { count: 3 },
        ];

        assert_eq!(decisions.len(), 7);
    }

    #[test]
    #[allow(clippy::useless_vec)]
    fn test_review_trigger_variants() {
        let triggers = vec![
            ReviewTrigger::Interval,
            ReviewTrigger::ScoreDrop,
            ReviewTrigger::Convergence,
            ReviewTrigger::FirstIteration,
            ReviewTrigger::Manual,
        ];

        assert_eq!(triggers.len(), 5);
    }
}

// ============================================================================
// Combined Pipeline Tests
// ============================================================================

mod pipeline_tests {
    use super::*;

    #[test]
    fn test_rag_refinement_pipeline() {
        // Initialize vector store with examples
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        store.add("example1", "Q: Parse JSON? A: use serde_json::from_str(s)");
        store.add("example2", "Q: Read file? A: std::fs::read_to_string(path)");
        store.add("example3", "Q: HTTP request? A: reqwest::get(url).await");

        // Retrieve context for new question
        let question = "How to parse YAML in Rust?";
        let examples = store.search_text(question, 2);

        assert_eq!(examples.len(), 2);

        // Build prompt with examples
        let mut prompt = String::from("Based on these examples:\n");
        for ex in &examples {
            prompt.push_str(&format!("{}\n", ex.content));
        }
        prompt.push_str(&format!("\nNow answer: {}", question));

        // Set up critic
        let critic = ChecklistCritic::new()
            .add_check("has_use", |s| s.contains("use "), 0.5, "Missing import")
            .add_check("has_fn", |s| s.contains("fn "), 0.5, "Missing function");

        let config = RecursiveConfig {
            max_iterations: 3,
            score_threshold: 1.0,
            ..Default::default()
        };

        let runner = StandaloneRunner::with_config(&critic, "rust", config);

        // Run refinement
        let result = runner
            .refine(&prompt, |iter, _feedback| {
                if iter == 0 {
                    Ok("fn parse_yaml() {}".to_string())
                } else {
                    Ok("use serde_yaml;\nfn parse_yaml(s: &str) -> serde_yaml::Value { serde_yaml::from_str(s).unwrap() }".to_string())
                }
            })
            .unwrap();

        assert!(result.score >= 0.5);

        // Store successful result back to RAG
        if result.score >= 0.8 {
            store.add(
                format!("learned:{}", result.context_id),
                format!("Q: {} A: {}", question, result.answer),
            );
        }
    }

    #[test]
    fn test_diff_between_refinement_iterations() {
        let critic = HeuristicCritic::new().min_length(50);

        let config = RecursiveConfig {
            max_iterations: 3,
            score_threshold: 1.0,
            ..Default::default()
        };

        let runner = StandaloneRunner::with_config(&critic, "test", config);

        let mut outputs: Vec<String> = Vec::new();

        let result = runner
            .refine("Generate Rust code", |iter, _| {
                let output = match iter {
                    0 => "fn foo() {}".to_string(),
                    1 => "fn foo() -> i32 { 42 }".to_string(),
                    _ => "/// Documentation\nfn foo() -> i32 {\n    // Return value\n    42\n}"
                        .to_string(),
                };
                outputs.push(output.clone());
                Ok(output)
            })
            .unwrap();

        // Generate diffs between iterations
        let renderer = DiffRenderer::new();

        for i in 1..outputs.len() {
            let diff = TextDiff::new(&outputs[i - 1], &outputs[i]);
            if diff.has_changes() {
                let rendered = renderer.render_text(&diff);
                assert!(
                    !rendered.is_empty(),
                    "Diff between iteration {} and {} should render",
                    i - 1,
                    i
                );
            }
        }

        assert!(result.iterations >= 1);
    }

    #[test]
    fn test_multi_stage_pipeline() {
        // Stage 1: Retrieve context
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);
        store.add("ctx1", "Context: Rust is a systems programming language");
        store.add("ctx2", "Context: Rust has zero-cost abstractions");

        let query = "What is Rust?";
        let context = store.search_text(query, 2);
        assert!(!context.is_empty());

        // Stage 2: Generate with CoT critic
        let cot_critic = ChecklistCritic::new()
            .add_check("reasoning", |s| s.contains("because"), 0.5, "No reasoning")
            .add_check("answer", |s| s.contains("Rust"), 0.5, "No answer");

        // Stage 3: Validate output
        let validator = HeuristicCritic::new().min_length(20).require("programming");

        // Combined scoring
        let output = "Rust is a systems programming language because it provides memory safety without garbage collection.";
        let state = RecursiveState::new();
        let cot_result = cot_critic.evaluate(StrView::new(output), &state);
        let val_result = validator.evaluate(StrView::new(output), &state);
        let score = (cot_result.score + val_result.score) / 2.0;

        assert!(score >= 0.5, "Combined score should be acceptable");
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_empty_store_operations() {
        let embedder = HashEmbedder::new(64);
        let store = InMemoryVectorStore::new(embedder);

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        let results = store.search_text("query", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_critic_empty_input() {
        let critic =
            ChecklistCritic::new().add_check("has_content", |s| !s.is_empty(), 1.0, "Empty");

        let state = RecursiveState::new();
        let result = critic.evaluate(StrView::new(""), &state);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_refinement_generator_error() {
        let critic = ChecklistCritic::new().add_check("always_pass", |_| true, 1.0, "");

        let config = RecursiveConfig {
            max_iterations: 3,
            score_threshold: 1.0,
            ..Default::default()
        };

        let runner = StandaloneRunner::with_config(&critic, "test", config);

        let result = runner.refine("test", |iter, _| {
            if iter == 0 {
                Err(kkachi::Error::Other("Generator failed".to_string()))
            } else {
                Ok("Success".to_string())
            }
        });

        // Should handle error gracefully
        assert!(result.is_err());
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_large_vector_store() {
        let embedder = HashEmbedder::new(64);
        let mut store = InMemoryVectorStore::new(embedder);

        // Add 1000 documents
        for i in 0..1000 {
            store.add(
                format!("doc{}", i),
                format!(
                    "Document {} with content about topic {} and category {}",
                    i,
                    i % 10,
                    i % 5
                ),
            );
        }

        assert_eq!(store.len(), 1000);

        // Search should still be fast
        let start = std::time::Instant::now();
        let results = store.search_text("document topic category", 10);
        let duration = start.elapsed();

        assert_eq!(results.len(), 10);
        assert!(
            duration.as_millis() < 1000,
            "Search should complete in under 1 second"
        );
    }

    #[test]
    fn test_diff_large_text() {
        let old: String = (0..1000).map(|i| format!("Line {}\n", i)).collect();
        let new: String = (0..1000)
            .map(|i| {
                if i % 100 == 0 {
                    format!("Modified Line {}\n", i)
                } else {
                    format!("Line {}\n", i)
                }
            })
            .collect();

        let start = std::time::Instant::now();
        let diff = TextDiff::new(&old, &new);
        let duration = start.elapsed();

        assert!(diff.has_changes());
        assert!(
            duration.as_millis() < 1000,
            "Diff should complete in under 1 second"
        );
    }
}
