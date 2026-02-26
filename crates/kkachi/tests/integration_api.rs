//! Integration tests using real LLM APIs.
//!
//! Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY or claude CLI) to run.
//! Tests are skipped at runtime if no provider is available.
//!
//! Run with:
//!   cargo test -p kkachi --features api --test integration_api -- --nocapture

#![cfg(feature = "api")]

use kkachi::recursive::*;

fn get_llm() -> Option<ApiLlm> {
    ApiLlm::from_env().ok()
}

#[test]
fn test_refine_improves_score() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    let result = refine(&llm, "Write a Rust function that adds two i32 numbers")
        .validate(checks().require("fn ").require("->").forbid(".unwrap()"))
        .max_iter(3)
        .target(0.8)
        .go()
        .unwrap();

    assert!(
        result.score >= 0.8,
        "Score should reach 0.8, got {}",
        result.score
    );
    assert!(
        result.output.contains("fn "),
        "Should contain a function definition"
    );
}

#[test]
fn test_reason_produces_answer() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    let result = reason(&llm, "What is 15 * 4?")
        .validate(checks().regex(r"60"))
        .max_iter(2)
        .go();

    assert!(
        result.output.contains("60"),
        "Should contain 60, got: {}",
        result.output
    );
}

#[test]
fn test_best_of_selects_best() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    let (result, _pool) = best_of(
        &llm,
        "Write a one-line Rust function that doubles an integer",
    )
    .validate(checks().require("fn ").min_len(10))
    .go_with_pool();

    assert!(result.score > 0.0, "Should have non-zero score");
    assert!(
        result.output.contains("fn "),
        "Should contain a function, got: {}",
        result.output
    );
}

#[test]
fn test_ensemble_reaches_consensus() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    let (result, consensus) = ensemble(
        &llm,
        "What is the capital of France? Reply with just the city name.",
    )
    .aggregate(Aggregate::MajorityVote)
    .go_with_consensus();

    assert!(
        result.output.to_lowercase().contains("paris"),
        "Should answer Paris, got: {}",
        result.output
    );
    assert!(
        consensus.agreement_ratio() > 0.5,
        "Should have majority agreement, got: {}",
        consensus.agreement_ratio()
    );
}

#[test]
fn test_refine_exercises_feedback_loop() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    // Strict validators that likely require multiple iterations:
    // Require specific function name, return type, match usage, and tests
    let result = refine(
        &llm,
        "Write a Rust function called 'fibonacci' that computes the nth fibonacci number",
    )
    .validate(
        checks()
            .require("fn fibonacci")
            .require("-> u64")
            .require("#[test]")
            .forbid(".unwrap()")
            .forbid("panic!")
            .min_len(200),
    )
    .max_iter(5)
    .target(1.0)
    .go()
    .unwrap();

    assert!(
        result.score >= 0.8,
        "Score should reach at least 0.8, got {}",
        result.score
    );
    // Verify the output actually contains the required elements
    assert!(
        result.output.contains("fn fibonacci"),
        "Should contain fn fibonacci"
    );
}

#[test]
fn test_refine_feedback_is_specific() {
    // Test that the feedback mechanism provides specific pattern info
    let validator = checks()
        .require("fn parse")
        .require("-> Result")
        .forbid(".unwrap()");

    let bad_input = "fn foo() { x.unwrap() }";
    let score = validator.validate(bad_input);

    assert!(score.value < 1.0);
    let feedback = score.feedback_str().unwrap();
    // Feedback should mention the specific patterns
    assert!(
        feedback.contains("fn parse"),
        "Feedback should mention 'fn parse', got: {}",
        feedback
    );
    assert!(
        feedback.contains(".unwrap()"),
        "Feedback should mention '.unwrap()', got: {}",
        feedback
    );
}

#[test]
fn test_best_of_with_extraction() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    // Use extract("rust") to pull code from markdown fences
    let (result, pool) = best_of(
        &llm,
        "Write a Rust function that returns the maximum of two numbers",
    )
    .extract("rust")
    .validate(checks().require("fn ").require("->").forbid("```"))
    .go_with_pool();

    // With extraction, the validator shouldn't see backticks
    assert!(
        result.score > 0.0,
        "Score should be positive with extraction"
    );
    let stats = pool.stats();
    assert!(stats.count > 0, "Should have generated candidates");
}

#[test]
fn test_validators_work_without_llm() {
    let validator = checks()
        .require("fn ")
        .require("Result")
        .forbid(".unwrap()")
        .min_len(30);

    let good = "fn parse(s: &str) -> Result<i32, std::num::ParseIntError> { s.parse() }";
    let bad = "fn parse(s: &str) -> i32 { s.parse().unwrap() }";

    let good_score = validator.validate(good);
    let bad_score = validator.validate(bad);

    assert!(
        good_score.value > bad_score.value,
        "Good code ({}) should score higher than bad code ({})",
        good_score.value,
        bad_score.value
    );
}

#[test]
fn test_optimizer_bootstrap_few_shot() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    let dataset = Dataset::new()
        .example("What is 2+2?", "4")
        .example("What is 5*3?", "15")
        .example("What is 10-4?", "6");

    let result = Optimizer::new(&llm, "Answer the math question with just the number.")
        .dataset(&dataset)
        .metric(|output, expected| {
            if output.trim().contains(expected) {
                1.0
            } else {
                0.0
            }
        })
        .strategy(Strategy::BootstrapFewShot { max_examples: 2 })
        .go();

    assert!(
        result.score > 0.0,
        "Optimizer should achieve non-zero score, got {}",
        result.score
    );
    assert!(result.evaluations > 0, "Should have performed evaluations");
    assert!(
        !result.prompt.is_empty(),
        "Optimized prompt should not be empty"
    );
}

#[test]
fn test_optimizer_instruction_search() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    let dataset = Dataset::new()
        .example("What is the capital of France?", "Paris")
        .example("What is the capital of Japan?", "Tokyo");

    let result = Optimizer::new(&llm, "Answer with just the city name.")
        .dataset(&dataset)
        .metric(|output, expected| {
            if output.to_lowercase().contains(&expected.to_lowercase()) {
                1.0
            } else {
                0.0
            }
        })
        .strategy(Strategy::InstructionSearch { num_candidates: 3 })
        .go();

    assert!(result.evaluations > 0, "Should have performed evaluations");
    assert!(
        result.candidate_scores.len() > 1,
        "Should have tried multiple candidates, got {}",
        result.candidate_scores.len()
    );
}

// ============================================================================
// CLI validation + Agent tool integration tests
// ============================================================================

#[test]
fn test_refine_with_cli_validation() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    // LLM generates bash code, cli() validates by executing it
    let result = refine(&llm, "Write a bash one-liner that prints exactly 'hello'")
        .validate(cli("bash").stdin())
        .extract("bash")
        .max_iter(3)
        .go()
        .unwrap();

    assert!(
        result.score > 0.0,
        "CLI validation should produce a non-zero score, got {}",
        result.score
    );
}

#[test]
fn test_refine_with_combined_cli_and_checks() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    // CLI validation combined with pattern checks
    let result = refine(
        &llm,
        "Write a bash script that computes 5 + 3 and prints the result",
    )
    .validate(
        cli("bash")
            .stdin()
            .and(checks().require_all(["echo", "+"]).forbid("TODO")),
    )
    .extract("bash")
    .max_iter(3)
    .go()
    .unwrap();

    assert!(
        result.score > 0.0,
        "Combined CLI+checks should produce a non-zero score, got {}",
        result.score
    );
}

#[test]
fn test_agent_with_cli_tool() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    // Agent using cli().as_tool() — same cli() pattern, as a tool
    let calc = cli("bc")
        .arg("-l")
        .stdin()
        .as_tool("calculator", "Evaluate math expressions");

    let result = agent(&llm, "What is 23 * 47? Use the calculator tool.")
        .tool(&calc)
        .max_steps(5)
        .go();

    assert!(
        result.output.contains("1081"),
        "Agent should find 23*47=1081, got: {}",
        result.output
    );
}

#[test]
fn test_agent_with_multiple_cli_tools() {
    let Some(llm) = get_llm() else {
        eprintln!("Skipping: no LLM provider available");
        return;
    };

    // Agent with multiple CLI-backed tools
    let wc = cli("wc")
        .arg("-w")
        .stdin()
        .as_tool("word_count", "Count words in text");

    let result = agent(
        &llm,
        "How many words are in the phrase 'hello world foo bar'? Use the word_count tool.",
    )
    .tool(&wc)
    .max_steps(3)
    .go();

    assert!(
        result.output.contains("4"),
        "Agent should find 4 words, got: {}",
        result.output
    );
}
