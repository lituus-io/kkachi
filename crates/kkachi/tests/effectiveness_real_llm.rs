// Real-LLM effectiveness tests for kkachi library.
// Uses CliLlm (Claude CLI subprocess) for all tests.
// Tests all core patterns: refine, reason, program, agent, best_of, ensemble, semantic, optimize.
//
// These tests require the `claude` CLI to be installed locally.
// Run with: cargo test -p kkachi --test effectiveness_real_llm -- --ignored --nocapture

use kkachi::recursive::prelude::*;

fn get_llm() -> CliLlm {
    CliLlm::new().expect("Claude Code CLI must be available")
}

// =============================================================================
// 1. REFINE: Iterative prompt refinement with validation
// =============================================================================

#[test]
#[ignore]
fn test_refine_basic_with_checks() {
    let llm = get_llm();
    let result = refine(&llm, "Write a Rust function named 'fibonacci' that takes n: u64 and returns the nth Fibonacci number. Include a doc comment.")
        .validate(
            checks()
                .require_all(["fn fibonacci", "u64", "///"])
                .forbid_all(["todo!", "unimplemented!"])
                .min_len(50)
        )
        .max_iter(3)
        .go()
        .unwrap();

    println!(
        "[refine_basic] score={:.2}, iterations={}, output_len={}",
        result.score,
        result.iterations,
        result.output.len()
    );
    println!(
        "[refine_basic] output:\n{}",
        &result.output[..result.output.len().min(300)]
    );
    assert!(
        result.score > 0.5,
        "Expected score > 0.5, got {}",
        result.score
    );
    assert!(
        result.output.contains("fibonacci"),
        "Output should contain 'fibonacci'"
    );
}

#[test]
#[ignore]
fn test_refine_with_cli_validation() {
    let llm = get_llm();
    let result = refine(
        &llm,
        "Write a bash one-liner that prints exactly the text 'hello world' (nothing else)",
    )
    .validate(cli("bash").stdin())
    .extract("bash")
    .max_iter(3)
    .go()
    .unwrap();

    println!(
        "[refine_cli] score={:.2}, iterations={}, output_len={}",
        result.score,
        result.iterations,
        result.output.len()
    );
    println!("[refine_cli] output: {}", result.output.trim());
    assert!(
        result.score > 0.0,
        "CLI validation should produce non-zero score"
    );
}

#[test]
#[ignore]
fn test_refine_with_combined_validation() {
    let llm = get_llm();
    let result = refine(
        &llm,
        "Write a bash script that computes 5 factorial (5!) and prints just the number 120",
    )
    .validate(cli("bash").stdin().and(checks().min_len(5)))
    .extract("bash")
    .max_iter(3)
    .go()
    .unwrap();

    println!(
        "[refine_combined] score={:.2}, iterations={}",
        result.score, result.iterations
    );
    println!("[refine_combined] output: {}", result.output.trim());
    assert!(
        result.score > 0.0,
        "Combined validation should produce non-zero score"
    );
}

#[test]
#[ignore]
fn test_refine_extract_code() {
    let llm = get_llm();
    let result = refine(&llm, "Write a Python function called 'is_prime' that checks if a number is prime and returns a boolean")
        .validate(checks().require_all(["def is_prime", "return"]).min_len(30))
        .extract("python")
        .max_iter(3)
        .go()
        .unwrap();

    println!(
        "[refine_extract] score={:.2}, output_len={}",
        result.score,
        result.output.len()
    );
    println!(
        "[refine_extract] output:\n{}",
        &result.output[..result.output.len().min(300)]
    );
    // Output should be clean code without markdown fences
    assert!(
        !result.output.contains("```"),
        "extract() should remove code fences"
    );
}

#[test]
#[ignore]
fn test_refine_adaptive_mode() {
    let llm = get_llm();
    let result = refine(
        &llm,
        "Write a concise haiku about programming (5-7-5 syllables)",
    )
    .validate(checks().min_len(10).max_len(200))
    .adaptive()
    .min_iter(2)
    .max_iter(5)
    .go()
    .unwrap();

    println!(
        "[refine_adaptive] score={:.2}, iterations={}, stop={:?}",
        result.score, result.iterations, result.stop_reason
    );
    println!("[refine_adaptive] output: {}", result.output.trim());
    assert!(result.iterations >= 1, "Should run at least 1 iteration");
}

// =============================================================================
// 2. REASON: Chain of Thought reasoning
// =============================================================================

#[test]
#[ignore]
fn test_reason_basic() {
    let llm = get_llm();
    let result = reason(
        &llm,
        "What is 17 * 23? Think step by step and give the final numeric answer after 'Therefore:'",
    )
    .go();

    println!(
        "[reason_basic] output='{}', reasoning_len={}, iterations={}",
        result.output.trim(),
        result.reasoning().len(),
        result.iterations
    );
    assert!(!result.output.is_empty(), "Should produce an answer");
    // 17 * 23 = 391
    assert!(
        result.output.contains("391"),
        "Expected 391, got: {}",
        result.output
    );
}

#[test]
#[ignore]
fn test_reason_with_validation() {
    let llm = get_llm();
    let result = reason(
        &llm,
        "What is the sum of the first 10 positive integers? Give just the number after 'Answer:'",
    )
    .validate(checks().regex(r"\b55\b"))
    .max_iter(3)
    .go();

    println!(
        "[reason_validated] output='{}', score={:.2}",
        result.output.trim(),
        result.score
    );
    assert!(
        result.score > 0.0,
        "Should pass validation with correct answer"
    );
}

#[test]
#[ignore]
fn test_reason_with_extract() {
    let llm = get_llm();
    let result = reason(&llm, "Write a Python one-liner that prints the square of 7. Put it in a ```python code block, then say 'Therefore: done'")
        .extract("python")
        .go();

    println!(
        "[reason_extract] output='{}', output_len={}",
        result.output.trim(),
        result.output.len()
    );
    // Should extract the code, not the reasoning text
    assert!(
        !result.output.contains("```"),
        "extract should remove fences from output"
    );
}

// =============================================================================
// 3. PROGRAM: Code generation and execution
// =============================================================================

#[test]
#[ignore]
fn test_program_bash() {
    let llm = get_llm();
    let result = program(
        &llm,
        "Calculate 2^10 (two to the power of ten) and print the result",
    )
    .executor(bash_executor())
    .max_iter(3)
    .go();

    println!(
        "[program_bash] success={}, output='{}', attempts={}",
        result.success,
        result.output.trim(),
        result.attempts
    );
    assert!(result.success, "Program should execute successfully");
    assert!(
        result.output.contains("1024"),
        "Expected 1024, got: {}",
        result.output
    );
}

#[test]
#[ignore]
fn test_program_with_validation() {
    let llm = get_llm();
    let result = program(
        &llm,
        "Print the first 5 Fibonacci numbers (1,1,2,3,5), one per line",
    )
    .executor(bash_executor())
    .validate(checks().require_all(["1", "2", "3", "5"]))
    .max_iter(3)
    .go();

    println!(
        "[program_validated] success={}, output='{}'",
        result.success,
        result.output.trim()
    );
    assert!(result.success, "Program should succeed");
}

#[test]
#[ignore]
fn test_program_with_cli_executor() {
    let llm = get_llm();
    let result = program(&llm, "Print the text 'hello from cli' exactly")
        .executor(cli("bash").stdin())
        .max_iter(2)
        .go();

    println!(
        "[program_cli] success={}, output='{}'",
        result.success,
        result.output.trim()
    );
    assert!(result.success, "Program with CLI executor should succeed");
}

// =============================================================================
// 4. AGENT: ReAct agent with tool calling
// =============================================================================

#[test]
#[ignore]
fn test_agent_with_cli_tool() {
    let llm = get_llm();
    let calc = cli("bc").stdin().as_tool("calculator", "Evaluate mathematical expressions using bc. Input should be a math expression like '23*47'");

    let result = agent(
        &llm,
        "What is 23 * 47? Use the calculator tool with input '23*47'.",
    )
    .tool(&calc)
    .max_steps(5)
    .go();

    println!(
        "[agent_tool] success={}, output='{}', steps={}",
        result.success,
        result.output.trim(),
        result.steps
    );
    println!("[agent_tool] trajectory len={}", result.trajectory.len());
    for (i, step) in result.trajectory.iter().enumerate() {
        println!(
            "  step {}: action='{}', obs='{}'",
            i,
            step.action,
            &step.observation[..step.observation.len().min(80)]
        );
    }
    assert!(result.success, "Agent should find an answer");
    // 23 * 47 = 1081
    assert!(
        result.output.contains("1081"),
        "Expected 1081, got: {}",
        result.output
    );
}

#[test]
#[ignore]
fn test_agent_with_validation() {
    let llm = get_llm();
    let wc = cli("wc")
        .arg("-c")
        .stdin()
        .as_tool("char_count", "Count bytes in text passed via stdin");

    let result = agent(&llm, "How many bytes are in the word 'hello'? Use char_count tool with input 'hello'. Give just the number.")
        .tool(&wc)
        .validate(checks().regex(r"\d+"))
        .max_steps(5)
        .go();

    println!(
        "[agent_validate] success={}, output='{}'",
        result.success,
        result.output.trim()
    );
    assert!(result.success, "Agent with validation should succeed");
}

#[test]
#[ignore]
fn test_agent_max_iter_alias() {
    let llm = get_llm();
    let result = agent(&llm, "What is 2+2? Answer with 'Final Answer: 4'")
        .max_iter(3) // u32 alias for max_steps
        .go();

    println!(
        "[agent_max_iter] success={}, output='{}'",
        result.success,
        result.output.trim()
    );
    assert!(result.success, "Agent should produce an answer");
}

// =============================================================================
// 5. BEST-OF-N: Generate multiple candidates, pick best
// =============================================================================

#[test]
#[ignore]
fn test_best_of_with_scorer() {
    let llm = get_llm();
    let result = best_of(
        &llm,
        "Write a one-sentence explanation of what Rust's borrow checker does",
    )
    .metric(|text: &str| {
        let mut score = 0.0f64;
        if text.contains("borrow") || text.contains("ownership") {
            score += 0.3;
        }
        if text.contains("memory") || text.contains("safety") {
            score += 0.3;
        }
        if text.len() > 30 && text.len() < 200 {
            score += 0.4;
        }
        score
    })
    .go();

    println!(
        "[best_of] output_len={}, score={:.2}",
        result.output.len(),
        result.score
    );
    println!(
        "[best_of] output: {}",
        &result.output[..result.output.len().min(200)]
    );
    assert!(!result.output.is_empty(), "Should produce a best candidate");
    assert!(result.score > 0.0, "Score should be positive");
}

#[test]
#[ignore]
fn test_best_of_with_pool() {
    let llm = get_llm();
    let (result, pool) = best_of(&llm, "Give a creative name for a programming language")
        .n(4)
        .go_with_pool();

    println!(
        "[best_of_pool] best='{}', pool size={}",
        result.output.trim(),
        pool.candidates().len()
    );
    let stats = pool.stats();
    println!(
        "[best_of_pool] stats: mean={:.2}, std={:.2}, min={:.2}, max={:.2}",
        stats.mean, stats.std_dev, stats.min, stats.max
    );
    assert!(
        pool.candidates().len() >= 2,
        "Pool should have at least 2 candidates"
    );
}

#[test]
#[ignore]
fn test_best_of_diverse() {
    let llm = get_llm();
    // Diversity is enabled by default in BestOf
    let (_, pool) = best_of(
        &llm,
        "Write a short greeting message for a programming blog",
    )
    .go_with_pool();

    println!("[best_of_diverse] pool size={}", pool.candidates().len());
    for (i, c) in pool.top_k(3).iter().enumerate() {
        println!(
            "  candidate {}: score={:.2}, text='{}'",
            i,
            c.combined_score,
            &c.output[..c.output.len().min(60)]
        );
    }
    // Check that candidates are reasonably different
    if pool.candidates().len() >= 2 {
        let candidates = pool.top_k(3);
        let c0 = &candidates[0].output;
        let c1 = &candidates[1].output;
        assert_ne!(c0, c1, "Diverse candidates should be different");
    }
}

// =============================================================================
// 6. ENSEMBLE: Multiple LLM calls with consensus
// =============================================================================

#[test]
#[ignore]
fn test_ensemble_majority_vote() {
    let llm = get_llm();
    let result = ensemble(
        &llm,
        "What is the capital of France? Answer with just the city name, nothing else.",
    )
    .go();

    println!("[ensemble] output='{}'", result.output.trim());
    assert!(
        result.output.to_lowercase().contains("paris"),
        "Ensemble should agree on Paris"
    );
}

#[test]
#[ignore]
fn test_ensemble_with_consensus_pool() {
    let llm = get_llm();
    let (result, pool) = ensemble(
        &llm,
        "Is Rust a compiled or interpreted language? Answer with exactly one word.",
    )
    .go_with_consensus();

    println!(
        "[ensemble_pool] consensus output='{}'",
        result.output.trim()
    );
    println!(
        "[ensemble_pool] agreement_ratio={:.2}, distinct={}",
        pool.agreement_ratio(),
        pool.vote_counts().len()
    );
    assert!(
        result.output.to_lowercase().contains("compiled"),
        "Should agree on 'compiled'"
    );
}

// =============================================================================
// 7. SEMANTIC VALIDATION: LLM-as-judge
// =============================================================================

#[test]
#[ignore]
fn test_semantic_validation() {
    let llm = get_llm();
    let validator = semantic(&llm)
        .criterion("The output should be a valid Python function definition with proper syntax")
        .build();

    // Good input
    let good_score = validator.validate("def hello():\n    print('hello world')");
    println!("[semantic] good_score={:.2}", good_score.value);

    // Bad input
    let bad_score =
        validator.validate("this is not python code at all, just random text about cats");
    println!("[semantic] bad_score={:.2}", bad_score.value);

    assert!(
        good_score.value > bad_score.value,
        "Good code ({:.2}) should score higher than random text ({:.2})",
        good_score.value,
        bad_score.value
    );
}

// =============================================================================
// 8. COMPOSITION: Validator combinators
// =============================================================================

#[test]
#[ignore]
fn test_composition_and_or() {
    let llm = get_llm();

    // AND composition
    let result = refine(&llm, "Write a Python function called 'greet' that takes a name parameter and returns a greeting string")
        .validate(
            checks().require("def greet").min_len(20)
                .and(checks().require("name").forbid("TODO"))
        )
        .max_iter(2)
        .go()
        .unwrap();

    println!("[compose_and] score={:.2}", result.score);
    assert!(result.score > 0.5, "AND composition should work");

    // OR composition
    let or_validator = checks()
        .require("def greet")
        .or(checks().require("fn greet"));

    let score = or_validator.validate("def greet(name):\n    pass");
    println!("[compose_or] score={:.2}", score.value);
    assert!(score.value >= 1.0, "OR should pass when first arm matches");
}

#[test]
#[ignore]
fn test_composition_all_any() {
    // ALL - all must pass
    let all_v = all([
        checks().min_len(5),
        checks().require("hello"),
        checks().forbid("goodbye"),
    ]);
    let score = all_v.validate("hello world");
    println!("[compose_all] score={:.2}", score.value);
    assert_eq!(score.value, 1.0);

    // ANY - at least one must pass
    let any_v = any([
        checks().require("xyz"),
        checks().require("hello"),
        checks().require("abc"),
    ]);
    let score = any_v.validate("hello world");
    println!("[compose_any] score={:.2}", score.value);
    assert_eq!(score.value, 1.0);
}

// =============================================================================
// 9. MEMORY/RAG: Vector search
// =============================================================================

#[test]
#[ignore]
fn test_memory_search() {
    let mut mem: Memory<HashEmbedder> = memory();

    mem.add("Rust is a systems programming language focused on safety");
    mem.add("Python is great for data science and machine learning");
    mem.add("JavaScript runs in web browsers and Node.js");
    mem.add("Go is designed for concurrent programming");

    let results = mem.search("memory safe language", 2);
    println!("[memory] top result: '{}'", results[0].content);
    assert!(
        results[0].content.contains("Rust"),
        "Should find Rust for 'memory safe'"
    );
}

#[test]
#[ignore]
fn test_memory_diverse_search() {
    let mut mem: Memory<HashEmbedder> = memory();

    mem.add_tagged("animals", "Cats are independent pets");
    mem.add_tagged("animals", "Dogs are loyal companions");
    mem.add_tagged("tech", "Python is a programming language");
    mem.add_tagged("tech", "Rust ensures memory safety");

    let results = mem.search_diverse("pets and programming", 3, 0.5);
    println!("[memory_diverse] found {} results", results.len());
    for r in &results {
        println!("  - '{}'", r.content);
    }
    assert!(results.len() >= 2, "Should find diverse results");
}

// =============================================================================
// 10. OPTIMIZER: Prompt optimization with dataset
// =============================================================================

#[test]
#[ignore]
fn test_optimizer_bootstrap() {
    let llm = get_llm();
    let dataset = Dataset::new()
        .example("What is 2+2?", "4")
        .example("What is 3+3?", "6")
        .example("What is 5+5?", "10");

    let result = Optimizer::new(&llm, "Answer math questions with just the number")
        .dataset(&dataset)
        .strategy(Strategy::BootstrapFewShot { max_examples: 2 })
        .metric(|output: &str, expected: &str| {
            if output.trim() == expected.trim() {
                1.0
            } else {
                0.0
            }
        })
        .go();

    println!(
        "[optimizer] score={:.2}, evaluations={}",
        result.score, result.evaluations
    );
    println!(
        "[optimizer] optimized prompt: '{}'",
        &result.prompt[..result.prompt.len().min(100)]
    );
    println!("[optimizer] examples selected: {}", result.examples.len());
}

// =============================================================================
// 11. TEMPLATE: Structured prompting
// =============================================================================

#[test]
#[ignore]
fn test_template_usage() {
    let llm = get_llm();
    use kkachi::recursive::template::Template;

    let template = Template::from_str(
        r#"---
name: explainer
signature: topic -> explanation
format:
  type: plain
---
Given a topic, provide a brief explanation in 1-2 sentences.
"#,
    )
    .unwrap();

    let prompt = template.assemble_prompt("quantum entanglement", 0, None);

    println!("[template] assembled prompt len={}", prompt.len());
    assert!(
        prompt.contains("quantum entanglement"),
        "Template should include input"
    );

    // Use it with the LLM
    let output = futures::executor::block_on(llm.generate(&prompt, "", None)).unwrap();
    println!(
        "[template] llm output: {}",
        &output.text[..output.text.len().min(200)]
    );
    assert!(
        !output.text.is_empty(),
        "LLM should respond to template prompt"
    );
}

// =============================================================================
// 12. REWRITE: Markdown manipulation
// =============================================================================

#[test]
#[ignore]
fn test_rewrite_operations() {
    let doc = "# Header\n\nSome content here.\n\n## Section A\n\nDetails about A.\n\n## Section B\n\nDetails about B.";

    let result = rewrite(doc)
        .section("Section A", "Updated details about A with more info.")
        .section_after("Section B", "Section C", "New section C content.")
        .build();

    println!("[rewrite] result:\n{}", result);
    assert!(
        result.contains("Updated details"),
        "Should have replaced section A"
    );
    assert!(result.contains("Section C"), "Should have added section C");
}

// =============================================================================
// 13. CLI: Subprocess validation and tools
// =============================================================================

#[test]
#[ignore]
fn test_cli_multi_stage() {
    let validator = cli("bash").stdin().then("grep").arg("-c").arg("hello");

    let score = validator.validate("echo hello; echo hello; echo world");
    println!(
        "[cli_multi] score={:.2}, feedback={:?}",
        score.value,
        score.feedback_str()
    );
    assert!(score.value > 0.0, "Multi-stage CLI should work");
}

#[test]
#[ignore]
fn test_cli_as_tool() {
    let date_tool = cli("date")
        .arg("+%Y")
        .as_tool("year", "Get the current year");
    let result = futures::executor::block_on(date_tool.execute(""));
    println!("[cli_tool] result: {:?}", result);
    assert!(result.is_ok(), "CLI tool should execute");
    assert!(
        result.unwrap().contains("202"),
        "Should return current year"
    );
}

// =============================================================================
// 14. FULL PIPELINE: End-to-end workflows
// =============================================================================

#[test]
#[ignore]
fn test_full_pipeline_refine_then_validate() {
    let llm = get_llm();

    // Step 1: Generate code with refine
    let code_result = refine(&llm, "Write a bash function named 'add' that takes two positional args and echoes their sum using $(( )) arithmetic")
        .validate(checks().require_all(["add", "echo"]).min_len(20))
        .max_iter(3)
        .go()
        .unwrap();

    println!("[pipeline] step1 score={:.2}", code_result.score);

    // Step 2: Validate the code runs
    if code_result.score >= 1.0 {
        let cli_score = cli("bash")
            .stdin()
            .validate(&format!("{}\nadd 3 4", code_result.output));
        println!("[pipeline] step2 cli_score={:.2}", cli_score.value);
    }
}

#[test]
#[ignore]
fn test_full_pipeline_reason_then_program() {
    let llm = get_llm();

    // Step 1: Reason about the approach
    let reason_result = reason(&llm, "What is the formula for the sum of squares of first N natural numbers? Give the formula after 'Therefore:'")
        .go();

    println!(
        "[pipeline2] reasoning len={}, output='{}'",
        reason_result.reasoning().len(),
        reason_result.output.trim()
    );

    // Step 2: Use that knowledge to write and execute code
    let prog_result = program(&llm, &format!(
        "Context: {}\nWrite bash code to compute the sum of squares of 1 to 10 using the formula or a loop. Print just the number (385).",
        reason_result.output
    ))
        .executor(bash_executor())
        .max_iter(2)
        .go();

    println!(
        "[pipeline2] program success={}, output='{}'",
        prog_result.success,
        prog_result.output.trim()
    );
    if prog_result.success {
        // Sum of squares 1..10 = 385
        assert!(
            prog_result.output.contains("385"),
            "Expected 385, got: {}",
            prog_result.output
        );
    }
}
