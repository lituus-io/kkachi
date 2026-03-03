// Iterative refinement with validation using a real LLM.
//
// Run with: cargo run --example refine --features api
// Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable,
//           or `claude` CLI installed locally.

use kkachi::recursive::{checks, refine, ApiLlm, Llm};

fn main() -> anyhow::Result<()> {
    let llm = ApiLlm::from_env()?;

    println!("Using model: {}", llm.model_name());
    println!("Prompt: Write a Rust function that parses a URL into its components\n");

    let result = refine(
        &llm,
        "Write a Rust function that parses a URL into its components (scheme, host, path)",
    )
    .validate(
        checks()
            .require("fn ")
            .require("->")
            .require("Result")
            .forbid(".unwrap()")
            .min_len(80),
    )
    .max_iter(5)
    .target(0.9)
    .go()?;

    println!("Score: {:.0}%", result.score * 100.0);
    println!("Iterations: {}", result.iterations);
    println!("Output:\n{}", result.output);
    Ok(())
}
