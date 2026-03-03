// Chain of Thought reasoning using a real LLM.
//
// Run with: cargo run --example reason --features api
// Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable,
//           or `claude` CLI installed locally.

use kkachi::recursive::{checks, reason, ApiLlm, Llm};

fn main() -> anyhow::Result<()> {
    let llm = ApiLlm::from_env()?;

    println!("Using model: {}", llm.model_name());
    println!("Prompt: A farmer has 17 sheep. All but 9 die. How many are left?\n");

    let result = reason(
        &llm,
        "A farmer has 17 sheep. All but 9 die. How many are left?",
    )
    .validate(checks().regex(r"\d+"))
    .max_iter(3)
    .go();

    println!("Reasoning:\n{}", result.reasoning());
    println!("\nAnswer: {}", result.output);
    println!("Score: {:.2}", result.score);
    Ok(())
}
