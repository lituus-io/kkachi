// Best-of-N candidate selection using a real LLM.
//
// Run with: cargo run --example best_of --features api
// Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable,
//           or `claude` CLI installed locally.

use kkachi::recursive::{best_of, checks, ApiLlm, Llm};

fn main() -> anyhow::Result<()> {
    let llm = ApiLlm::from_env()?;

    println!("Using model: {}", llm.model_name());
    println!("Prompt: Write a haiku about Rust programming");
    println!("Generating 5 candidates...\n");

    let (result, pool) = best_of(&llm, "Write a haiku about Rust programming").n(5)
        .metric(|output| {
            let lines: Vec<_> = output.trim().lines().collect();
            if lines.len() == 3 {
                0.8
            } else {
                0.2
            }
        })
        .validate(checks().min_len(10).forbid("```"))
        .go_with_pool();

    println!("Best (score={:.2}):\n{}", result.score, result.output);
    let stats = pool.stats();
    println!(
        "\nPool: count={}, mean={:.2}, std_dev={:.2}",
        stats.count, stats.mean, stats.std_dev
    );
    Ok(())
}
