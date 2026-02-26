// Multi-chain ensemble voting using a real LLM.
//
// Run with: cargo run --example ensemble --features api
// Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable,
//           or `claude` CLI installed locally.

use kkachi::recursive::{ensemble, Aggregate, ApiLlm, Llm};

fn main() -> anyhow::Result<()> {
    let llm = ApiLlm::from_env()?;

    println!("Using model: {}", llm.model_name());
    println!("Prompt: What is the capital of Australia?");
    println!("Generating 5 chains with majority vote...\n");

    let (result, consensus) = ensemble(&llm, "What is the capital of Australia?").n(5)
        .aggregate(Aggregate::MajorityVote)
        .go_with_consensus();

    println!("Answer: {}", result.output);
    println!(
        "Agreement: {:.0}%",
        consensus.agreement_ratio() * 100.0
    );
    println!("Chains generated: {}", result.chains_generated);
    Ok(())
}
