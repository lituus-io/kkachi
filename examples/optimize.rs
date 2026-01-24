// True prompt optimization via dataset evaluation.
//
// Unlike `refine` (which retries outputs), this optimizes the *prompt itself*
// by evaluating different strategies against a training dataset.
//
// Run with: cargo run --example optimize --features api
// Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable,
//           or `claude` CLI installed locally.

use kkachi::recursive::{ApiLlm, Dataset, Llm, Optimizer, Strategy};

fn main() -> anyhow::Result<()> {
    let llm = ApiLlm::from_env()?;

    println!("Using model: {}", llm.model_name());
    println!("Strategy: BootstrapFewShot + InstructionSearch (Combined)\n");

    // Define a training dataset of input/output pairs
    let dataset = Dataset::new()
        .example("What is 2+2?", "4")
        .example("What is 7*8?", "56")
        .example("What is 100/5?", "20")
        .example("What is 15-9?", "6")
        .example("What is 3^3?", "27");

    // Run the optimizer: finds the best instruction + few-shot examples
    let result = Optimizer::new(&llm, "Answer the math question with just the number.")
        .dataset(&dataset)
        .metric(|output, expected| {
            if output.trim().contains(expected) {
                1.0
            } else {
                0.0
            }
        })
        .strategy(Strategy::Combined {
            max_examples: 2,
            num_candidates: 3,
        })
        .go();

    println!("Optimized prompt:\n{}\n", result.prompt);
    println!("Score: {:.0}%", result.score * 100.0);
    println!("Evaluations: {}", result.evaluations);
    println!(
        "Candidates tried: {} (scores: {:?})",
        result.candidate_scores.len(),
        result
            .candidate_scores
            .iter()
            .map(|s| format!("{:.2}", s))
            .collect::<Vec<_>>()
    );
    if !result.examples.is_empty() {
        println!("\nFew-shot examples selected:");
        for ex in &result.examples {
            println!("  {} → {}", ex.input, ex.expected);
        }
    }
    Ok(())
}
