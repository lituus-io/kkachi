// Parallel fan-out pipeline with multiple branches.
//
// Run with: cargo run --example fan_out --features api
// Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable,
//           or `claude` CLI installed locally.

use kkachi::recursive::{checks, pipeline, ApiLlm, BranchBuilder, Llm, MergeStrategy};

fn main() -> anyhow::Result<()> {
    let llm = ApiLlm::from_env()?;

    println!("Using model: {}", llm.model_name());
    println!("Fan-out: generating solutions in Rust, Python, and Go concurrently\n");

    let result = pipeline(&llm, "Write a function that checks if a string is a palindrome")
        .fan_out(
            vec![
                BranchBuilder::new("rust")
                    .refine(checks().require("fn ").require("bool").forbid("todo!")),
                BranchBuilder::new("python")
                    .refine(checks().require("def ").require("return")),
                BranchBuilder::new("go")
                    .refine(checks().require("func ").require("bool")),
            ],
            MergeStrategy::Concat {
                separator: "\n---\n".to_string(),
            },
        )
        .go();

    println!("Steps: {}", result.steps.len());
    println!("Output:\n{}", result.output);
    Ok(())
}
