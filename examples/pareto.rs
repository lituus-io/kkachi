// Multi-objective optimization with Pareto front exploration.
//
// Run with: cargo run --example pareto --features api
// Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable,
//           or `claude` CLI installed locally.

use kkachi::recursive::{
    checks, multi_objective, refine_pareto_sync, ApiLlm, Llm, Objective, Scalarization,
};

fn main() -> anyhow::Result<()> {
    let llm = ApiLlm::from_env()?;

    println!("Using model: {}", llm.model_name());
    println!("Multi-objective: correctness vs brevity vs safety\n");

    // Define competing objectives with different validators
    let validator = multi_objective()
        .scalarize(Scalarization::Chebyshev)
        .objectives([
            (
                Objective::new("correctness").weight(2.0).target(0.9),
                checks().require("fn ").require("->").require("Result"),
            ),
            (
                Objective::new("brevity").weight(1.0).target(0.8),
                checks().max_len(300),
            ),
            (
                Objective::new("safety").weight(1.5).target(0.9),
                checks().forbid(".unwrap()").forbid("panic!").forbid("unsafe"),
            ),
        ]);

    let result = refine_pareto_sync(
        &llm,
        "Write a Rust function that reads a file and returns its lines as a Vec<String>",
        &validator,
        5,
    );

    println!("Iterations: {}", result.iterations);
    println!("Pareto front size: {}", result.front.len());
    println!("\nPer-objective bests:");
    for (name, score) in &result.objective_bests {
        println!("  {}: {:.0}%", name, score * 100.0);
    }
    println!("\nBest output (scalarized):\n{}", result.best_output);
    Ok(())
}
