// Rate-limited LLM calls with composable wrappers.
//
// Run with: cargo run --example rate_limit --features api
// Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable,
//           or `claude` CLI installed locally.

use kkachi::recursive::{checks, refine, ApiLlm, CacheExt, LlmExt, Llm, RateLimitConfig, RateLimitExt};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // Compose: cache (innermost) → rate limit → retry (outermost)
    let llm = ApiLlm::from_env()?
        .with_cache(50)
        .with_rate_limit_config(RateLimitConfig::new(5.0).with_burst(2))
        .with_retry(3);

    println!("Using model: {}", llm.model_name());
    println!("Config: 5 req/s, burst=2, cache=50, retry=3\n");

    let start = Instant::now();

    // Run 3 refinements — rate limiter paces the calls
    for i in 1..=3 {
        let result = refine(
            &llm,
            &format!("Write a short Rust function #{} that converts Celsius to Fahrenheit", i),
        )
        .validate(checks().require("fn ").require("f64").forbid("todo!"))
        .max_iter(3)
        .go()?;

        println!(
            "#{}: score={:.0}%, iters={}, elapsed={:.1}s",
            i,
            result.score * 100.0,
            result.iterations,
            start.elapsed().as_secs_f64()
        );
    }

    println!("\nTotal time: {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}
