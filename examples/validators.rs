// Validator composition demo (no LLM needed).
//
// Run with: cargo run --example validators
// No API key required.

use kkachi::recursive::{checks, Validate};

fn main() {
    // Batch API: require_all/forbid_all for concise validation
    let strict = checks()
        .require_all(["fn ", "Result", "->"])
        .forbid_all([".unwrap()", "panic!", "todo!"])
        .regex(r"fn \w+\(")
        .min_len(50);

    let good =
        "/// Parses input.\nfn parse(s: &str) -> Result<i32, std::num::ParseIntError> { s.parse() }";
    let bad = "fn parse(s: &str) -> i32 { s.parse().unwrap() }";

    let good_score = strict.validate(good);
    let bad_score = strict.validate(bad);

    println!("Good code score: {:.2}", good_score.value);
    if let Some(ref feedback) = good_score.feedback {
        println!("  Feedback: {}", feedback);
    }

    println!("\nBad code score: {:.2}", bad_score.value);
    if let Some(ref feedback) = bad_score.feedback {
        println!("  Feedback: {}", feedback);
    }

    // Weighted batch: regex patterns with lower weight
    let flexible = checks()
        .require_all(["fn ", "->"])
        .regex_all_weighted([r"\w+::\w+", r"pub\s+fn"], 0.5);

    let score = flexible.validate("pub fn add(a: i32, b: i32) -> i32 { std::cmp::max(a, b) }");
    println!("\nFlexible score: {:.2}", score.value);
}
