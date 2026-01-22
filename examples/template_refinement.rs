// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Example demonstrating template-based recursive refinement.
//!
//! Templates provide a structured way to define output formats, system prompts,
//! and few-shot examples for consistent LLM outputs.
//!
//! Run with: `cargo run --example template_refinement`

use kkachi::recursive::{Template, FormatType, TemplateExample, TemplateCritic, HeuristicCritic};
use kkachi::kkachi;
use kkachi::error::Result;

fn main() -> Result<()> {
    println!("=== Template-Based Refinement Example ===\n");

    // Example 1: Inline template with JSON output format
    example_inline_template()?;

    // Example 2: Template with few-shot examples
    example_template_with_examples()?;

    // Example 3: Using TemplateCritic for format validation
    example_template_critic()?;

    Ok(())
}

/// Example 1: Using an inline template for JSON output
fn example_inline_template() -> Result<()> {
    println!("--- Example 1: Inline Template ---\n");

    let result = Kkachi::refine("question -> answer")
        .template_content(r#"
---
name: qa_json
version: "1.0"
format:
  type: json
  schema:
    type: object
    required:
      - answer
      - confidence
options:
  strict: false
  include_in_prompt: true
---
You are a helpful assistant that answers questions.

## Output Format

Return your answer as JSON with the following fields:
- `answer`: Your answer to the question
- `confidence`: A number from 0 to 1 indicating how confident you are
"#)
        .max_iterations(3)
        .until_score(0.9)
        .critic_always_pass()
        .run("What is the capital of France?", |iteration, feedback| {
            // Simulate LLM response - in production, this would call an actual LLM
            let response = match iteration {
                0 => r#"{"answer": "Paris", "confidence": 0.95}"#.to_string(),
                _ => {
                    if let Some(fb) = feedback {
                        println!("  Received feedback: {}", fb);
                    }
                    r#"{"answer": "Paris", "confidence": 0.99}"#.to_string()
                }
            };
            println!("  Iteration {}: {}", iteration, response);
            Ok(response)
        });

    println!("\nResult: {}\n", result.answer);
    Ok(())
}

/// Example 2: Template with few-shot examples
fn example_template_with_examples() -> Result<()> {
    println!("--- Example 2: Template with Few-Shot Examples ---\n");

    let template_content = r#"
---
name: code_generator
version: "1.0"
signature: "requirement -> code"
format:
  type: json
  schema:
    type: object
    required:
      - code
      - explanation
options:
  strict: true
  include_in_prompt: true
---
You are an expert Rust programmer. Generate clean, idiomatic code.

## Output Format

Return your response as JSON with:
- `code`: The generated Rust code
- `explanation`: Brief explanation of the implementation

---examples---

## Example 1

**Input:** Write a function that checks if a number is even

**Output:**
```json
{
  "code": "fn is_even(n: i32) -> bool { n % 2 == 0 }",
  "explanation": "Uses modulo operator to check divisibility by 2"
}
```

## Example 2

**Input:** Write a function to reverse a string

**Output:**
```json
{
  "code": "fn reverse(s: &str) -> String { s.chars().rev().collect() }",
  "explanation": "Iterates over characters in reverse and collects into a new String"
}
```
"#;

    // Parse the template
    let template = Template::from_str(template_content)?;

    println!("Template name: {}", template.name);
    println!("Format type: {:?}", template.format.format_type);
    println!("Number of examples: {}", template.examples.len());
    println!("Strict mode: {}", template.options.strict);

    // Demonstrate prompt assembly
    let prompt = template.assemble_prompt("Write a function to calculate factorial", 0, None);
    println!("\n--- Assembled Prompt Preview ---");
    println!("{}", &prompt[..prompt.len().min(500)]);
    println!("...\n");

    // Use with Kkachi
    let result = Kkachi::refine("requirement -> code")
        .with_template(template.into_owned())
        .max_iterations(3)
        .critic_always_pass()
        .run("Write a function to calculate factorial", |iteration, _| {
            let response = r#"{"code": "fn factorial(n: u64) -> u64 { (1..=n).product() }", "explanation": "Uses iterator product for concise implementation"}"#;
            println!("  Iteration {}: Generated code response", iteration);
            Ok(response.to_string())
        });

    println!("\nFinal result: {}\n", result.answer);
    Ok(())
}

/// Example 3: Using TemplateCritic for format validation
fn example_template_critic() -> Result<()> {
    println!("--- Example 3: TemplateCritic for Format Validation ---\n");

    use kkachi::recursive::{RecursiveState, Critic};
    use kkachi::str_view::StrView;

    // Create a template with strict JSON schema
    let template = Template::new("test")
        .with_format(FormatType::Json)
        .strict(true)
        .with_example(TemplateExample::new(
            "test input",
            r#"{"result": "test output"}"#,
        ));

    // Create a template critic wrapping a heuristic critic
    let inner_critic = HeuristicCritic::new().min_length(10);
    let critic = TemplateCritic::new(&template, inner_critic);

    // Test with valid JSON
    let state = RecursiveState::new();
    let valid_output = r#"{"result": "hello world"}"#;
    let result = critic.evaluate(StrView::new(valid_output), &state);
    println!("Valid JSON output:");
    println!("  Score: {:.2}", result.score);
    println!("  Feedback: {:?}", result.feedback);

    // Test with invalid JSON
    let invalid_output = "not valid json";
    let result = critic.evaluate(StrView::new(invalid_output), &state);
    println!("\nInvalid JSON output:");
    println!("  Score: {:.2}", result.score);
    println!("  Feedback: {:?}", result.feedback);

    // Test with valid JSON but missing required fields (using schema)
    let template_with_schema_content = r#"
---
format:
  type: json
  schema:
    type: object
    required:
      - answer
options:
  strict: true
---
Test template
"#;

    let schema_template = Template::from_str(template_with_schema_content)?;
    let schema_critic = TemplateCritic::new(&schema_template, HeuristicCritic::new());

    let missing_field = r#"{"other": "value"}"#;
    let result = schema_critic.evaluate(StrView::new(missing_field), &state);
    println!("\nJSON missing required field:");
    println!("  Score: {:.2}", result.score);
    println!("  Feedback: {:?}", result.feedback);

    let has_required = r#"{"answer": "42"}"#;
    let result = schema_critic.evaluate(StrView::new(has_required), &state);
    println!("\nJSON with required field:");
    println!("  Score: {:.2}", result.score);
    println!("  Feedback: {:?}\n", result.feedback);

    Ok(())
}
