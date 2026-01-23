// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! GCS Bucket RAG-Assisted Generation Example
//!
//! Demonstrates building Pulumi YAML infrastructure code using:
//! 1. RAG lookup from memory store for similar examples
//! 2. Recursive refinement with `pulumi preview` validation
//! 3. CLI validators with GCP credentials
//!
//! # GCP Credentials
//!
//! This example demonstrates authentication methods:
//! - Application Default Credentials (ADC) via `GOOGLE_APPLICATION_CREDENTIALS`
//! - Project ID via `GOOGLE_PROJECT` or `CLOUDSDK_CORE_PROJECT`
//!
//! # Prerequisites
//!
//! ```bash
//! # Set GCP credentials
//! export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
//! export GOOGLE_PROJECT=my-gcp-project
//!
//! # Run the example
//! cargo run --example pulumi_gcs_rag_optimizer
//! ```

use kkachi::recursive::{cli, Validate};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   GCS Bucket RAG-Assisted Generation - Pulumi YAML");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Run synchronous examples (demonstrating API setup)
    example_1_validator_with_gcp_credentials();
    example_2_two_stage_preview_deploy();
    example_3_memory_based_rag();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   Examples Complete - API Demonstration");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("To run the actual workflow, you need:");
    println!("  1. A real LLM implementation (e.g., OpenAI, Anthropic)");
    println!("  2. GCP credentials configured");
    println!("  3. Pulumi CLI installed and initialized");
}

/// Example 1: Pulumi YAML Validator with GCP Credentials
fn example_1_validator_with_gcp_credentials() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 1: Pulumi YAML Validator with GCP Credentials         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Method 1: Application Default Credentials (ADC)
    // Inherits GOOGLE_APPLICATION_CREDENTIALS from environment
    let adc_validator = cli("pulumi")
        .args(&["preview", "--non-interactive", "--stack", "dev"])
        .env_from("GOOGLE_APPLICATION_CREDENTIALS") // JSON key path
        .env_from("CLOUDSDK_CORE_PROJECT")          // Project ID
        .env("PULUMI_CONFIG_PASSPHRASE", "")        // Non-interactive mode
        .ext("yaml")
        .required();

    println!("  ADC Validator:");
    println!("    Command: pulumi preview --non-interactive --stack dev");
    println!("    Env inherited: GOOGLE_APPLICATION_CREDENTIALS, CLOUDSDK_CORE_PROJECT");
    println!("    Env set: PULUMI_CONFIG_PASSPHRASE=\"\"");
    println!();

    // Method 2: Explicit project configuration
    let explicit_validator = cli("pulumi")
        .args(&["preview", "--non-interactive"])
        .env("GOOGLE_PROJECT", "my-gcp-project")
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .ext("yaml")
        .required();

    println!("  Explicit Project Validator:");
    println!("    Command: pulumi preview --non-interactive");
    println!("    Env set: GOOGLE_PROJECT=my-gcp-project");
    println!();

    // Method 3: Workload Identity Federation (WIF)
    let wif_validator = cli("pulumi")
        .args(&["preview", "--non-interactive"])
        .env_from("GOOGLE_APPLICATION_CREDENTIALS") // Points to WIF config
        .env_from("GOOGLE_CLOUD_PROJECT")
        .ext("yaml")
        .required();

    println!("  WIF Validator:");
    println!("    Env inherited: GOOGLE_APPLICATION_CREDENTIALS (WIF config)");
    println!();

    fn assert_validates<V: Validate>(_v: &V) {}
    assert_validates(&adc_validator);
    assert_validates(&explicit_validator);
    assert_validates(&wif_validator);
}

/// Example 2: Two-Stage Workflow (Preview + Deploy)
fn example_2_two_stage_preview_deploy() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 2: Two-Stage Workflow (Preview + Deploy)              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Stage 1: Preview (validate without applying)
    let preview_validator = cli("pulumi")
        .args(&["preview", "--non-interactive"])
        .env_from("GOOGLE_APPLICATION_CREDENTIALS")
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .weight(0.4)
        .required()
        // Stage 2: Deploy (apply changes) - only runs if preview passes
        .then("pulumi")
        .args(&["up", "--yes", "--non-interactive"])
        .env_from("GOOGLE_APPLICATION_CREDENTIALS")
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .weight(0.6)
        .required()
        .ext("yaml");

    println!("  Two-stage workflow:");
    println!("    Stage 1 (preview): pulumi preview --non-interactive (weight: 0.4)");
    println!("    Stage 2 (deploy):  pulumi up --yes --non-interactive (weight: 0.6)");
    println!();
    println!("  Stage execution:");
    println!("    1. Run preview to validate infrastructure");
    println!("    2. Only deploy if preview passes");
    println!();

    fn assert_validates<V: Validate>(_v: &V) {}
    assert_validates(&preview_validator);
}

/// Example 3: Memory-Based RAG for Infrastructure Examples
fn example_3_memory_based_rag() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 3: Memory-Based RAG for Infrastructure Examples       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // The new API uses memory() for RAG functionality
    // Here's how you would set up RAG with the new API:

    println!("  Using the new memory() API for RAG:");
    println!();
    println!("    use kkachi::recursive::{{memory, refine}};");
    println!();
    println!("    // Create memory store with seed examples");
    println!("    let mut mem = memory()");
    println!("        .seed_if_empty([");
    println!("            (\"gcs:bucket\", \"name: gcs-bucket\\nruntime: yaml\\nresources:...\"),");
    println!("            (\"gcs:iam\", \"name: gcs-iam\\nruntime: yaml\\n...\"),");
    println!("        ]);");
    println!();
    println!("    // Use memory for RAG-assisted refinement");
    println!("    let result = refine(&llm, \"Create a GCS bucket\")");
    println!("        .memory(&mut mem)");
    println!("        .k(3)  // Retrieve top-3 similar examples");
    println!("        .validate(preview_validator)");
    println!("        .learn_above(0.9)  // Store successful outputs");
    println!("        .max_iter(10)");
    println!("        .target(0.95)");
    println!("        .go_full()?;");
    println!();
    println!("  Benefits of RAG-assisted generation:");
    println!("    - Similar examples provide context");
    println!("    - Successful patterns are stored for future use");
    println!("    - Continuous learning from successful refinements");
    println!();
}

/// Mock LLM for demonstration
#[allow(dead_code)]
fn mock_pulumi_yaml() -> &'static str {
    r#"name: gcs-bucket-project
runtime: yaml

resources:
  my-bucket:
    type: gcp:storage:Bucket
    properties:
      name: my-unique-bucket-name-12345
      location: US
      uniformBucketLevelAccess: true
      forceDestroy: true
      labels:
        environment: dev
        managed-by: pulumi
"#
}

/// Full workflow documentation
#[allow(dead_code)]
fn full_workflow_docs() {
    // Complete workflow with the new API:
    //
    // ```rust
    // use kkachi::recursive::{cli, memory, refine, IterativeMockLlm};
    //
    // // 1. Create validator with GCP credentials
    // let validator = cli("pulumi")
    //     .args(&["preview", "--non-interactive"])
    //     .env_from("GOOGLE_APPLICATION_CREDENTIALS")
    //     .env("PULUMI_CONFIG_PASSPHRASE", "")
    //     .ext("yaml")
    //     .required();
    //
    // // 2. Create memory store
    // let mut mem = memory();
    //
    // // 3. Create LLM (replace with real implementation)
    // let llm = IterativeMockLlm::new(|iter, _prompt, _feedback| {
    //     mock_pulumi_yaml().to_string()
    // });
    //
    // // 4. Run refinement with RAG and validation
    // let result = refine(&llm, "Create a GCS bucket with security best practices")
    //     .memory(&mut mem)
    //     .k(3)
    //     .validate(validator)
    //     .learn_above(0.9)
    //     .max_iter(10)
    //     .target(0.95)
    //     .go_full()?;
    //
    // println!("Generated YAML:\n{}", result.output);
    // println!("Score: {:.2}", result.score);
    // println!("Iterations: {}", result.iterations);
    // ```
}
