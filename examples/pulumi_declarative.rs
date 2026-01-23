// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Pulumi Infrastructure Pipeline - Generic CLI Validation
//!
//! This example demonstrates building Pulumi TypeScript infrastructure code
//! using kkachi's **generic CLI validation** primitives.
//!
//! Key concepts:
//! - **User-defined validators**: Build your own `cli()` pipelines
//! - **No hardcoded tool methods**: The library provides only primitives
//! - **Composable**: Create reusable validators in your application code
//!
//! The library provides:
//! - `cli()`: CLI command validator builder
//! - `Validate` trait: For custom validators
//!
//! You define your own:
//! - Which tools to run
//! - What arguments to pass
//! - How to weight stages
//! - When to stop on failure

use kkachi::recursive::{cli, Validate};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   Pulumi Infrastructure Pipeline - Generic CLI Validation");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Run examples
    example_1_single_command();
    example_2_multi_stage_pipeline();
    example_3_reusable_validators();
    example_4_custom_error_parsing();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   All Examples Complete!");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}

/// Example 1: Single CLI Command Validation
fn example_1_single_command() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 1: Single CLI Command Validation                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Simple single-command validation
    // The library provides cli() - you specify the command
    let pulumi_validator = cli("pulumi")
        .args(&["preview", "--non-interactive", "--json"])
        .ext("ts");

    println!("  Created Pulumi preview validator:");
    println!("    - Command: pulumi preview --non-interactive --json");
    println!("    - File extension: .ts");
    println!();

    // TypeScript compilation validator
    let ts_validator = cli("npx")
        .args(&["tsc", "--noEmit", "--strict"])
        .ext("ts");

    println!("  TypeScript compilation validator:");
    println!("    - Command: npx tsc --noEmit --strict");
    println!("    - File extension: .ts");
    println!();

    // Verify validators implement Validate trait
    fn assert_validates<V: Validate>(_v: &V) {}
    assert_validates(&pulumi_validator);
    assert_validates(&ts_validator);
}

/// Example 2: Multi-Stage Validation Pipeline
fn example_2_multi_stage_pipeline() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 2: Multi-Stage Validation Pipeline                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Build a custom multi-stage pipeline using .then() chaining
    // Each stage has a weight and optional required flag
    let validator = cli("npx")
        .args(&["tsc", "--noEmit", "--strict"])
        .weight(0.3)
        .required()
        // Stage 2: ESLint for code quality
        .then("npx")
        .args(&["eslint", "--max-warnings", "0"])
        .weight(0.2)
        // Stage 3: Pulumi preview
        .then("pulumi")
        .args(&["preview", "--non-interactive", "--json"])
        .weight(0.5)
        .required()
        .ext("ts");

    println!("  Created 3-stage validation pipeline:");
    println!("    1. TypeScript (tsc --noEmit --strict) - weight: 0.3, required");
    println!("    2. ESLint (--max-warnings 0) - weight: 0.2");
    println!("    3. Pulumi preview - weight: 0.5, required");
    println!();
    println!("  The validator will:");
    println!("    -> Run TypeScript compilation first");
    println!("    -> Then run ESLint for code quality");
    println!("    -> Finally validate with Pulumi preview");
    println!();

    fn assert_validates<V: Validate>(_v: &V) {}
    assert_validates(&validator);
}

/// Example 3: Reusable Validators (User Defined)
///
/// This is the recommended pattern: define validators in your application code
/// and reuse them across pipelines.
fn example_3_reusable_validators() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 3: Reusable Validators (User Defined)                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Define reusable validators in your application:
    mod my_validators {
        use kkachi::recursive::{cli, Cli};

        /// Rust code validator (user-defined, not in library)
        pub fn rust() -> Cli {
            cli("rustfmt")
                .args(&["--check"])
                .weight(0.1)
                .then("rustc")
                .args(&["--emit=metadata", "-o", "/dev/null"])
                .weight(0.6)
                .required()
                .then("cargo")
                .args(&["clippy", "--", "-D", "warnings"])
                .weight(0.3)
                .ext("rs")
        }

        /// Python code validator (user-defined)
        pub fn python() -> Cli {
            cli("python")
                .args(&["-m", "py_compile"])
                .required()
                .then("ruff")
                .args(&["check"])
                .weight(0.3)
                .then("mypy")
                .args(&["--strict"])
                .weight(0.3)
                .ext("py")
        }

        /// Pulumi TypeScript validator (user-defined)
        pub fn pulumi() -> Cli {
            cli("npx")
                .args(&["tsc", "--noEmit"])
                .weight(0.3)
                .required()
                .then("pulumi")
                .args(&["preview", "--non-interactive"])
                .weight(0.7)
                .required()
                .ext("ts")
        }

        /// Terraform validator (user-defined example)
        #[allow(dead_code)]
        pub fn terraform() -> Cli {
            cli("terraform")
                .args(&["fmt", "-check"])
                .weight(0.1)
                .then("terraform")
                .args(&["validate"])
                .weight(0.4)
                .required()
                .then("terraform")
                .args(&["plan", "-detailed-exitcode"])
                .weight(0.5)
                .required()
                .ext("tf")
        }
    }

    // Create instances of the validators
    let _rust_validator = my_validators::rust();
    let _python_validator = my_validators::python();
    let _pulumi_validator = my_validators::pulumi();

    println!("  User-defined reusable validators:");
    println!("    - my_validators::rust()     -> rustfmt + rustc + clippy");
    println!("    - my_validators::python()   -> py_compile + ruff + mypy");
    println!("    - my_validators::pulumi()   -> tsc + pulumi preview");
    println!("    - my_validators::terraform()-> fmt + validate + plan");
    println!();
    println!("  Benefits:");
    println!("    - Define once, use everywhere");
    println!("    - Customize for your project's tools and settings");
    println!("    - No library updates needed for new tools");
    println!("    - Share validators across team/organization");
    println!();
}

/// Example 4: Validators with Environment and Capture
fn example_4_custom_error_parsing() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 4: Validators with Environment and Capture            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Validator with environment variables
    let validator_with_env = cli("pulumi")
        .args(&["preview", "--non-interactive"])
        .env("PULUMI_CONFIG_PASSPHRASE", "")       // Set env var
        .env_from("GOOGLE_APPLICATION_CREDENTIALS") // Inherit from process
        .ext("yaml")
        .required();

    println!("  Validator with environment:");
    println!("    - Sets PULUMI_CONFIG_PASSPHRASE=\"\"");
    println!("    - Inherits GOOGLE_APPLICATION_CREDENTIALS from process");
    println!();

    // Validator with output capture for debugging
    let validator_with_capture = cli("rustc")
        .args(&["--emit=metadata"])
        .capture()   // Enable output capture
        .ext("rs");

    println!("  Validator with capture:");
    println!("    - Command: rustc --emit=metadata");
    println!("    - Output captured for post-validation analysis");
    println!();

    // Validator with working directory
    let validator_with_workdir = cli("npm")
        .args(&["run", "lint"])
        .workdir("./frontend")  // Run in specific directory
        .timeout(60)            // 60 second timeout
        .ext("ts");

    println!("  Validator with workdir and timeout:");
    println!("    - Command: npm run lint");
    println!("    - Working directory: ./frontend");
    println!("    - Timeout: 60 seconds");
    println!();

    fn assert_validates<V: Validate>(_v: &V) {}
    assert_validates(&validator_with_env);
    assert_validates(&validator_with_capture);
    assert_validates(&validator_with_workdir);
}

/// Bonus: Complete Production Pattern
#[allow(dead_code)]
fn production_pattern_docs() {
    // In production, you would define validators in a shared module:
    //
    // ```rust
    // // In your crate: validators.rs
    // mod validators {
    //     use kkachi::recursive::cli;
    //
    //     pub fn pulumi() -> impl Validate {
    //         cli("npx").args(&["tsc", "--noEmit"]).required()
    //             .then("pulumi").args(&["preview"]).required()
    //             .ext("ts")
    //     }
    //
    //     pub fn k8s_manifests() -> impl Validate {
    //         cli("kubectl")
    //             .args(&["apply", "--dry-run=client", "-f"])
    //             .ext("yaml")
    //     }
    // }
    //
    // // Then use everywhere:
    // let result = refine(&llm, "requirement -> infra")
    //     .validate(validators::pulumi())
    //     .max_iter(10)
    //     .target(0.95)
    //     .go_full()?;
    // ```
    //
    // Benefits:
    // - Library is generic, not tied to any specific tools
    // - You control the exact CLI commands and arguments
    // - Easy to add new tools without waiting for library updates
    // - Share validators across your organization
    // - Version control your validator configurations
}
