// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Pulumi Infrastructure Pipeline - Generic CLI Validation
//!
//! This example demonstrates building Pulumi TypeScript infrastructure code
//! using kkachi's **generic CLI validation** primitives.
//!
//! Key concepts:
//! - **User-defined validators**: Build your own `Cli` and `CliPipeline`
//! - **No hardcoded tool methods**: The library provides only primitives
//! - **Composable**: Create reusable validators in your application code
//!
//! The library provides:
//! - `Cli`: Single CLI command validator
//! - `CliPipeline`: Multi-stage validation pipeline
//! - `Validator` trait: For custom validators
//!
//! You define your own:
//! - Which tools to run
//! - What arguments to pass
//! - How to weight stages
//! - When to stop on failure

use kkachi::{
    declarative::{Cli, CliPipeline, Step, Steps},
    // New declarative API
    pipeline,
};

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
    example_5_multi_step_infrastructure();

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
    // The library provides Cli - you specify the command
    let _pipeline = pipeline("requirement -> pulumi_code")
        .validate(
            Cli::new("pulumi")
                .args(["preview", "--non-interactive", "--json"])
                .file_ext("ts"),
        )
        .refine(10, 0.95);

    println!("  Created pipeline with single CLI validator:");
    println!("    - Command: pulumi preview --non-interactive --json");
    println!("    - File extension: .ts");
    println!("    - Refinement: 10 iterations, 0.95 threshold");
    println!();

    // Another example: TypeScript compilation
    let _ts_pipeline = pipeline("requirement -> typescript_code")
        .validate(
            Cli::new("npx")
                .args(["tsc", "--noEmit", "--strict"])
                .file_ext("ts"),
        )
        .refine(5, 0.9);

    println!("  TypeScript compilation validator:");
    println!("    - Command: npx tsc --noEmit --strict");
    println!("    - File extension: .ts");
    println!();
}

/// Example 2: Multi-Stage Validation Pipeline
fn example_2_multi_stage_pipeline() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 2: Multi-Stage Validation Pipeline                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Build a custom multi-stage pipeline
    // Each stage has a name, weight, and optional required flag
    let validator = CliPipeline::new()
        // Stage 1: TypeScript compilation
        .stage(
            "typescript",
            Cli::new("npx")
                .args(["tsc", "--noEmit", "--strict"])
                .weight(0.3)
                .required(),
        )
        // Stage 2: ESLint for code quality
        .stage(
            "eslint",
            Cli::new("npx")
                .args(["eslint", "--max-warnings", "0"])
                .weight(0.2),
        )
        // Stage 3: Pulumi preview
        .stage(
            "pulumi",
            Cli::new("pulumi")
                .args(["preview", "--non-interactive", "--json"])
                .weight(0.5)
                .required(),
        )
        .file_ext("ts");

    let _pipeline = pipeline("requirement -> pulumi_code")
        .validate(validator)
        .fix_per_stage() // Fix each stage before proceeding
        .strict(); // 10 iterations, 0.95 threshold

    println!("  Created 3-stage validation pipeline:");
    println!("    1. TypeScript (tsc --noEmit --strict) - weight: 0.3, required");
    println!("    2. ESLint (--max-warnings 0) - weight: 0.2");
    println!("    3. Pulumi preview - weight: 0.5, required");
    println!();
    println!("  With fix_per_stage(), each stage is fixed before proceeding:");
    println!("    -> Fix TypeScript errors first");
    println!("    -> Then fix ESLint warnings");
    println!("    -> Finally validate with Pulumi");
    println!();
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
        use kkachi::declarative::{Cli, CliPipeline};

        /// Rust code validator (user-defined, not in library)
        pub fn rust() -> CliPipeline {
            CliPipeline::new()
                .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
                .stage(
                    "compile",
                    Cli::new("rustc")
                        .args(["--emit=metadata", "-o", "/dev/null"])
                        .weight(0.6)
                        .required(),
                )
                .stage(
                    "lint",
                    Cli::new("cargo")
                        .args(["clippy", "--", "-D", "warnings"])
                        .weight(0.3),
                )
                .file_ext("rs")
        }

        /// Python code validator (user-defined)
        pub fn python() -> CliPipeline {
            CliPipeline::new()
                .stage(
                    "syntax",
                    Cli::new("python").args(["-m", "py_compile"]).required(),
                )
                .stage("lint", Cli::new("ruff").args(["check"]).weight(0.3))
                .stage("types", Cli::new("mypy").args(["--strict"]).weight(0.3))
                .file_ext("py")
        }

        /// Pulumi TypeScript validator (user-defined)
        pub fn pulumi() -> CliPipeline {
            CliPipeline::new()
                .stage(
                    "typescript",
                    Cli::new("npx")
                        .args(["tsc", "--noEmit"])
                        .weight(0.3)
                        .required(),
                )
                .stage(
                    "preview",
                    Cli::new("pulumi")
                        .args(["preview", "--non-interactive"])
                        .weight(0.7)
                        .required(),
                )
                .file_ext("ts")
        }

        /// Terraform validator (user-defined example)
        #[allow(dead_code)]
        pub fn terraform() -> CliPipeline {
            CliPipeline::new()
                .stage(
                    "format",
                    Cli::new("terraform").args(["fmt", "-check"]).weight(0.1),
                )
                .stage(
                    "validate",
                    Cli::new("terraform")
                        .args(["validate"])
                        .weight(0.4)
                        .required(),
                )
                .stage(
                    "plan",
                    Cli::new("terraform")
                        .args(["plan", "-detailed-exitcode"])
                        .weight(0.5)
                        .required(),
                )
                .file_ext("tf")
        }
    }

    // Use your reusable validators
    let _rust_pipeline = pipeline("requirement -> rust_code")
        .validate(my_validators::rust())
        .refine(5, 0.9);

    let _python_pipeline = pipeline("requirement -> python_code")
        .validate(my_validators::python())
        .refine(5, 0.9);

    let _pulumi_pipeline = pipeline("requirement -> pulumi_code")
        .validate(my_validators::pulumi())
        .refine(10, 0.95);

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

/// Example 4: Custom Error Parsing
fn example_4_custom_error_parsing() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 4: Custom Error Parsing                               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Custom error parser for better feedback
    let validator = CliPipeline::new()
        .stage(
            "compile",
            Cli::new("rustc")
                .args(["--emit=metadata"])
                .required()
                // Custom error parser extracts relevant lines
                .with_error_parser(|result| {
                    result
                        .stderr
                        .lines()
                        .filter(|line| line.starts_with("error") || line.contains("error[E"))
                        .map(|s| s.to_string())
                        .collect()
                }),
        )
        .file_ext("rs");

    let _pipeline = pipeline("requirement -> rust_code")
        .validate(validator)
        .refine(5, 0.9);

    println!("  Custom error parser example:");
    println!("    - Filters stderr to only show error lines");
    println!("    - Matches: 'error' or 'error[E...]'");
    println!("    - Better feedback for LLM refinement");
    println!();

    // Another example: JSON output parsing
    let _json_validator = Cli::new("pulumi")
        .args(["preview", "--json"])
        .file_ext("ts")
        .with_error_parser(|result| {
            // Parse JSON output for structured errors
            if result.success {
                return Vec::new();
            }
            // In real code, parse JSON for diagnostics
            result
                .stderr
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|s| s.to_string())
                .collect()
        });

    println!("  JSON error parser example:");
    println!("    - Parse structured JSON output for errors");
    println!("    - Extract diagnostics from tool-specific formats");
    println!();
}

/// Helper function to create Pulumi validator (used in example 5)
fn make_pulumi_validator() -> CliPipeline {
    CliPipeline::new()
        .stage(
            "typescript",
            Cli::new("npx").args(["tsc", "--noEmit"]).required(),
        )
        .stage("preview", Cli::new("pulumi").args(["preview"]).required())
        .file_ext("ts")
}

/// Example 5: Multi-Step Infrastructure Pipeline
fn example_5_multi_step_infrastructure() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 5: Multi-Step Infrastructure Pipeline                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Multi-step pipeline with per-step validation
    // Each step creates a fresh validator (or define a factory function)
    let _pipeline = Steps::new()
        // Step 1: Design (no validation, just quality threshold)
        .then(Step::new("design", "requirement -> architecture_design").quick()) // 3 iterations, 0.8 threshold
        // Step 2: Network layer with Pulumi validation
        .then(
            Step::new("network", "architecture -> network_code")
                .with_validator(make_pulumi_validator())
                .standard(),
        ) // 5 iterations, 0.9 threshold
        // Step 3: Compute layer with Pulumi validation
        .then(
            Step::new("compute", "network_code -> compute_code")
                .with_validator(make_pulumi_validator())
                .standard(),
        )
        // Step 4: Final validation
        .then(
            Step::new("validate", "compute_code -> validated_code")
                .with_validator(make_pulumi_validator())
                .strict(),
        ); // 10 iterations, 0.95 threshold

    println!("  Created 4-step infrastructure pipeline:");
    println!("    1. design     -> architecture (quick: 3 iter, 0.8)");
    println!("    2. network    -> network layer (standard: 5 iter, 0.9)");
    println!("    3. compute    -> compute layer (standard: 5 iter, 0.9)");
    println!("    4. validate   -> final validation (strict: 10 iter, 0.95)");
    println!();
    println!("  Each step uses user-defined Pulumi validator:");
    println!("    - TypeScript compilation check");
    println!("    - Pulumi preview for infrastructure validation");
    println!();
}

/// Bonus: Complete Production Pattern
#[allow(dead_code)]
fn production_pattern() {
    // In production, you would define validators in a shared module:
    //
    // ```rust
    // // In your crate: validators.rs
    // mod validators {
    //     use kkachi::declarative::{Cli, CliPipeline};
    //
    //     pub fn pulumi() -> CliPipeline {
    //         CliPipeline::new()
    //             .stage("tsc", Cli::new("npx").args(["tsc", "--noEmit"]).required())
    //             .stage("preview", Cli::new("pulumi").args(["preview"]).required())
    //             .file_ext("ts")
    //     }
    //
    //     pub fn k8s_manifests() -> CliPipeline {
    //         CliPipeline::new()
    //             .stage("validate", Cli::new("kubectl").args(["apply", "--dry-run=client", "-f"]))
    //             .file_ext("yaml")
    //     }
    // }
    //
    // // Then use everywhere:
    // let result = pipeline("requirement -> infra")
    //     .validate(validators::pulumi())
    //     .refine(10, 0.95)
    //     .run(requirement, &llm)
    //     .await?;
    // ```
    //
    // Benefits:
    // - Library is generic, not tied to any specific tools
    // - You control the exact CLI commands and arguments
    // - Easy to add new tools without waiting for library updates
    // - Share validators across your organization
    // - Version control your validator configurations
}
