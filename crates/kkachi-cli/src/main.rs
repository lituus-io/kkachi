// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Kkachi CLI tool

#![allow(clippy::manual_strip)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::trim_split_whitespace)]

mod repl;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "kkachi")]
#[command(about = "High-performance LM optimization tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive REPL for prompt engineering
    Repl,

    /// Refine prompts interactively
    Refine {
        /// Path to examples file
        #[arg(short, long)]
        examples: String,

        /// Metric to use
        #[arg(short, long, default_value = "exact_match")]
        metric: String,
    },

    /// Compile an optimized module
    Compile {
        /// Input module path
        #[arg(short, long)]
        input: String,

        /// Output path
        #[arg(short, long)]
        output: String,
    },

    /// Evaluate a module
    Eval {
        /// Module path
        #[arg(short, long)]
        module: String,

        /// Dataset path
        #[arg(short, long)]
        dataset: String,

        /// Number of parallel threads
        #[arg(short, long, default_value_t = 8)]
        parallel: usize,
    },

    /// Serve as API
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Repl => {
            let mut repl = repl::Repl::new()?;
            repl.run()?;
        }
        Commands::Refine { examples, metric } => {
            println!("Refining prompts from: {}", examples);
            println!("Using metric: {}", metric);
            // Implementation would go here
        }
        Commands::Compile { input, output } => {
            println!("Compiling module from: {}", input);
            println!("Output to: {}", output);
            // Implementation would go here
        }
        Commands::Eval {
            module,
            dataset,
            parallel,
        } => {
            println!("Evaluating module: {}", module);
            println!("On dataset: {}", dataset);
            println!("Parallel threads: {}", parallel);
            // Implementation would go here
        }
        Commands::Serve { port } => {
            println!("Starting server on port: {}", port);
            // Implementation would go here
        }
    }

    Ok(())
}
