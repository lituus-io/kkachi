// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Execution commands (predict, refine).

use super::{Command, ExecutionContext, Output};
use crate::repl::SessionState;
use console::style;

/// Predict command - run a single prediction.
pub struct PredictCommand;

impl Command for PredictCommand {
    fn name(&self) -> &str {
        "predict"
    }

    fn aliases(&self) -> &[&str] {
        &["pred", "run"]
    }

    fn description(&self) -> &str {
        "Run a prediction"
    }

    fn help(&self) -> &str {
        "Usage: predict <input>\n\n\
         Run a single prediction with the current module configuration.\n\n\
         Examples:\n\
           predict What is the capital of France?\n\
           predict \"How do I parse JSON in Rust?\""
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let input = args.trim();

        if input.is_empty() {
            return Output::error("Please provide input for prediction.");
        }

        if !state.is_ready() {
            return Output::error(
                "Module not ready. Set signature and instruction first.\n\
                 Use 'show' to see current state.",
            );
        }

        // Store the input
        state.current_input = Some(input.to_string());

        // In a real implementation, this would call the LLM
        // For now, we return a placeholder
        let mut output = String::new();
        output.push_str(&format!("{}\n\n", style("PREDICTION").bold().underlined()));
        output.push_str(&format!("  {} {}\n\n", style("Input:").bold(), input));

        output.push_str(&format!(
            "  {} [{}]\n\n",
            style("Status:").yellow(),
            "LLM integration pending"
        ));

        output.push_str(&format!(
            "  To connect to an LLM, configure your API key:\n\
             \n\
             \n    {}",
            match state.provider {
                crate::repl::ProviderType::OpenAI => "export OPENAI_API_KEY=your-key",
                crate::repl::ProviderType::Anthropic => "export ANTHROPIC_API_KEY=your-key",
                crate::repl::ProviderType::Google => "export GOOGLE_API_KEY=your-key",
                crate::repl::ProviderType::Local => "# Start Ollama: ollama serve",
            }
        ));

        Output::text(output)
    }
}

/// Refine command - run recursive refinement.
pub struct RefineCommand;

impl Command for RefineCommand {
    fn name(&self) -> &str {
        "refine"
    }

    fn aliases(&self) -> &[&str] {
        &["ref", "iterate"]
    }

    fn description(&self) -> &str {
        "Run recursive refinement"
    }

    fn help(&self) -> &str {
        "Usage: refine <input>\n\n\
         Run recursive refinement with critic feedback.\n\n\
         Options:\n\
           --max <n>      Maximum iterations (default: 5)\n\
           --threshold <f>  Score threshold to stop (default: 0.9)\n\
           --domain <s>   Domain for critic selection\n\n\
         Examples:\n\
           refine \"Write a URL parser in Rust\"\n\
           refine --max 10 --threshold 0.95 \"Parse JSON\""
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        if args.is_empty() {
            return Output::error("Please provide input for refinement.");
        }

        if !state.is_ready() {
            return Output::error(
                "Module not ready. Set signature and instruction first.\n\
                 Use 'show' to see current state.",
            );
        }

        // Parse arguments (simple version)
        let (input, max_iter, threshold) = Self::parse_args(args);

        // Store the input
        state.current_input = Some(input.to_string());

        // In a real implementation, this would run the refinement loop
        let mut output = String::new();
        output.push_str(&format!("{}\n\n", style("REFINEMENT").bold().underlined()));
        output.push_str(&format!("  {} {}\n", style("Input:").bold(), input));
        output.push_str(&format!(
            "  {} {}\n",
            style("Max Iterations:").bold(),
            max_iter
        ));
        output.push_str(&format!(
            "  {} {}\n\n",
            style("Threshold:").bold(),
            threshold
        ));

        output.push_str(&format!(
            "  {} [{}]\n\n",
            style("Status:").yellow(),
            "Refinement loop pending LLM integration"
        ));

        // Simulated iteration display
        output.push_str(&format!("  {} (simulated)\n", style("Progress:").dim()));
        for i in 0..3 {
            let score = 0.5 + (i as f64 * 0.2);
            output.push_str(&format!(
                "    [Iter {}] Score: {:.2} {}\n",
                i,
                score,
                if score >= threshold {
                    style("✓").green().to_string()
                } else {
                    String::new()
                }
            ));
        }

        output.push_str(&format!(
            "\n  {} Use 'diff' to see changes, 'iterations' to see history.\n",
            style("Tip:").cyan()
        ));

        Output::text(output)
    }
}

impl RefineCommand {
    fn parse_args(args: &str) -> (&str, u32, f64) {
        let mut max_iter = 5u32;
        let mut threshold = 0.9f64;
        let mut input = args;

        let parts: Vec<&str> = args.split_whitespace().collect();
        let mut skip_next = false;

        for (i, part) in parts.iter().enumerate() {
            if skip_next {
                skip_next = false;
                continue;
            }

            if *part == "--max" {
                if let Some(n) = parts.get(i + 1) {
                    if let Ok(m) = n.parse() {
                        max_iter = m;
                    }
                }
                skip_next = true;
            } else if *part == "--threshold" {
                if let Some(t) = parts.get(i + 1) {
                    if let Ok(th) = t.parse() {
                        threshold = th;
                    }
                }
                skip_next = true;
            } else if !part.starts_with("--") {
                // First non-flag argument starts the input
                input = &args[args.find(part).unwrap_or(0)..];
                break;
            }
        }

        // Remove quotes if present
        let input = input
            .strip_prefix('"')
            .unwrap_or(input)
            .strip_suffix('"')
            .unwrap_or(input);

        (input, max_iter, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::StateHistory;
    use kkachi::DiffRenderer;

    fn make_ctx<'a>(
        history: &'a mut StateHistory,
        renderer: &'a DiffRenderer,
    ) -> ExecutionContext<'a> {
        ExecutionContext { renderer, history }
    }

    fn setup_ready_state() -> SessionState {
        let mut state = SessionState::default();
        state.set_signature("question -> answer");
        state.instruction = "You are helpful.".to_string();
        state
    }

    #[test]
    fn test_predict_command_not_ready() {
        let cmd = PredictCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("test", &mut state, &mut ctx);
        assert!(matches!(output, Output::Error(_)));
    }

    #[test]
    fn test_predict_command_ready() {
        let cmd = PredictCommand;
        let mut state = setup_ready_state();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("What is 2+2?", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
        assert_eq!(state.current_input, Some("What is 2+2?".to_string()));
    }

    #[test]
    fn test_refine_command() {
        let cmd = RefineCommand;
        let mut state = setup_ready_state();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("Write a URL parser", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_refine_parse_args() {
        let (input, max, threshold) =
            RefineCommand::parse_args("--max 10 --threshold 0.95 test input");
        assert_eq!(max, 10);
        assert_eq!(threshold, 0.95);
        assert!(input.contains("test"));
    }
}
