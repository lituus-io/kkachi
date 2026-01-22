// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Module configuration commands.

use super::{Command, ExecutionContext, Output};
use crate::repl::SessionState;
use console::style;

/// Signature command - set the module signature.
pub struct SignatureCommand;

impl Command for SignatureCommand {
    fn name(&self) -> &str {
        "signature"
    }

    fn aliases(&self) -> &[&str] {
        &["sig"]
    }

    fn description(&self) -> &str {
        "Set the module signature"
    }

    fn help(&self) -> &str {
        "Usage: signature <input_fields> -> <output_fields>\n\n\
         Examples:\n\
           signature question -> answer\n\
           signature context, question -> answer, reasoning\n\n\
         Fields are comma-separated. Use -> to separate inputs from outputs."
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        if args.is_empty() {
            return if let Some(ref sig) = state.signature {
                Output::text(format!("Current signature: {}", sig.raw))
            } else {
                Output::warning("No signature set. Use: signature <inputs> -> <outputs>")
            };
        }

        // Save state for undo
        ctx.history.push(state.snapshot());

        if state.set_signature(args) {
            Output::success(format!("Signature set: {}", args))
        } else {
            Output::error("Invalid signature format. Use: inputs -> outputs")
        }
    }
}

/// Instruction command - set the instruction/system prompt.
pub struct InstructionCommand;

impl Command for InstructionCommand {
    fn name(&self) -> &str {
        "instruction"
    }

    fn aliases(&self) -> &[&str] {
        &["inst", "system"]
    }

    fn description(&self) -> &str {
        "Set the instruction/system prompt"
    }

    fn help(&self) -> &str {
        "Usage: instruction <text>\n\n\
         Set the system prompt or instruction for the module.\n\n\
         Examples:\n\
           instruction You are a helpful coding assistant.\n\
           instruction \"You are an expert in Rust programming.\""
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        if args.is_empty() {
            return if state.instruction.is_empty() {
                Output::warning("No instruction set.")
            } else {
                let preview: String = state.instruction.chars().take(100).collect();
                let suffix = if state.instruction.len() > 100 {
                    "..."
                } else {
                    ""
                };
                Output::text(format!("Current instruction: {}{}", preview, suffix))
            };
        }

        // Remove surrounding quotes if present
        let instruction = args
            .strip_prefix('"')
            .unwrap_or(args)
            .strip_suffix('"')
            .unwrap_or(args)
            .to_string();

        // Save state for undo
        ctx.history.push(state.snapshot());

        state.instruction = instruction.clone();
        Output::success(format!("Instruction set ({} chars)", instruction.len()))
    }
}

/// Demo command - manage demonstrations.
pub struct DemoCommand;

impl Command for DemoCommand {
    fn name(&self) -> &str {
        "demo"
    }

    fn aliases(&self) -> &[&str] {
        &["d", "demos"]
    }

    fn description(&self) -> &str {
        "Manage few-shot demonstrations"
    }

    fn help(&self) -> &str {
        "Usage: demo <subcommand>\n\n\
         Subcommands:\n\
           demo add <input> -> <output>   Add a demonstration\n\
           demo remove <index>            Remove demo by index (0-based)\n\
           demo list                      List all demonstrations\n\
           demo clear                     Remove all demonstrations\n\n\
         Examples:\n\
           demo add \"What is 2+2?\" -> \"4\"\n\
           demo remove 0\n\
           demo list"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        if args.is_empty() || args == "list" {
            return self.list_demos(state);
        }

        if args == "clear" {
            ctx.history.push(state.snapshot());
            state.clear_demos();
            return Output::success("All demonstrations cleared.");
        }

        if args.starts_with("add ") {
            let rest = &args[4..].trim();
            return self.add_demo(rest, state, ctx);
        }

        if args.starts_with("remove ") || args.starts_with("rm ") {
            let rest = args.split_whitespace().nth(1).unwrap_or("");
            return self.remove_demo(rest, state, ctx);
        }

        Output::error("Unknown subcommand. Use: demo add/remove/list/clear")
    }
}

impl DemoCommand {
    fn list_demos(&self, state: &SessionState) -> Output {
        if state.demos.is_empty() {
            return Output::text(
                "No demonstrations. Use 'demo add <input> -> <output>' to add one.",
            );
        }

        let mut output = String::new();
        output.push_str(&format!(
            "{} ({} total)\n\n",
            style("DEMONSTRATIONS").bold(),
            state.demos.len()
        ));

        for (i, demo) in state.demos.iter().enumerate() {
            let input_preview: String = demo.input.chars().take(40).collect();
            let output_preview: String = demo.output.chars().take(40).collect();
            let input_suffix = if demo.input.len() > 40 { "..." } else { "" };
            let output_suffix = if demo.output.len() > 40 { "..." } else { "" };

            output.push_str(&format!(
                "  {} Q: {}{}\n     A: {}{}\n\n",
                style(format!("[{}]", i)).dim(),
                input_preview,
                input_suffix,
                output_preview,
                output_suffix,
            ));
        }

        Output::text(output)
    }

    fn add_demo(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        // Parse: "input" -> "output" or input -> output
        let parts: Vec<&str> = args.split("->").collect();
        if parts.len() != 2 {
            return Output::error("Invalid format. Use: demo add <input> -> <output>");
        }

        let input = parts[0]
            .trim()
            .strip_prefix('"')
            .unwrap_or(parts[0].trim())
            .strip_suffix('"')
            .unwrap_or(parts[0].trim())
            .to_string();

        let output = parts[1]
            .trim()
            .strip_prefix('"')
            .unwrap_or(parts[1].trim())
            .strip_suffix('"')
            .unwrap_or(parts[1].trim())
            .to_string();

        if input.is_empty() || output.is_empty() {
            return Output::error("Both input and output are required.");
        }

        ctx.history.push(state.snapshot());
        state.add_demo(input, output);
        Output::success(format!("Demo added ({} total)", state.demos.len()))
    }

    fn remove_demo(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let index: usize = match args.parse() {
            Ok(i) => i,
            Err(_) => return Output::error("Invalid index. Use: demo remove <number>"),
        };

        if index >= state.demos.len() {
            return Output::error(format!(
                "Index {} out of range. Valid range: 0-{}",
                index,
                state.demos.len().saturating_sub(1)
            ));
        }

        ctx.history.push(state.snapshot());
        state.remove_demo(index);
        Output::success(format!(
            "Demo {} removed ({} remaining)",
            index,
            state.demos.len()
        ))
    }
}

/// Show command - display current module state.
pub struct ShowCommand;

impl Command for ShowCommand {
    fn name(&self) -> &str {
        "show"
    }

    fn aliases(&self) -> &[&str] {
        &["status", "s"]
    }

    fn description(&self) -> &str {
        "Show current module state"
    }

    fn execute(
        &self,
        _args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let mut output = String::new();

        output.push_str(&format!("{}\n", style("MODULE STATE").bold().underlined()));
        output.push('\n');

        // Signature
        output.push_str(&format!(
            "  {} {}\n",
            style("Signature:").bold(),
            state
                .signature
                .as_ref()
                .map(|s| s.raw.as_str())
                .unwrap_or("<not set>")
        ));

        // Instruction
        let instruction = if state.instruction.is_empty() {
            "<not set>".to_string()
        } else {
            let preview: String = state.instruction.chars().take(50).collect();
            let suffix = if state.instruction.len() > 50 {
                "..."
            } else {
                ""
            };
            format!("{}{}  ({} chars)", preview, suffix, state.instruction.len())
        };
        output.push_str(&format!(
            "  {} {}\n",
            style("Instruction:").bold(),
            instruction
        ));

        // Demos
        output.push_str(&format!(
            "  {} {} demonstrations\n",
            style("Demos:").bold(),
            state.demos.len()
        ));

        // Model
        output.push_str(&format!(
            "  {} {} ({})\n",
            style("Model:").bold(),
            state.lm_config.model,
            state.provider.display_name()
        ));

        // Temperature
        output.push_str(&format!(
            "  {} {}\n",
            style("Temperature:").bold(),
            state.lm_config.temperature
        ));

        // Domain
        output.push_str(&format!("  {} {}\n", style("Domain:").bold(), state.domain));

        // HITL
        output.push_str(&format!(
            "  {} {}\n",
            style("HITL:").bold(),
            if state.hitl.enabled {
                "enabled"
            } else {
                "disabled"
            }
        ));

        // Ready status
        output.push('\n');
        if state.is_ready() {
            output.push_str(&format!(
                "  {} Ready to run predictions.\n",
                style("✓").green()
            ));
        } else {
            output.push_str(&format!(
                "  {} {}.\n",
                style("⚠").yellow(),
                if state.signature.is_none() {
                    "Set a signature to get started"
                } else {
                    "Set an instruction to complete setup"
                }
            ));
        }

        Output::text(output)
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

    #[test]
    fn test_signature_command() {
        let cmd = SignatureCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("question -> answer", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert!(state.signature.is_some());
    }

    #[test]
    fn test_instruction_command() {
        let cmd = InstructionCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("You are helpful.", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert_eq!(state.instruction, "You are helpful.");
    }

    #[test]
    fn test_demo_command_add() {
        let cmd = DemoCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("add \"Q1\" -> \"A1\"", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert_eq!(state.demos.len(), 1);
    }

    #[test]
    fn test_demo_command_remove() {
        let cmd = DemoCommand;
        let mut state = SessionState::default();
        state.add_demo("Q1".to_string(), "A1".to_string());

        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("remove 0", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert_eq!(state.demos.len(), 0);
    }

    #[test]
    fn test_show_command() {
        let cmd = ShowCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }
}
