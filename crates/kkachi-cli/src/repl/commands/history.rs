// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! History and diff commands.

use super::{Command, ExecutionContext, Output};
use crate::repl::SessionState;
use console::style;
use kkachi::TextDiff;

/// History command - show command history.
pub struct HistoryCommand;

impl Command for HistoryCommand {
    fn name(&self) -> &str {
        "history"
    }

    fn aliases(&self) -> &[&str] {
        &["hist"]
    }

    fn description(&self) -> &str {
        "Show state history"
    }

    fn execute(
        &self,
        _args: &str,
        _state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let history = &ctx.history;

        if history.is_empty() {
            return Output::text("No history yet. Make changes to build history.");
        }

        let mut output = String::new();
        output.push_str(&format!(
            "{} ({} states)\n\n",
            style("STATE HISTORY").bold().underlined(),
            history.len()
        ));

        output.push_str(&format!(
            "  Current position: {}/{}\n",
            history.position(),
            history.len()
        ));

        output.push_str(&format!(
            "  Can undo: {}  Can redo: {}\n",
            if history.can_undo() { "yes" } else { "no" },
            if history.can_redo() { "yes" } else { "no" }
        ));

        output.push_str("\n  Use 'undo' and 'redo' to navigate history.\n");

        Output::text(output)
    }
}

/// Iterations command - show iteration history from last operation.
pub struct IterationsCommand;

impl Command for IterationsCommand {
    fn name(&self) -> &str {
        "iterations"
    }

    fn aliases(&self) -> &[&str] {
        &["iters", "iter"]
    }

    fn description(&self) -> &str {
        "Show iteration history from last refinement"
    }

    fn execute(
        &self,
        _args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if state.iterations.is_empty() {
            return Output::text("No iteration history. Run 'refine' to create iteration history.");
        }

        let mut output = String::new();
        output.push_str(&format!(
            "{} ({} iterations)\n\n",
            style("ITERATION HISTORY").bold().underlined(),
            state.iterations.len()
        ));

        for iter in &state.iterations {
            let score_style = if iter.score >= 0.9 {
                style(format!("{:.2}", iter.score)).green()
            } else if iter.score >= 0.7 {
                style(format!("{:.2}", iter.score)).yellow()
            } else {
                style(format!("{:.2}", iter.score)).red()
            };

            output.push_str(&format!(
                "  {} {} Score: {}\n",
                style(format!("[{}]", iter.iteration)).bold(),
                if iter.score >= 0.9 {
                    style("✓").green().to_string()
                } else {
                    " ".to_string()
                },
                score_style
            ));

            // Output preview
            let output_preview: String = iter.output.chars().take(50).collect();
            let suffix = if iter.output.len() > 50 { "..." } else { "" };
            output.push_str(&format!("      Output: {}{}\n", output_preview, suffix));

            // Feedback
            if let Some(ref feedback) = iter.feedback {
                let fb_preview: String = feedback.chars().take(40).collect();
                let fb_suffix = if feedback.len() > 40 { "..." } else { "" };
                output.push_str(&format!(
                    "      Feedback: {}\n",
                    style(format!("{}{}", fb_preview, fb_suffix)).dim()
                ));
            }

            output.push('\n');
        }

        output.push_str("  Use 'diff <n>' to see changes between iterations.\n");

        Output::text(output)
    }
}

/// Diff command - show diff between iterations.
pub struct DiffCommand;

impl Command for DiffCommand {
    fn name(&self) -> &str {
        "diff"
    }

    fn aliases(&self) -> &[&str] {
        &["d"]
    }

    fn description(&self) -> &str {
        "Show diff between iterations"
    }

    fn help(&self) -> &str {
        "Usage: diff [iteration]\n\n\
         Show the diff between iterations.\n\n\
         Examples:\n\
           diff        Show diff for latest iteration\n\
           diff 2      Show diff between iteration 1 and 2\n\
           diff 1 3    Show diff between iteration 1 and 3"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if state.iterations.is_empty() {
            return Output::text("No iterations to diff. Run 'refine' first.");
        }

        let args = args.trim();
        let parts: Vec<&str> = args.split_whitespace().collect();

        let (from_idx, to_idx) = match parts.len() {
            0 => {
                // Show latest diff
                if state.iterations.len() < 2 {
                    return Output::text("Need at least 2 iterations to show diff.");
                }
                (state.iterations.len() - 2, state.iterations.len() - 1)
            }
            1 => {
                // Show diff to specified iteration
                let to: usize = match parts[0].parse() {
                    Ok(n) => n,
                    Err(_) => return Output::error("Invalid iteration number."),
                };
                if to == 0 || to >= state.iterations.len() {
                    return Output::error(format!(
                        "Iteration {} out of range (1-{}).",
                        to,
                        state.iterations.len() - 1
                    ));
                }
                (to - 1, to)
            }
            2 => {
                // Show diff between two iterations
                let from: usize = match parts[0].parse() {
                    Ok(n) => n,
                    Err(_) => return Output::error("Invalid 'from' iteration."),
                };
                let to: usize = match parts[1].parse() {
                    Ok(n) => n,
                    Err(_) => return Output::error("Invalid 'to' iteration."),
                };
                if from >= state.iterations.len() || to >= state.iterations.len() {
                    return Output::error(format!(
                        "Iteration out of range (0-{}).",
                        state.iterations.len() - 1
                    ));
                }
                (from, to)
            }
            _ => return Output::error("Too many arguments. Use: diff [from] [to]"),
        };

        let from_iter = &state.iterations[from_idx];
        let to_iter = &state.iterations[to_idx];

        let mut output = String::new();

        // Header
        output.push_str(&ctx.renderer.render_iteration_header(
            from_iter.iteration,
            to_iter.iteration,
            from_iter.score,
            to_iter.score,
        ));

        // Output diff
        let text_diff = TextDiff::new(&from_iter.output, &to_iter.output);
        if text_diff.has_changes() {
            output.push_str(&format!(
                "{}\n",
                style("OUTPUT CHANGES").bold().underlined()
            ));
            output.push_str(&ctx.renderer.render_text(&text_diff));
        } else {
            output.push_str(&format!("{}\n", style("No changes in output.").dim()));
        }

        // Feedback
        if let Some(ref feedback) = to_iter.feedback {
            output.push_str(&format!(
                "\n{}\n  {}\n",
                style("FEEDBACK").bold().underlined(),
                style(feedback).yellow()
            ));
        }

        Output::text(output)
    }
}

/// Undo command - undo last change.
pub struct UndoCommand;

impl Command for UndoCommand {
    fn name(&self) -> &str {
        "undo"
    }

    fn aliases(&self) -> &[&str] {
        &["u"]
    }

    fn description(&self) -> &str {
        "Undo last change"
    }

    fn execute(
        &self,
        _args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if let Some(snapshot) = ctx.history.undo() {
            state.restore(snapshot);
            Output::success("Undone.")
        } else {
            Output::warning("Nothing to undo.")
        }
    }
}

/// Redo command - redo undone change.
pub struct RedoCommand;

impl Command for RedoCommand {
    fn name(&self) -> &str {
        "redo"
    }

    fn aliases(&self) -> &[&str] {
        &["r"]
    }

    fn description(&self) -> &str {
        "Redo undone change"
    }

    fn execute(
        &self,
        _args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if let Some(snapshot) = ctx.history.redo() {
            state.restore(snapshot);
            Output::success("Redone.")
        } else {
            Output::warning("Nothing to redo.")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::{DemoData, IterationSnapshot, StateHistory};
    use kkachi::DiffRenderer;

    fn make_ctx<'a>(
        history: &'a mut StateHistory,
        renderer: &'a DiffRenderer,
    ) -> ExecutionContext<'a> {
        ExecutionContext { renderer, history }
    }

    #[test]
    fn test_history_command_empty() {
        let cmd = HistoryCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_iterations_command_empty() {
        let cmd = IterationsCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_iterations_command_with_data() {
        let cmd = IterationsCommand;
        let mut state = SessionState::default();
        state.iterations.push(IterationSnapshot {
            iteration: 0,
            instruction: String::new(),
            demos: vec![],
            output: "output 0".to_string(),
            score: 0.5,
            feedback: Some("needs work".to_string()),
        });
        state.iterations.push(IterationSnapshot {
            iteration: 1,
            instruction: String::new(),
            demos: vec![],
            output: "output 1".to_string(),
            score: 0.9,
            feedback: None,
        });

        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_diff_command_no_iterations() {
        let cmd = DiffCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_undo_redo() {
        let mut state = SessionState::default();
        state.instruction = "v1".to_string();

        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();

        // Push initial state
        history.push(state.snapshot());

        // Modify and push
        state.instruction = "v2".to_string();
        history.push(state.snapshot());

        let mut ctx = make_ctx(&mut history, &renderer);

        // Undo
        let undo_cmd = UndoCommand;
        let output = undo_cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert_eq!(state.instruction, "v1");

        // Redo
        let redo_cmd = RedoCommand;
        let output = redo_cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert_eq!(state.instruction, "v2");
    }
}
