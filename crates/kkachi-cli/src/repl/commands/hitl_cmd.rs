// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! HITL (Human-in-the-Loop) commands.

use super::{Command, ExecutionContext, Output};
use crate::repl::SessionState;
use console::style;
use kkachi::HITLConfig;

/// HITL command - configure human-in-the-loop.
pub struct HitlCommand;

impl Command for HitlCommand {
    fn name(&self) -> &str {
        "hitl"
    }

    fn aliases(&self) -> &[&str] {
        &["human"]
    }

    fn description(&self) -> &str {
        "Configure human-in-the-loop"
    }

    fn help(&self) -> &str {
        "Usage: hitl <subcommand>\n\n\
         Subcommands:\n\
           hitl on              Enable HITL (review every iteration)\n\
           hitl off             Disable HITL\n\
           hitl interval <n>    Review every N iterations\n\
           hitl convergence     Review only at convergence\n\
           hitl regression      Review on score drops\n\
           hitl status          Show current HITL configuration\n\n\
         Examples:\n\
           hitl on\n\
           hitl interval 3"
    }

    fn completions(&self, partial: &str, _state: &SessionState) -> Vec<String> {
        vec![
            "on",
            "off",
            "interval",
            "convergence",
            "regression",
            "status",
        ]
        .into_iter()
        .filter(|c| c.starts_with(partial))
        .map(|s| s.to_string())
        .collect()
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();
        let parts: Vec<&str> = args.split_whitespace().collect();

        match parts.first().copied() {
            None | Some("status") => self.show_status(state),
            Some("on") => {
                state.hitl = HITLConfig::every_iteration();
                Output::success("HITL enabled (review every iteration)")
            }
            Some("off") => {
                state.hitl = HITLConfig::disabled();
                Output::success("HITL disabled")
            }
            Some("interval") => {
                if let Some(n) = parts.get(1) {
                    if let Ok(interval) = n.parse::<u32>() {
                        state.hitl = HITLConfig::every(interval);
                        Output::success(format!("HITL set to review every {} iterations", interval))
                    } else {
                        Output::error("Invalid interval number")
                    }
                } else {
                    Output::error("Usage: hitl interval <number>")
                }
            }
            Some("convergence") => {
                state.hitl = HITLConfig::on_completion();
                Output::success("HITL set to review at convergence only")
            }
            Some("regression") => {
                state.hitl = HITLConfig::on_regression();
                Output::success("HITL set to review on score drops")
            }
            Some(other) => Output::error(format!("Unknown subcommand: {}", other)),
        }
    }
}

impl HitlCommand {
    fn show_status(&self, state: &SessionState) -> Output {
        let config = &state.hitl;
        let mut output = String::new();

        output.push_str(&format!(
            "{}\n\n",
            style("HITL CONFIGURATION").bold().underlined()
        ));

        output.push_str(&format!(
            "  {} {}\n",
            style("Enabled:").bold(),
            if config.enabled {
                style("yes").green()
            } else {
                style("no").red()
            }
        ));

        if config.enabled {
            output.push_str(&format!(
                "  {} {}\n",
                style("Interval:").bold(),
                if config.interval > 0 {
                    format!("every {} iterations", config.interval)
                } else {
                    "disabled".to_string()
                }
            ));

            output.push_str(&format!(
                "  {} {}\n",
                style("On Score Drop:").bold(),
                if config.on_score_drop { "yes" } else { "no" }
            ));

            output.push_str(&format!(
                "  {} {}\n",
                style("On Convergence:").bold(),
                if config.on_convergence { "yes" } else { "no" }
            ));

            output.push_str(&format!(
                "  {} {}\n",
                style("On First:").bold(),
                if config.on_first { "yes" } else { "no" }
            ));

            output.push_str(&format!(
                "  {} {}\n",
                style("Show Diff:").bold(),
                if config.show_diff { "yes" } else { "no" }
            ));

            if let Some(ref timeout) = config.timeout {
                output.push_str(&format!("  {} {:?}\n", style("Timeout:").bold(), timeout));
            }
        }

        output.push_str("\n  Use 'hitl on/off/interval/convergence/regression' to configure.\n");

        Output::text(output)
    }
}

/// Review command - trigger manual review.
pub struct ReviewCommand;

impl Command for ReviewCommand {
    fn name(&self) -> &str {
        "review"
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }

    fn description(&self) -> &str {
        "Trigger manual review of current state"
    }

    fn help(&self) -> &str {
        "Usage: review\n\n\
         Trigger a manual HITL review of the current state.\n\
         This shows the current output and allows you to accept, reject, or edit."
    }

    fn execute(
        &self,
        _args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if state.last_output.is_none() {
            return Output::warning("No output to review. Run 'predict' or 'refine' first.");
        }

        let mut output = String::new();

        output.push_str(&format!(
            "{}\n\n",
            style("MANUAL REVIEW").bold().underlined()
        ));

        // Show current state
        if let Some(ref input) = state.current_input {
            output.push_str(&format!("  {} {}\n", style("Input:").bold(), input));
        }

        if let Some(ref last_output) = state.last_output {
            output.push_str(&format!("\n  {}\n", style("Output:").bold()));
            for line in last_output.lines().take(20) {
                output.push_str(&format!("    {}\n", line));
            }
            if last_output.lines().count() > 20 {
                output.push_str(&format!(
                    "    {} more lines...\n",
                    style(format!("... {} ", last_output.lines().count() - 20)).dim()
                ));
            }
        }

        if let Some(score) = state.last_score {
            output.push_str(&format!("\n  {} {:.2}\n", style("Score:").bold(), score));
        }

        output.push_str(&format!(
            "\n  {}\n",
            style("In interactive mode, you would be able to:").dim()
        ));
        output.push_str("    [a] Accept  [r] Reject  [e] Edit  [s] Stop\n");

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
    fn test_hitl_command_on() {
        let cmd = HitlCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("on", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert!(state.hitl.enabled);
    }

    #[test]
    fn test_hitl_command_off() {
        let cmd = HitlCommand;
        let mut state = SessionState::default();
        state.hitl = HITLConfig::every_iteration();

        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("off", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert!(!state.hitl.enabled);
    }

    #[test]
    fn test_hitl_command_interval() {
        let cmd = HitlCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("interval 5", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert!(state.hitl.enabled);
        assert_eq!(state.hitl.interval, 5);
    }

    #[test]
    fn test_hitl_command_status() {
        let cmd = HitlCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("status", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_review_command_no_output() {
        let cmd = ReviewCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Warning(_)));
    }

    #[test]
    fn test_review_command_with_output() {
        let cmd = ReviewCommand;
        let mut state = SessionState::default();
        state.last_output = Some("test output".to_string());
        state.last_score = Some(0.85);

        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }
}
