// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Terminal-based interactive reviewer.

use super::{HITLConfig, HumanReviewer, ReviewContext, ReviewDecision};
use crate::diff::{DiffRenderer, DiffStyle};
use console::{style, Term};
use std::io::{self, Write};

/// Interactive terminal-based reviewer.
///
/// Displays diffs and prompts for user input in the terminal.
pub struct TerminalReviewer {
    /// Diff renderer.
    renderer: DiffRenderer,
    /// HITL configuration.
    config: HITLConfig,
    /// Terminal handle.
    term: Term,
}

impl Default for TerminalReviewer {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalReviewer {
    /// Create a new terminal reviewer with default settings.
    pub fn new() -> Self {
        Self {
            renderer: DiffRenderer::new(),
            config: HITLConfig::every_iteration(),
            term: Term::stdout(),
        }
    }

    /// Create with custom HITL config.
    pub fn with_config(config: HITLConfig) -> Self {
        let renderer = DiffRenderer::new().with_style(config.diff_style);
        Self {
            renderer,
            config,
            term: Term::stdout(),
        }
    }

    /// Set diff style.
    pub fn with_diff_style(mut self, style: DiffStyle) -> Self {
        self.renderer = self.renderer.with_style(style);
        self
    }

    /// Set context lines for diff.
    pub fn with_context_lines(mut self, lines: usize) -> Self {
        self.renderer = self.renderer.with_context(lines);
        self
    }

    /// Display the review header.
    fn display_header(&self, ctx: &ReviewContext<'_>) {
        let width = self.term.size().1 as usize;

        // Clear screen for better UX
        let _ = self.term.clear_screen();

        // Title bar
        println!(
            "{}",
            style(format!("{:═^width$}", " HUMAN REVIEW ", width = width)).cyan()
        );
        println!();

        // Iteration info
        println!(
            "  {} {}/{}",
            style("Iteration:").bold(),
            ctx.iteration,
            ctx.max_iterations
        );

        // Score
        let score_color = if ctx.score_improved() {
            console::Color::Green
        } else if ctx.score < ctx.prev_score {
            console::Color::Red
        } else {
            console::Color::Yellow
        };

        let score_change = ctx.score_change();
        let change_str = if score_change > 0.0 {
            format!("+{:.2}", score_change)
        } else {
            format!("{:.2}", score_change)
        };

        println!(
            "  {} {:.2} → {} ({})",
            style("Score:").bold(),
            ctx.prev_score,
            style(format!("{:.2}", ctx.score)).fg(score_color),
            style(change_str).fg(score_color)
        );

        // Trigger reason
        println!(
            "  {} {}",
            style("Trigger:").bold(),
            if ctx.trigger.is_warning() {
                style(ctx.trigger.description()).red()
            } else {
                style(ctx.trigger.description()).dim()
            }
        );

        // Domain and question
        if !ctx.domain.is_empty() {
            println!("  {} {}", style("Domain:").bold(), ctx.domain);
        }
        if !ctx.question.is_empty() {
            let preview: String = ctx.question.chars().take(60).collect();
            let suffix = if ctx.question.len() > 60 { "..." } else { "" };
            println!("  {} {}{}", style("Question:").bold(), preview, suffix);
        }

        println!();
        println!("{}", "─".repeat(width));
        println!();
    }

    /// Display the diff.
    fn display_diff(&self, ctx: &ReviewContext<'_>) {
        if !self.config.show_diff {
            return;
        }

        if ctx.diff.has_changes() {
            println!("{}", self.renderer.render_module(&ctx.diff));
        } else if let (Some(prev), output) = (ctx.prev_output, ctx.output) {
            // Show inline diff of just the output
            let diff = crate::diff::TextDiff::new(prev, output);
            if diff.has_changes() {
                println!("{}", style("OUTPUT CHANGES:").bold().underlined());
                println!("{}", self.renderer.render_text(&diff));
            } else {
                println!("{}", style("No changes from previous iteration.").dim());
            }
        }
    }

    /// Display the current output.
    fn display_output(&self, ctx: &ReviewContext<'_>) {
        let width = self.term.size().1 as usize;

        println!("{}", "─".repeat(width));
        println!("{}", style("CURRENT OUTPUT:").bold().underlined());
        println!();

        // Show output with line limit
        let lines: Vec<&str> = ctx.output.lines().collect();
        let max_lines = 30;

        if lines.len() > max_lines {
            for line in lines.iter().take(max_lines) {
                println!("  {}", line);
            }
            println!(
                "  {} {} more lines...",
                style("...").dim(),
                lines.len() - max_lines
            );
        } else {
            for line in &lines {
                println!("  {}", line);
            }
        }

        println!();
    }

    /// Display feedback if present.
    fn display_feedback(&self, ctx: &ReviewContext<'_>) {
        if let Some(feedback) = ctx.feedback {
            println!("{}", style("FEEDBACK:").bold().underlined());
            println!("  {}", style(feedback).yellow());
            println!();
        }
    }

    /// Display available options.
    fn display_options(&self, ctx: &ReviewContext<'_>) {
        let width = self.term.size().1 as usize;
        println!("{}", "─".repeat(width));
        println!("{}", style("OPTIONS:").bold());
        println!();
        println!("  {} - Accept and continue", style("[a]").green().bold());
        println!("  {} - Reject (try alternative)", style("[r]").red().bold());
        println!("  {} - Edit output", style("[e]").yellow().bold());
        println!(
            "  {} - Stop and return current best",
            style("[s]").cyan().bold()
        );
        println!("  {} - Accept this as final", style("[f]").magenta().bold());

        if ctx.iteration > 0 {
            println!("  {} - Rollback to iteration N", style("[b N]").dim());
        }

        println!("  {} - Skip next N reviews", style("[k N]").dim());
        println!("  {} - Show help", style("[h]").dim());
        println!();
    }

    /// Prompt for user decision.
    fn prompt_decision(&self, ctx: &ReviewContext<'_>) -> ReviewDecision {
        loop {
            print!("{} ", style("Your choice:").bold());
            io::stdout().flush().ok();

            let input = self.read_line().trim().to_lowercase();

            match input.as_str() {
                "a" | "accept" => return ReviewDecision::Accept,
                "r" | "reject" => return ReviewDecision::Reject,
                "e" | "edit" => return self.prompt_edit(ctx),
                "s" | "stop" => return ReviewDecision::Stop,
                "f" | "final" => return ReviewDecision::AcceptFinal,
                "h" | "help" => {
                    self.display_help();
                    continue;
                }
                _ => {
                    // Check for rollback (b N)
                    if input.starts_with("b ") || input.starts_with("back ") {
                        if let Some(n) = input.split_whitespace().nth(1) {
                            if let Ok(iter) = n.parse::<u32>() {
                                if iter < ctx.iteration {
                                    return ReviewDecision::Rollback { to_iteration: iter };
                                } else {
                                    println!(
                                        "{}",
                                        style("Cannot rollback to future iteration.").red()
                                    );
                                }
                            }
                        }
                        continue;
                    }

                    // Check for skip (k N)
                    if input.starts_with("k ") || input.starts_with("skip ") {
                        if let Some(n) = input.split_whitespace().nth(1) {
                            if let Ok(count) = n.parse::<u32>() {
                                return ReviewDecision::SkipNext { count };
                            }
                        }
                        continue;
                    }

                    println!("{} Type 'h' for help.", style("Invalid option.").red());
                }
            }
        }
    }

    /// Prompt for edit.
    fn prompt_edit(&self, ctx: &ReviewContext<'_>) -> ReviewDecision {
        println!();
        println!("{}", style("EDIT MODE").bold().underlined());
        println!("Enter new output (empty line to finish, 'cancel' to abort):");
        println!();

        let mut lines = Vec::new();
        loop {
            let line = self.read_line();
            if line.trim() == "cancel" {
                return self.prompt_decision(ctx);
            }
            if line.is_empty() && !lines.is_empty() {
                break;
            }
            lines.push(line);
        }

        if lines.is_empty() {
            println!("{}", style("No changes made.").dim());
            return self.prompt_decision(ctx);
        }

        let output = lines.join("\n");

        // Ask for optional guidance
        print!("Additional guidance (optional, press Enter to skip): ");
        io::stdout().flush().ok();
        let guidance = self.read_line();

        ReviewDecision::Edit {
            instruction: None,
            output: Some(output),
            guidance: if guidance.is_empty() {
                None
            } else {
                Some(guidance)
            },
        }
    }

    /// Display help text.
    fn display_help(&self) {
        println!();
        println!("{}", style("DETAILED HELP").bold().underlined());
        println!();
        println!(
            "  {} Accept the current output and continue to next iteration.",
            style("a/accept:").green()
        );
        println!("            The LLM will continue refining based on critic feedback.");
        println!();
        println!(
            "  {} Reject this output. The LLM will try a different approach.",
            style("r/reject:").red()
        );
        println!();
        println!(
            "  {} Manually edit the output. You can make corrections",
            style("e/edit:").yellow()
        );
        println!("            and provide guidance for the next iteration.");
        println!();
        println!(
            "  {} Stop iteration and return the best result so far.",
            style("s/stop:").cyan()
        );
        println!("            Useful if you're satisfied with a previous iteration.");
        println!();
        println!(
            "  {} Accept the current output as the final answer and stop.",
            style("f/final:").magenta()
        );
        println!();
        println!(
            "  {} Go back to iteration N and try a different path.",
            style("b N/back N:").dim()
        );
        println!();
        println!(
            "  {} Skip the next N reviews (auto-accept).",
            style("k N/skip N:").dim()
        );
        println!();
    }

    /// Read a line from stdin.
    fn read_line(&self) -> String {
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        input.trim_end().to_string()
    }
}

impl HumanReviewer for TerminalReviewer {
    fn review(&self, ctx: ReviewContext<'_>) -> ReviewDecision {
        self.display_header(&ctx);
        self.display_diff(&ctx);
        self.display_output(&ctx);
        self.display_feedback(&ctx);
        self.display_options(&ctx);
        self.prompt_decision(&ctx)
    }

    fn on_progress(&self, iteration: u32, score: f64) {
        println!(
            "  {} Iteration {} - Score: {:.2}",
            style("→").dim(),
            iteration,
            score
        );
    }

    fn on_iteration_start(&self, iteration: u32) {
        println!(
            "{}",
            style(format!("Starting iteration {}...", iteration)).dim()
        );
    }

    fn on_iteration_complete(&self, iteration: u32, score: f64) {
        let score_style = if score >= 0.9 {
            style(format!("{:.2}", score)).green()
        } else if score >= 0.7 {
            style(format!("{:.2}", score)).yellow()
        } else {
            style(format!("{:.2}", score)).red()
        };

        println!(
            "  {} Iteration {} complete - Score: {}",
            style("✓").green(),
            iteration,
            score_style
        );
    }

    fn on_complete(&self, final_score: f64, total_iterations: u32) {
        println!();
        println!(
            "{} Final score: {:.2} after {} iterations",
            style("✓").green().bold(),
            final_score,
            total_iterations
        );
    }

    fn on_error(&self, error: &str) {
        println!("{} {}", style("Error:").red().bold(), error);
    }
}

/// A non-interactive terminal reviewer that just displays progress.
///
/// Does not prompt for input - always accepts.
pub struct ProgressReviewer {
    /// Terminal handle for future progress display
    #[allow(dead_code)]
    term: Term,
}

impl Default for ProgressReviewer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReviewer {
    /// Create a new progress reviewer.
    pub fn new() -> Self {
        Self {
            term: Term::stdout(),
        }
    }
}

impl HumanReviewer for ProgressReviewer {
    fn review(&self, _ctx: ReviewContext<'_>) -> ReviewDecision {
        // Non-interactive: always accept
        ReviewDecision::Accept
    }

    fn on_progress(&self, iteration: u32, score: f64) {
        let bar_width = 20;
        let filled = (score * bar_width as f64) as usize;
        let empty = bar_width - filled;

        let bar = format!("[{}{}]", "█".repeat(filled), "░".repeat(empty));

        println!(
            "  Iter {} {} {:.1}%",
            style(format!("{:3}", iteration)).bold(),
            style(bar).cyan(),
            score * 100.0
        );
    }

    fn on_complete(&self, final_score: f64, total_iterations: u32) {
        println!();
        println!(
            "{} Complete: {:.1}% ({} iterations)",
            style("✓").green().bold(),
            final_score * 100.0,
            total_iterations
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hitl::ReviewTrigger;

    #[test]
    fn test_terminal_reviewer_creation() {
        let reviewer = TerminalReviewer::new();
        assert!(reviewer.config.enabled);
    }

    #[test]
    fn test_terminal_reviewer_with_config() {
        let config = HITLConfig::every(5);
        let reviewer = TerminalReviewer::with_config(config);
        assert_eq!(reviewer.config.interval, 5);
    }

    #[test]
    fn test_progress_reviewer() {
        let reviewer = ProgressReviewer::new();
        let ctx = ReviewContext::new(0, 10, 0.5, "output", ReviewTrigger::Interval);
        let decision = reviewer.review(ctx);
        assert!(matches!(decision, ReviewDecision::Accept));
    }
}
