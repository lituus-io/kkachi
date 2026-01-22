// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Dynamic prompt builder for the REPL.

use super::SessionState;
use console::style;

/// Builder for the REPL prompt.
pub struct PromptBuilder {
    /// Whether to use colors.
    use_colors: bool,
    /// Whether to show state info.
    show_state: bool,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptBuilder {
    /// Create a new prompt builder.
    pub fn new() -> Self {
        Self {
            use_colors: true,
            show_state: true,
        }
    }

    /// Disable colors.
    pub fn without_colors(mut self) -> Self {
        self.use_colors = false;
        self
    }

    /// Disable state info.
    pub fn without_state(mut self) -> Self {
        self.show_state = false;
        self
    }

    /// Build the prompt string.
    pub fn build(&self, state: &SessionState) -> String {
        let mut prompt = String::new();

        if self.use_colors {
            prompt.push_str(&format!("{}", style("kkachi").cyan().bold()));
        } else {
            prompt.push_str("kkachi");
        }

        if self.show_state {
            // Domain
            if self.use_colors {
                prompt.push_str(&format!(" [{}]", style(&state.domain).yellow()));
            } else {
                prompt.push_str(&format!(" [{}]", state.domain));
            }

            // Demos count
            if !state.demos.is_empty() {
                if self.use_colors {
                    prompt.push_str(&format!(
                        " ({})",
                        style(format!("{} demos", state.demos.len())).dim()
                    ));
                } else {
                    prompt.push_str(&format!(" ({} demos)", state.demos.len()));
                }
            }

            // HITL indicator
            if state.hitl.enabled {
                if self.use_colors {
                    prompt.push_str(&format!(" {}", style("⚡").yellow()));
                } else {
                    prompt.push_str(" [HITL]");
                }
            }
        }

        prompt.push_str("> ");
        prompt
    }

    /// Build a continuation prompt (for multi-line input).
    pub fn build_continuation(&self) -> String {
        if self.use_colors {
            format!("{} ", style("...").dim())
        } else {
            "... ".to_string()
        }
    }

    /// Build an error indicator.
    pub fn error_indicator(&self) -> String {
        if self.use_colors {
            format!("{}", style("✗").red().bold())
        } else {
            "ERROR".to_string()
        }
    }

    /// Build a success indicator.
    pub fn success_indicator(&self) -> String {
        if self.use_colors {
            format!("{}", style("✓").green().bold())
        } else {
            "OK".to_string()
        }
    }

    /// Build a warning indicator.
    pub fn warning_indicator(&self) -> String {
        if self.use_colors {
            format!("{}", style("⚠").yellow().bold())
        } else {
            "WARN".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_builder_basic() {
        let builder = PromptBuilder::new();
        let state = SessionState::default();

        let prompt = builder.build(&state);
        assert!(prompt.contains("kkachi"));
        assert!(prompt.contains(">"));
    }

    #[test]
    fn test_prompt_builder_no_colors() {
        let builder = PromptBuilder::new().without_colors();
        let state = SessionState::default();

        let prompt = builder.build(&state);
        assert!(prompt.contains("kkachi"));
        assert!(!prompt.contains("\x1b")); // No ANSI codes
    }

    #[test]
    fn test_prompt_builder_with_demos() {
        let builder = PromptBuilder::new().without_colors();
        let mut state = SessionState::default();
        state.add_demo("Q".to_string(), "A".to_string());
        state.add_demo("Q2".to_string(), "A2".to_string());

        let prompt = builder.build(&state);
        assert!(prompt.contains("2 demos"));
    }

    #[test]
    fn test_prompt_builder_with_hitl() {
        let builder = PromptBuilder::new().without_colors();
        let mut state = SessionState::default();
        state.hitl = kkachi::HITLConfig::every_iteration();

        let prompt = builder.build(&state);
        assert!(prompt.contains("HITL"));
    }
}
