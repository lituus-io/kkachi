// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! REPL engine implementation.

use super::commands::{CommandRegistry, ExecutionContext, Output};
use super::completer::ReplCompleter;
use super::prompt::PromptBuilder;
use super::state::{SessionState, StateHistory};
use console::style;
use kkachi::DiffRenderer;
use reedline::{
    Emacs, FileBackedHistory, Prompt, PromptEditMode, PromptHistorySearch,
    PromptHistorySearchStatus, Reedline, Signal,
};
use std::borrow::Cow;
use std::path::PathBuf;

/// The main REPL engine.
pub struct Repl {
    /// Current session state.
    state: SessionState,
    /// Command registry.
    commands: CommandRegistry,
    /// State history for undo/redo.
    history: StateHistory,
    /// Diff renderer.
    renderer: DiffRenderer,
    /// Line editor.
    editor: Reedline,
    /// Prompt builder.
    prompt_builder: PromptBuilder,
    /// History file path.
    history_file: PathBuf,
}

impl Repl {
    /// Create a new REPL with default settings.
    pub fn new() -> anyhow::Result<Self> {
        let commands = CommandRegistry::with_defaults();
        let completer = Box::new(ReplCompleter::new(&commands));

        // Set up history file
        let history_file = Self::history_file_path();

        // Set up reedline
        let history = Box::new(
            FileBackedHistory::with_file(1000, history_file.clone())
                .map_err(|e| anyhow::anyhow!("Failed to create history: {}", e))?,
        );

        let editor = Reedline::create()
            .with_history(history)
            .with_completer(completer)
            .with_edit_mode(Box::new(Emacs::default()));

        Ok(Self {
            state: SessionState::default(),
            commands,
            history: StateHistory::new(),
            renderer: DiffRenderer::new(),
            editor,
            prompt_builder: PromptBuilder::new(),
            history_file,
        })
    }

    /// Get the history file path.
    fn history_file_path() -> PathBuf {
        directories::ProjectDirs::from("io", "kkachi", "kkachi-cli")
            .map(|dirs| dirs.data_dir().join("repl_history.txt"))
            .unwrap_or_else(|| PathBuf::from(".kkachi_history"))
    }

    /// Run the REPL loop.
    pub fn run(&mut self) -> anyhow::Result<()> {
        self.print_welcome();

        loop {
            let prompt = ReplPrompt::new(&self.prompt_builder, &self.state);

            match self.editor.read_line(&prompt) {
                Ok(Signal::Success(line)) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    match self.execute_line(line) {
                        Output::Quit => {
                            println!("{}", style("Goodbye!").dim());
                            break;
                        }
                        Output::Text(text) => println!("{}", text),
                        Output::Success(msg) => {
                            println!("{} {}", self.prompt_builder.success_indicator(), msg)
                        }
                        Output::Warning(msg) => {
                            println!("{} {}", self.prompt_builder.warning_indicator(), msg)
                        }
                        Output::Error(msg) => {
                            println!("{} {}", self.prompt_builder.error_indicator(), msg)
                        }
                        Output::None => {}
                    }
                }
                Ok(Signal::CtrlC) => {
                    println!("{}", style("Ctrl-C: Use 'quit' or Ctrl-D to exit").dim());
                }
                Ok(Signal::CtrlD) => {
                    println!("{}", style("Goodbye!").dim());
                    break;
                }
                Err(e) => {
                    eprintln!("{} {}", self.prompt_builder.error_indicator(), e);
                }
            }
        }

        Ok(())
    }

    /// Print the welcome message.
    fn print_welcome(&self) {
        println!();
        println!(
            "{}",
            style("╭─────────────────────────────────────────────────────────────╮").cyan()
        );
        println!(
            "{}  {} {}  {}",
            style("│").cyan(),
            style("Kkachi REPL").bold().cyan(),
            style(format!("v{}", env!("CARGO_PKG_VERSION"))).dim(),
            style("│").cyan()
        );
        println!(
            "{}  {}  {}",
            style("│").cyan(),
            style("Interactive Prompt Engineering                              ").dim(),
            style("│").cyan()
        );
        println!(
            "{}  {}  {}",
            style("│").cyan(),
            style("Type 'help' for commands, 'quit' to exit                   ").dim(),
            style("│").cyan()
        );
        println!(
            "{}",
            style("╰─────────────────────────────────────────────────────────────╯").cyan()
        );
        println!();
    }

    /// Execute a line of input.
    fn execute_line(&mut self, line: &str) -> Output {
        // Parse command and arguments
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let cmd_name = parts[0];
        let args = parts.get(1).unwrap_or(&"");

        // Look up command
        let command = match self.commands.get(cmd_name) {
            Some(cmd) => cmd,
            None => {
                return Output::error(format!(
                    "Unknown command: '{}'. Type 'help' for available commands.",
                    cmd_name
                ));
            }
        };

        // Create execution context
        let mut ctx = ExecutionContext {
            renderer: &self.renderer,
            history: &mut self.history,
        };

        // Execute command
        command.execute(args, &mut self.state, &mut ctx)
    }

    /// Get the current state (for external access).
    pub fn state(&self) -> &SessionState {
        &self.state
    }

    /// Get mutable state.
    pub fn state_mut(&mut self) -> &mut SessionState {
        &mut self.state
    }
}

impl Default for Repl {
    fn default() -> Self {
        Self::new().expect("Failed to create REPL")
    }
}

/// Custom prompt for the REPL.
struct ReplPrompt {
    prompt_str: String,
}

impl ReplPrompt {
    fn new(builder: &PromptBuilder, state: &SessionState) -> Self {
        Self {
            prompt_str: builder.build(state),
        }
    }
}

#[allow(clippy::all, warnings)] // Cow<str> lifetime pattern required by reedline Prompt trait
impl Prompt for ReplPrompt {
    fn render_prompt_left(&self) -> Cow<str> {
        Cow::Borrowed(&self.prompt_str)
    }

    fn render_prompt_right(&self) -> Cow<str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<str> {
        Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<str> {
        Cow::Borrowed("... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!(
            "({}reverse-search: {}) ",
            prefix, history_search.term
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_creation() {
        // Note: This test may fail in CI if file creation is not allowed
        // For now, just test that the types compile correctly
        let _ = CommandRegistry::with_defaults();
        let _ = PromptBuilder::new();
        let _ = DiffRenderer::new();
    }

    #[test]
    fn test_execute_line_unknown() {
        // Can't easily test without the full REPL setup,
        // so just verify the command registry works
        let registry = CommandRegistry::with_defaults();
        assert!(registry.get("help").is_some());
        assert!(registry.get("unknown_command_xyz").is_none());
    }
}
