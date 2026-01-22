// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! REPL command system.
//!
//! Provides the Command trait and command registry for the REPL.

mod execute;
mod help;
mod history;
mod hitl_cmd;
mod io;
mod model;
mod module;
mod pipeline_cmd;
mod store_cmd;

pub use execute::{PredictCommand, RefineCommand};
pub use help::{HelpCommand, QuitCommand};
pub use history::{DiffCommand, HistoryCommand, IterationsCommand, RedoCommand, UndoCommand};
pub use hitl_cmd::{HitlCommand, ReviewCommand};
pub use io::{ExportCommand, ImportCommand, LoadCommand, SaveCommand};
pub use model::{ConfigCommand, ModelCommand, ProviderCommand, TemperatureCommand};
pub use module::{DemoCommand, InstructionCommand, ShowCommand, SignatureCommand};
pub use pipeline_cmd::{
    BreakpointCommand, InspectCommand, PipelineCommand, RunCommand, StagesCommand,
};
pub use store_cmd::{SearchCommand, StoreCommand};

use super::{SessionState, StateHistory};
use kkachi::DiffRenderer;
use std::collections::HashMap;

/// Output from a command.
#[derive(Debug)]
pub enum Output {
    /// Plain text output.
    Text(String),
    /// Success message.
    Success(String),
    /// Warning message.
    Warning(String),
    /// Error message.
    Error(String),
    /// No output (silent success).
    None,
    /// Quit signal.
    Quit,
}

impl Output {
    /// Create a success output.
    pub fn success(msg: impl Into<String>) -> Self {
        Self::Success(msg.into())
    }

    /// Create a warning output.
    pub fn warning(msg: impl Into<String>) -> Self {
        Self::Warning(msg.into())
    }

    /// Create an error output.
    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error(msg.into())
    }

    /// Create a text output.
    pub fn text(msg: impl Into<String>) -> Self {
        Self::Text(msg.into())
    }
}

/// Execution context for commands.
pub struct ExecutionContext<'a> {
    /// Diff renderer.
    pub renderer: &'a DiffRenderer,
    /// State history for undo/redo.
    pub history: &'a mut StateHistory,
}

/// Command trait for REPL commands.
pub trait Command: Send + Sync {
    /// Command name (e.g., "predict").
    fn name(&self) -> &str;

    /// Command aliases (e.g., ["p", "pred"]).
    fn aliases(&self) -> &[&str] {
        &[]
    }

    /// Short description for help.
    fn description(&self) -> &str;

    /// Detailed help text.
    fn help(&self) -> &str {
        self.description()
    }

    /// Argument completions for tab completion.
    fn completions(&self, _partial: &str, _state: &SessionState) -> Vec<String> {
        vec![]
    }

    /// Execute the command.
    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output;
}

/// Command registry for looking up and executing commands.
pub struct CommandRegistry {
    /// Registered commands.
    commands: HashMap<String, Box<dyn Command>>,
    /// Alias to command name mapping.
    aliases: HashMap<String, String>,
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            commands: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    /// Create a registry with all default commands.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Module commands
        registry.register(Box::new(SignatureCommand));
        registry.register(Box::new(InstructionCommand));
        registry.register(Box::new(DemoCommand));
        registry.register(Box::new(ShowCommand));

        // Model commands
        registry.register(Box::new(ModelCommand));
        registry.register(Box::new(ProviderCommand));
        registry.register(Box::new(TemperatureCommand));
        registry.register(Box::new(ConfigCommand));

        // Execute commands
        registry.register(Box::new(PredictCommand));
        registry.register(Box::new(RefineCommand));

        // History commands
        registry.register(Box::new(HistoryCommand));
        registry.register(Box::new(IterationsCommand));
        registry.register(Box::new(DiffCommand));
        registry.register(Box::new(UndoCommand));
        registry.register(Box::new(RedoCommand));

        // HITL commands
        registry.register(Box::new(HitlCommand));
        registry.register(Box::new(ReviewCommand));

        // I/O commands
        registry.register(Box::new(SaveCommand));
        registry.register(Box::new(LoadCommand));
        registry.register(Box::new(ExportCommand));
        registry.register(Box::new(ImportCommand));

        // Pipeline commands
        registry.register(Box::new(PipelineCommand));
        registry.register(Box::new(RunCommand));
        registry.register(Box::new(StagesCommand));
        registry.register(Box::new(InspectCommand));
        registry.register(Box::new(BreakpointCommand));

        // Vector store commands
        registry.register(Box::new(StoreCommand));
        registry.register(Box::new(SearchCommand));

        // Help commands
        registry.register(Box::new(HelpCommand));
        registry.register(Box::new(QuitCommand));

        registry
    }

    /// Register a command.
    pub fn register(&mut self, command: Box<dyn Command>) {
        let name = command.name().to_string();

        // Register aliases
        for alias in command.aliases() {
            self.aliases.insert(alias.to_string(), name.clone());
        }

        // Register command
        self.commands.insert(name, command);
    }

    /// Look up a command by name or alias.
    pub fn get(&self, name: &str) -> Option<&dyn Command> {
        // Try direct lookup
        if let Some(cmd) = self.commands.get(name) {
            return Some(cmd.as_ref());
        }

        // Try alias lookup
        if let Some(cmd_name) = self.aliases.get(name) {
            if let Some(cmd) = self.commands.get(cmd_name) {
                return Some(cmd.as_ref());
            }
        }

        None
    }

    /// Get all command names.
    pub fn command_names(&self) -> Vec<&str> {
        self.commands.keys().map(|s| s.as_str()).collect()
    }

    /// Get all commands.
    pub fn commands(&self) -> impl Iterator<Item = &dyn Command> {
        self.commands.values().map(|b| b.as_ref())
    }

    /// Get completions for a partial command.
    pub fn completions(&self, partial: &str) -> Vec<String> {
        let mut results = Vec::new();

        for name in self.commands.keys() {
            if name.starts_with(partial) {
                results.push(name.clone());
            }
        }

        for alias in self.aliases.keys() {
            if alias.starts_with(partial) {
                results.push(alias.clone());
            }
        }

        results.sort();
        results.dedup();
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_constructors() {
        assert!(matches!(Output::success("ok"), Output::Success(_)));
        assert!(matches!(Output::warning("warn"), Output::Warning(_)));
        assert!(matches!(Output::error("err"), Output::Error(_)));
        assert!(matches!(Output::text("text"), Output::Text(_)));
    }

    #[test]
    fn test_command_registry() {
        let registry = CommandRegistry::with_defaults();

        // Test command lookup
        assert!(registry.get("help").is_some());
        assert!(registry.get("quit").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_command_registry_aliases() {
        let registry = CommandRegistry::with_defaults();

        // HelpCommand has alias "?"
        let help = registry.get("help").unwrap();
        assert_eq!(help.name(), "help");
    }

    #[test]
    fn test_command_registry_completions() {
        let registry = CommandRegistry::with_defaults();

        let completions = registry.completions("he");
        assert!(completions.contains(&"help".to_string()));
    }
}
