// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Help and quit commands.

use super::{Command, ExecutionContext, Output};
use crate::repl::SessionState;
use console::style;

/// Help command - shows available commands.
pub struct HelpCommand;

impl Command for HelpCommand {
    fn name(&self) -> &str {
        "help"
    }

    fn aliases(&self) -> &[&str] {
        &["?", "h"]
    }

    fn description(&self) -> &str {
        "Show help for commands"
    }

    fn help(&self) -> &str {
        "Usage: help [command]\n\n\
         Without arguments, shows all available commands.\n\
         With a command name, shows detailed help for that command."
    }

    fn execute(
        &self,
        args: &str,
        _state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        if args.is_empty() {
            Output::text(format!(
                "{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}",
                style("KKACHI REPL COMMANDS").bold().underlined(),
                format!(
                    "{}\n  {} - Set signature (e.g., \"question -> answer\")\n  {} - Set instruction/system prompt\n  {} - Manage demonstrations (add/remove/list)\n  {} - Show current module state",
                    style("MODULE").cyan().bold(),
                    style("signature").green(),
                    style("instruction").green(),
                    style("demo").green(),
                    style("show").green(),
                ),
                format!(
                    "{}\n  {} - Set model (e.g., \"gpt-4o\", \"anthropic-sonnet\")\n  {} - Set provider (openai, anthropic, google, local)\n  {} - Set temperature (0.0 - 2.0)\n  {} - Show current configuration",
                    style("MODEL").cyan().bold(),
                    style("model").green(),
                    style("provider").green(),
                    style("temperature").green(),
                    style("config").green(),
                ),
                format!(
                    "{}\n  {} - Run a prediction\n  {} - Run recursive refinement",
                    style("EXECUTE").cyan().bold(),
                    style("predict").green(),
                    style("refine").green(),
                ),
                format!(
                    "{}\n  {} - Show command history\n  {} - Show iteration history\n  {} - Show diff between iterations\n  {} - Undo last change\n  {} - Redo undone change",
                    style("HISTORY").cyan().bold(),
                    style("history").green(),
                    style("iterations").green(),
                    style("diff").green(),
                    style("undo").green(),
                    style("redo").green(),
                ),
                format!(
                    "{}\n  {} - Configure human-in-the-loop\n  {} - Trigger manual review",
                    style("HITL").cyan().bold(),
                    style("hitl").green(),
                    style("review").green(),
                ),
                format!(
                    "{}\n  {} - Save session to file\n  {} - Load session from file\n  {} - Export module\n  {} - Import module",
                    style("I/O").cyan().bold(),
                    style("save").green(),
                    style("load").green(),
                    style("export").green(),
                    style("import").green(),
                ),
                format!(
                    "{}\n  {} - Show this help\n  {} - Exit the REPL",
                    style("HELP").cyan().bold(),
                    style("help").green(),
                    style("quit").green(),
                ),
            ))
        } else {
            Output::text(format!(
                "Help for '{}': Use 'help' without arguments to see all commands.",
                args
            ))
        }
    }
}

/// Quit command - exits the REPL.
pub struct QuitCommand;

impl Command for QuitCommand {
    fn name(&self) -> &str {
        "quit"
    }

    fn aliases(&self) -> &[&str] {
        &["exit", "q"]
    }

    fn description(&self) -> &str {
        "Exit the REPL"
    }

    fn execute(
        &self,
        _args: &str,
        _state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        Output::Quit
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
    fn test_help_command() {
        let cmd = HelpCommand;
        assert_eq!(cmd.name(), "help");
        assert!(cmd.aliases().contains(&"?"));

        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_quit_command() {
        let cmd = QuitCommand;
        assert_eq!(cmd.name(), "quit");
        assert!(cmd.aliases().contains(&"exit"));

        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Quit));
    }
}
