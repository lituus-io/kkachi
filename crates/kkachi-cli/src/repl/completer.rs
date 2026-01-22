// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Tab completion for the REPL.

use super::CommandRegistry;
use reedline::{Completer, Span, Suggestion};

/// REPL tab completer.
pub struct ReplCompleter {
    /// Reference to command registry (owned copy of names).
    command_names: Vec<String>,
}

impl ReplCompleter {
    /// Create a new completer from a command registry.
    pub fn new(registry: &CommandRegistry) -> Self {
        Self {
            command_names: registry
                .command_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    /// Get completions for the current input.
    fn complete_command(&self, partial: &str) -> Vec<String> {
        self.command_names
            .iter()
            .filter(|name| name.starts_with(partial))
            .cloned()
            .collect()
    }
}

impl Completer for ReplCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let line_to_cursor = &line[..pos];
        let words: Vec<&str> = line_to_cursor.split_whitespace().collect();

        let (word_start, partial) = if line_to_cursor.ends_with(' ') {
            // Cursor after space - completing new word
            (pos, "")
        } else if let Some(last_word) = words.last() {
            // Cursor in a word - find word start
            let start = line_to_cursor.rfind(last_word).unwrap_or(0);
            (start, *last_word)
        } else {
            // Empty line
            (0, "")
        };

        let span = Span::new(word_start, pos);

        if words.len() <= 1 && !line_to_cursor.ends_with(' ') {
            // Completing command name
            self.complete_command(partial)
                .into_iter()
                .map(|value| Suggestion {
                    value,
                    description: None,
                    extra: None,
                    span,
                    append_whitespace: true,
                })
                .collect()
        } else {
            // Could complete arguments based on command
            // For now, return empty
            vec![]
        }
    }
}

/// Simple completer that completes file paths.
pub struct PathCompleter;

impl PathCompleter {
    /// Complete a partial path.
    pub fn complete(partial: &str) -> Vec<String> {
        use std::fs;
        use std::path::Path;

        let path = Path::new(partial);

        // Determine directory to list
        let (dir, prefix) = if partial.ends_with('/') || partial.ends_with('\\') {
            (path.to_path_buf(), String::new())
        } else if let Some(parent) = path.parent() {
            let prefix = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            (
                if parent.as_os_str().is_empty() {
                    ".".into()
                } else {
                    parent.to_path_buf()
                },
                prefix,
            )
        } else {
            (".".into(), partial.to_string())
        };

        // List directory
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => return vec![],
        };

        let mut completions = Vec::new();

        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&prefix) && !name.starts_with('.') {
                    let mut completion = if dir.as_os_str() == "." {
                        name.to_string()
                    } else {
                        dir.join(name).to_string_lossy().to_string()
                    };

                    // Add trailing slash for directories
                    if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        completion.push('/');
                    }

                    completions.push(completion);
                }
            }
        }

        completions.sort();
        completions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_completer_command() {
        let registry = CommandRegistry::with_defaults();
        let mut completer = ReplCompleter::new(&registry);

        let suggestions = completer.complete("he", 2);
        let values: Vec<&str> = suggestions.iter().map(|s| s.value.as_str()).collect();
        assert!(values.contains(&"help"));
    }

    #[test]
    fn test_repl_completer_empty() {
        let registry = CommandRegistry::with_defaults();
        let mut completer = ReplCompleter::new(&registry);

        let suggestions = completer.complete("", 0);
        // Should suggest all commands
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_path_completer() {
        // This test depends on the filesystem, so just test the function runs
        let completions = PathCompleter::complete("./");
        // May or may not have completions depending on current directory
        let _ = completions;
    }
}
