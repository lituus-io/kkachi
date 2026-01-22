// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Vector store commands for the REPL.

use super::{Command, ExecutionContext, Output};
use crate::repl::SessionState;
use console::style;

/// Store command - manage the vector store.
pub struct StoreCommand;

impl Command for StoreCommand {
    fn name(&self) -> &str {
        "store"
    }

    fn aliases(&self) -> &[&str] {
        &["vs", "vector"]
    }

    fn description(&self) -> &str {
        "Manage the in-memory vector store"
    }

    fn help(&self) -> &str {
        "Usage: store <subcommand> [args]\n\n\
         Subcommands:\n\
           add <id> <content>    Add a document to the store\n\
           remove <id>           Remove a document by ID\n\
           get <id>              Get a document by ID\n\
           search <query> [k]    Search for similar documents (default k=5)\n\
           list                  List document count\n\
           clear                 Clear all documents\n\
           info                  Show store information\n\n\
         Examples:\n\
           store add doc1 \"This is a test document\"\n\
           store search \"test query\" 10\n\
           store remove doc1"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let parts: Vec<&str> = args.trim().splitn(3, char::is_whitespace).collect();

        if parts.is_empty() || parts[0].is_empty() {
            return self.show_info(state);
        }

        match parts[0] {
            "add" => {
                if parts.len() < 3 {
                    return Output::error("Usage: store add <id> <content>");
                }
                self.add_document(parts[1], parts[2], state)
            }
            "remove" | "rm" | "delete" => {
                if parts.len() < 2 {
                    return Output::error("Usage: store remove <id>");
                }
                self.remove_document(parts[1], state)
            }
            "get" => {
                if parts.len() < 2 {
                    return Output::error("Usage: store get <id>");
                }
                self.get_document(parts[1], state)
            }
            "search" | "query" | "find" => {
                if parts.len() < 2 {
                    return Output::error("Usage: store search <query> [k]");
                }
                // Parse k from the end of the query if it's a number
                let (query, k) = self.parse_search_args(&parts[1..]);
                self.search_documents(&query, k, state)
            }
            "list" | "ls" => self.list_documents(state),
            "clear" => self.clear_store(state),
            "info" | "status" => self.show_info(state),
            _ => Output::error(format!(
                "Unknown subcommand: {}. Use 'help store' for usage.",
                parts[0]
            )),
        }
    }
}

impl StoreCommand {
    fn add_document(&self, id: &str, content: &str, state: &mut SessionState) -> Output {
        // Remove quotes if present
        let content = content.trim_matches('"').trim_matches('\'');

        state.store_add(id, content);
        Output::success(format!("Added document '{}' ({} chars)", id, content.len()))
    }

    fn remove_document(&self, id: &str, state: &mut SessionState) -> Output {
        if state.store_remove(id) {
            Output::success(format!("Removed document '{}'", id))
        } else {
            Output::warning(format!("Document '{}' not found", id))
        }
    }

    fn get_document(&self, id: &str, state: &SessionState) -> Output {
        match state.store_get(id) {
            Some(content) => {
                let mut output = String::new();
                output.push_str(&format!(
                    "{}\n\n",
                    style(format!("DOCUMENT: {}", id)).bold().underlined()
                ));
                output.push_str(&content);
                Output::text(output)
            }
            None => Output::warning(format!("Document '{}' not found", id)),
        }
    }

    fn parse_search_args(&self, args: &[&str]) -> (String, usize) {
        if args.is_empty() {
            return (String::new(), 5);
        }

        // If there's only one arg, it's the query
        if args.len() == 1 {
            return (args[0].to_string(), 5);
        }

        // Check if last part is a number (k value)
        let last = args.last().unwrap();
        if let Ok(k) = last.parse::<usize>() {
            let query = args[..args.len() - 1].join(" ");
            (query, k)
        } else {
            (args.join(" "), 5)
        }
    }

    fn search_documents(&self, query: &str, k: usize, state: &SessionState) -> Output {
        if state.store_is_empty() {
            return Output::warning("Store is empty. Add documents first with 'store add'.");
        }

        let query = query.trim_matches('"').trim_matches('\'');
        let results = state.store_search(query, k);

        if results.is_empty() {
            return Output::text("No results found.");
        }

        let mut output = String::new();
        output.push_str(&format!(
            "{} (query: \"{}\", k={})\n\n",
            style("SEARCH RESULTS").bold().underlined(),
            query,
            k
        ));

        for (i, result) in results.iter().enumerate() {
            let score_color = if result.score > 0.8 {
                style(format!("{:.3}", result.score)).green()
            } else if result.score > 0.5 {
                style(format!("{:.3}", result.score)).yellow()
            } else {
                style(format!("{:.3}", result.score)).red()
            };

            output.push_str(&format!(
                "  [{}] {} (score: {})\n",
                i + 1,
                style(&result.id).cyan(),
                score_color
            ));

            // Show content preview
            let preview: String = result.content.chars().take(80).collect();
            let suffix = if result.content.len() > 80 { "..." } else { "" };
            output.push_str(&format!("      {}{}\n", style(&preview).dim(), suffix));
        }

        Output::text(output)
    }

    fn list_documents(&self, state: &SessionState) -> Output {
        let count = state.store_len();
        if count == 0 {
            Output::text("Store is empty.")
        } else {
            Output::text(format!("Store contains {} document(s).", count))
        }
    }

    fn clear_store(&self, state: &mut SessionState) -> Output {
        let count = state.store_len();
        state.store_clear();
        Output::success(format!("Cleared {} document(s) from store.", count))
    }

    fn show_info(&self, state: &SessionState) -> Output {
        let mut output = String::new();

        output.push_str(&format!(
            "{}\n\n",
            style("VECTOR STORE INFO").bold().underlined()
        ));

        let count = state.store_len();
        output.push_str(&format!("  Documents: {}\n", count));
        output.push_str("  Embedding dimension: 64\n");
        output.push_str("  Embedder: HashEmbedder\n");

        if count == 0 {
            output.push_str(&format!(
                "\n  {} Use 'store add <id> <content>' to add documents.\n",
                style("Tip:").yellow()
            ));
        }

        Output::text(output)
    }
}

/// Search command - shorthand for store search.
pub struct SearchCommand;

impl Command for SearchCommand {
    fn name(&self) -> &str {
        "search"
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }

    fn description(&self) -> &str {
        "Search the vector store for similar documents"
    }

    fn help(&self) -> &str {
        "Usage: search <query> [k]\n\n\
         Search for documents similar to the query.\n\
         Default k=5 (number of results).\n\n\
         Examples:\n\
           search \"machine learning concepts\"\n\
           search \"API endpoints\" 10"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if args.trim().is_empty() {
            return Output::error("Usage: search <query> [k]");
        }
        StoreCommand.execute(&format!("search {}", args), state, ctx)
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
    fn test_store_info_empty() {
        let cmd = StoreCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("info", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_store_add_and_search() {
        let cmd = StoreCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        // Add documents
        let output = cmd.execute(
            "add doc1 \"This is about machine learning\"",
            &mut state,
            &mut ctx,
        );
        assert!(matches!(output, Output::Success(_)));

        let output = cmd.execute(
            "add doc2 \"Deep learning neural networks\"",
            &mut state,
            &mut ctx,
        );
        assert!(matches!(output, Output::Success(_)));

        // Search
        let output = cmd.execute("search machine 2", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));

        // List
        let output = cmd.execute("list", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_store_remove() {
        let cmd = StoreCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        // Add then remove
        cmd.execute("add test_doc \"Test content\"", &mut state, &mut ctx);
        let output = cmd.execute("remove test_doc", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));

        // Remove non-existent
        let output = cmd.execute("remove nonexistent", &mut state, &mut ctx);
        assert!(matches!(output, Output::Warning(_)));
    }

    #[test]
    fn test_store_clear() {
        let cmd = StoreCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        // Add documents
        cmd.execute("add doc1 \"Content 1\"", &mut state, &mut ctx);
        cmd.execute("add doc2 \"Content 2\"", &mut state, &mut ctx);

        // Clear
        let output = cmd.execute("clear", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert!(state.store_is_empty());
    }

    #[test]
    fn test_search_command() {
        let cmd = SearchCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        // Empty search
        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Error(_)));

        // Search empty store
        let output = cmd.execute("test query", &mut state, &mut ctx);
        assert!(matches!(output, Output::Warning(_)));
    }
}
