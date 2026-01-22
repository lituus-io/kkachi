// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! I/O commands (save, load, export, import).

use super::{Command, ExecutionContext, Output};
use crate::repl::SessionState;
use std::fs;
use std::path::Path;

/// Save command - save session to file.
pub struct SaveCommand;

impl Command for SaveCommand {
    fn name(&self) -> &str {
        "save"
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }

    fn description(&self) -> &str {
        "Save session to file"
    }

    fn help(&self) -> &str {
        "Usage: save <filename>\n\n\
         Save the current session state to a JSON file.\n\
         This includes signature, instruction, demos, model config, etc.\n\n\
         Examples:\n\
           save session.json\n\
           save ~/projects/my-module.json"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let filename = args.trim();

        if filename.is_empty() {
            return Output::error("Please specify a filename. Usage: save <filename>");
        }

        // Ensure .json extension
        let filename = if filename.ends_with(".json") {
            filename.to_string()
        } else {
            format!("{}.json", filename)
        };

        // Serialize state
        let json = match serde_json::to_string_pretty(state) {
            Ok(j) => j,
            Err(e) => return Output::error(format!("Failed to serialize: {}", e)),
        };

        // Write to file
        if let Err(e) = fs::write(&filename, json) {
            return Output::error(format!("Failed to write file: {}", e));
        }

        Output::success(format!("Session saved to: {}", filename))
    }
}

/// Load command - load session from file.
pub struct LoadCommand;

impl Command for LoadCommand {
    fn name(&self) -> &str {
        "load"
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }

    fn description(&self) -> &str {
        "Load session from file"
    }

    fn help(&self) -> &str {
        "Usage: load <filename>\n\n\
         Load a session state from a JSON file.\n\
         This replaces the current session state.\n\n\
         Examples:\n\
           load session.json\n\
           load ~/projects/my-module.json"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let filename = args.trim();

        if filename.is_empty() {
            return Output::error("Please specify a filename. Usage: load <filename>");
        }

        // Check if file exists
        if !Path::new(filename).exists() {
            return Output::error(format!("File not found: {}", filename));
        }

        // Read file
        let json = match fs::read_to_string(filename) {
            Ok(j) => j,
            Err(e) => return Output::error(format!("Failed to read file: {}", e)),
        };

        // Deserialize
        let new_state: SessionState = match serde_json::from_str(&json) {
            Ok(s) => s,
            Err(e) => return Output::error(format!("Failed to parse JSON: {}", e)),
        };

        // Save current state for undo
        ctx.history.push(state.snapshot());

        // Replace state
        *state = new_state;

        Output::success(format!("Session loaded from: {}", filename))
    }
}

/// Export command - export module definition.
pub struct ExportCommand;

impl Command for ExportCommand {
    fn name(&self) -> &str {
        "export"
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }

    fn description(&self) -> &str {
        "Export module definition"
    }

    fn help(&self) -> &str {
        "Usage: export <filename>\n\n\
         Export the current module definition (signature, instruction, demos)\n\
         to a portable format that can be used in code.\n\n\
         Formats:\n\
           .json  - JSON format\n\
           .yaml  - YAML format (if available)\n\
           .rs    - Rust code snippet\n\
           .py    - Python code snippet\n\n\
         Examples:\n\
           export module.json\n\
           export my-module.rs"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let filename = args.trim();

        if filename.is_empty() {
            return Output::error("Please specify a filename. Usage: export <filename>");
        }

        let content = if filename.ends_with(".rs") {
            self.export_rust(state)
        } else if filename.ends_with(".py") {
            self.export_python(state)
        } else {
            self.export_json(state)
        };

        if let Err(e) = fs::write(filename, content) {
            return Output::error(format!("Failed to write file: {}", e));
        }

        Output::success(format!("Module exported to: {}", filename))
    }
}

impl ExportCommand {
    fn export_json(&self, state: &SessionState) -> String {
        let module = serde_json::json!({
            "signature": state.signature.as_ref().map(|s| &s.raw),
            "instruction": &state.instruction,
            "demos": state.demos.iter().map(|d| {
                serde_json::json!({
                    "input": &d.input,
                    "output": &d.output
                })
            }).collect::<Vec<_>>(),
        });

        serde_json::to_string_pretty(&module).unwrap_or_default()
    }

    fn export_rust(&self, state: &SessionState) -> String {
        let mut code = String::new();

        code.push_str("use kkachi::prelude::*;\n\n");

        // Signature
        if let Some(ref sig) = state.signature {
            code.push_str(&format!(
                "let module = ChainOfThought::new(signature!(\"{}\"))\n",
                sig.raw
            ));
        } else {
            code.push_str("let module = ChainOfThought::new(signature!(\"input -> output\"))\n");
        }

        // Instruction
        if !state.instruction.is_empty() {
            code.push_str(&format!(
                "    .with_instruction(\"{}\")\n",
                state.instruction.replace('"', "\\\"")
            ));
        }

        // Demos
        for demo in &state.demos {
            code.push_str(&format!(
                "    .with_demo(\"{}\", \"{}\")\n",
                demo.input.replace('"', "\\\""),
                demo.output.replace('"', "\\\"")
            ));
        }

        code.push_str(";\n");

        code
    }

    fn export_python(&self, state: &SessionState) -> String {
        let mut code = String::new();

        code.push_str("import dspy\n\n");

        // Signature
        if let Some(ref sig) = state.signature {
            code.push_str(&format!(
                "class MyModule(dspy.Module):\n    \"\"\"{}.\"\"\"\n\n",
                sig.raw
            ));
        } else {
            code.push_str("class MyModule(dspy.Module):\n    \"\"\"Custom module.\"\"\"\n\n");
        }

        code.push_str("    def __init__(self):\n");
        code.push_str("        super().__init__()\n");

        // Signature
        if let Some(ref sig) = state.signature {
            code.push_str(&format!(
                "        self.predict = dspy.ChainOfThought(\"{}\")\n",
                sig.raw
            ));
        }

        code.push_str("\n    def forward(self, **kwargs):\n");
        code.push_str("        return self.predict(**kwargs)\n");

        // Instruction as comment
        if !state.instruction.is_empty() {
            code.push_str(&format!("\n# Instruction: {}\n", state.instruction));
        }

        // Demos as comment
        if !state.demos.is_empty() {
            code.push_str("\n# Demonstrations:\n");
            for (i, demo) in state.demos.iter().enumerate() {
                code.push_str(&format!("# {}. Input: {}\n", i + 1, demo.input));
                code.push_str(&format!("#    Output: {}\n", demo.output));
            }
        }

        code
    }
}

/// Import command - import module definition.
pub struct ImportCommand;

impl Command for ImportCommand {
    fn name(&self) -> &str {
        "import"
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }

    fn description(&self) -> &str {
        "Import module definition"
    }

    fn help(&self) -> &str {
        "Usage: import <filename>\n\n\
         Import a module definition from a JSON file.\n\
         This merges with or replaces the current module.\n\n\
         Examples:\n\
           import module.json"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let filename = args.trim();

        if filename.is_empty() {
            return Output::error("Please specify a filename. Usage: import <filename>");
        }

        // Check if file exists
        if !Path::new(filename).exists() {
            return Output::error(format!("File not found: {}", filename));
        }

        // Read file
        let json = match fs::read_to_string(filename) {
            Ok(j) => j,
            Err(e) => return Output::error(format!("Failed to read file: {}", e)),
        };

        // Parse JSON
        let module: serde_json::Value = match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(e) => return Output::error(format!("Failed to parse JSON: {}", e)),
        };

        // Save current state for undo
        ctx.history.push(state.snapshot());

        // Apply module
        if let Some(sig) = module.get("signature").and_then(|v| v.as_str()) {
            state.set_signature(sig);
        }

        if let Some(inst) = module.get("instruction").and_then(|v| v.as_str()) {
            state.instruction = inst.to_string();
        }

        if let Some(demos) = module.get("demos").and_then(|v| v.as_array()) {
            state.demos.clear();
            for demo in demos {
                if let (Some(input), Some(output)) = (
                    demo.get("input").and_then(|v| v.as_str()),
                    demo.get("output").and_then(|v| v.as_str()),
                ) {
                    state.add_demo(input.to_string(), output.to_string());
                }
            }
        }

        Output::success(format!("Module imported from: {}", filename))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::StateHistory;
    use kkachi::DiffRenderer;
    use std::fs;
    use tempfile::tempdir;

    fn make_ctx<'a>(
        history: &'a mut StateHistory,
        renderer: &'a DiffRenderer,
    ) -> ExecutionContext<'a> {
        ExecutionContext { renderer, history }
    }

    #[test]
    fn test_save_command_no_filename() {
        let cmd = SaveCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Error(_)));
    }

    #[test]
    fn test_export_json() {
        let cmd = ExportCommand;
        let mut state = SessionState::default();
        state.set_signature("question -> answer");
        state.instruction = "Be helpful".to_string();
        state.add_demo("Q1".to_string(), "A1".to_string());

        let json = cmd.export_json(&state);
        assert!(json.contains("question -> answer"));
        assert!(json.contains("Be helpful"));
    }

    #[test]
    fn test_export_rust() {
        let cmd = ExportCommand;
        let mut state = SessionState::default();
        state.set_signature("question -> answer");

        let rust = cmd.export_rust(&state);
        assert!(rust.contains("use kkachi::prelude"));
        assert!(rust.contains("signature!"));
    }

    #[test]
    fn test_export_python() {
        let cmd = ExportCommand;
        let mut state = SessionState::default();
        state.set_signature("question -> answer");

        let python = cmd.export_python(&state);
        assert!(python.contains("import dspy"));
        assert!(python.contains("class MyModule"));
    }
}
