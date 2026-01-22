// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Pipeline commands for the REPL.

use super::{Command, ExecutionContext, Output};
use crate::repl::pipeline::{Pipeline, StageResult};
use crate::repl::SessionState;
use console::style;
use std::collections::HashMap;
use std::time::Instant;

/// Pipeline command - manage and run pipelines.
pub struct PipelineCommand;

impl Command for PipelineCommand {
    fn name(&self) -> &str {
        "pipeline"
    }

    fn aliases(&self) -> &[&str] {
        &["pipe", "p"]
    }

    fn description(&self) -> &str {
        "Manage and run pipelines"
    }

    fn help(&self) -> &str {
        "Usage: pipeline <subcommand> [args]\n\n\
         Subcommands:\n\
           load <file>       Load a pipeline from JSON file\n\
           unload            Unload the current pipeline\n\
           info              Show loaded pipeline information\n\
           stages            List pipeline stages\n\
           inspect <stage>   Inspect a specific stage\n\
           breakpoint <stage> [on|off]  Set/toggle breakpoint\n\
           breakpoints       List all breakpoints\n\n\
         Examples:\n\
           pipeline load qa_pipeline.json\n\
           pipeline stages\n\
           pipeline breakpoint generator on\n\
           pipeline inspect retriever"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let parts: Vec<&str> = args.trim().split_whitespace().collect();

        if parts.is_empty() {
            return self.show_info(state);
        }

        match parts[0] {
            "load" => {
                if parts.len() < 2 {
                    return Output::error("Usage: pipeline load <file>");
                }
                self.load_pipeline(parts[1], state)
            }
            "unload" => self.unload_pipeline(state),
            "info" => self.show_info(state),
            "stages" | "list" => self.list_stages(state),
            "inspect" => {
                if parts.len() < 2 {
                    return Output::error("Usage: pipeline inspect <stage>");
                }
                self.inspect_stage(parts[1], state)
            }
            "breakpoint" | "bp" => {
                if parts.len() < 2 {
                    return self.list_breakpoints(state);
                }
                let enabled = parts.get(2).map(|s| *s == "on").unwrap_or(true);
                self.set_breakpoint(parts[1], enabled, state)
            }
            "breakpoints" | "bps" => self.list_breakpoints(state),
            _ => Output::error(format!(
                "Unknown subcommand: {}. Use 'help pipeline' for usage.",
                parts[0]
            )),
        }
    }
}

impl PipelineCommand {
    fn load_pipeline(&self, path: &str, state: &mut SessionState) -> Output {
        match Pipeline::load(path) {
            Ok(pipeline) => {
                if let Err(e) = pipeline.validate() {
                    return Output::error(format!("Invalid pipeline: {}", e));
                }

                let name = pipeline.name.clone();
                let stage_count = pipeline.stages.len();
                state.load_pipeline(pipeline);

                Output::success(format!(
                    "Pipeline '{}' loaded ({} stages)",
                    name, stage_count
                ))
            }
            Err(e) => Output::error(format!("Failed to load pipeline: {}", e)),
        }
    }

    fn unload_pipeline(&self, state: &mut SessionState) -> Output {
        if state.has_pipeline() {
            let name = state
                .get_pipeline()
                .map(|p| p.name.clone())
                .unwrap_or_default();
            state.unload_pipeline();
            Output::success(format!("Pipeline '{}' unloaded", name))
        } else {
            Output::warning("No pipeline loaded")
        }
    }

    fn show_info(&self, state: &SessionState) -> Output {
        match state.get_pipeline() {
            Some(pipeline) => {
                let mut output = String::new();

                output.push_str(&format!(
                    "{}\n\n",
                    style("PIPELINE INFO").bold().underlined()
                ));

                output.push_str(&format!("  Name: {}\n", style(&pipeline.name).cyan()));
                output.push_str(&format!("  Description: {}\n", pipeline.description));
                output.push_str(&format!("  Stages: {}\n", pipeline.stages.len()));

                // Defaults
                if pipeline.defaults.model.is_some()
                    || pipeline.defaults.temperature.is_some()
                    || pipeline.defaults.max_tokens.is_some()
                {
                    output.push_str("\n  Defaults:\n");
                    if let Some(ref model) = pipeline.defaults.model {
                        output.push_str(&format!("    Model: {}\n", model));
                    }
                    if let Some(temp) = pipeline.defaults.temperature {
                        output.push_str(&format!("    Temperature: {}\n", temp));
                    }
                    if let Some(tokens) = pipeline.defaults.max_tokens {
                        output.push_str(&format!("    Max tokens: {}\n", tokens));
                    }
                }

                // Breakpoints
                let breakpoints = pipeline.breakpoints();
                if !breakpoints.is_empty() {
                    output.push_str(&format!("\n  Breakpoints: {}\n", breakpoints.join(", ")));
                }

                // Execution state
                if let Some(exec) = state.get_execution_state() {
                    output.push_str(&format!(
                        "\n  {} Execution paused at stage {}/{}\n",
                        style("▶").yellow(),
                        exec.current_stage + 1,
                        pipeline.stages.len()
                    ));
                }

                Output::text(output)
            }
            None => Output::text("No pipeline loaded. Use 'pipeline load <file>' to load one."),
        }
    }

    fn list_stages(&self, state: &SessionState) -> Output {
        match state.get_pipeline() {
            Some(pipeline) => {
                let mut output = String::new();

                output.push_str(&format!(
                    "{} ({})\n\n",
                    style("PIPELINE STAGES").bold().underlined(),
                    pipeline.name
                ));

                for (i, stage) in pipeline.stages.iter().enumerate() {
                    let bp_marker = if stage.breakpoint {
                        style("●").red().to_string()
                    } else {
                        " ".to_string()
                    };

                    let stage_type = style(stage.type_display()).dim();

                    output.push_str(&format!(
                        "  {} [{}] {} ({})\n",
                        bp_marker,
                        i + 1,
                        style(&stage.name).cyan(),
                        stage_type
                    ));

                    // Show signature if present
                    if let Some(ref sig) = stage.config.signature {
                        output.push_str(&format!("        signature: {}\n", style(sig).dim()));
                    }
                }

                output.push_str(&format!("\n  {} = breakpoint\n", style("●").red()));

                Output::text(output)
            }
            None => Output::error("No pipeline loaded"),
        }
    }

    fn inspect_stage(&self, stage_name: &str, state: &SessionState) -> Output {
        match state.get_pipeline() {
            Some(pipeline) => {
                match pipeline.get_stage(stage_name) {
                    Some(stage) => {
                        let mut output = String::new();

                        output.push_str(&format!(
                            "{}\n\n",
                            style(format!("STAGE: {}", stage.name)).bold().underlined()
                        ));

                        output.push_str(&format!("  Type: {}\n", stage.type_display()));
                        output.push_str(&format!(
                            "  Breakpoint: {}\n",
                            if stage.breakpoint { "yes" } else { "no" }
                        ));

                        // Config details
                        output.push_str("\n  Configuration:\n");

                        if let Some(ref sig) = stage.config.signature {
                            output.push_str(&format!("    Signature: {}\n", sig));
                        }
                        if let Some(ref inst) = stage.config.instruction {
                            let preview: String = inst.chars().take(60).collect();
                            let suffix = if inst.len() > 60 { "..." } else { "" };
                            output.push_str(&format!("    Instruction: {}{}\n", preview, suffix));
                        }
                        if let Some(k) = stage.config.k {
                            output.push_str(&format!("    k: {}\n", k));
                        }
                        if let Some(threshold) = stage.config.threshold {
                            output.push_str(&format!("    Threshold: {}\n", threshold));
                        }
                        if let Some(max_iter) = stage.config.max_iterations {
                            output.push_str(&format!("    Max iterations: {}\n", max_iter));
                        }
                        if let Some(score_thresh) = stage.config.score_threshold {
                            output.push_str(&format!("    Score threshold: {}\n", score_thresh));
                        }
                        if let Some(ref tools) = stage.config.tools {
                            output.push_str(&format!("    Tools: {}\n", tools.join(", ")));
                        }

                        // Custom params
                        if !stage.config.params.is_empty() {
                            output.push_str("\n    Custom params:\n");
                            for (key, value) in &stage.config.params {
                                output.push_str(&format!("      {}: {}\n", key, value));
                            }
                        }

                        Output::text(output)
                    }
                    None => Output::error(format!("Stage '{}' not found", stage_name)),
                }
            }
            None => Output::error("No pipeline loaded"),
        }
    }

    fn set_breakpoint(&self, stage_name: &str, enabled: bool, state: &mut SessionState) -> Output {
        match state.get_pipeline_mut() {
            Some(pipeline) => {
                if pipeline.set_breakpoint(stage_name, enabled) {
                    let action = if enabled { "set" } else { "removed" };
                    Output::success(format!("Breakpoint {} on stage '{}'", action, stage_name))
                } else {
                    Output::error(format!("Stage '{}' not found", stage_name))
                }
            }
            None => Output::error("No pipeline loaded"),
        }
    }

    fn list_breakpoints(&self, state: &SessionState) -> Output {
        match state.get_pipeline() {
            Some(pipeline) => {
                let breakpoints = pipeline.breakpoints();
                if breakpoints.is_empty() {
                    Output::text("No breakpoints set")
                } else {
                    Output::text(format!("Breakpoints: {}", breakpoints.join(", ")))
                }
            }
            None => Output::error("No pipeline loaded"),
        }
    }
}

/// Run command - execute the loaded pipeline.
pub struct RunCommand;

impl Command for RunCommand {
    fn name(&self) -> &str {
        "run"
    }

    fn aliases(&self) -> &[&str] {
        &["exec"]
    }

    fn description(&self) -> &str {
        "Execute the loaded pipeline"
    }

    fn help(&self) -> &str {
        "Usage: run <input>\n\n\
         Execute the loaded pipeline with the given input.\n\
         Pipeline will pause at breakpoints for HITL review.\n\n\
         Options:\n\
           --continue, -c    Continue from paused execution\n\
           --skip, -s        Skip current stage and continue\n\n\
         Examples:\n\
           run What is machine learning?\n\
           run -c             (continue after breakpoint)\n\
           run --skip         (skip current stage)"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        // Check for continuation flags
        if args == "-c" || args == "--continue" {
            return self.continue_execution(state, ctx);
        }

        if args == "-s" || args == "--skip" {
            return self.skip_and_continue(state, ctx);
        }

        // Check if pipeline is loaded
        if !state.has_pipeline() {
            return Output::error("No pipeline loaded. Use 'pipeline load <file>' first.");
        }

        if args.is_empty() {
            return Output::error("Please provide an input. Usage: run <input>");
        }

        // Start fresh execution
        state.start_pipeline_execution(args.to_string());
        self.execute_pipeline(state, ctx)
    }
}

impl RunCommand {
    fn execute_pipeline(
        &self,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let pipeline = match state.get_pipeline() {
            Some(p) => p.clone(),
            None => return Output::error("No pipeline loaded"),
        };

        let exec_state = match state.get_execution_state_mut() {
            Some(e) => e,
            None => return Output::error("No execution state"),
        };

        let start_time = Instant::now();
        let mut output = String::new();

        output.push_str(&format!(
            "{} Running pipeline '{}'\n\n",
            style("▶").green(),
            pipeline.name
        ));

        // Execute stages
        while exec_state.current_stage < pipeline.stages.len() {
            let stage = &pipeline.stages[exec_state.current_stage];
            let stage_start = Instant::now();

            output.push_str(&format!(
                "  [{}] {} ",
                exec_state.current_stage + 1,
                style(&stage.name).cyan()
            ));

            // Simulate stage execution (in real implementation, this would call the actual module)
            let stage_output = self.simulate_stage_execution(stage, &exec_state.current_input);
            let duration = stage_start.elapsed().as_millis() as u64;

            // Record result
            let result = StageResult {
                stage_name: stage.name.clone(),
                stage_type: stage.stage_type.clone(),
                output: stage_output.clone(),
                score: Some(0.85), // Simulated score
                duration_ms: duration,
                metadata: HashMap::new(),
            };

            exec_state.results.push(result);
            exec_state.current_input = stage_output.clone();

            output.push_str(&format!("{} ({}ms)\n", style("✓").green(), duration));

            // Check for breakpoint
            if stage.breakpoint && exec_state.current_stage < pipeline.stages.len() - 1 {
                exec_state.current_stage += 1;

                output.push_str(&format!(
                    "\n  {} Breakpoint reached at stage '{}'\n",
                    style("⏸").yellow(),
                    stage.name
                ));
                output.push_str("  Use 'run -c' to continue or 'run -s' to skip next stage\n");

                // Store last output
                state.last_output = Some(stage_output);

                return Output::text(output);
            }

            exec_state.current_stage += 1;
        }

        // Pipeline completed
        let total_duration = start_time.elapsed().as_millis() as u64;
        let final_output = exec_state.current_input.clone();

        output.push_str(&format!(
            "\n{} Pipeline completed in {}ms\n",
            style("✓").green().bold(),
            total_duration
        ));

        // Show output preview
        let preview: String = final_output.chars().take(200).collect();
        let suffix = if final_output.len() > 200 { "..." } else { "" };
        output.push_str(&format!(
            "\n{}\n  {}{}\n",
            style("OUTPUT").bold().underlined(),
            preview,
            suffix
        ));

        // Update state
        state.last_output = Some(final_output);
        state.clear_execution_state();

        Output::text(output)
    }

    fn continue_execution(
        &self,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if state.get_execution_state().is_none() {
            return Output::error("No paused execution to continue");
        }

        self.execute_pipeline(state, ctx)
    }

    fn skip_and_continue(
        &self,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        // Get pipeline info first to avoid borrow issues
        let (stage_count, skipped_name) = match (state.get_pipeline(), state.get_execution_state())
        {
            (Some(pipeline), Some(exec_state)) => {
                if exec_state.current_stage < pipeline.stages.len() {
                    (
                        pipeline.stages.len(),
                        Some(pipeline.stages[exec_state.current_stage].name.clone()),
                    )
                } else {
                    return Output::error("No stages to skip");
                }
            }
            (None, _) => return Output::error("No pipeline loaded"),
            (_, None) => return Output::error("No paused execution to skip"),
        };

        let skipped = match skipped_name {
            Some(name) => name,
            None => return Output::error("No stages to skip"),
        };

        // Now mutably borrow to update stage
        if let Some(exec_state) = state.get_execution_state_mut() {
            exec_state.current_stage += 1;

            if exec_state.current_stage < stage_count {
                return Output::success(format!(
                    "Skipped '{}'. Use 'run -c' to continue.",
                    skipped
                ));
            }
        }

        Output::text(format!(
            "{} Skipped stage '{}'\n{} Pipeline completed (with skip)\n",
            style("→").yellow(),
            skipped,
            style("✓").green()
        ))
    }

    fn simulate_stage_execution(
        &self,
        stage: &crate::repl::pipeline::PipelineStage,
        input: &str,
    ) -> String {
        // This is a simulation. In a real implementation, this would:
        // 1. Create the appropriate module (ChainOfThought, Retriever, etc.)
        // 2. Execute it with the input
        // 3. Return the actual output

        match &stage.stage_type {
            crate::repl::pipeline::StageType::Retriever => {
                format!("[Retrieved documents for: {}]", input)
            }
            crate::repl::pipeline::StageType::ChainOfThought => {
                format!("[Chain of thought reasoning for: {}]", input)
            }
            crate::repl::pipeline::StageType::Predict => {
                format!("[Prediction for: {}]", input)
            }
            crate::repl::pipeline::StageType::Validator => {
                format!("[Validated: {}]", input)
            }
            _ => format!("[Processed: {}]", input),
        }
    }
}

/// Stages command - shorthand for pipeline stages.
pub struct StagesCommand;

impl Command for StagesCommand {
    fn name(&self) -> &str {
        "stages"
    }

    fn aliases(&self) -> &[&str] {
        &[]
    }

    fn description(&self) -> &str {
        "List stages in the loaded pipeline"
    }

    fn execute(
        &self,
        _args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        PipelineCommand.execute("stages", state, ctx)
    }
}

/// Inspect command - shorthand for pipeline inspect.
pub struct InspectCommand;

impl Command for InspectCommand {
    fn name(&self) -> &str {
        "inspect"
    }

    fn aliases(&self) -> &[&str] {
        &["i"]
    }

    fn description(&self) -> &str {
        "Inspect a pipeline stage"
    }

    fn help(&self) -> &str {
        "Usage: inspect <stage>\n\n\
         Show detailed information about a pipeline stage.\n\n\
         Examples:\n\
           inspect retriever\n\
           inspect generator"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if args.trim().is_empty() {
            return Output::error("Usage: inspect <stage>");
        }
        PipelineCommand.execute(&format!("inspect {}", args), state, ctx)
    }
}

/// Breakpoint command - shorthand for pipeline breakpoint.
pub struct BreakpointCommand;

impl Command for BreakpointCommand {
    fn name(&self) -> &str {
        "breakpoint"
    }

    fn aliases(&self) -> &[&str] {
        &["bp"]
    }

    fn description(&self) -> &str {
        "Set a breakpoint on a pipeline stage"
    }

    fn help(&self) -> &str {
        "Usage: breakpoint <stage> [on|off]\n\n\
         Set or toggle a breakpoint on a pipeline stage.\n\
         Breakpoints pause execution for HITL review.\n\n\
         Examples:\n\
           breakpoint generator on\n\
           breakpoint retriever off\n\
           breakpoint validator"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        if args.trim().is_empty() {
            return PipelineCommand.execute("breakpoints", state, ctx);
        }
        PipelineCommand.execute(&format!("breakpoint {}", args), state, ctx)
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
    fn test_pipeline_command_no_pipeline() {
        let cmd = PipelineCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }

    #[test]
    fn test_run_command_no_pipeline() {
        let cmd = RunCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("test input", &mut state, &mut ctx);
        assert!(matches!(output, Output::Error(_)));
    }

    #[test]
    fn test_stages_command() {
        let cmd = StagesCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        // No pipeline - should show error
        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Error(_)));
    }

    #[test]
    fn test_breakpoint_command() {
        let cmd = BreakpointCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        // No pipeline - should show error
        let output = cmd.execute("test_stage on", &mut state, &mut ctx);
        assert!(matches!(output, Output::Error(_)));
    }
}
