// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! ReAct (Reasoning + Acting) module
//!
//! Implements the ReAct pattern where the LM alternates between:
//! - Thought: reasoning about the current state
//! - Action: selecting and calling a tool
//! - Observation: receiving tool output
//!
//! ## Zero-Copy Design
//!
//! - Tools receive `StrView<'a>` inputs
//! - Tool outputs are written to a shared buffer
//! - No string allocations during the reasoning loop

use crate::buffer::BufferView;
use crate::error::Result;
use crate::intern::{sym, Sym};
use crate::module::Module;
use crate::predict::LMClient;
use crate::prediction::Prediction;
use crate::signature::Signature;
use crate::str_view::StrView;
use crate::types::Inputs;
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;

/// Tool trait - zero-copy input/output.
pub trait Tool: Send + Sync {
    /// Tool name (interned).
    fn name(&self) -> Sym;

    /// Tool description for the prompt.
    fn description(&self) -> &str;

    /// Execute the tool with the given input.
    ///
    /// Returns the result written to the provided buffer.
    fn execute<'a>(
        &'a self,
        input: StrView<'a>,
        output_buffer: &'a mut Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult<'a>>> + Send + 'a>>;
}

/// Zero-copy tool result.
pub struct ToolResult<'a> {
    /// Output as view into buffer
    pub output: StrView<'a>,
    /// Whether the tool succeeded
    pub success: bool,
}

/// ReAct module configuration.
#[derive(Clone, Copy)]
pub struct ReActConfig {
    /// Maximum iterations (thought-action-observation cycles)
    pub max_iterations: u8,
    /// Whether to include full trajectory in output
    pub include_trajectory: bool,
}

impl Default for ReActConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            include_trajectory: true,
        }
    }
}

/// ReAct module - Reasoning + Acting with tools.
///
/// ## Example
///
/// ```ignore
/// let tools: &[&dyn Tool] = &[&search_tool, &calculator_tool];
/// let react = ReAct::new(&signature, tools);
/// let result = react_with_lm(&react, &inputs, &lm, &mut buffer).await?;
/// ```
pub struct ReAct<'sig, 'tools> {
    /// The signature defining inputs/outputs
    signature: &'sig Signature<'sig>,
    /// Available tools
    tools: &'tools [&'tools dyn Tool],
    /// Configuration
    config: ReActConfig,
    /// Symbols for ReAct fields (reserved for future optimization)
    #[allow(dead_code)]
    thought_sym: Sym,
    #[allow(dead_code)]
    action_sym: Sym,
    #[allow(dead_code)]
    action_input_sym: Sym,
    #[allow(dead_code)]
    observation_sym: Sym,
}

impl<'sig, 'tools> ReAct<'sig, 'tools> {
    /// Create a new ReAct module.
    pub fn new(signature: &'sig Signature<'sig>, tools: &'tools [&'tools dyn Tool]) -> Self {
        Self {
            signature,
            tools,
            config: ReActConfig::default(),
            thought_sym: sym("thought"),
            action_sym: sym("action"),
            action_input_sym: sym("action_input"),
            observation_sym: sym("observation"),
        }
    }

    /// Configure max iterations.
    pub fn with_max_iterations(mut self, n: u8) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Configure trajectory inclusion.
    pub fn with_trajectory(mut self, include: bool) -> Self {
        self.config.include_trajectory = include;
        self
    }

    /// Get the signature.
    #[inline]
    pub fn signature(&self) -> &'sig Signature<'sig> {
        self.signature
    }

    /// Get available tools.
    #[inline]
    pub fn tools(&self) -> &[&'tools dyn Tool] {
        self.tools
    }

    /// Build the ReAct prompt.
    pub fn build_prompt_into<'buf>(
        &self,
        inputs: &Inputs<'_>,
        trajectory: &[TrajectoryStep<'_>],
        buffer: &'buf mut Vec<u8>,
    ) -> StrView<'buf> {
        buffer.clear();

        // Instructions
        buffer.extend_from_slice(self.signature.instructions.as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Tool descriptions
        buffer.extend_from_slice(b"Available tools:\n");
        for tool in self.tools {
            buffer.extend_from_slice(b"- ");
            buffer.extend_from_slice(crate::intern::resolve(tool.name()).as_bytes());
            buffer.extend_from_slice(b": ");
            buffer.extend_from_slice(tool.description().as_bytes());
            buffer.push(b'\n');
        }
        buffer.extend_from_slice(b"\n");

        // ReAct format instructions
        buffer.extend_from_slice(
            b"Use the following format:\n\
            Thought: reason about what to do\n\
            Action: tool_name\n\
            Action Input: input to the tool\n\
            Observation: tool output\n\
            ... (repeat as needed)\n\
            Thought: I now know the final answer\n\
            Final Answer: your answer\n\n",
        );

        // Current input
        buffer.extend_from_slice(b"Question: ");
        if let Some(question) = inputs.get("question").or_else(|| inputs.get("input")) {
            buffer.extend_from_slice(question.as_bytes());
        }
        buffer.push(b'\n');

        // Add trajectory so far
        for step in trajectory {
            buffer.extend_from_slice(b"Thought: ");
            buffer.extend_from_slice(step.thought.as_bytes());
            buffer.push(b'\n');

            buffer.extend_from_slice(b"Action: ");
            buffer.extend_from_slice(crate::intern::resolve(step.action).as_bytes());
            buffer.push(b'\n');

            buffer.extend_from_slice(b"Action Input: ");
            buffer.extend_from_slice(step.action_input.as_bytes());
            buffer.push(b'\n');

            buffer.extend_from_slice(b"Observation: ");
            buffer.extend_from_slice(step.observation.as_bytes());
            buffer.push(b'\n');
        }

        // Prompt for next thought
        buffer.extend_from_slice(b"Thought: ");

        // SAFETY: We only write valid UTF-8
        unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) }
    }

    /// Parse a single step from LM response.
    pub fn parse_step(&self, response: &str) -> Option<ParsedStep> {
        // Check for final answer
        if let Some(final_idx) = response.find("Final Answer:") {
            let answer_start = final_idx + "Final Answer:".len();
            let answer_start = skip_whitespace(response, answer_start);
            let answer_end = response[answer_start..]
                .find('\n')
                .map(|i| answer_start + i)
                .unwrap_or(response.len());

            return Some(ParsedStep::FinalAnswer(answer_start..answer_end));
        }

        // Parse thought
        let thought_start = 0; // Starts immediately
        let thought_end = response.find("Action:")?.saturating_sub(1);
        let thought = thought_start..thought_end;

        // Parse action
        let action_start = response.find("Action:")? + "Action:".len();
        let action_start = skip_whitespace(response, action_start);
        let action_end = response[action_start..]
            .find('\n')
            .map(|i| action_start + i)
            .unwrap_or(response.len());

        // Parse action input
        let input_start = response.find("Action Input:")? + "Action Input:".len();
        let input_start = skip_whitespace(response, input_start);
        let input_end = response[input_start..]
            .find('\n')
            .map(|i| input_start + i)
            .unwrap_or(response.len());

        Some(ParsedStep::Action {
            thought,
            action: action_start..action_end,
            action_input: input_start..input_end,
        })
    }

    /// Find tool by name.
    pub fn find_tool(&self, name: &str) -> Option<&'tools dyn Tool> {
        let name_sym = sym(name);
        self.tools.iter().find(|t| t.name() == name_sym).copied()
    }
}

/// Skip whitespace in text starting from pos.
#[inline]
fn skip_whitespace(text: &str, mut pos: usize) -> usize {
    let bytes = text.as_bytes();
    while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\n') {
        pos += 1;
    }
    pos
}

/// A step in the ReAct trajectory.
#[derive(Clone)]
pub struct TrajectoryStep<'a> {
    /// The thought/reasoning
    pub thought: StrView<'a>,
    /// Tool name (as symbol)
    pub action: Sym,
    /// Input to the tool
    pub action_input: StrView<'a>,
    /// Tool output
    pub observation: StrView<'a>,
}

/// Parsed step from LM response.
pub enum ParsedStep {
    /// Continue with an action.
    Action {
        /// Range of the thought text in the buffer.
        thought: Range<usize>,
        /// Range of the action name in the buffer.
        action: Range<usize>,
        /// Range of the action input in the buffer.
        action_input: Range<usize>,
    },
    /// Final answer reached.
    FinalAnswer(Range<usize>),
}

/// Execute ReAct with an LM client.
///
/// This simplified version executes a single step. For the full iterative loop,
/// use react_step() in a loop managed by the caller.
pub async fn react_with_lm<'a, L>(
    react: &ReAct<'_, '_>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<ReActOutput<'a>>
where
    L: LMClient,
{
    // Build initial prompt (no trajectory)
    let prompt = react.build_prompt_into(inputs, &[], prompt_buffer);

    // Call LM
    let output = lm.generate(prompt).await?;
    let response_text = output.text()?.as_str();

    // Parse the step
    match react.parse_step(response_text) {
        Some(ParsedStep::FinalAnswer(range)) => Ok(ReActOutput {
            buffer: output.buffer,
            answer_range: Some(range),
            trajectory_count: 0,
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
        }),
        Some(ParsedStep::Action { .. }) => {
            // For single-step execution, return without answer
            // Caller should use react_step() for full loop
            Ok(ReActOutput {
                buffer: output.buffer,
                answer_range: None,
                trajectory_count: 0,
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
            })
        }
        None => Err(crate::error::Error::module(
            "Failed to parse ReAct response",
        )),
    }
}

/// Execute a single ReAct step with trajectory context.
pub async fn react_step<'a, L>(
    react: &ReAct<'_, '_>,
    inputs: &Inputs<'_>,
    trajectory: &[TrajectoryStep<'_>],
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
) -> Result<ReActStepOutput<'a>>
where
    L: LMClient,
{
    // Build prompt with trajectory
    let prompt = react.build_prompt_into(inputs, trajectory, prompt_buffer);

    // Call LM
    let output = lm.generate(prompt).await?;
    let response_text = output.text()?.as_str();

    // Parse the step
    match react.parse_step(response_text) {
        Some(ParsedStep::FinalAnswer(range)) => Ok(ReActStepOutput {
            buffer: output.buffer,
            step: StepResult::FinalAnswer(range),
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
        }),
        Some(ParsedStep::Action {
            thought,
            action,
            action_input,
        }) => Ok(ReActStepOutput {
            buffer: output.buffer,
            step: StepResult::Action {
                thought,
                action,
                action_input,
            },
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
        }),
        None => Err(crate::error::Error::module(
            "Failed to parse ReAct response",
        )),
    }
}

/// Output from a single ReAct step.
pub struct ReActStepOutput<'a> {
    /// Response buffer
    pub buffer: BufferView<'a>,
    /// The parsed step result.
    pub step: StepResult,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

/// Result of parsing a ReAct step.
pub enum StepResult {
    /// Continue with an action.
    Action {
        /// Range of the thought text in the buffer.
        thought: Range<usize>,
        /// Range of the action name in the buffer.
        action: Range<usize>,
        /// Range of the action input in the buffer.
        action_input: Range<usize>,
    },
    /// Final answer reached.
    FinalAnswer(Range<usize>),
}

/// Zero-copy ReAct output.
pub struct ReActOutput<'a> {
    /// Response buffer containing final answer.
    pub buffer: BufferView<'a>,
    /// Range for the final answer.
    pub answer_range: Option<Range<usize>>,
    /// Number of steps in trajectory.
    pub trajectory_count: usize,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

impl<'a> ReActOutput<'a> {
    /// Get the final answer.
    pub fn answer(&self) -> Option<StrView<'a>> {
        let range = self.answer_range.as_ref()?;
        let text = self.buffer.as_str().ok()?;
        Some(StrView::new(&text[range.clone()]))
    }
}

impl Module for ReAct<'_, '_> {
    type ForwardFut<'a>
        = std::future::Ready<Result<Prediction<'a>>>
    where
        Self: 'a;

    fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
        std::future::ready(Err(crate::error::Error::module(
            "Use react_with_lm() instead of forward() for zero-copy execution",
        )))
    }

    fn name(&self) -> &str {
        "ReAct"
    }

    fn id(&self) -> Sym {
        sym("react")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockCalculator;

    impl Tool for MockCalculator {
        fn name(&self) -> Sym {
            sym("calculator")
        }

        fn description(&self) -> &str {
            "Perform arithmetic calculations"
        }

        fn execute<'a>(
            &'a self,
            _input: StrView<'a>,
            output_buffer: &'a mut Vec<u8>,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult<'a>>> + Send + 'a>> {
            Box::pin(async move {
                // Simple mock: always return "42"
                output_buffer.clear();
                output_buffer.extend_from_slice(b"42");

                Ok(ToolResult {
                    output: unsafe {
                        StrView::from_raw_parts(output_buffer.as_ptr(), output_buffer.len())
                    },
                    success: true,
                })
            })
        }
    }

    #[test]
    fn test_react_creation() {
        let sig = Signature::parse("question -> answer").unwrap();
        let calc = MockCalculator;
        let tools: &[&dyn Tool] = &[&calc];
        let react = ReAct::new(&sig, tools);

        assert_eq!(react.name(), "ReAct");
        assert_eq!(react.tools().len(), 1);
    }

    #[test]
    fn test_parse_step_action() {
        let sig = Signature::parse("question -> answer").unwrap();
        let calc = MockCalculator;
        let tools: &[&dyn Tool] = &[&calc];
        let react = ReAct::new(&sig, tools);

        let response = "I need to calculate this\nAction: calculator\nAction Input: 2 + 2\n";
        let step = react.parse_step(response);

        assert!(matches!(step, Some(ParsedStep::Action { .. })));
    }

    #[test]
    fn test_parse_step_final() {
        let sig = Signature::parse("question -> answer").unwrap();
        let calc = MockCalculator;
        let tools: &[&dyn Tool] = &[&calc];
        let react = ReAct::new(&sig, tools);

        let response = "I now know the answer\nFinal Answer: 42\n";
        let step = react.parse_step(response);

        assert!(matches!(step, Some(ParsedStep::FinalAnswer(_))));
    }

    #[test]
    fn test_find_tool() {
        let sig = Signature::parse("question -> answer").unwrap();
        let calc = MockCalculator;
        let tools: &[&dyn Tool] = &[&calc];
        let react = ReAct::new(&sig, tools);

        assert!(react.find_tool("calculator").is_some());
        assert!(react.find_tool("unknown").is_none());
    }
}
