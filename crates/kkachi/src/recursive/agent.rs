// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! ReAct agent with tool calling.
//!
//! This module provides the [`agent`] entry point for creating agents that
//! can use tools to accomplish goals through iterative reasoning and action.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, agent};
//! use kkachi::recursive::tool::{tool, MockTool};
//!
//! let calc = MockTool::new("calculator", "Perform calculations", "42");
//!
//! let llm = MockLlm::new(|_, _| {
//!     "I need to calculate this.\nFinal Answer: 42".to_string()
//! });
//!
//! let result = agent(&llm, "What is 2 + 2?")
//!     .tool(&calc)
//!     .max_steps(5)
//!     .go();
//! ```

use crate::recursive::llm::Llm;
use crate::recursive::tool::Tool;
use smallvec::SmallVec;

/// Entry point for creating a ReAct agent.
///
/// Creates a builder for an agent that can use tools to accomplish goals.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, agent};
/// use kkachi::recursive::tool::MockTool;
///
/// let search = MockTool::new("search", "Search the web", "results...");
///
/// let llm = MockLlm::new(|_, _| "Final Answer: found it".to_string());
///
/// let result = agent(&llm, "Find information about Rust")
///     .tool(&search)
///     .go();
/// ```
pub fn agent<'a, L: Llm>(llm: &'a L, goal: &'a str) -> Agent<'a, L> {
    Agent::new(llm, goal)
}

/// Configuration for the agent.
#[derive(Clone)]
pub struct AgentConfig {
    /// Maximum number of reasoning steps.
    pub max_steps: usize,
    /// Whether to include the full trajectory in the result.
    pub include_trajectory: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            include_trajectory: true,
        }
    }
}

/// ReAct agent builder.
///
/// Builds an agent that can use tools through iterative reasoning and action.
pub struct Agent<'a, L: Llm> {
    llm: &'a L,
    goal: &'a str,
    tools: SmallVec<[&'a dyn Tool; 4]>,
    config: AgentConfig,
    on_step: Option<Box<dyn Fn(&Step) + Send + Sync + 'a>>,
}

impl<'a, L: Llm> Agent<'a, L> {
    /// Create a new agent builder.
    pub fn new(llm: &'a L, goal: &'a str) -> Self {
        Self {
            llm,
            goal,
            tools: SmallVec::new(),
            config: AgentConfig::default(),
            on_step: None,
        }
    }

    /// Add a tool to the agent.
    pub fn tool<T: Tool + 'a>(mut self, tool: &'a T) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools to the agent.
    pub fn tools(mut self, tools: &'a [&'a dyn Tool]) -> Self {
        for tool in tools {
            self.tools.push(*tool);
        }
        self
    }

    /// Set maximum number of reasoning steps.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.config.max_steps = n.max(1);
        self
    }

    /// Set a callback to be invoked on each step.
    ///
    /// Useful for logging or debugging the agent's thought process.
    pub fn on_step<F: Fn(&Step) + Send + Sync + 'a>(mut self, f: F) -> Self {
        self.on_step = Some(Box::new(f));
        self
    }

    /// Disable trajectory inclusion in result.
    pub fn no_trajectory(mut self) -> Self {
        self.config.include_trajectory = false;
        self
    }

    /// Execute the agent synchronously.
    pub fn go(self) -> AgentResult {
        futures::executor::block_on(self.run())
    }

    /// Execute the agent asynchronously.
    pub async fn run(self) -> AgentResult {
        let mut trajectory: SmallVec<[Step; 8]> = SmallVec::new();
        let mut total_tokens = 0u32;

        for step_num in 0..self.config.max_steps {
            // Build prompt with current trajectory
            let prompt = self.build_prompt(&trajectory);

            // Get LLM response
            let output = match self.llm.generate(&prompt, "", None).await {
                Ok(out) => out,
                Err(e) => {
                    return AgentResult {
                        output: String::new(),
                        trajectory: if self.config.include_trajectory {
                            trajectory
                        } else {
                            SmallVec::new()
                        },
                        steps: step_num,
                        tokens: total_tokens,
                        success: false,
                        error: Some(e.to_string()),
                    };
                }
            };

            total_tokens += output.prompt_tokens + output.completion_tokens;

            // Parse the response
            match self.parse_response(&output.text) {
                ParsedResponse::FinalAnswer(answer) => {
                    return AgentResult {
                        output: answer,
                        trajectory: if self.config.include_trajectory {
                            trajectory
                        } else {
                            SmallVec::new()
                        },
                        steps: step_num + 1,
                        tokens: total_tokens,
                        success: true,
                        error: None,
                    };
                }
                ParsedResponse::Action { thought, action, input } => {
                    // Find and execute the tool
                    let observation = match self.find_tool(&action) {
                        Some(tool) => {
                            match tool.execute(&input).await {
                                Ok(result) => result,
                                Err(e) => format!("Tool error: {}", e),
                            }
                        }
                        None => format!("Unknown tool: {}", action),
                    };

                    let step = Step {
                        thought,
                        action,
                        action_input: input,
                        observation,
                    };

                    // Call the step callback if set
                    if let Some(ref on_step) = self.on_step {
                        on_step(&step);
                    }

                    trajectory.push(step);
                }
                ParsedResponse::Invalid(reason) => {
                    // LLM gave invalid response, add it as a step with error
                    let step = Step {
                        thought: output.text.clone(),
                        action: String::new(),
                        action_input: String::new(),
                        observation: format!("Parse error: {}", reason),
                    };

                    if let Some(ref on_step) = self.on_step {
                        on_step(&step);
                    }

                    trajectory.push(step);
                }
            }
        }

        // Max steps reached without finding answer
        AgentResult {
            output: String::new(),
            trajectory: if self.config.include_trajectory {
                trajectory
            } else {
                SmallVec::new()
            },
            steps: self.config.max_steps,
            tokens: total_tokens,
            success: false,
            error: Some("Max steps reached without finding answer".to_string()),
        }
    }

    /// Build the prompt with trajectory.
    fn build_prompt(&self, trajectory: &[Step]) -> String {
        let mut prompt = format!("Goal: {}\n\n", self.goal);

        // Add tool descriptions
        if !self.tools.is_empty() {
            prompt.push_str("Available tools:\n");
            for tool in &self.tools {
                prompt.push_str(&format!("- {}: {}\n", tool.name(), tool.description()));
            }
            prompt.push('\n');
        }

        // Add format instructions
        prompt.push_str(
            "Use the following format:\n\
             Thought: reason about what to do\n\
             Action: tool_name\n\
             Action Input: input to the tool\n\
             Observation: tool output\n\
             ... (repeat as needed)\n\
             Thought: I now know the final answer\n\
             Final Answer: your answer\n\n"
        );

        // Add trajectory
        for step in trajectory {
            prompt.push_str(&format!("Thought: {}\n", step.thought));
            if !step.action.is_empty() {
                prompt.push_str(&format!("Action: {}\n", step.action));
                prompt.push_str(&format!("Action Input: {}\n", step.action_input));
            }
            prompt.push_str(&format!("Observation: {}\n\n", step.observation));
        }

        // Prompt for next thought
        prompt.push_str("Thought: ");
        prompt
    }

    /// Find a tool by name.
    fn find_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).copied()
    }

    /// Parse the LLM response.
    fn parse_response(&self, response: &str) -> ParsedResponse {
        // Check for final answer first
        if let Some(idx) = response.find("Final Answer:") {
            let answer_start = idx + "Final Answer:".len();
            let answer = response[answer_start..].trim();
            // Find end of answer (next newline or end)
            let answer_end = answer.find('\n').unwrap_or(answer.len());
            return ParsedResponse::FinalAnswer(answer[..answer_end].trim().to_string());
        }

        // Try to parse action
        let thought_end = response.find("Action:").unwrap_or(response.len());
        let thought = response[..thought_end].trim().to_string();

        if let Some(action_idx) = response.find("Action:") {
            let action_start = action_idx + "Action:".len();
            let action_text = &response[action_start..];

            // Find action name (up to newline)
            let action_end = action_text.find('\n').unwrap_or(action_text.len());
            let action = action_text[..action_end].trim().to_string();

            // Find action input
            if let Some(input_idx) = response.find("Action Input:") {
                let input_start = input_idx + "Action Input:".len();
                let input_text = &response[input_start..];
                let input_end = input_text.find('\n').unwrap_or(input_text.len());
                let input = input_text[..input_end].trim().to_string();

                return ParsedResponse::Action { thought, action, input };
            } else {
                return ParsedResponse::Action {
                    thought,
                    action,
                    input: String::new(),
                };
            }
        }

        // No valid format found
        ParsedResponse::Invalid("Could not parse response format".to_string())
    }
}

/// Parsed response from the LLM.
enum ParsedResponse {
    /// Final answer found.
    FinalAnswer(String),
    /// Action to execute.
    Action {
        thought: String,
        action: String,
        input: String,
    },
    /// Invalid response.
    Invalid(String),
}

/// A single step in the agent's trajectory.
#[derive(Debug, Clone)]
pub struct Step {
    /// The agent's thought/reasoning.
    pub thought: String,
    /// The tool name being called.
    pub action: String,
    /// Input provided to the tool.
    pub action_input: String,
    /// Output from the tool.
    pub observation: String,
}

/// Result from agent execution.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The final answer.
    pub output: String,
    /// The full trajectory of steps.
    pub trajectory: SmallVec<[Step; 8]>,
    /// Number of steps taken.
    pub steps: usize,
    /// Total tokens used.
    pub tokens: u32,
    /// Whether the agent succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

impl AgentResult {
    /// Get the trajectory as a slice.
    pub fn trajectory(&self) -> &[Step] {
        &self.trajectory
    }

    /// Check if the agent found an answer.
    pub fn has_answer(&self) -> bool {
        self.success && !self.output.is_empty()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::MockLlm;
    use crate::recursive::tool::MockTool;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_agent_direct_answer() {
        let llm = MockLlm::new(|_, _| {
            "I know the answer immediately.\nFinal Answer: 42".to_string()
        });

        let result = agent(&llm, "What is the answer?").go();

        assert!(result.success);
        assert_eq!(result.output, "42");
        assert_eq!(result.steps, 1);
    }

    #[test]
    fn test_agent_with_tool() {
        let calc = MockTool::new("calculator", "Calculate things", "8");

        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => "I need to calculate.\nAction: calculator\nAction Input: 2 + 2 * 2".to_string(),
                _ => "Now I know.\nFinal Answer: 8".to_string(),
            }
        });

        let result = agent(&llm, "Calculate 2 + 2 * 2")
            .tool(&calc)
            .go();

        assert!(result.success);
        assert_eq!(result.output, "8");
        assert_eq!(result.trajectory.len(), 1);
        assert_eq!(result.trajectory[0].action, "calculator");
    }

    #[test]
    fn test_agent_unknown_tool() {
        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => "Try this.\nAction: unknown_tool\nAction Input: test".to_string(),
                _ => "Final Answer: done".to_string(),
            }
        });

        let result = agent(&llm, "Test")
            .max_steps(3)
            .go();

        assert!(result.success);
        assert!(result.trajectory[0].observation.contains("Unknown tool"));
    }

    #[test]
    fn test_agent_max_steps() {
        let llm = MockLlm::new(|_, _| {
            "Still thinking.\nAction: think\nAction Input: more".to_string()
        });

        let result = agent(&llm, "Never ends")
            .max_steps(3)
            .go();

        assert!(!result.success);
        assert_eq!(result.steps, 3);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_agent_on_step_callback() {
        use std::sync::atomic::AtomicUsize;

        let callback_count = std::sync::Arc::new(AtomicUsize::new(0));
        let callback_count_clone = callback_count.clone();

        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => "Step 1.\nAction: test\nAction Input: a".to_string(),
                1 => "Step 2.\nAction: test\nAction Input: b".to_string(),
                _ => "Final Answer: done".to_string(),
            }
        });

        let test_tool = MockTool::new("test", "Test tool", "ok");

        let result = agent(&llm, "Do steps")
            .tool(&test_tool)
            .on_step(move |_| {
                callback_count_clone.fetch_add(1, Ordering::SeqCst);
            })
            .go();

        assert!(result.success);
        assert_eq!(callback_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_agent_no_trajectory() {
        let llm = MockLlm::new(|_, _| "Final Answer: 42".to_string());

        let result = agent(&llm, "Answer")
            .no_trajectory()
            .go();

        assert!(result.success);
        assert!(result.trajectory.is_empty());
    }

    #[test]
    fn test_parse_response() {
        let llm = MockLlm::new(|_, _| String::new());
        let builder = agent(&llm, "test");

        // Test final answer parsing
        match builder.parse_response("I figured it out.\nFinal Answer: The answer is 42") {
            ParsedResponse::FinalAnswer(answer) => assert_eq!(answer, "The answer is 42"),
            _ => panic!("Expected FinalAnswer"),
        }

        // Test action parsing
        match builder.parse_response("Need to search.\nAction: search\nAction Input: query") {
            ParsedResponse::Action { thought, action, input } => {
                assert!(thought.contains("search"));
                assert_eq!(action, "search");
                assert_eq!(input, "query");
            }
            _ => panic!("Expected Action"),
        }
    }
}
