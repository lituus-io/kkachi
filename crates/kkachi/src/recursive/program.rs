// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Program of Thought - code generation and execution.
//!
//! This module provides the [`program`] entry point for generating code to
//! solve problems, executing it, and using the results.
//!
//! # Examples
//!
//! ```rust,ignore
//! use kkachi::recursive::{MockLlm, program};
//! use kkachi::recursive::executor::python_executor;
//! use std::time::Duration;
//!
//! let llm = MockLlm::new(|_, _| {
//!     "```python\nprint(2 + 2)\n```".to_string()
//! });
//!
//! let result = program(&llm, "Calculate 2 + 2")
//!     .executor(python_executor().timeout(Duration::from_secs(5)))
//!     .go();
//! ```

use crate::recursive::executor::CodeExecutor;
use crate::recursive::llm::Llm;
use crate::recursive::validate::{NoValidation, Validate};

/// Entry point for Program of Thought.
///
/// Creates a builder that generates code to solve problems.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::{MockLlm, program};
/// use kkachi::recursive::executor::bash_executor;
///
/// let llm = MockLlm::new(|_, _| "```bash\necho 42\n```".to_string());
///
/// // Build a program executor (call .go() to run)
/// let builder = program(&llm, "Print 42")
///     .executor(bash_executor())
///     .max_attempts(3);
/// ```
pub fn program<'a, L: Llm>(llm: &'a L, problem: &'a str) -> Program<'a, L, NoValidation> {
    Program::new(llm, problem)
}

/// Configuration for Program of Thought.
#[derive(Clone)]
pub struct ProgramConfig {
    /// Maximum code generation attempts.
    pub max_attempts: usize,
    /// Whether to include the generated code in the result.
    pub include_code: bool,
    /// Target programming language.
    pub language: String,
}

impl Default for ProgramConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            include_code: true,
            language: "python".to_string(),
        }
    }
}

/// Program of Thought builder.
///
/// Generates code to solve problems, executes it, and refines if needed.
pub struct Program<'a, L: Llm, V: Validate> {
    llm: &'a L,
    problem: &'a str,
    executor: Option<Box<dyn CodeExecutor + Send + Sync + 'static>>,
    validator: V,
    config: ProgramConfig,
}

impl<'a, L: Llm> Program<'a, L, NoValidation> {
    /// Create a new Program of Thought builder.
    pub fn new(llm: &'a L, problem: &'a str) -> Self {
        Self {
            llm,
            problem,
            executor: None,
            validator: NoValidation,
            config: ProgramConfig::default(),
        }
    }
}

impl<'a, L: Llm, V: Validate> Program<'a, L, V> {
    /// Set a validator for the execution output.
    pub fn validate<V2: Validate>(self, validator: V2) -> Program<'a, L, V2> {
        Program {
            llm: self.llm,
            problem: self.problem,
            executor: self.executor,
            validator,
            config: self.config,
        }
    }

    /// Set the code executor.
    ///
    /// This is required - without an executor, the code cannot be run.
    pub fn executor<E: CodeExecutor + Send + Sync + 'static>(mut self, executor: E) -> Self {
        self.config.language = executor.language().to_string();
        self.executor = Some(Box::new(executor));
        self
    }

    /// Set maximum code generation attempts.
    ///
    /// If execution fails, the LLM will be asked to fix the code up to this
    /// many times.
    pub fn max_attempts(mut self, n: usize) -> Self {
        self.config.max_attempts = n.max(1);
        self
    }

    /// Set the target programming language.
    ///
    /// This is usually auto-detected from the executor but can be overridden.
    pub fn language(mut self, lang: &str) -> Self {
        self.config.language = lang.to_string();
        self
    }

    /// Disable code inclusion in result.
    pub fn no_code(mut self) -> Self {
        self.config.include_code = false;
        self
    }

    /// Execute synchronously.
    pub fn go(self) -> ProgramResult {
        futures::executor::block_on(self.run())
    }

    /// Execute asynchronously.
    pub async fn run(self) -> ProgramResult {
        let executor = match self.executor {
            Some(ref e) => e,
            None => {
                return ProgramResult {
                    output: String::new(),
                    code: String::new(),
                    attempts: 0,
                    tokens: 0,
                    success: false,
                    error: Some("No executor configured. Use .executor() to set one.".to_string()),
                };
            }
        };
        // Clone necessary values for the loop since we'll be using self
        let max_attempts = self.config.max_attempts;
        let include_code = self.config.include_code;

        let mut last_error: Option<String> = None;
        let mut last_code = String::new();
        let mut total_tokens = 0u32;

        for attempt in 0..max_attempts {
            // Build the prompt
            let prompt = self.build_prompt(last_error.as_deref());

            // Generate code
            let output = match self.llm.generate(&prompt, "", None).await {
                Ok(out) => out,
                Err(e) => {
                    return ProgramResult {
                        output: String::new(),
                        code: last_code,
                        attempts: attempt + 1,
                        tokens: total_tokens,
                        success: false,
                        error: Some(e.to_string()),
                    };
                }
            };

            total_tokens += output.prompt_tokens + output.completion_tokens;

            // Extract code from response
            let code = self.extract_code(&output.text);
            last_code = code.to_string();

            // Execute the code
            let result = executor.execute(code).await;

            if result.success {
                // Validate the output
                let score = self.validator.validate(result.output());

                if score.value >= 1.0 || attempt == max_attempts - 1 {
                    return ProgramResult {
                        output: result.stdout.trim().to_string(),
                        code: if include_code {
                            last_code
                        } else {
                            String::new()
                        },
                        attempts: attempt + 1,
                        tokens: total_tokens,
                        success: true,
                        error: None,
                    };
                }

                // Validation failed, try again with feedback
                last_error = score.feedback_str().map(|s| s.to_string());
            } else {
                // Execution failed, try again with error
                last_error = Some(result.stderr.clone());
            }
        }

        // All attempts failed
        ProgramResult {
            output: String::new(),
            code: if include_code {
                last_code
            } else {
                String::new()
            },
            attempts: max_attempts,
            tokens: total_tokens,
            success: false,
            error: last_error,
        }
    }

    /// Build the prompt for code generation.
    fn build_prompt(&self, previous_error: Option<&str>) -> String {
        let mut prompt = format!(
            "Write {} code to solve the following problem:\n\n{}\n\n",
            self.config.language, self.problem
        );

        if let Some(error) = previous_error {
            prompt.push_str(&format!(
                "Previous attempt failed with error:\n```\n{}\n```\n\n\
                 Please fix the code and try again.\n\n",
                error
            ));
        }

        prompt.push_str(&format!(
            "Provide your solution in a code block:\n```{}\n",
            self.config.language
        ));

        prompt
    }

    /// Extract code from the LLM response.
    fn extract_code<'b>(&self, response: &'b str) -> &'b str {
        // Find code block with language tag
        let lang_marker = format!("```{}", self.config.language);
        if let Some(start) = response.find(&lang_marker) {
            let code_start = start + lang_marker.len();
            // Skip to newline
            let code_start = response[code_start..]
                .find('\n')
                .map(|i| code_start + i + 1)
                .unwrap_or(code_start);

            if let Some(end) = response[code_start..].find("```") {
                return &response[code_start..code_start + end];
            }
        }

        // Try generic code block
        if let Some(start) = response.find("```") {
            let code_start = start + 3;
            // Skip to newline (past language marker if any)
            let code_start = response[code_start..]
                .find('\n')
                .map(|i| code_start + i + 1)
                .unwrap_or(code_start);

            if let Some(end) = response[code_start..].find("```") {
                return &response[code_start..code_start + end];
            }
        }

        // No code block found, return whole response
        response.trim()
    }
}

/// Result of Program of Thought execution.
#[derive(Debug, Clone)]
pub struct ProgramResult {
    /// The output from executing the code.
    pub output: String,
    /// The generated code (if include_code is true).
    pub code: String,
    /// Number of attempts made.
    pub attempts: usize,
    /// Total tokens used.
    pub tokens: u32,
    /// Whether execution succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

impl ProgramResult {
    /// Get the generated code.
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Check if the program ran successfully.
    pub fn is_success(&self) -> bool {
        self.success
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::executor::bash_executor;
    use crate::recursive::llm::MockLlm;

    #[test]
    fn test_program_basic() {
        let llm = MockLlm::new(|_, _| "Here's the code:\n```bash\necho 42\n```".to_string());

        let result = program(&llm, "Print 42").executor(bash_executor()).go();

        assert!(result.success);
        assert!(result.output.contains("42"));
        assert!(result.code.contains("echo"));
    }

    #[test]
    fn test_program_no_executor() {
        let llm = MockLlm::new(|_, _| "print(42)".to_string());

        let result = program(&llm, "Print 42").go();

        assert!(!result.success);
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("No executor"));
    }

    #[test]
    fn test_program_extract_code() {
        let llm = MockLlm::new(|_, _| String::new());
        let builder = program(&llm, "test");

        // Test with language-specific code block
        let code =
            builder.extract_code("Here's the solution:\n```python\nprint('hello')\n```\nDone!");
        assert_eq!(code, "print('hello')\n");

        // Test with generic code block
        let code = builder.extract_code("```\necho test\n```");
        assert_eq!(code, "echo test\n");

        // Test without code block
        let code = builder.extract_code("just plain text");
        assert_eq!(code, "just plain text");
    }

    #[test]
    fn test_program_with_error_retry() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = AtomicUsize::new(0);
        let llm = MockLlm::new(move |_prompt, _| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => "```bash\nexit 1\n```".to_string(),       // Fails
                _ => "```bash\necho success\n```".to_string(), // Succeeds
            }
        });

        let result = program(&llm, "Succeed")
            .executor(bash_executor())
            .max_attempts(3)
            .go();

        assert!(result.success);
        assert_eq!(result.attempts, 2);
    }

    #[test]
    fn test_program_no_code() {
        let llm = MockLlm::new(|_, _| "```bash\necho test\n```".to_string());

        let result = program(&llm, "Test")
            .executor(bash_executor())
            .no_code()
            .go();

        assert!(result.success);
        assert!(result.code.is_empty());
    }

    #[test]
    fn test_program_config() {
        let llm = MockLlm::new(|_, _| String::new());

        let builder = program(&llm, "test").max_attempts(5).language("python");

        assert_eq!(builder.config.max_attempts, 5);
        assert_eq!(builder.config.language, "python");
    }

    #[test]
    fn test_program_result_methods() {
        let result = ProgramResult {
            output: "42".to_string(),
            code: "print(42)".to_string(),
            attempts: 1,
            tokens: 100,
            success: true,
            error: None,
        };

        assert!(result.is_success());
        assert_eq!(result.code(), "print(42)");
    }
}
