// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Program of Thought module
//!
//! Implements the Program of Thought (PoT) strategy where the LM generates
//! code to solve problems, which is then executed to get the answer.
//!
//! ## Zero-Copy Design
//!
//! - Code and results use `StrView<'a>`
//! - Executor outputs to shared buffer
//! - No string allocations in hot path

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

/// Code executor trait.
pub trait CodeExecutor: Send + Sync {
    /// Execute code and return the result.
    ///
    /// The output is written to the provided buffer.
    fn execute<'a>(
        &'a self,
        code: StrView<'a>,
        output_buffer: &'a mut Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<ExecutionResult<'a>>> + Send + 'a>>;

    /// Language supported by this executor.
    fn language(&self) -> &str;
}

/// Result of code execution.
pub struct ExecutionResult<'a> {
    /// Output/result of execution
    pub output: StrView<'a>,
    /// Whether execution succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<StrView<'a>>,
}

/// ProgramOfThought configuration.
#[derive(Clone, Copy)]
pub struct PoTConfig {
    /// Maximum code generation attempts
    pub max_attempts: u8,
    /// Whether to include code in output
    pub include_code: bool,
    /// Language hint for code generation
    pub language: &'static str,
}

impl Default for PoTConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            include_code: true,
            language: "python",
        }
    }
}

/// Program of Thought module.
///
/// Generates code to solve problems and executes it.
///
/// ## Example
///
/// ```ignore
/// let executor = PythonExecutor::new();
/// let pot = ProgramOfThought::new(&signature, &executor);
/// let result = pot_with_lm(&pot, &inputs, &lm, &mut buffer).await?;
/// ```
pub struct ProgramOfThought<'sig, 'exec, E: CodeExecutor> {
    /// The signature defining inputs/outputs
    signature: &'sig Signature<'sig>,
    /// Code executor
    executor: &'exec E,
    /// Configuration
    config: PoTConfig,
    /// Field symbols (reserved for future optimization)
    #[allow(dead_code)]
    code_sym: Sym,
    #[allow(dead_code)]
    result_sym: Sym,
}

impl<'sig, 'exec, E: CodeExecutor> ProgramOfThought<'sig, 'exec, E> {
    /// Create a new ProgramOfThought module.
    pub fn new(signature: &'sig Signature<'sig>, executor: &'exec E) -> Self {
        Self {
            signature,
            executor,
            config: PoTConfig::default(),
            code_sym: sym("code"),
            result_sym: sym("result"),
        }
    }

    /// Configure max attempts.
    pub fn with_max_attempts(mut self, n: u8) -> Self {
        self.config.max_attempts = n;
        self
    }

    /// Configure code inclusion.
    pub fn with_code_output(mut self, include: bool) -> Self {
        self.config.include_code = include;
        self
    }

    /// Configure language.
    pub fn with_language(mut self, lang: &'static str) -> Self {
        self.config.language = lang;
        self
    }

    /// Get the signature.
    #[inline]
    pub fn signature(&self) -> &'sig Signature<'sig> {
        self.signature
    }

    /// Get the executor.
    #[inline]
    pub fn executor(&self) -> &'exec E {
        self.executor
    }

    /// Build the code generation prompt.
    pub fn build_prompt_into<'buf>(
        &self,
        inputs: &Inputs<'_>,
        previous_error: Option<&str>,
        buffer: &'buf mut Vec<u8>,
    ) -> StrView<'buf> {
        buffer.clear();

        // Instructions
        buffer.extend_from_slice(self.signature.instructions.as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // PoT instructions
        buffer.extend_from_slice(
            b"Solve the following problem by writing code.\n\
              Write the code in a code block with the language specified.\n\
              The code should print the final answer.\n\n",
        );

        // Language hint
        buffer.extend_from_slice(b"Language: ");
        buffer.extend_from_slice(self.config.language.as_bytes());
        buffer.extend_from_slice(b"\n\n");

        // Previous error context (for retry)
        if let Some(error) = previous_error {
            buffer.extend_from_slice(b"Previous attempt failed with error:\n");
            buffer.extend_from_slice(error.as_bytes());
            buffer.extend_from_slice(b"\n\nPlease fix the error and try again.\n\n");
        }

        // Current input
        buffer.extend_from_slice(b"Problem: ");
        for field in &self.signature.input_fields {
            if let Some(value) = inputs.get(&field.name) {
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b' ');
            }
        }
        buffer.extend_from_slice(b"\n\nCode:\n```");
        buffer.extend_from_slice(self.config.language.as_bytes());
        buffer.push(b'\n');

        // SAFETY: We only write valid UTF-8
        unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) }
    }

    /// Extract code from LM response.
    pub fn extract_code(&self, response: &str) -> Option<Range<usize>> {
        // Find code block
        let start_marker = format!("```{}", self.config.language);
        let code_start = response.find(&start_marker)?;
        let code_start = code_start + start_marker.len();

        // Skip any newline after language tag
        let code_start = if response.as_bytes().get(code_start) == Some(&b'\n') {
            code_start + 1
        } else {
            code_start
        };

        // Find end of code block
        let code_end = response[code_start..].find("```")?;
        let code_end = code_start + code_end;

        Some(code_start..code_end)
    }
}

/// Execute ProgramOfThought with an LM client.
///
/// Generates code and executes it. This simplified version runs a single attempt.
/// For retry logic, use pot_attempt() in a loop.
pub async fn pot_with_lm<'a, L, E>(
    pot: &ProgramOfThought<'_, '_, E>,
    inputs: &Inputs<'_>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
    exec_buffer: &'a mut Vec<u8>,
) -> Result<PoTOutput<'a>>
where
    L: LMClient,
    E: CodeExecutor,
{
    // Build prompt (no previous error for initial attempt)
    let prompt = pot.build_prompt_into(inputs, None, prompt_buffer);

    // Generate code
    let output = lm.generate(prompt).await?;
    let response_text = output.text()?.as_str();

    // Extract code
    let code_range = pot
        .extract_code(response_text)
        .ok_or_else(|| crate::error::Error::module("Failed to extract code from response"))?;

    let code = &response_text[code_range.clone()];

    // Execute code
    exec_buffer.clear();
    let exec_result = pot
        .executor
        .execute(StrView::new(code), exec_buffer)
        .await?;

    if exec_result.success {
        Ok(PoTOutput {
            buffer: output.buffer,
            code_range: Some(code_range),
            result: Some(exec_result.output.as_str().to_string()),
            attempts: 1,
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
        })
    } else {
        Err(crate::error::Error::module(format!(
            "Code execution failed: {}",
            exec_result
                .error
                .map(|e| e.as_str().to_string())
                .unwrap_or_default()
        )))
    }
}

/// Execute a single attempt with optional previous error context.
pub async fn pot_attempt<'a, L, E>(
    pot: &ProgramOfThought<'_, '_, E>,
    inputs: &Inputs<'_>,
    previous_error: Option<&str>,
    lm: &'a L,
    prompt_buffer: &'a mut Vec<u8>,
    exec_buffer: &'a mut Vec<u8>,
) -> Result<PoTOutput<'a>>
where
    L: LMClient,
    E: CodeExecutor,
{
    // Build prompt
    let prompt = pot.build_prompt_into(inputs, previous_error, prompt_buffer);

    // Generate code
    let output = lm.generate(prompt).await?;
    let response_text = output.text()?.as_str();

    // Extract code
    let code_range = pot
        .extract_code(response_text)
        .ok_or_else(|| crate::error::Error::module("Failed to extract code from response"))?;

    let code = &response_text[code_range.clone()];

    // Execute code
    exec_buffer.clear();
    let exec_result = pot
        .executor
        .execute(StrView::new(code), exec_buffer)
        .await?;

    if exec_result.success {
        Ok(PoTOutput {
            buffer: output.buffer,
            code_range: Some(code_range),
            result: Some(exec_result.output.as_str().to_string()),
            attempts: 1,
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
        })
    } else {
        Err(crate::error::Error::module(format!(
            "Code execution failed: {}",
            exec_result
                .error
                .map(|e| e.as_str().to_string())
                .unwrap_or_default()
        )))
    }
}

/// Zero-copy PoT output.
pub struct PoTOutput<'a> {
    /// Response buffer containing code
    pub buffer: BufferView<'a>,
    /// Range for the generated code
    pub code_range: Option<Range<usize>>,
    /// Execution result.
    pub result: Option<String>,
    /// Number of attempts.
    pub attempts: u8,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    pub completion_tokens: u32,
}

impl<'a> PoTOutput<'a> {
    /// Get the generated code.
    pub fn code(&self) -> Option<StrView<'a>> {
        let range = self.code_range.as_ref()?;
        let text = self.buffer.as_str().ok()?;
        Some(StrView::new(&text[range.clone()]))
    }

    /// Get the execution result.
    pub fn result(&self) -> Option<&str> {
        self.result.as_deref()
    }
}

impl<E: CodeExecutor> Module for ProgramOfThought<'_, '_, E> {
    type ForwardFut<'a>
        = std::future::Ready<Result<Prediction<'a>>>
    where
        Self: 'a;

    fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
        std::future::ready(Err(crate::error::Error::module(
            "Use pot_with_lm() instead of forward() for zero-copy execution",
        )))
    }

    fn name(&self) -> &str {
        "ProgramOfThought"
    }

    fn id(&self) -> Sym {
        sym("program_of_thought")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockPythonExecutor;

    impl CodeExecutor for MockPythonExecutor {
        fn execute<'a>(
            &'a self,
            _code: StrView<'a>,
            output_buffer: &'a mut Vec<u8>,
        ) -> Pin<Box<dyn Future<Output = Result<ExecutionResult<'a>>> + Send + 'a>> {
            Box::pin(async move {
                output_buffer.clear();
                output_buffer.extend_from_slice(b"42");

                Ok(ExecutionResult {
                    output: unsafe {
                        StrView::from_raw_parts(output_buffer.as_ptr(), output_buffer.len())
                    },
                    success: true,
                    error: None,
                })
            })
        }

        fn language(&self) -> &str {
            "python"
        }
    }

    #[test]
    fn test_pot_creation() {
        let sig = Signature::parse("question -> answer").unwrap();
        let exec = MockPythonExecutor;
        let pot = ProgramOfThought::new(&sig, &exec);

        assert_eq!(pot.name(), "ProgramOfThought");
        assert_eq!(pot.executor().language(), "python");
    }

    #[test]
    fn test_extract_code() {
        let sig = Signature::parse("question -> answer").unwrap();
        let exec = MockPythonExecutor;
        let pot = ProgramOfThought::new(&sig, &exec);

        let response = "Here's the code:\n```python\nprint(2 + 2)\n```\n";
        let range = pot.extract_code(response);

        assert!(range.is_some());
        let code = &response[range.unwrap()];
        assert_eq!(code, "print(2 + 2)\n");
    }

    #[test]
    fn test_extract_code_no_block() {
        let sig = Signature::parse("question -> answer").unwrap();
        let exec = MockPythonExecutor;
        let pot = ProgramOfThought::new(&sig, &exec);

        let response = "Here's some text without code";
        let range = pot.extract_code(response);

        assert!(range.is_none());
    }
}
