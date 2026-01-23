// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! LLM trait using Generic Associated Types (GATs).
//!
//! This module provides the [`Llm`] trait which defines the interface for
//! language model providers. Using GATs instead of `async_trait` allows
//! zero-cost async without boxing.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{MockLlm, Llm};
//!
//! // Create a mock LLM for testing
//! let llm = MockLlm::new(|prompt, _feedback| {
//!     format!("Generated response for: {}", prompt)
//! });
//! ```

use crate::error::{Error, Result};
use std::future::Future;
use std::pin::Pin;

/// Output from an LLM generation request.
#[derive(Debug, Clone)]
pub struct LmOutput {
    /// The generated text.
    pub text: String,
    /// Number of prompt tokens used.
    pub prompt_tokens: u32,
    /// Number of completion tokens generated.
    pub completion_tokens: u32,
}

impl LmOutput {
    /// Create a new LmOutput with just the text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            prompt_tokens: 0,
            completion_tokens: 0,
        }
    }

    /// Create a new LmOutput with token counts.
    pub fn with_tokens(text: impl Into<String>, prompt: u32, completion: u32) -> Self {
        Self {
            text: text.into(),
            prompt_tokens: prompt,
            completion_tokens: completion,
        }
    }

    /// Get the total token count.
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Trait for language model providers.
///
/// This trait uses GATs for zero-cost async without boxing. Implementations
/// can be synchronous (returning `Ready<T>`) or asynchronous (returning
/// custom futures).
///
/// # Type Parameters
///
/// The associated type `GenerateFut<'a>` represents the future returned by
/// `generate()`. Using GATs allows each implementation to define its own
/// future type without boxing.
pub trait Llm: Send + Sync {
    /// The future type returned by `generate()`.
    type GenerateFut<'a>: Future<Output = Result<LmOutput>> + Send + 'a
    where
        Self: 'a;

    /// Generate a response from the LLM.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The main prompt/question
    /// * `context` - Additional context (e.g., from RAG)
    /// * `feedback` - Optional feedback from previous iteration
    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a>;

    /// Get the model name for logging.
    fn model_name(&self) -> &str {
        "unknown"
    }

    /// Get the maximum context length.
    fn max_context(&self) -> usize {
        4096
    }
}

/// A mock LLM for testing and examples.
///
/// This LLM uses a closure to generate responses synchronously.
/// It's useful for testing refinement loops without actual API calls.
pub struct MockLlm<F>
where
    F: Fn(&str, Option<&str>) -> String + Send + Sync,
{
    generator: F,
    name: &'static str,
}

impl<F> MockLlm<F>
where
    F: Fn(&str, Option<&str>) -> String + Send + Sync,
{
    /// Create a new mock LLM with the given generator function.
    ///
    /// The function receives the prompt and optional feedback, and returns
    /// the generated text.
    pub fn new(generator: F) -> Self {
        Self {
            generator,
            name: "mock",
        }
    }

    /// Set a custom name for the mock LLM.
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
}

impl<F> Llm for MockLlm<F>
where
    F: Fn(&str, Option<&str>) -> String + Send + Sync,
{
    type GenerateFut<'a>
        = std::future::Ready<Result<LmOutput>>
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        _context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        let text = (self.generator)(prompt, feedback);
        std::future::ready(Ok(LmOutput::new(text)))
    }

    fn model_name(&self) -> &str {
        self.name
    }
}

/// A mock LLM that generates responses based on iteration count.
///
/// This is useful for testing refinement loops where the output should
/// improve over iterations.
pub struct IterativeMockLlm<F>
where
    F: Fn(u32, &str, Option<&str>) -> String + Send + Sync,
{
    generator: F,
    iteration: std::sync::atomic::AtomicU32,
    name: &'static str,
}

impl<F> IterativeMockLlm<F>
where
    F: Fn(u32, &str, Option<&str>) -> String + Send + Sync,
{
    /// Create a new iterative mock LLM.
    ///
    /// The generator function receives the iteration number (0-based),
    /// prompt, and optional feedback.
    pub fn new(generator: F) -> Self {
        Self {
            generator,
            iteration: std::sync::atomic::AtomicU32::new(0),
            name: "iterative_mock",
        }
    }

    /// Set a custom name for the mock LLM.
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    /// Reset the iteration counter.
    pub fn reset(&self) {
        self.iteration.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get the current iteration count.
    pub fn current_iteration(&self) -> u32 {
        self.iteration.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl<F> Llm for IterativeMockLlm<F>
where
    F: Fn(u32, &str, Option<&str>) -> String + Send + Sync,
{
    type GenerateFut<'a>
        = std::future::Ready<Result<LmOutput>>
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        _context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        let iteration = self
            .iteration
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let text = (self.generator)(iteration, prompt, feedback);
        std::future::ready(Ok(LmOutput::new(text)))
    }

    fn model_name(&self) -> &str {
        self.name
    }
}

/// An LLM that fails with a specific error.
///
/// Useful for testing error handling in the refinement loop.
#[derive(Debug, Clone)]
pub struct FailingLlm {
    message: String,
}

impl FailingLlm {
    /// Create a new failing LLM with the given error message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Llm for FailingLlm {
    type GenerateFut<'a>
        = std::future::Ready<Result<LmOutput>>
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        _prompt: &'a str,
        _context: &'a str,
        _feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        std::future::ready(Err(Error::module(&self.message)))
    }

    fn model_name(&self) -> &str {
        "failing"
    }
}

/// Wrapper for boxed async LLM implementations.
///
/// This allows using dynamic dispatch when needed, at the cost of
/// boxing the future. Use this only when you need runtime polymorphism.
pub struct BoxedLlm<'a> {
    generate_fn: Box<
        dyn Fn(
                &str,
                &str,
                Option<&str>,
            ) -> Pin<Box<dyn Future<Output = Result<LmOutput>> + Send + 'static>>
            + Send
            + Sync
            + 'a,
    >,
    name: &'static str,
}

impl<'a> BoxedLlm<'a> {
    /// Create a new boxed LLM from any Llm implementation.
    pub fn new<L: Llm + 'static>(llm: L) -> Self {
        let llm = std::sync::Arc::new(llm);
        let name = L::model_name(&*llm);
        // Note: We need to use a static name here
        let static_name: &'static str = Box::leak(name.to_string().into_boxed_str());
        Self {
            generate_fn: Box::new(move |prompt: &str, context: &str, feedback: Option<&str>| {
                let llm = llm.clone();
                let prompt = prompt.to_owned();
                let context = context.to_owned();
                let feedback = feedback.map(|s| s.to_owned());
                Box::pin(async move { llm.generate(&prompt, &context, feedback.as_deref()).await })
            }),
            name: static_name,
        }
    }
}

impl Llm for BoxedLlm<'_> {
    type GenerateFut<'b>
        = Pin<Box<dyn Future<Output = Result<LmOutput>> + Send + 'b>>
    where
        Self: 'b;

    fn generate<'b>(
        &'b self,
        prompt: &'b str,
        context: &'b str,
        feedback: Option<&'b str>,
    ) -> Self::GenerateFut<'b> {
        (self.generate_fn)(prompt, context, feedback)
    }

    fn model_name(&self) -> &str {
        self.name
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_llm() {
        let llm = MockLlm::new(|prompt, _| format!("Response: {}", prompt));

        let output = llm.generate("test prompt", "", None).await.unwrap();
        assert_eq!(output.text, "Response: test prompt");
    }

    #[tokio::test]
    async fn test_mock_llm_with_feedback() {
        let llm = MockLlm::new(|prompt, feedback| match feedback {
            Some(fb) => format!("Improved: {} (feedback: {})", prompt, fb),
            None => format!("Initial: {}", prompt),
        });

        let output = llm.generate("test", "", None).await.unwrap();
        assert!(output.text.starts_with("Initial:"));

        let output = llm.generate("test", "", Some("do better")).await.unwrap();
        assert!(output.text.starts_with("Improved:"));
        assert!(output.text.contains("do better"));
    }

    #[tokio::test]
    async fn test_iterative_mock_llm() {
        let llm = IterativeMockLlm::new(|iter, _prompt, _| match iter {
            0 => "first try".to_string(),
            1 => "second try".to_string(),
            _ => "final answer".to_string(),
        });

        let out1 = llm.generate("test", "", None).await.unwrap();
        assert_eq!(out1.text, "first try");

        let out2 = llm.generate("test", "", Some("improve")).await.unwrap();
        assert_eq!(out2.text, "second try");

        let out3 = llm.generate("test", "", Some("more")).await.unwrap();
        assert_eq!(out3.text, "final answer");
    }

    #[tokio::test]
    async fn test_failing_llm() {
        let llm = FailingLlm::new("intentional failure");

        let result = llm.generate("test", "", None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("intentional failure"));
    }

    #[test]
    fn test_lm_output() {
        let output = LmOutput::new("test");
        assert_eq!(output.text, "test");
        assert_eq!(output.total_tokens(), 0);

        let output = LmOutput::with_tokens("test", 10, 20);
        assert_eq!(output.total_tokens(), 30);
    }
}
