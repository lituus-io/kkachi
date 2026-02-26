// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Tool abstraction for agent patterns.
//!
//! This module provides the [`Tool`] trait which defines the interface for
//! tools that can be used by agents.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::tool::{tool, Tool};
//!
//! // Create a tool from a closure
//! let calc = tool("calculator")
//!     .description("Perform arithmetic calculations")
//!     .execute(|input| Ok(format!("Result: {}", input)));
//! ```

use crate::error::Result;
use std::future::Future;
use std::pin::Pin;

/// Trait for tools that can be used by agents.
///
/// This trait is object-safe to allow storing multiple tools of different
/// types in an agent. For async execution, it uses boxed futures.
pub trait Tool: Send + Sync {
    /// Get the tool name.
    fn name(&self) -> &str;

    /// Get the tool description for prompts.
    fn description(&self) -> &str;

    /// Execute the tool with the given input.
    ///
    /// Returns a boxed future to maintain object safety.
    fn execute<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

/// Builder for creating tools from closures.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::tool::tool;
///
/// let search = tool("search")
///     .description("Search the web")
///     .execute(|query| Ok(format!("Results for: {}", query)));
/// ```
pub fn tool(name: &'static str) -> ToolBuilder {
    ToolBuilder::new(name)
}

/// Builder for constructing tools.
pub struct ToolBuilder {
    name: &'static str,
    description: &'static str,
}

impl ToolBuilder {
    /// Create a new tool builder with the given name.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            description: "",
        }
    }

    /// Set the tool description.
    pub fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }

    /// Build a tool with a synchronous executor.
    ///
    /// The executor receives the input string and returns a Result<String>.
    pub fn execute<F>(self, f: F) -> FnTool<F>
    where
        F: Fn(&str) -> Result<String> + Send + Sync,
    {
        FnTool {
            name: self.name,
            description: self.description,
            executor: f,
        }
    }

    /// Build a tool with an async executor.
    ///
    /// The executor receives the input string and returns a Future<Output = Result<String>>.
    pub fn execute_async<F, Fut>(self, f: F) -> AsyncFnTool<F>
    where
        F: Fn(String) -> Fut + Send + Sync,
        Fut: Future<Output = Result<String>> + Send,
    {
        AsyncFnTool {
            name: self.name,
            description: self.description,
            executor: f,
        }
    }
}

/// A tool created from a synchronous closure.
pub struct FnTool<F> {
    name: &'static str,
    description: &'static str,
    executor: F,
}

impl<F> Tool for FnTool<F>
where
    F: Fn(&str) -> Result<String> + Send + Sync,
{
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn execute<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        let result = (self.executor)(input);
        Box::pin(std::future::ready(result))
    }
}

/// A tool created from an async closure.
pub struct AsyncFnTool<F> {
    name: &'static str,
    description: &'static str,
    executor: F,
}

impl<F, Fut> Tool for AsyncFnTool<F>
where
    F: Fn(String) -> Fut + Send + Sync,
    Fut: Future<Output = Result<String>> + Send + 'static,
{
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn execute<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        let input_owned = input.to_owned();
        let fut = (self.executor)(input_owned);
        Box::pin(fut)
    }
}

/// A mock tool for testing that always returns the same response.
pub struct MockTool {
    name: &'static str,
    description: &'static str,
    response: &'static str,
}

impl MockTool {
    /// Create a new mock tool.
    pub fn new(name: &'static str, description: &'static str, response: &'static str) -> Self {
        Self {
            name,
            description,
            response,
        }
    }
}

impl Tool for MockTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn execute<'a>(
        &'a self,
        _input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(std::future::ready(Ok(self.response.to_string())))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fn_tool() {
        let calc = tool("calculator")
            .description("Perform calculations")
            .execute(|input| Ok(format!("Result: {}", input)));

        assert_eq!(calc.name(), "calculator");
        assert_eq!(calc.description(), "Perform calculations");

        let result = calc.execute("2 + 2").await.unwrap();
        assert_eq!(result, "Result: 2 + 2");
    }

    #[tokio::test]
    async fn test_mock_tool() {
        let t = MockTool::new("test", "A test tool", "mock response");

        assert_eq!(t.name(), "test");
        let result = t.execute("any input").await.unwrap();
        assert_eq!(result, "mock response");
    }

    #[tokio::test]
    async fn test_async_fn_tool() {
        let search = tool("search")
            .description("Search for information")
            .execute_async(|query: String| async move { Ok(format!("Found: {}", query)) });

        assert_eq!(search.name(), "search");
        let result = search.execute("rust async").await.unwrap();
        assert_eq!(result, "Found: rust async");
    }

    #[test]
    fn test_tool_is_object_safe() {
        // This test verifies that Tool is object-safe
        let mock = MockTool::new("test", "desc", "response");
        let _: &dyn Tool = &mock;
    }
}
