// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Type-erased LLM wrapper using dynamic dispatch.
//!
//! [`BoxedLlm`] wraps any `Llm` implementation behind `dyn ErasedLlm`,
//! boxing the returned future. This is only needed at FFI boundaries
//! (e.g. Python bindings) where type erasure is required. Pure Rust
//! code should use generics (`L: Llm`) to avoid the boxing overhead.

use crate::error::Result;
use crate::recursive::llm::{Llm, LmOutput};
use std::future::Future;
use std::pin::Pin;

/// Object-safe erased trait for LLM implementations.
///
/// This bridges the GAT-based [`Llm`] trait to dynamic dispatch by
/// returning `Pin<Box<dyn Future>>` instead of associated types.
pub(crate) trait ErasedLlm: Send + Sync {
    fn generate_erased<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<LmOutput>> + Send + 'a>>;

    #[allow(dead_code)]
    fn model_name_erased(&self) -> &str;

    fn max_context_erased(&self) -> usize;
}

impl<L: Llm> ErasedLlm for L {
    fn generate_erased<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<LmOutput>> + Send + 'a>> {
        Box::pin(self.generate(prompt, context, feedback))
    }

    fn model_name_erased(&self) -> &str {
        self.model_name()
    }

    fn max_context_erased(&self) -> usize {
        self.max_context()
    }
}

/// Wrapper for boxed async LLM implementations.
///
/// This allows using dynamic dispatch when needed, at the cost of
/// boxing the future. Use this only when you need runtime polymorphism
/// (e.g. at the Python FFI boundary).
///
/// Owns the LLM directly (no `Arc`), stores the model name as an owned
/// `String` (no `Box::leak`).
pub struct BoxedLlm {
    inner: Box<dyn ErasedLlm>,
    name: String,
}

impl BoxedLlm {
    /// Create a new boxed LLM from any Llm implementation.
    pub fn new<L: Llm + 'static>(llm: L) -> Self {
        let name = llm.model_name().to_string();
        Self {
            inner: Box::new(llm),
            name,
        }
    }
}

impl Llm for BoxedLlm {
    type GenerateFut<'a>
        = Pin<Box<dyn Future<Output = Result<LmOutput>> + Send + 'a>>
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        self.inner.generate_erased(prompt, context, feedback)
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn max_context(&self) -> usize {
        self.inner.max_context_erased()
    }
}
