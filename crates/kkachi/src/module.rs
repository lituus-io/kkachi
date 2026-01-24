// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Module trait and base implementations
//!
//! This module provides the core [`Module`] trait which uses Generic Associated Types (GATs)
//! for zero-cost async without boxing.
//!
//! ## GAT-based Design
//!
//! Unlike `async_trait` which requires heap allocation, the GAT-based approach
//! allows each implementation to specify its own future type, enabling:
//!
//! - Zero heap allocation for async execution
//! - Compiler inlining across await points
//! - Type-level Send bounds checking
//!
//! ## Example
//!
//! ```ignore
//! struct MyModule;
//!
//! impl Module for MyModule {
//!     type ForwardFut<'a> = impl Future<Output = Result<Prediction<'a>>> + Send + 'a;
//!
//!     fn forward<'a>(&'a self, inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
//!         async move {
//!             // Process inputs
//!             Ok(Prediction::new())
//!         }
//!     }
//! }
//! ```

use crate::error::Result;
use crate::intern::Sym;
use crate::prediction::Prediction;
use crate::types::Inputs;
use std::future::Future;

/// Core trait for executable modules.
///
/// This is the fundamental abstraction for DSPy-style modules in Kkachi.
/// Each module takes structured inputs and produces a prediction.
///
/// ## Implementation
///
/// Implementations must specify a future type using GATs. The simplest
/// approach is to use `impl Future` in nightly Rust, or return a concrete
/// future type like `std::future::Ready` for sync operations.
pub trait Module: Send + Sync {
    /// Future type returned by forward.
    ///
    /// This GAT allows zero-cost async by letting each implementation
    /// specify its own future type.
    type ForwardFut<'a>: Future<Output = Result<Prediction<'a>>> + Send + 'a
    where
        Self: 'a;

    /// Execute the module with given inputs.
    ///
    /// This is the main entry point for module execution. Implementations
    /// should process the inputs and return a prediction.
    fn forward<'a>(&'a self, inputs: Inputs<'a>) -> Self::ForwardFut<'a>;

    /// Get module name for debugging and tracing.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get module identifier as an interned symbol.
    fn id(&self) -> Sym {
        crate::intern::sym(self.name())
    }
}

/// Base module implementation with common functionality.
#[derive(Debug, Clone)]
pub struct BaseModule {
    /// Module name
    pub name: String,
}

impl BaseModule {
    /// Create a new base module.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// A module that wraps a function.
///
/// This allows using closures and functions as modules without
/// defining a new struct.
pub struct FnModule<F> {
    f: F,
    name: &'static str,
}

impl<F> FnModule<F> {
    /// Create a new function module.
    pub const fn new(name: &'static str, f: F) -> Self {
        Self { f, name }
    }
}

impl<F> Clone for FnModule<F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            name: self.name,
        }
    }
}

/// Extension trait for Module providing additional combinators.
pub trait ModuleExt: Module + Sized {
    /// Chain another module after this one.
    fn then<M: Module>(self, next: M) -> ChainedModule<Self, M> {
        ChainedModule {
            first: self,
            second: next,
        }
    }

    /// Map the output with a function.
    fn map<F, O>(self, f: F) -> MappedModule<Self, F>
    where
        F: Fn(Prediction<'_>) -> O + Send + Sync,
    {
        MappedModule { inner: self, f }
    }
}

impl<M: Module> ModuleExt for M {}

/// Two modules chained sequentially.
pub struct ChainedModule<A, B> {
    first: A,
    second: B,
}

impl<A, B> Clone for ChainedModule<A, B>
where
    A: Clone,
    B: Clone,
{
    fn clone(&self) -> Self {
        Self {
            first: self.first.clone(),
            second: self.second.clone(),
        }
    }
}

/// A module with mapped output.
pub struct MappedModule<M, F> {
    inner: M,
    f: F,
}

impl<M, F> Clone for MappedModule<M, F>
where
    M: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            f: self.f.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Inputs;

    struct TestModule;

    impl Module for TestModule {
        type ForwardFut<'a> = std::future::Ready<Result<Prediction<'a>>>;

        fn forward<'a>(&'a self, _inputs: Inputs<'a>) -> Self::ForwardFut<'a> {
            std::future::ready(Ok(Prediction::new()))
        }
    }

    #[tokio::test]
    async fn test_module_execution() {
        let module = TestModule;
        let inputs = Inputs::new();
        let result = module.forward(inputs).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_module_name() {
        let module = TestModule;
        assert!(module.name().contains("TestModule"));
    }

    #[test]
    fn test_base_module() {
        let base = BaseModule::new("my_module");
        assert_eq!(base.name, "my_module");
    }
}
