// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Core Node trait for computation graph nodes
//!
//! This module contains infrastructure types that may not be used internally
//! but are provided for API consumers building custom graphs.

#![allow(dead_code)]

use crate::error::Result;
use crate::intern::Sym;
use std::future::Future;

/// A node in the computation DAG.
///
/// This is the fundamental building block for computation graphs.
/// Each node has:
/// - An input type
/// - An output type
/// - An async execution method
///
/// ## GAT-based Design
///
/// The `Fut` associated type uses Generic Associated Types (GATs) to enable
/// zero-cost async without boxing. This means:
/// - No heap allocation for futures
/// - Compiler can inline and optimize across await points
/// - Type-level guarantee of Send + 'a lifetime bounds
///
/// ## Example
///
/// ```ignore
/// struct MyNode;
///
/// impl Node for MyNode {
///     type Input<'a> = &'a str;
///     type Output<'a> = String;
///     type Fut<'a> = impl Future<Output = Result<String>> + Send + 'a;
///
///     fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
///         async move {
///             Ok(input.to_uppercase())
///         }
///     }
///
///     fn id(&self) -> Sym {
///         sym("my_node")
///     }
/// }
/// ```
pub trait Node: Send + Sync {
    /// Input type for this node.
    ///
    /// Can be a tuple `(A, B)` for nodes that combine multiple inputs.
    type Input<'a>
    where
        Self: 'a;

    /// Output type for this node.
    type Output<'a>
    where
        Self: 'a;

    /// Future type returned by `call`.
    ///
    /// This GAT allows each implementation to specify its own future type,
    /// enabling zero-cost async without boxing.
    type Fut<'a>: Future<Output = Result<Self::Output<'a>>> + Send + 'a
    where
        Self: 'a;

    /// Execute the node with the given input.
    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a>;

    /// Get the node identifier for tracing and debugging.
    fn id(&self) -> Sym;
}

/// Extension trait for nodes that can be cloned.
pub trait CloneNode: Node + Clone {}

impl<T: Node + Clone> CloneNode for T {}

/// A boxed node for dynamic dispatch.
///
/// Use this when you need to store nodes of different types together,
/// at the cost of heap allocation and dynamic dispatch.
pub type BoxNode<'a, I, O> = Box<
    dyn for<'b> Node<Input<'b> = I, Output<'b> = O, Fut<'b> = BoxFuture<'b, O>> + Send + Sync + 'a,
>;

/// Boxed future type for dynamic dispatch.
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// Identity node that passes input through unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub struct Identity;

impl Node for Identity {
    type Input<'a> = ();
    type Output<'a> = ();
    type Fut<'a> = std::future::Ready<Result<()>>;

    fn call<'a>(&'a self, _input: Self::Input<'a>) -> Self::Fut<'a> {
        std::future::ready(Ok(()))
    }

    fn id(&self) -> Sym {
        crate::intern::sym("identity")
    }
}

/// A function wrapper that implements Node.
///
/// This allows using closures and functions as nodes.
pub struct FnNode<F, I, O>
where
    F: Fn(I) -> O + Send + Sync,
{
    f: F,
    name: Sym,
    _marker: std::marker::PhantomData<fn(I) -> O>,
}

impl<F, I, O> FnNode<F, I, O>
where
    F: Fn(I) -> O + Send + Sync,
{
    /// Create a new function node.
    pub fn new(name: &str, f: F) -> Self {
        Self {
            f,
            name: crate::intern::sym(name),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, I, O> Clone for FnNode<F, I, O>
where
    F: Fn(I) -> O + Send + Sync + Clone,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            name: self.name,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_node() {
        let node = Identity;
        assert_eq!(node.id().as_str(), "identity");
    }

    #[test]
    fn test_node_size() {
        // Identity should be zero-sized
        assert_eq!(std::mem::size_of::<Identity>(), 0);
    }
}
