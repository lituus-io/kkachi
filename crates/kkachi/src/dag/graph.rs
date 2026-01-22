// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Graph builder for composing nodes
//!
//! This module contains infrastructure types for graph composition.

#![allow(dead_code)]

use super::combinators::{Branch, Chain, Fanout, Feedback, Par};
use super::node::Node;
use crate::error::Result;
use crate::intern::Sym;
use std::future::Future;

/// A computation graph built from composable nodes.
///
/// `Graph` wraps a root node and provides methods for composing
/// it with other nodes to build complex pipelines.
///
/// ## Composition Methods
///
/// - [`then`](Graph::then): Sequential composition `A >> B`
/// - [`par`](Graph::par): Parallel composition `A & B`
/// - [`branch`](Graph::branch): Conditional `if cond then A else B`
/// - [`fanout`](Graph::fanout): Fan-out `A -> [B, C, D]`
/// - [`feedback`](Graph::feedback): Iteration with feedback loop
///
/// ## Example
///
/// ```ignore
/// let pipeline = Graph::new(preprocess)
///     .then(model)
///     .then(postprocess);
///
/// let result = pipeline.execute(input).await?;
/// ```
#[derive(Clone)]
pub struct Graph<N: Node> {
    root: N,
}

impl<N: Node> Graph<N> {
    /// Create a new graph with the given root node.
    #[inline]
    pub fn new(root: N) -> Self {
        Self { root }
    }

    /// Get a reference to the root node.
    #[inline]
    pub fn root(&self) -> &N {
        &self.root
    }

    /// Chain another node after this graph.
    ///
    /// The output of the current graph becomes the input to `next`.
    ///
    /// # Type Requirements
    ///
    /// `M::Input<'a>` must be constructable from `N::Output<'a>` for all lifetimes `'a`.
    #[inline]
    pub fn then<M>(self, next: M) -> Graph<Chain<N, M>>
    where
        M: Node,
        for<'a> M::Input<'a>: From<N::Output<'a>>,
    {
        Graph {
            root: Chain::new(self.root, next),
        }
    }

    /// Run another node in parallel with this graph.
    ///
    /// Both nodes receive the same input and their outputs are combined
    /// into a tuple.
    #[inline]
    pub fn par<M>(self, other: M) -> Graph<Par<N, M>>
    where
        M: Node,
        for<'a> N::Input<'a>: Clone,
        for<'a> M::Input<'a>: From<N::Input<'a>>,
        for<'a> N::Output<'a>: Send,
        for<'a> M::Output<'a>: Send,
    {
        Graph {
            root: Par::new(self.root, other),
        }
    }

    /// Conditional branching based on the output.
    ///
    /// If `cond` returns true, executes `then_node`, otherwise `else_node`.
    #[inline]
    pub fn branch<C, T, F>(self, cond: C, then_node: T, else_node: F) -> Graph<Branch<N, C, T, F>>
    where
        C: Fn(&N::Output<'_>) -> bool + Send + Sync,
        T: Node,
        F: Node,
        for<'a> T::Input<'a>: From<N::Output<'a>>,
        for<'a> F::Input<'a>: From<N::Output<'a>>,
        for<'a> T::Output<'a>: Into<F::Output<'a>>,
    {
        Graph {
            root: Branch::new(self.root, cond, then_node, else_node),
        }
    }

    /// Fan-out to multiple nodes.
    ///
    /// The same input is sent to all target nodes, which run in parallel.
    #[inline]
    pub fn fanout<M>(self, targets: M) -> Graph<Fanout<N, M>>
    where
        M: Node,
        for<'a> M::Input<'a>: From<N::Output<'a>>,
    {
        Graph {
            root: Fanout::new(self.root, targets),
        }
    }

    /// Create a feedback loop for iterative optimization.
    ///
    /// The node is executed repeatedly until `should_continue` returns false
    /// or `max_iters` is reached.
    #[inline]
    pub fn feedback<F>(self, should_continue: F, max_iters: usize) -> Graph<Feedback<N, F>>
    where
        F: Fn(&N::Output<'_>, usize) -> bool + Send + Sync,
        for<'a> N::Input<'a>: From<N::Output<'a>> + Clone,
    {
        Graph {
            root: Feedback::new(self.root, should_continue, max_iters),
        }
    }

    /// Execute the graph with the given input.
    #[inline]
    pub fn execute<'a>(&'a self, input: N::Input<'a>) -> N::Fut<'a> {
        self.root.call(input)
    }
}

impl<N: Node> Node for Graph<N> {
    type Input<'a>
        = N::Input<'a>
    where
        Self: 'a;
    type Output<'a>
        = N::Output<'a>
    where
        Self: 'a;
    type Fut<'a>
        = N::Fut<'a>
    where
        Self: 'a;

    #[inline]
    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
        self.root.call(input)
    }

    fn id(&self) -> Sym {
        self.root.id()
    }
}

/// Builder for constructing graphs from a sequence of operations.
pub struct GraphBuilder<N: Node> {
    graph: Graph<N>,
}

impl<N: Node> GraphBuilder<N> {
    /// Start building a graph with the given root node.
    pub fn start(root: N) -> Self {
        Self {
            graph: Graph::new(root),
        }
    }

    /// Add a sequential step.
    pub fn step<M>(self, next: M) -> GraphBuilder<Chain<N, M>>
    where
        M: Node,
        for<'a> M::Input<'a>: From<N::Output<'a>>,
    {
        GraphBuilder {
            graph: self.graph.then(next),
        }
    }

    /// Finish building and return the graph.
    pub fn build(self) -> Graph<N> {
        self.graph
    }
}

/// Type alias for graphs with boxed futures (for dynamic dispatch).
pub type DynGraph<'a, I, O> = Graph<BoxedNode<'a, I, O>>;

/// A node wrapper that boxes the future for dynamic dispatch.
pub struct BoxedNode<'a, I, O> {
    inner: Box<dyn DynNode<I, O> + Send + Sync + 'a>,
}

/// Helper trait for dynamic node dispatch.
trait DynNode<I, O> {
    fn call_boxed<'a>(
        &'a self,
        input: I,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<O>> + Send + 'a>>;
    fn id(&self) -> Sym;
}

impl<'a, I, O> Node for BoxedNode<'a, I, O>
where
    I: 'a,
    O: 'a,
{
    type Input<'b>
        = I
    where
        Self: 'b;
    type Output<'b>
        = O
    where
        Self: 'b;
    type Fut<'b>
        = std::pin::Pin<Box<dyn Future<Output = Result<O>> + Send + 'b>>
    where
        Self: 'b;

    fn call<'b>(&'b self, input: Self::Input<'b>) -> Self::Fut<'b> {
        self.inner.call_boxed(input)
    }

    fn id(&self) -> Sym {
        self.inner.id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::sym;

    struct AddOne;

    impl Node for AddOne {
        type Input<'a> = i32;
        type Output<'a> = i32;
        type Fut<'a> = std::future::Ready<Result<i32>>;

        fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
            std::future::ready(Ok(input + 1))
        }

        fn id(&self) -> Sym {
            sym("add_one")
        }
    }

    struct Double;

    impl Node for Double {
        type Input<'a> = i32;
        type Output<'a> = i32;
        type Fut<'a> = std::future::Ready<Result<i32>>;

        fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
            std::future::ready(Ok(input * 2))
        }

        fn id(&self) -> Sym {
            sym("double")
        }
    }

    #[tokio::test]
    async fn test_graph_execute() {
        let graph = Graph::new(AddOne);
        let result = graph.execute(5).await.unwrap();
        assert_eq!(result, 6);
    }

    #[tokio::test]
    async fn test_graph_chain() {
        let graph = Graph::new(AddOne).then(Double);
        let result = graph.execute(5).await.unwrap();
        assert_eq!(result, 12); // (5 + 1) * 2
    }

    #[tokio::test]
    async fn test_graph_multiple_chain() {
        let graph = Graph::new(AddOne).then(AddOne).then(Double);
        let result = graph.execute(5).await.unwrap();
        assert_eq!(result, 14); // ((5 + 1) + 1) * 2
    }

    #[test]
    fn test_graph_builder() {
        let graph = GraphBuilder::start(AddOne).step(Double).build();
        assert_eq!(graph.root().id().as_str(), "chain");
    }
}
