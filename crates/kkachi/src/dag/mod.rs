// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! DAG-based computation graph for composable pipelines
//!
//! This module provides the core abstractions for building computation graphs:
//!
//! - [`Node`]: Trait for computation nodes with GAT-based async
//! - [`Graph`]: Builder for composing nodes into graphs
//! - Combinators: [`Chain`], [`Par`], [`Branch`], [`Fanout`], [`Feedback`]
//!
//! ## Design Principles
//!
//! - **Zero-cost abstractions**: GATs enable async without boxing
//! - **Compile-time composition**: Graph structure known at compile time
//! - **Type-safe**: Input/output types are checked at compile time
//!
//! ## Example
//!
//! ```ignore
//! use kkachi::dag::{Node, Graph};
//!
//! // Create a pipeline: A >> B >> C
//! let pipeline = Graph::new(node_a)
//!     .then(node_b)
//!     .then(node_c);
//!
//! // Execute
//! let result = pipeline.execute(input).await?;
//! ```

mod combinators;
mod graph;
mod node;

pub use combinators::{Branch, Chain, Fanout, Feedback, Filter, Map, Par};
pub use graph::Graph;
pub use node::Node;

/// Helper macro for building graphs ergonomically.
///
/// # Examples
///
/// ```ignore
/// // Simple chain: chain!(a, b, c)
/// let pipeline = chain!(a, b, c);
///
/// // Parallel execution: par!(a, b)
/// let parallel = par!(a, b);
/// ```
#[macro_export]
macro_rules! chain {
    // Two nodes
    ($a:expr, $b:expr) => {
        $crate::dag::Graph::new($a).then($b)
    };
    // Three or more nodes
    ($a:expr, $b:expr, $($rest:expr),+) => {
        chain!($crate::dag::Graph::new($a).then($b), $($rest),+)
    };
    // Single node
    ($a:expr) => {
        $crate::dag::Graph::new($a)
    };
}

/// Macro for parallel node execution.
#[macro_export]
macro_rules! par {
    ($a:expr, $b:expr) => {
        $crate::dag::Graph::new($a).par($b)
    };
}
