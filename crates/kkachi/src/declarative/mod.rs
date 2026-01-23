// Copyright 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Declarative Pipeline API
//!
//! This module provides the same API as the recursive module with additional
//! Jinja template support. In the simplified API, declarative is a thin
//! re-export of recursive.
//!
//! # Example
//!
//! ```rust,ignore
//! use kkachi::declarative::*;
//!
//! let llm = MockLlm::new(|prompt, _| "fn add(a: i32, b: i32) -> i32 { a + b }".to_string());
//!
//! // Define your own validator
//! let validator = cli("rustfmt").arg("--check")
//!     .then("rustc").args(&["--emit=metadata"]).required()
//!     .ext("rs");
//!
//! let result = refine(&llm, "question -> code")
//!     .validate(validator)
//!     .max_iter(5)
//!     .go();
//! ```

// Jinja template support
pub mod jinja;
pub use jinja::JinjaTemplate;

// Re-export minijinja::Value for template rendering
pub use minijinja::Value as JinjaValue;

// Re-export everything from recursive prelude
pub use crate::recursive::prelude::*;

// Re-export StrView for compatibility
pub use crate::str_view::StrView;
