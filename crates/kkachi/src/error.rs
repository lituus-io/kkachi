// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Error types for Kkachi

use crate::intern::Sym;
use thiserror::Error;

/// Result type alias for Kkachi operations
pub type Result<T> = core::result::Result<T, Error>;

/// Optimization-specific error details.
///
/// This type captures errors that occur during prompt optimization,
/// distinct from library/infrastructure errors. These errors are
/// *expected* and contribute to the optimization process.
#[derive(Debug, Clone)]
pub struct OptimizationDetails {
    /// Final score achieved before failure.
    pub score: f64,
    /// Number of iterations completed.
    pub iterations: u32,
    /// Target threshold that wasn't met.
    pub threshold: f64,
    /// Feedback from the critic.
    pub feedback: Option<String>,
    /// Whether this was a convergence failure vs. other issue.
    pub convergence_failure: bool,
}

impl OptimizationDetails {
    /// Create new optimization details.
    pub fn new(score: f64, iterations: u32, threshold: f64) -> Self {
        Self {
            score,
            iterations,
            threshold,
            feedback: None,
            convergence_failure: true,
        }
    }

    /// Add feedback from the critic.
    pub fn with_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }

    /// Mark as non-convergence failure (e.g., max iterations reached).
    pub fn max_iterations_reached(mut self) -> Self {
        self.convergence_failure = false;
        self
    }

    /// Check if the optimization made progress.
    pub fn made_progress(&self) -> bool {
        self.score > 0.0 && self.iterations > 0
    }

    /// Get the gap between achieved score and threshold.
    pub fn score_gap(&self) -> f64 {
        self.threshold - self.score
    }
}

impl core::fmt::Display for OptimizationDetails {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "score={:.2} (threshold={:.2}), iterations={}",
            self.score, self.threshold, self.iterations
        )?;
        if let Some(ref fb) = self.feedback {
            write!(f, ", feedback: {}", fb)?;
        }
        Ok(())
    }
}

/// Main error type for Kkachi library
#[derive(Error, Debug)]
pub enum Error {
    /// Signature-related errors
    #[error("Signature error: {0}")]
    Signature(String),

    /// Field-related errors
    #[error("Field error: {0}")]
    Field(String),

    /// Module execution errors
    #[error("Module error: {0}")]
    Module(String),

    /// Prediction errors
    #[error("Prediction error: {0}")]
    Prediction(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Assertion failure
    #[error("Assertion failed for field {field:?}: {description}")]
    AssertionFailed {
        /// The field that failed assertion
        field: Sym,
        /// Description of the assertion
        description: &'static str,
    },

    /// Invalid pattern (regex)
    #[error("Invalid pattern: {0}")]
    InvalidPattern(String),

    /// I/O errors (only with std feature)
    #[cfg(feature = "std")]
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// UTF-8 encoding errors
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// Storage errors (DuckDB, etc.)
    #[cfg(any(feature = "storage", feature = "storage-bundled"))]
    #[error("Storage error: {0}")]
    Storage(String),

    /// Parse errors (for template parsing, etc.)
    #[error("Parse error: {0}")]
    Parse(String),

    /// Validation errors (for format validation, etc.)
    #[error("Validation error: {0}")]
    Validation(String),

    /// Optimization failure - prompt did not converge.
    ///
    /// This is DISTINCT from library errors. Optimization errors are
    /// *expected* outcomes that can inform the optimization process.
    /// Use `Error::is_optimization_error()` to check, and
    /// `Error::optimization_details()` to extract context.
    #[error("Optimization failed: {0}")]
    Optimization(OptimizationDetails),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Create a storage error
    #[cfg(any(feature = "storage", feature = "storage-bundled"))]
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage(msg.into())
    }

    /// Create a signature error
    pub fn signature(msg: impl Into<String>) -> Self {
        Self::Signature(msg.into())
    }

    /// Create a field error
    pub fn field(msg: impl Into<String>) -> Self {
        Self::Field(msg.into())
    }

    /// Create a module error
    pub fn module(msg: impl Into<String>) -> Self {
        Self::Module(msg.into())
    }

    /// Create a prediction error
    pub fn prediction(msg: impl Into<String>) -> Self {
        Self::Prediction(msg.into())
    }

    /// Create a parse error
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }

    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    /// Create an I/O error with a custom message
    #[cfg(feature = "std")]
    pub fn io(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Create an optimization error (prompt did not converge).
    ///
    /// Use this when optimization fails to meet the threshold,
    /// not for library/infrastructure errors.
    pub fn optimization(score: f64, iterations: u32, threshold: f64) -> Self {
        Self::Optimization(OptimizationDetails::new(score, iterations, threshold))
    }

    /// Create an optimization error with feedback.
    pub fn optimization_with_feedback(
        score: f64,
        iterations: u32,
        threshold: f64,
        feedback: impl Into<String>,
    ) -> Self {
        Self::Optimization(
            OptimizationDetails::new(score, iterations, threshold).with_feedback(feedback),
        )
    }

    // =========================================================================
    // Error Classification Methods
    // =========================================================================

    /// Check if this is an optimization error (prompt didn't converge).
    ///
    /// Optimization errors are *expected* and should inform the optimization
    /// process. They are distinct from library errors which indicate bugs
    /// or infrastructure issues.
    ///
    /// # Example
    ///
    /// ```ignore
    /// match result {
    ///     Ok(output) => handle_success(output),
    ///     Err(e) if e.is_optimization_error() => {
    ///         // Use the feedback to improve the prompt
    ///         let details = e.optimization_details().unwrap();
    ///         improve_prompt(details.feedback.as_deref());
    ///     }
    ///     Err(e) => {
    ///         // This is a library bug - report it
    ///         panic!("Library error: {}", e);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn is_optimization_error(&self) -> bool {
        matches!(self, Self::Optimization(_))
    }

    /// Check if this is a library/infrastructure error (not optimization).
    ///
    /// Library errors indicate bugs or configuration issues in kkachi itself,
    /// not failures in the optimization process.
    #[inline]
    pub fn is_library_error(&self) -> bool {
        !self.is_optimization_error()
    }

    /// Get optimization details if this is an optimization error.
    ///
    /// Returns `None` for library errors.
    #[inline]
    pub fn optimization_details(&self) -> Option<&OptimizationDetails> {
        match self {
            Self::Optimization(details) => Some(details),
            _ => None,
        }
    }

    /// Get mutable optimization details if this is an optimization error.
    #[inline]
    pub fn optimization_details_mut(&mut self) -> Option<&mut OptimizationDetails> {
        match self {
            Self::Optimization(details) => Some(details),
            _ => None,
        }
    }

    /// Check if this error can be recovered from (optimization errors can).
    #[inline]
    pub fn is_recoverable(&self) -> bool {
        self.is_optimization_error()
    }

    /// Get the error category for logging/metrics.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Signature(_) => "signature",
            Self::Field(_) => "field",
            Self::Module(_) => "module",
            Self::Prediction(_) => "prediction",
            Self::Serialization(_) => "serialization",
            Self::AssertionFailed { .. } => "assertion",
            Self::InvalidPattern(_) => "pattern",
            #[cfg(feature = "std")]
            Self::Io(_) => "io",
            Self::Json(_) => "json",
            Self::Utf8(_) => "utf8",
            #[cfg(any(feature = "storage", feature = "storage-bundled"))]
            Self::Storage(_) => "storage",
            Self::Parse(_) => "parse",
            Self::Validation(_) => "validation",
            Self::Optimization(_) => "optimization",
            Self::Other(_) => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_signature() {
        let err = Error::signature("invalid format");
        assert!(matches!(err, Error::Signature(_)));
        assert_eq!(err.to_string(), "Signature error: invalid format");
    }

    #[test]
    fn test_error_field() {
        let err = Error::field("missing field");
        assert!(matches!(err, Error::Field(_)));
        assert_eq!(err.to_string(), "Field error: missing field");
    }

    #[test]
    fn test_error_module() {
        let err = Error::module("execution failed");
        assert!(matches!(err, Error::Module(_)));
        assert_eq!(err.to_string(), "Module error: execution failed");
    }

    #[test]
    fn test_error_prediction() {
        let err = Error::prediction("parsing failed");
        assert!(matches!(err, Error::Prediction(_)));
        assert_eq!(err.to_string(), "Prediction error: parsing failed");
    }

    #[test]
    fn test_result_type() {
        let ok: Result<i32> = Ok(42);
        assert!(ok.is_ok());

        let err: Result<i32> = Err(Error::Other("failed".to_string()));
        assert!(err.is_err());
    }

    // =========================================================================
    // Optimization Error Tests
    // =========================================================================

    #[test]
    fn test_optimization_error_creation() {
        let err = Error::optimization(0.75, 5, 0.9);
        assert!(err.is_optimization_error());
        assert!(!err.is_library_error());
        assert!(err.is_recoverable());

        let details = err.optimization_details().unwrap();
        assert!((details.score - 0.75).abs() < 0.001);
        assert_eq!(details.iterations, 5);
        assert!((details.threshold - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_optimization_error_with_feedback() {
        let err = Error::optimization_with_feedback(0.6, 3, 0.8, "Missing return type annotation");

        let details = err.optimization_details().unwrap();
        assert_eq!(
            details.feedback.as_deref(),
            Some("Missing return type annotation")
        );
    }

    #[test]
    fn test_optimization_details_methods() {
        let details = OptimizationDetails::new(0.7, 4, 0.9)
            .with_feedback("Test feedback")
            .max_iterations_reached();

        assert!((details.score_gap() - 0.2).abs() < 0.001);
        assert!(details.made_progress());
        assert!(!details.convergence_failure);
    }

    #[test]
    fn test_optimization_details_display() {
        let details = OptimizationDetails::new(0.75, 5, 0.9).with_feedback("Needs docstring");

        let display = details.to_string();
        assert!(display.contains("0.75"));
        assert!(display.contains("0.9"));
        assert!(display.contains("5"));
        assert!(display.contains("Needs docstring"));
    }

    #[test]
    fn test_is_library_error() {
        // These are library errors (bugs/infrastructure)
        assert!(Error::signature("bad").is_library_error());
        assert!(Error::field("bad").is_library_error());
        assert!(Error::module("bad").is_library_error());
        assert!(Error::parse("bad").is_library_error());

        // This is an optimization error (expected)
        assert!(!Error::optimization(0.5, 3, 0.9).is_library_error());
    }

    #[test]
    fn test_error_category() {
        assert_eq!(Error::signature("x").category(), "signature");
        assert_eq!(Error::field("x").category(), "field");
        assert_eq!(Error::module("x").category(), "module");
        assert_eq!(Error::prediction("x").category(), "prediction");
        assert_eq!(Error::parse("x").category(), "parse");
        assert_eq!(Error::validation("x").category(), "validation");
        assert_eq!(Error::optimization(0.5, 1, 0.9).category(), "optimization");
        assert_eq!(Error::Other("x".to_string()).category(), "other");
    }

    #[test]
    fn test_optimization_error_message() {
        let err = Error::optimization_with_feedback(0.65, 10, 0.9, "Missing tests");
        let msg = err.to_string();
        assert!(msg.contains("Optimization failed"));
        assert!(msg.contains("0.65"));
    }

    #[test]
    fn test_mutable_optimization_details() {
        let mut err = Error::optimization(0.5, 2, 0.9);
        if let Some(details) = err.optimization_details_mut() {
            details.feedback = Some("Updated feedback".to_string());
        }

        let details = err.optimization_details().unwrap();
        assert_eq!(details.feedback.as_deref(), Some("Updated feedback"));
    }
}
