// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Validator composition for Python bindings.
//!
//! Provides `and()`/`or()`/`all()`/`any()` composition of validators using
//! `Box<dyn Validate>` for runtime polymorphism at the Python boundary.
//!
//! The `ValidatorNode` enum stores validator configurations as a tree,
//! materialized into `Box<dyn Validate>` at execution time.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use kkachi::recursive::{All, And, Any, Checks, Or, Score, Validate};

// ============================================================================
// DynValidator — Bridge type for dynamic dispatch at the Python boundary
// ============================================================================

/// Bridge type wrapping `Box<dyn Validate>`.
///
/// Only used in the Python binding layer where dynamic typing is unavoidable.
/// Python call overhead already dwarfs vtable cost.
pub(crate) struct DynValidator(pub Box<dyn Validate>);

impl Validate for DynValidator {
    fn validate(&self, text: &str) -> Score<'static> {
        self.0.validate(text)
    }

    fn name(&self) -> &'static str {
        self.0.name()
    }
}

// Safety: The inner Box<dyn Validate> is already Send + Sync (trait bound).
unsafe impl Send for DynValidator {}
unsafe impl Sync for DynValidator {}

// ============================================================================
// ValidatorNode — Tree representation of composed validators
// ============================================================================

/// Internal representation of composed validators.
///
/// Stored as a tree and materialized into `Box<dyn Validate>` at execution time.
/// This avoids holding trait objects that can't be cloned for the builder pattern.
pub(crate) enum ValidatorNode {
    /// A pattern-based checks validator.
    Checks(Checks),
    /// A semantic (LLM-as-judge) validator.
    Semantic(SemanticConfig),
    /// A CLI-based validator (external tool validation).
    Cli(kkachi::recursive::Cli),
    /// Both children must pass.
    And(Box<ValidatorNode>, Box<ValidatorNode>),
    /// At least one child must pass.
    Or(Box<ValidatorNode>, Box<ValidatorNode>),
    /// All children must pass.
    All(Vec<ValidatorNode>),
    /// At least one child must pass.
    Any(Vec<ValidatorNode>),
}

impl ValidatorNode {
    /// Materialize this node tree into a concrete validator.
    pub fn materialize(&self, py: Python<'_>) -> Box<dyn Validate> {
        match self {
            Self::Checks(checks) => Box::new(checks.clone()),
            Self::Semantic(cfg) => cfg.materialize(py),
            Self::Cli(cli) => Box::new(cli.clone()),
            Self::And(a, b) => {
                let va = DynValidator(a.materialize(py));
                let vb = DynValidator(b.materialize(py));
                Box::new(And(va, vb))
            }
            Self::Or(a, b) => {
                let va = DynValidator(a.materialize(py));
                let vb = DynValidator(b.materialize(py));
                Box::new(Or(va, vb))
            }
            Self::All(nodes) => {
                let validators: Vec<DynValidator> = nodes
                    .iter()
                    .map(|n| DynValidator(n.materialize(py)))
                    .collect();
                Box::new(All(validators))
            }
            Self::Any(nodes) => {
                let validators: Vec<DynValidator> = nodes
                    .iter()
                    .map(|n| DynValidator(n.materialize(py)))
                    .collect();
                Box::new(Any(validators))
            }
        }
    }
}

impl Clone for ValidatorNode {
    fn clone(&self) -> Self {
        match self {
            Self::Checks(c) => Self::Checks(c.clone()),
            Self::Semantic(cfg) => Self::Semantic(cfg.clone()),
            Self::Cli(cli) => Self::Cli(cli.clone()),
            Self::And(a, b) => Self::And(a.clone(), b.clone()),
            Self::Or(a, b) => Self::Or(a.clone(), b.clone()),
            Self::All(nodes) => Self::All(nodes.clone()),
            Self::Any(nodes) => Self::Any(nodes.clone()),
        }
    }
}

// ============================================================================
// SemanticConfig — Configuration for OwnedSemanticValidator
// ============================================================================

/// Configuration for a semantic validator, stored in the node tree.
///
/// The PyObject (LLM callable) requires GIL for cloning.
pub(crate) struct SemanticConfig {
    pub llm: PyObject,
    pub criteria: Vec<String>,
    pub threshold: f64,
    pub system_prompt: Option<String>,
}

impl SemanticConfig {
    /// Materialize into an OwnedSemanticValidator.
    fn materialize(&self, py: Python<'_>) -> Box<dyn Validate> {
        Box::new(crate::semantic::OwnedSemanticValidator::new(
            py,
            &self.llm,
            self.criteria.clone(),
            self.threshold,
            self.system_prompt.clone(),
        ))
    }
}

impl Clone for SemanticConfig {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            criteria: self.criteria.clone(),
            threshold: self.threshold,
            system_prompt: self.system_prompt.clone(),
        })
    }
}

// ============================================================================
// PyScoreResult — Validation score returned to Python
// ============================================================================

/// Result from calling `.validate()` directly.
#[pyclass(name = "ScoreResult")]
#[derive(Clone)]
pub struct PyScoreResult {
    /// The validation score (0.0 to 1.0).
    #[pyo3(get)]
    pub value: f64,
    /// Optional feedback explaining the score.
    #[pyo3(get)]
    pub feedback: Option<String>,
    /// Confidence level (0.0 to 1.0).
    #[pyo3(get)]
    pub confidence: Option<f64>,
}

#[pymethods]
impl PyScoreResult {
    /// Check if the score passes a threshold.
    fn passes(&self, threshold: f64) -> bool {
        self.value >= threshold
    }

    /// Check if the score is perfect (1.0).
    fn is_perfect(&self) -> bool {
        (self.value - 1.0).abs() < f64::EPSILON
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoreResult(value={:.3}, feedback={:?})",
            self.value,
            self.feedback.as_deref().map(|s| truncate(s, 50))
        )
    }
}

impl From<Score<'_>> for PyScoreResult {
    fn from(score: Score<'_>) -> Self {
        Self {
            value: score.value,
            feedback: score.feedback.map(|f| f.into_owned()),
            confidence: if (score.confidence - 1.0).abs() < f64::EPSILON {
                None
            } else {
                Some(score.confidence)
            },
        }
    }
}

// ============================================================================
// PyValidator — Python-facing composable validator
// ============================================================================

/// A composable validator that can combine Checks and Semantic validators.
///
/// Example:
/// ```python
/// strict = checks.and(semantic)   # Both must pass
/// relaxed = checks.or(semantic)   # At least one passes
/// combined = Validator.all([checks, semantic])
/// ```
#[pyclass(name = "Validator")]
pub struct PyValidator {
    pub(crate) node: ValidatorNode,
}

impl Clone for PyValidator {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
        }
    }
}

#[pymethods]
impl PyValidator {
    /// Compose with AND semantics (both must pass).
    #[pyo3(name = "and_")]
    fn and_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other_node = extract_validator_node(other)?;
        Ok(Self {
            node: ValidatorNode::And(Box::new(self.node.clone()), Box::new(other_node)),
        })
    }

    /// Compose with OR semantics (at least one must pass).
    #[pyo3(name = "or_")]
    fn or_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other_node = extract_validator_node(other)?;
        Ok(Self {
            node: ValidatorNode::Or(Box::new(self.node.clone()), Box::new(other_node)),
        })
    }

    /// All validators must pass.
    #[staticmethod]
    fn all(validators: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let nodes: PyResult<Vec<ValidatorNode>> = validators
            .iter()
            .map(|v| extract_validator_node(v))
            .collect();
        Ok(Self {
            node: ValidatorNode::All(nodes?),
        })
    }

    /// At least one validator must pass.
    #[staticmethod]
    fn any(validators: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let nodes: PyResult<Vec<ValidatorNode>> = validators
            .iter()
            .map(|v| extract_validator_node(v))
            .collect();
        Ok(Self {
            node: ValidatorNode::Any(nodes?),
        })
    }

    /// Test validation directly.
    fn validate(&self, text: String) -> PyResult<PyScoreResult> {
        Python::with_gil(|py| {
            let validator = self.node.materialize(py);
            let score = validator.validate(&text);
            Ok(PyScoreResult::from(score))
        })
    }

    fn __repr__(&self) -> String {
        let desc = match &self.node {
            ValidatorNode::Checks(_) => "Checks",
            ValidatorNode::Semantic(_) => "Semantic",
            ValidatorNode::Cli(_) => "Cli",
            ValidatorNode::And(_, _) => "And(...)",
            ValidatorNode::Or(_, _) => "Or(...)",
            ValidatorNode::All(v) => return format!("Validator.all(n={})", v.len()),
            ValidatorNode::Any(v) => return format!("Validator.any(n={})", v.len()),
        };
        format!("Validator({})", desc)
    }
}

// ============================================================================
// extract_validator_node — Utility to extract from any Python validator type
// ============================================================================

/// Extract a `ValidatorNode` from a Python object.
///
/// Accepts `Checks`, `Semantic`, or `Validator` instances.
pub(crate) fn extract_validator_node(obj: &Bound<'_, PyAny>) -> PyResult<ValidatorNode> {
    use crate::checks::PyChecks;
    use crate::semantic::PySemantic;
    use crate::validator::PyCliValidator;

    if let Ok(checks) = obj.downcast::<PyChecks>() {
        Ok(ValidatorNode::Checks(checks.borrow().inner_ref().clone()))
    } else if let Ok(semantic) = obj.downcast::<PySemantic>() {
        Ok(semantic.borrow().to_validator_node())
    } else if let Ok(cli_validator) = obj.downcast::<PyCliValidator>() {
        Ok(ValidatorNode::Cli(
            cli_validator.borrow().inner_ref().clone(),
        ))
    } else if let Ok(validator) = obj.downcast::<PyValidator>() {
        Ok(validator.borrow().node.clone())
    } else {
        Err(PyRuntimeError::new_err(
            "Expected Checks, Semantic, CliValidator, or Validator instance",
        ))
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
