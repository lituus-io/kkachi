// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for evaluation metrics.

use pyo3::prelude::*;

use kkachi::metric::{Contains, ExactMatch, F1Token, Metric as MetricTrait};

/// Internal enum representing the different metric variants.
#[derive(Debug)]
enum MetricKind {
    ExactMatch,
    Contains,
    F1Token,
    Custom { name: String, func: PyObject },
}

impl Clone for MetricKind {
    fn clone(&self) -> Self {
        match self {
            MetricKind::ExactMatch => MetricKind::ExactMatch,
            MetricKind::Contains => MetricKind::Contains,
            MetricKind::F1Token => MetricKind::F1Token,
            MetricKind::Custom { name, func } => Python::with_gil(|py| MetricKind::Custom {
                name: name.clone(),
                func: func.clone_ref(py),
            }),
        }
    }
}

/// A metric for evaluating predictions against expected values.
///
/// Provides several built-in metrics as well as the ability to define
/// custom metrics using Python callables.
///
/// Examples:
///     >>> m = Metric.exact_match()
///     >>> m.evaluate("hello", "hello")
///     1.0
///
///     >>> m = Metric.contains()
///     >>> m.evaluate("the answer is 42", "42")
///     1.0
///
///     >>> m = Metric.f1_token()
///     >>> m.evaluate("the quick brown fox", "the quick brown fox")
///     1.0
///
///     >>> m = Metric.custom("my_metric", lambda pred, exp: 1.0 if pred == exp else 0.0)
///     >>> m.evaluate("hello", "hello")
///     1.0
///     >>> m.name()
///     'my_metric'
#[pyclass(name = "Metric")]
#[derive(Clone)]
pub struct PyMetric {
    kind: MetricKind,
}

#[pymethods]
impl PyMetric {
    /// Create an exact-match metric.
    ///
    /// Returns 1.0 if the trimmed prediction exactly equals the trimmed
    /// expected string, 0.0 otherwise.
    #[staticmethod]
    fn exact_match() -> Self {
        PyMetric {
            kind: MetricKind::ExactMatch,
        }
    }

    /// Create a substring-containment metric.
    ///
    /// Returns 1.0 if the expected string is contained within the
    /// prediction, 0.0 otherwise.
    #[staticmethod]
    fn contains() -> Self {
        PyMetric {
            kind: MetricKind::Contains,
        }
    }

    /// Create a word-level F1 token metric.
    ///
    /// Computes the F1 score based on word-level overlap between the
    /// prediction and the expected string.
    #[staticmethod]
    fn f1_token() -> Self {
        PyMetric {
            kind: MetricKind::F1Token,
        }
    }

    /// Create a custom metric from a Python callable.
    ///
    /// Args:
    ///     name: The name of the metric.
    ///     func: A callable that takes (prediction, expected) and returns a float.
    #[staticmethod]
    fn custom(name: String, func: PyObject) -> Self {
        PyMetric {
            kind: MetricKind::Custom { name, func },
        }
    }

    /// Evaluate the metric on a prediction/expected pair.
    ///
    /// Args:
    ///     prediction: The predicted string.
    ///     expected: The expected (ground-truth) string.
    ///
    /// Returns:
    ///     A float score, typically between 0.0 and 1.0.
    fn evaluate(&self, prediction: &str, expected: &str) -> f64 {
        match &self.kind {
            MetricKind::ExactMatch => {
                let m = ExactMatch;
                m.evaluate(prediction, expected)
            }
            MetricKind::Contains => {
                let m = Contains;
                m.evaluate(prediction, expected)
            }
            MetricKind::F1Token => {
                let m = F1Token;
                m.evaluate(prediction, expected)
            }
            MetricKind::Custom { func, .. } => Python::with_gil(|py| {
                func.call1(py, (prediction, expected))
                    .and_then(|r| r.extract::<f64>(py))
                    .unwrap_or(0.0)
            }),
        }
    }

    /// Return the name of this metric.
    ///
    /// Returns:
    ///     The metric name as a string.
    fn name(&self) -> String {
        match &self.kind {
            MetricKind::ExactMatch => {
                let m = ExactMatch;
                m.name().to_string()
            }
            MetricKind::Contains => {
                let m = Contains;
                m.name().to_string()
            }
            MetricKind::F1Token => {
                let m = F1Token;
                m.name().to_string()
            }
            MetricKind::Custom { name, .. } => name.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Metric(\"{}\")", self.name())
    }
}
