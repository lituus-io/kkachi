// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for semantic validation (LLM-as-judge).
//!
//! Provides `Semantic(llm)` builder that creates an LLM-as-judge validator,
//! and `OwnedSemanticValidator` which owns its data for the Python bridge.

use pyo3::prelude::*;

use kkachi::recursive::{Score, Validate};

use crate::compose::{PyScoreResult, PyValidator, SemanticConfig, ValidatorNode};

// ============================================================================
// OwnedSemanticValidator — Owns LLM callable and criteria for the bridge layer
// ============================================================================

/// Semantic validator that owns its LLM callable and criteria.
///
/// The core library's `SemanticValidator<'a, L>` borrows; this owned version
/// is necessary for the Python bridge where lifetimes can't be expressed.
pub(crate) struct OwnedSemanticValidator {
    llm: PyObject,
    criteria: Vec<String>,
    threshold: f64,
    #[allow(dead_code)]
    system_prompt: Option<String>,
}

impl OwnedSemanticValidator {
    /// Create a new owned semantic validator.
    pub fn new(
        py: Python<'_>,
        llm: &PyObject,
        criteria: Vec<String>,
        threshold: f64,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            llm: llm.clone_ref(py),
            criteria,
            threshold,
            system_prompt,
        }
    }

    /// Build the judge prompt for evaluating text.
    fn build_judge_prompt(&self, text: &str) -> String {
        let criteria_list = self
            .criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"You are evaluating code/text against specific criteria.
Rate each criterion from 0.0 to 1.0.

TEXT TO EVALUATE:
```
{text}
```

CRITERIA:
{criteria}

Respond ONLY with a JSON object in this exact format:
{{"scores": [{{"criterion": "criterion text", "score": 0.0-1.0, "reason": "brief explanation"}}], "overall": 0.0-1.0, "confidence": 0.0-1.0}}

Important:
- "overall" is the weighted average of all criterion scores
- "confidence" indicates how certain you are about your assessment (1.0 = very certain, 0.5 = uncertain)
- Be strict but fair in your evaluation"#,
            text = text,
            criteria = criteria_list
        )
    }

    /// Parse the judge's response to extract scores.
    fn parse_judgment(&self, response: &str) -> Score<'static> {
        let json_str = self.extract_json(response);
        let (overall, confidence) = self.parse_scores(json_str);

        let feedback = if overall >= self.threshold {
            format!("Semantic validation passed with score {:.2}", overall)
        } else {
            format!(
                "Semantic validation failed: score {:.2} < threshold {:.2}",
                overall, self.threshold
            )
        };

        Score::with_feedback(overall, feedback).with_confidence(confidence)
    }

    /// Extract JSON from the response (handles markdown code blocks).
    fn extract_json<'b>(&self, response: &'b str) -> &'b str {
        if let Some(start) = response.find("```json") {
            let after_marker = &response[start + 7..];
            if let Some(end) = after_marker.find("```") {
                return after_marker[..end].trim();
            }
        }

        if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                return &response[start..=end];
            }
        }

        response
    }

    /// Parse overall and confidence scores from JSON.
    fn parse_scores(&self, json_str: &str) -> (f64, f64) {
        let overall = self.extract_number(json_str, "overall").unwrap_or(0.5);
        let confidence = self.extract_number(json_str, "confidence").unwrap_or(0.5);
        (overall.clamp(0.0, 1.0), confidence.clamp(0.0, 1.0))
    }

    /// Extract a number value from a JSON string.
    fn extract_number(&self, json: &str, key: &str) -> Option<f64> {
        let pattern = format!("\"{}\"", key);
        let start = json.find(&pattern)?;
        let after_key = &json[start + pattern.len()..];
        let colon_pos = after_key.find(':')?;
        let after_colon = &after_key[colon_pos + 1..];
        let trimmed = after_colon.trim_start();
        let end = trimmed
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(trimmed.len());
        trimmed[..end].parse().ok()
    }
}

impl Validate for OwnedSemanticValidator {
    fn validate(&self, text: &str) -> Score<'static> {
        let prompt = self.build_judge_prompt(text);

        let result = Python::with_gil(|py| -> PyResult<String> {
            let feedback_py: Option<String> = None;
            let result = self.llm.call1(py, (prompt, feedback_py))?;
            result.extract::<String>(py)
        });

        match result {
            Ok(output) => self.parse_judgment(&output),
            Err(e) => Score::with_feedback(
                0.5,
                format!("Semantic validation error: {}", e),
            )
            .with_confidence(0.0),
        }
    }

    fn name(&self) -> &'static str {
        "semantic"
    }
}

// Safety: PyObject is Send+Sync, and we always acquire GIL before using it.
unsafe impl Send for OwnedSemanticValidator {}
unsafe impl Sync for OwnedSemanticValidator {}

// ============================================================================
// PySemantic — Python-facing builder for semantic validation
// ============================================================================

/// Semantic validator using LLM-as-judge.
///
/// Example:
/// ```python
/// validator = Semantic(judge) \
///     .criterion("Code is idiomatic Rust") \
///     .criterion("Error handling is complete") \
///     .threshold(0.8)
///
/// score = validator.validate("fn parse() { ... }")
/// ```
#[pyclass(name = "Semantic")]
pub struct PySemantic {
    llm: PyObject,
    criteria: Vec<String>,
    threshold: f64,
    system_prompt: Option<String>,
}

impl Clone for PySemantic {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            criteria: self.criteria.clone(),
            threshold: self.threshold,
            system_prompt: self.system_prompt.clone(),
        })
    }
}

#[pymethods]
impl PySemantic {
    /// Create a new semantic validator builder.
    ///
    /// Args:
    ///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
    #[new]
    fn new(llm: PyObject) -> Self {
        Self {
            llm,
            criteria: Vec::new(),
            threshold: 0.7,
            system_prompt: None,
        }
    }

    /// Add a criterion for evaluation.
    fn criterion(&self, criterion: String) -> Self {
        let mut new = self.clone();
        new.criteria.push(criterion);
        new
    }

    /// Set minimum passing threshold (0.0-1.0).
    fn threshold(&self, threshold: f64) -> Self {
        let mut new = self.clone();
        new.threshold = threshold.clamp(0.0, 1.0);
        new
    }

    /// Set custom judge system prompt.
    fn system_prompt(&self, prompt: String) -> Self {
        let mut new = self.clone();
        new.system_prompt = Some(prompt);
        new
    }

    /// Compose with AND semantics.
    #[pyo3(name = "and_")]
    fn and_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValidator> {
        let self_node = self.to_validator_node();
        let other_node = crate::compose::extract_validator_node(other)?;
        Ok(PyValidator {
            node: ValidatorNode::And(Box::new(self_node), Box::new(other_node)),
        })
    }

    /// Compose with OR semantics.
    #[pyo3(name = "or_")]
    fn or_compose(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValidator> {
        let self_node = self.to_validator_node();
        let other_node = crate::compose::extract_validator_node(other)?;
        Ok(PyValidator {
            node: ValidatorNode::Or(Box::new(self_node), Box::new(other_node)),
        })
    }

    /// Test validation directly.
    fn validate(&self, text: String) -> PyResult<PyScoreResult> {
        Python::with_gil(|py| {
            let validator = OwnedSemanticValidator::new(
                py,
                &self.llm,
                self.criteria.clone(),
                self.threshold,
                self.system_prompt.clone(),
            );
            let score = validator.validate(&text);
            Ok(PyScoreResult::from(score))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Semantic(criteria={}, threshold={:.2})",
            self.criteria.len(),
            self.threshold
        )
    }
}

impl PySemantic {
    /// Convert to a ValidatorNode for composition.
    pub(crate) fn to_validator_node(&self) -> ValidatorNode {
        Python::with_gil(|py| {
            ValidatorNode::Semantic(SemanticConfig {
                llm: self.llm.clone_ref(py),
                criteria: self.criteria.clone(),
                threshold: self.threshold,
                system_prompt: self.system_prompt.clone(),
            })
        })
    }
}
