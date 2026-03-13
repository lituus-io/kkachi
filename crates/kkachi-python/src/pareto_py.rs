// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for multi-objective optimization with Pareto-front support.
//!
//! Provides Python access to:
//! - `Objective` - named optimization objective (weight, target, direction)
//! - `multi_objective()` - builder for multi-objective validators
//! - `MultiObjectiveValidator` - validates text against multiple criteria
//! - `ParetoFront` / `ParetoCandidate` - non-dominated solutions
//! - `refine_pareto()` - Pareto-aware iterative refinement

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use kkachi::recursive::{
    multi_objective, refine_pareto_sync, MultiObjective, MultiObjectiveValidate, Objective,
    ParetoCandidate, Scalarization, Validate,
};

use crate::compose::{extract_validator_node, PyScoreResult, ValidatorNode};
use crate::dspy::PyCallableLlm;

// ============================================================================
// PyObjective — Named optimization objective
// ============================================================================

/// A named optimization objective with weight, target, and direction.
///
/// Examples:
///     >>> obj = Objective("correctness").weight(2.0).target(0.9)
///     >>> obj = Objective("brevity").weight(1.0).minimize()
#[pyclass(name = "Objective")]
#[derive(Clone)]
pub struct PyObjective {
    name: String,
    weight: f64,
    target: f64,
    minimize: bool,
}

#[pymethods]
impl PyObjective {
    /// Create a new objective with default settings.
    ///
    /// Args:
    ///     name: The name of this objective.
    ///
    /// Defaults: weight=1.0, target=1.0, direction=Maximize.
    #[new]
    fn new(name: String) -> Self {
        Self {
            name,
            weight: 1.0,
            target: 1.0,
            minimize: false,
        }
    }

    /// Set the weight for scalarization.
    ///
    /// Higher weights give this objective more influence when combining
    /// multiple objectives into a single score.
    ///
    /// Args:
    ///     w: Weight value (default 1.0).
    ///
    /// Returns:
    ///     A new Objective with the updated weight.
    fn weight(&self, w: f64) -> Self {
        let mut obj = self.clone();
        obj.weight = w;
        obj
    }

    /// Set the target threshold for this objective.
    ///
    /// Refinement stops early when all objectives meet their targets.
    ///
    /// Args:
    ///     t: Target value in [0.0, 1.0] (default 1.0).
    ///
    /// Returns:
    ///     A new Objective with the updated target.
    fn target(&self, t: f64) -> Self {
        let mut obj = self.clone();
        obj.target = t;
        obj
    }

    /// Set direction to minimize (lower is better).
    ///
    /// By default, objectives maximize (higher is better). Calling minimize()
    /// inverts the score: value becomes 1.0 - original_value.
    ///
    /// Returns:
    ///     A new Objective with minimize direction.
    fn minimize(&self) -> Self {
        let mut obj = self.clone();
        obj.minimize = true;
        obj
    }

    /// Get the objective name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Get the objective weight.
    #[getter]
    fn get_weight(&self) -> f64 {
        self.weight
    }

    /// Get the target threshold.
    #[getter]
    fn get_target(&self) -> f64 {
        self.target
    }

    /// Whether this objective minimizes.
    #[getter]
    fn is_minimize(&self) -> bool {
        self.minimize
    }

    fn __repr__(&self) -> String {
        let dir = if self.minimize {
            "Minimize"
        } else {
            "Maximize"
        };
        format!(
            "Objective('{}', weight={:.1}, target={:.1}, {})",
            self.name, self.weight, self.target, dir
        )
    }
}

impl PyObjective {
    /// Convert to a Rust `Objective`.
    ///
    /// The Rust `Objective.name` is `&'static str`, so we leak the string.
    /// This is acceptable since Python objects live for the process lifetime
    /// and objectives are created infrequently.
    pub(crate) fn to_rust(&self) -> Objective {
        let name: &'static str = Box::leak(self.name.clone().into_boxed_str());
        let mut obj = Objective::new(name).weight(self.weight).target(self.target);
        if self.minimize {
            obj = obj.minimize();
        }
        obj
    }
}

// ============================================================================
// PyMultiObjectiveBuilder — Builder for multi-objective validators
// ============================================================================

/// Internal storage for an objective+validator pair before materialization.
struct ObjectivePair {
    objective: PyObjective,
    validator: ValidatorNode,
}

impl Clone for ObjectivePair {
    fn clone(&self) -> Self {
        Self {
            objective: self.objective.clone(),
            validator: self.validator.clone(),
        }
    }
}

/// Builder for multi-objective validators.
///
/// Created by `multi_objective()`. Collects objectives paired with validators
/// and builds a `MultiObjectiveValidator` that evaluates text against all criteria.
///
/// Examples:
///     >>> validator = multi_objective() \
///     ...     .scalarize("weighted_sum") \
///     ...     .objective(Objective("correctness").weight(2.0), Checks().require("fn ")) \
///     ...     .objective(Objective("brevity").weight(1.0), Checks().max_len(200)) \
///     ...     .build()
#[pyclass(name = "MultiObjectiveBuilder")]
pub struct PyMultiObjectiveBuilder {
    scalarization: Scalarization,
    pairs: Vec<ObjectivePair>,
}

impl Clone for PyMultiObjectiveBuilder {
    fn clone(&self) -> Self {
        Self {
            scalarization: self.scalarization.clone(),
            pairs: self.pairs.clone(),
        }
    }
}

#[pymethods]
impl PyMultiObjectiveBuilder {
    /// Set the scalarization strategy.
    ///
    /// Determines how multiple objective scores are combined into a single
    /// scalar value for ranking.
    ///
    /// Args:
    ///     strategy: One of "weighted_sum", "chebyshev", "weighted_product".
    ///
    /// Returns:
    ///     A new builder with the updated strategy.
    fn scalarize(&self, strategy: &str) -> PyResult<Self> {
        let s = parse_scalarization(strategy)?;
        let mut new = self.clone();
        new.scalarization = s;
        Ok(new)
    }

    /// Add an objective with its associated validator.
    ///
    /// Args:
    ///     obj: An Objective instance defining name, weight, target, direction.
    ///     validator: A Checks, Semantic, CliValidator, or Validator instance.
    ///
    /// Returns:
    ///     A new builder with the objective added.
    fn objective(&self, obj: &PyObjective, validator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let node = extract_validator_node(validator)?;
        let mut new = self.clone();
        new.pairs.push(ObjectivePair {
            objective: obj.clone(),
            validator: node,
        });
        Ok(new)
    }

    /// Build the multi-objective validator.
    ///
    /// Materializes all validator nodes into concrete validators and constructs
    /// the `MultiObjectiveValidator`.
    ///
    /// Returns:
    ///     A MultiObjectiveValidator ready for use.
    fn build(&self) -> PyResult<PyMultiObjectiveValidator> {
        let scalarization = self.scalarization.clone();
        let pairs = self.pairs.clone();

        Python::with_gil(|py| {
            let rust_pairs: Vec<(Objective, Box<dyn Validate>)> = pairs
                .iter()
                .map(|pair| {
                    let obj = pair.objective.to_rust();
                    let val = pair.validator.materialize(py);
                    (obj, val)
                })
                .collect();

            let validator = multi_objective()
                .scalarize(scalarization)
                .objectives(rust_pairs);

            Ok(PyMultiObjectiveValidator { inner: validator })
        })
    }

    fn __repr__(&self) -> String {
        let strategy = match &self.scalarization {
            Scalarization::WeightedSum => "weighted_sum",
            Scalarization::Chebyshev => "chebyshev",
            Scalarization::WeightedProduct => "weighted_product",
        };
        format!(
            "MultiObjectiveBuilder(scalarize='{}', objectives={})",
            strategy,
            self.pairs.len()
        )
    }
}

// ============================================================================
// PyMultiObjectiveValidator — Multi-objective validator
// ============================================================================

/// A multi-objective validator that evaluates text against multiple criteria.
///
/// Unlike simple validators which return a single score, this validator
/// preserves per-objective scores for Pareto analysis. When used as a
/// regular validator (via `validate()`), it scalarizes the result using
/// the configured strategy.
///
/// Examples:
///     >>> score = validator.validate("fn main() {}")
///     >>> multi = validator.validate_multi("fn main() {}")
///     >>> for obj_score in multi:
///     ...     print(f"{obj_score['name']}: {obj_score['value']:.2f}")
#[pyclass(name = "MultiObjectiveValidator")]
pub struct PyMultiObjectiveValidator {
    inner: MultiObjective<Box<dyn Validate>>,
}

#[pymethods]
impl PyMultiObjectiveValidator {
    /// Validate text and return a scalarized score.
    ///
    /// Evaluates the text against all objectives and combines the scores
    /// using the configured scalarization strategy.
    ///
    /// Args:
    ///     text: The text to validate.
    ///
    /// Returns:
    ///     A ScoreResult with the combined score.
    fn validate(&self, text: &str) -> PyScoreResult {
        let score = Validate::validate(&self.inner, text);
        PyScoreResult::from(score)
    }

    /// Validate text and return per-objective scores.
    ///
    /// Returns a list of dictionaries, one per objective, containing:
    /// - name: The objective name
    /// - value: The score value (0.0 to 1.0)
    /// - weight: The objective weight
    ///
    /// Args:
    ///     text: The text to validate.
    ///
    /// Returns:
    ///     A list of dicts with per-objective score details.
    fn validate_multi<'py>(
        &self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let multi = self.inner.validate_multi(text);
        let mut results = Vec::with_capacity(multi.objectives.len());
        for obj_score in &multi.objectives {
            let dict = PyDict::new_bound(py);
            dict.set_item("name", obj_score.name)?;
            dict.set_item("value", obj_score.value)?;
            dict.set_item("weight", obj_score.weight)?;
            results.push(dict);
        }
        Ok(results)
    }

    /// Get the number of objectives.
    #[getter]
    fn num_objectives(&self) -> usize {
        self.inner.num_objectives()
    }

    fn __repr__(&self) -> String {
        let strategy = match self.inner.scalarization() {
            Scalarization::WeightedSum => "weighted_sum",
            Scalarization::Chebyshev => "chebyshev",
            Scalarization::WeightedProduct => "weighted_product",
        };
        format!(
            "MultiObjectiveValidator(objectives={}, scalarize='{}')",
            self.inner.num_objectives(),
            strategy
        )
    }
}

// ============================================================================
// PyParetoCandidate — A single non-dominated solution
// ============================================================================

/// A candidate solution on the Pareto front.
///
/// Represents a non-dominated solution with its output text,
/// per-objective scores, and the iteration index when it was generated.
#[pyclass(name = "ParetoCandidate")]
#[derive(Clone)]
pub struct PyParetoCandidate {
    /// The generated output text.
    #[pyo3(get)]
    pub output: String,
    /// Per-objective scores.
    #[pyo3(get)]
    pub scores: Vec<f64>,
    /// Generation index (which iteration produced this candidate).
    #[pyo3(get)]
    pub index: usize,
}

#[pymethods]
impl PyParetoCandidate {
    fn __repr__(&self) -> String {
        let scores_str: Vec<String> = self.scores.iter().map(|s| format!("{:.3}", s)).collect();
        format!(
            "ParetoCandidate(index={}, scores=[{}], output='{}')",
            self.index,
            scores_str.join(", "),
            truncate(&self.output, 40)
        )
    }
}

impl From<&ParetoCandidate> for PyParetoCandidate {
    fn from(c: &ParetoCandidate) -> Self {
        Self {
            output: c.output.clone(),
            scores: c.scores.to_vec(),
            index: c.index,
        }
    }
}

// ============================================================================
// PyParetoFront — Non-dominated solutions
// ============================================================================

/// A Pareto front of non-dominated solutions.
///
/// Contains the set of solutions where no solution is strictly better
/// than any other in all objectives simultaneously. These represent
/// the optimal trade-off surface.
///
/// Examples:
///     >>> front = result.front
///     >>> print(f"Found {len(front)} Pareto-optimal solutions")
///     >>> for candidate in front.solutions:
///     ...     print(f"  scores={candidate.scores}")
///     >>> best = front.best
///     >>> print(f"Best by scalarization: {best.output[:50]}...")
#[pyclass(name = "ParetoFront")]
#[derive(Clone)]
pub struct PyParetoFront {
    solutions_list: Vec<PyParetoCandidate>,
    /// Scalarization strategy used for `best()` selection.
    scalarization: Scalarization,
    /// Weights for scalarization (from objective definitions).
    weights: Vec<f64>,
}

#[pymethods]
impl PyParetoFront {
    /// Get all non-dominated solutions.
    ///
    /// Returns:
    ///     A list of ParetoCandidate instances.
    #[getter]
    fn solutions(&self) -> Vec<PyParetoCandidate> {
        self.solutions_list.clone()
    }

    /// Number of Pareto-optimal solutions.
    fn __len__(&self) -> usize {
        self.solutions_list.len()
    }

    /// Whether the front is empty.
    #[getter]
    fn is_empty(&self) -> bool {
        self.solutions_list.is_empty()
    }

    /// Get the best solution according to the scalarization strategy.
    ///
    /// Returns:
    ///     The best ParetoCandidate, or None if the front is empty.
    #[getter]
    fn best(&self) -> Option<PyParetoCandidate> {
        if self.solutions_list.is_empty() {
            return None;
        }

        // Rebuild a temporary ParetoFront to use the Rust `best()` method.
        // This is simpler than re-implementing scalarization in Python bindings.
        let mut front = kkachi::recursive::ParetoFront::new(
            self.solutions_list
                .first()
                .map(|s| s.scores.len())
                .unwrap_or(0),
        );
        for candidate in &self.solutions_list {
            let rust_candidate = ParetoCandidate {
                output: candidate.output.clone(),
                scores: candidate.scores.iter().copied().collect(),
                index: candidate.index,
            };
            front.insert(rust_candidate);
        }

        front
            .best(&self.scalarization, &self.weights)
            .map(PyParetoCandidate::from)
    }

    fn __repr__(&self) -> String {
        format!("ParetoFront(solutions={})", self.solutions_list.len())
    }
}

// ============================================================================
// PyParetoRefineResult — Result of Pareto refinement
// ============================================================================

/// Result of multi-objective Pareto refinement.
///
/// Contains the Pareto front, the best output (by scalarization),
/// per-objective best scores, and execution statistics.
///
/// Examples:
///     >>> result = refine_pareto(llm, "Write code", validator, max_iter=5)
///     >>> print(result.best_output)
///     >>> print(f"Iterations: {result.iterations}, Tokens: {result.total_tokens}")
///     >>> for candidate in result.front.solutions:
///     ...     print(f"  {candidate.scores}")
#[pyclass(name = "ParetoRefineResult")]
#[derive(Clone)]
pub struct PyParetoRefineResult {
    /// The best output according to scalarization.
    #[pyo3(get)]
    pub best_output: String,
    /// Number of iterations performed.
    #[pyo3(get)]
    pub iterations: u32,
    /// Total tokens consumed across all iterations.
    #[pyo3(get)]
    pub total_tokens: u32,
    /// Elapsed time in milliseconds.
    #[pyo3(get)]
    pub elapsed_ms: u64,
    /// Per-objective best scores: list of (name, best_value) tuples.
    objective_bests_data: Vec<(String, f64)>,
    /// The Pareto front of non-dominated solutions.
    front_data: PyParetoFront,
}

#[pymethods]
impl PyParetoRefineResult {
    /// Get the Pareto front of non-dominated solutions.
    ///
    /// Returns:
    ///     A ParetoFront containing all non-dominated solutions.
    #[getter]
    fn front(&self) -> PyParetoFront {
        self.front_data.clone()
    }

    /// Get per-objective best scores as a list of (name, value) tuples.
    ///
    /// Returns:
    ///     A list of (objective_name, best_score) tuples.
    #[getter]
    fn objective_bests(&self) -> Vec<(String, f64)> {
        self.objective_bests_data.clone()
    }

    /// Get per-objective best scores as a dictionary.
    ///
    /// Returns:
    ///     A dict mapping objective names to their best scores.
    fn objective_bests_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (name, value) in &self.objective_bests_data {
            dict.set_item(name, value)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "ParetoRefineResult(best='{}', front={}, iterations={}, tokens={}, elapsed={}ms)",
            truncate(&self.best_output, 40),
            self.front_data.solutions_list.len(),
            self.iterations,
            self.total_tokens,
            self.elapsed_ms
        )
    }
}

// ============================================================================
// Entry point functions
// ============================================================================

/// Create a multi-objective builder.
///
/// Entry point for building multi-objective validators that evaluate text
/// against multiple criteria simultaneously.
///
/// # Python Example
///
/// ```python
/// validator = multi_objective() \
///     .scalarize("weighted_sum") \
///     .objective(Objective("correctness").weight(2.0), Checks().require("fn ")) \
///     .objective(Objective("brevity").weight(1.0), Checks().max_len(200)) \
///     .build()
///
/// score = validator.validate("fn main() {}")
/// ```
#[pyfunction]
#[pyo3(name = "multi_objective")]
pub fn py_multi_objective() -> PyMultiObjectiveBuilder {
    PyMultiObjectiveBuilder {
        scalarization: Scalarization::WeightedSum,
        pairs: Vec::new(),
    }
}

/// Run multi-objective Pareto refinement.
///
/// Iteratively generates outputs using the LLM, evaluates them against
/// multiple objectives, and maintains a Pareto front of non-dominated
/// solutions. Stops when all objectives meet their targets, the front
/// converges, or max iterations are reached.
///
/// Args:
///     llm: A Python callable with signature `(prompt: str, feedback: Optional[str]) -> str`.
///     prompt: The initial prompt to send to the LLM.
///     validator: A MultiObjectiveValidator (built via `multi_objective().build()`).
///     max_iter: Maximum number of refinement iterations (default 10).
///
/// Returns:
///     A ParetoRefineResult with the Pareto front, best output, and statistics.
///
/// # Python Example
///
/// ```python
/// result = refine_pareto(llm, "Write a function", validator, max_iter=5)
/// print(result.best_output)
/// print(f"Found {len(result.front)} Pareto-optimal solutions")
/// ```
#[pyfunction]
#[pyo3(name = "refine_pareto", signature = (llm, prompt, validator, max_iter=10))]
pub fn py_refine_pareto(
    llm: PyObject,
    prompt: String,
    validator: &PyMultiObjectiveValidator,
    max_iter: u32,
) -> PyResult<PyParetoRefineResult> {
    let callable_llm = Python::with_gil(|py| PyCallableLlm::new_from_ref(py, &llm));

    let result = refine_pareto_sync(&callable_llm, &prompt, &validator.inner, max_iter);

    // Extract weights from objectives for ParetoFront.best() support
    let objectives = validator.inner.objectives();
    let weights: Vec<f64> = if objectives.is_empty() {
        vec![1.0; validator.inner.num_objectives()]
    } else {
        objectives.iter().map(|o| o.weight).collect()
    };

    // Convert Pareto front solutions
    let solutions: Vec<PyParetoCandidate> = result
        .front
        .solutions()
        .iter()
        .map(PyParetoCandidate::from)
        .collect();

    let front = PyParetoFront {
        solutions_list: solutions,
        scalarization: validator.inner.scalarization().clone(),
        weights,
    };

    // Convert objective bests
    let objective_bests_data: Vec<(String, f64)> = result
        .objective_bests
        .iter()
        .map(|(name, val)| (name.to_string(), *val))
        .collect();

    Ok(PyParetoRefineResult {
        best_output: result.best_output,
        iterations: result.iterations,
        total_tokens: result.total_tokens,
        elapsed_ms: result.elapsed.as_millis() as u64,
        objective_bests_data,
        front_data: front,
    })
}

// ============================================================================
// Helpers
// ============================================================================

/// Parse a scalarization strategy from a string.
fn parse_scalarization(s: &str) -> PyResult<Scalarization> {
    match s {
        "weighted_sum" => Ok(Scalarization::WeightedSum),
        "chebyshev" => Ok(Scalarization::Chebyshev),
        "weighted_product" => Ok(Scalarization::WeightedProduct),
        _ => Err(PyRuntimeError::new_err(format!(
            "Unknown scalarization strategy '{}'. Expected one of: \
             'weighted_sum', 'chebyshev', 'weighted_product'",
            s
        ))),
    }
}

/// Truncate a string for display in __repr__.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
