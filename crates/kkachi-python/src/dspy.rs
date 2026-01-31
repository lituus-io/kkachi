// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for DSPy-style modules.
//!
//! Provides Python access to:
//! - `reason()` - Chain of Thought reasoning
//! - `best_of()` - Best of N candidate generation
//! - `ensemble()` - Multi-chain ensemble voting
//! - `agent()` - ReAct tool-calling agent
//! - `program()` - Program of Thought code execution
//! - Tool and Executor builders

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use kkachi::recursive::{
    // Agent
    agent,
    bash_executor,
    // Best of N
    best_of,
    // Validation
    checks,
    // Ensemble
    ensemble,
    node_executor,
    // Program
    program,
    // Executor
    python_executor,
    // Chain of Thought
    reason,
    ruby_executor,
    Aggregate,
    BestOfResult,
    ChainResult as RustChainResult,
    CodeExecutor,
    EnsembleResult,
    // LLM
    Llm,
    LmOutput,
    PoolStats as RustPoolStats,
    ProcessExecutor,
    ProgramResult,
    ReasonResult,
    ScoredCandidate as RustScoredCandidate,
    Step as RustStep,
    // Tool
    Tool,
};

use crate::compose::{extract_validator_node, DynValidator, ValidatorNode};
use crate::types::PyRefineResult;

// ============================================================================
// Python-callable LLM wrapper
// ============================================================================

/// An LLM implementation that delegates to a Python callable.
///
/// The Python callable signature: `(prompt: str, feedback: Optional[str]) -> str`
struct PyCallableLlm {
    callable: PyObject,
}

impl PyCallableLlm {
    fn new_from_ref(py: Python<'_>, callable: &PyObject) -> Self {
        Self {
            callable: callable.clone_ref(py),
        }
    }
}

impl Llm for PyCallableLlm {
    type GenerateFut<'a>
        = std::future::Ready<kkachi::error::Result<LmOutput>>
    where
        Self: 'a;

    fn generate<'a>(
        &'a self,
        prompt: &'a str,
        _context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::GenerateFut<'a> {
        let result = Python::with_gil(|py| -> PyResult<String> {
            let feedback_py = feedback.map(|s| s.to_string());
            let result = self.callable.call1(py, (prompt.to_string(), feedback_py))?;
            result.extract::<String>(py)
        });

        match result {
            Ok(text) => std::future::ready(Ok(LmOutput::new(text))),
            Err(e) => std::future::ready(Err(kkachi::error::Error::module(format!(
                "Python LLM error: {}",
                e
            )))),
        }
    }

    fn model_name(&self) -> &str {
        "python_callable"
    }
}

// Safety: PyCallableLlm is Send+Sync because PyObject is Send+Sync
// and we always acquire the GIL before using it.
unsafe impl Send for PyCallableLlm {}
unsafe impl Sync for PyCallableLlm {}

// ============================================================================
// Python Tool wrapper
// ============================================================================

/// A Tool implementation that delegates to a Python callable.
struct PyCallableTool {
    name: String,
    description: String,
    callable: PyObject,
}

impl Tool for PyCallableTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn execute<'a>(
        &'a self,
        input: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = kkachi::error::Result<String>> + Send + 'a>,
    > {
        let result = Python::with_gil(|py| -> PyResult<String> {
            let result = self.callable.call1(py, (input.to_string(),))?;
            result.extract::<String>(py)
        });

        Box::pin(std::future::ready(match result {
            Ok(text) => Ok(text),
            Err(e) => Err(kkachi::error::Error::module(format!(
                "Python tool error: {}",
                e
            ))),
        }))
    }
}

unsafe impl Send for PyCallableTool {}
unsafe impl Sync for PyCallableTool {}

// ============================================================================
// Result types
// ============================================================================

/// Result from Chain of Thought reasoning.
#[pyclass(name = "ReasonResult")]
#[derive(Clone)]
pub struct PyReasonResult {
    /// The final answer.
    #[pyo3(get)]
    pub output: String,
    /// The reasoning trace.
    #[pyo3(get)]
    pub reasoning: Option<String>,
    /// Validation score.
    #[pyo3(get)]
    pub score: f64,
    /// Number of iterations.
    #[pyo3(get)]
    pub iterations: u32,
    /// Total tokens used.
    #[pyo3(get)]
    pub tokens: u32,
    /// Error if reasoning failed.
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl PyReasonResult {
    /// Check if reasoning succeeded.
    fn success(&self) -> bool {
        self.error.is_none() && self.score >= 1.0
    }

    fn __repr__(&self) -> String {
        format!(
            "ReasonResult(score={:.3}, iterations={}, output='{}')",
            self.score,
            self.iterations,
            truncate_str(&self.output, 50)
        )
    }

    fn __str__(&self) -> String {
        self.output.clone()
    }
}

impl From<ReasonResult> for PyReasonResult {
    fn from(r: ReasonResult) -> Self {
        Self {
            output: r.output,
            reasoning: r.reasoning,
            score: r.score,
            iterations: r.iterations,
            tokens: r.tokens,
            error: r.error,
        }
    }
}

/// Result from Best of N generation.
#[pyclass(name = "BestOfResult")]
#[derive(Clone)]
pub struct PyBestOfResult {
    /// The best candidate output.
    #[pyo3(get)]
    pub output: String,
    /// Combined score of the best candidate.
    #[pyo3(get)]
    pub score: f64,
    /// Number of candidates generated.
    #[pyo3(get)]
    pub candidates_generated: usize,
    /// Total tokens used.
    #[pyo3(get)]
    pub tokens: u32,
    /// Error message if some generations failed.
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl PyBestOfResult {
    /// Check if generation succeeded.
    fn success(&self) -> bool {
        !self.output.is_empty() && self.error.is_none()
    }

    fn __repr__(&self) -> String {
        format!(
            "BestOfResult(score={:.3}, candidates={})",
            self.score, self.candidates_generated
        )
    }

    fn __str__(&self) -> String {
        self.output.clone()
    }
}

impl From<BestOfResult> for PyBestOfResult {
    fn from(r: BestOfResult) -> Self {
        Self {
            output: r.output,
            score: r.score,
            candidates_generated: r.candidates_generated,
            tokens: r.tokens,
            error: r.error,
        }
    }
}

/// A scored candidate from Best of N generation.
#[pyclass(name = "ScoredCandidate")]
#[derive(Clone)]
pub struct PyScoredCandidate {
    /// Index in generation order.
    #[pyo3(get)]
    pub index: usize,
    /// The generated output.
    #[pyo3(get)]
    pub output: String,
    /// Score from the scorer function.
    #[pyo3(get)]
    pub scorer_score: f64,
    /// Score from the validator.
    #[pyo3(get)]
    pub validator_score: f64,
    /// Combined score.
    #[pyo3(get)]
    pub combined_score: f64,
    /// Validation feedback if any.
    #[pyo3(get)]
    pub feedback: Option<String>,
}

#[pymethods]
impl PyScoredCandidate {
    fn __repr__(&self) -> String {
        format!(
            "ScoredCandidate(index={}, combined_score={:.3})",
            self.index, self.combined_score
        )
    }
}

impl From<&RustScoredCandidate> for PyScoredCandidate {
    fn from(c: &RustScoredCandidate) -> Self {
        Self {
            index: c.index,
            output: c.output.clone(),
            scorer_score: c.scorer_score,
            validator_score: c.validator_score,
            combined_score: c.combined_score,
            feedback: c.feedback.clone(),
        }
    }
}

/// Statistics about a candidate pool.
#[pyclass(name = "PoolStats")]
#[derive(Clone)]
pub struct PyPoolStats {
    /// Number of candidates.
    #[pyo3(get)]
    pub count: usize,
    /// Mean combined score.
    #[pyo3(get)]
    pub mean: f64,
    /// Standard deviation of scores.
    #[pyo3(get)]
    pub std_dev: f64,
    /// Minimum score.
    #[pyo3(get)]
    pub min: f64,
    /// Maximum score.
    #[pyo3(get)]
    pub max: f64,
}

#[pymethods]
impl PyPoolStats {
    fn __repr__(&self) -> String {
        format!(
            "PoolStats(count={}, mean={:.3}, std_dev={:.3})",
            self.count, self.mean, self.std_dev
        )
    }
}

impl From<RustPoolStats> for PyPoolStats {
    fn from(s: RustPoolStats) -> Self {
        Self {
            count: s.count,
            mean: s.mean,
            std_dev: s.std_dev,
            min: s.min,
            max: s.max,
        }
    }
}

/// Pool of all generated candidates for recall/precision tuning.
#[pyclass(name = "CandidatePool")]
#[derive(Clone)]
pub struct PyCandidatePool {
    candidates: Vec<PyScoredCandidate>,
    total_tokens: u32,
}

#[pymethods]
impl PyCandidatePool {
    /// Get all candidates.
    #[getter]
    fn candidates(&self) -> Vec<PyScoredCandidate> {
        self.candidates.clone()
    }

    /// Get total tokens used.
    #[getter]
    fn total_tokens(&self) -> u32 {
        self.total_tokens
    }

    /// Filter candidates by minimum combined score threshold.
    fn filter_by_threshold(&self, threshold: f64) -> Vec<PyScoredCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.combined_score >= threshold)
            .cloned()
            .collect()
    }

    /// Get the best candidate.
    fn best(&self) -> Option<PyScoredCandidate> {
        self.candidates
            .iter()
            .max_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap())
            .cloned()
    }

    /// Get the top K candidates sorted by score.
    fn top_k(&self, k: usize) -> Vec<PyScoredCandidate> {
        let mut sorted = self.candidates.clone();
        sorted.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        sorted.truncate(k);
        sorted
    }

    /// Get statistics about the pool.
    fn stats(&self) -> PyPoolStats {
        if self.candidates.is_empty() {
            return PyPoolStats {
                count: 0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }
        let scores: Vec<f64> = self.candidates.iter().map(|c| c.combined_score).collect();
        let count = scores.len();
        let mean = scores.iter().sum::<f64>() / count as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / count as f64;
        PyPoolStats {
            count,
            mean,
            std_dev: variance.sqrt(),
            min: scores.iter().copied().fold(f64::MAX, f64::min),
            max: scores.iter().copied().fold(f64::MIN, f64::max),
        }
    }

    /// Check if any candidate passed validation.
    fn has_valid(&self) -> bool {
        self.candidates.iter().any(|c| c.validator_score >= 1.0)
    }

    /// Get all passing candidates.
    fn passing(&self) -> Vec<PyScoredCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.validator_score >= 1.0)
            .cloned()
            .collect()
    }

    fn __len__(&self) -> usize {
        self.candidates.len()
    }

    fn __repr__(&self) -> String {
        format!("CandidatePool(len={})", self.candidates.len())
    }
}

/// Result from ensemble generation.
#[pyclass(name = "EnsembleResult")]
#[derive(Clone)]
pub struct PyEnsembleResult {
    /// The selected output.
    #[pyo3(get)]
    pub output: String,
    /// Number of chains generated.
    #[pyo3(get)]
    pub chains_generated: usize,
    /// Total tokens used.
    #[pyo3(get)]
    pub tokens: u32,
    /// Error if some chains failed.
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl PyEnsembleResult {
    fn success(&self) -> bool {
        !self.output.is_empty() && self.error.is_none()
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleResult(chains={}, output='{}')",
            self.chains_generated,
            truncate_str(&self.output, 50)
        )
    }

    fn __str__(&self) -> String {
        self.output.clone()
    }
}

impl From<EnsembleResult> for PyEnsembleResult {
    fn from(r: EnsembleResult) -> Self {
        Self {
            output: r.output,
            chains_generated: r.chains_generated,
            tokens: r.tokens,
            error: r.error,
        }
    }
}

/// A single chain result in the consensus pool.
#[pyclass(name = "ChainResult")]
#[derive(Clone)]
pub struct PyChainResult {
    /// Index in generation order.
    #[pyo3(get)]
    pub index: usize,
    /// The raw answer from this chain.
    #[pyo3(get)]
    pub answer: String,
    /// The normalized answer (trimmed + lowercased).
    #[pyo3(get)]
    pub normalized_answer: String,
    /// Whether this chain agrees with the majority.
    #[pyo3(get)]
    pub agrees_with_majority: bool,
}

#[pymethods]
impl PyChainResult {
    fn __repr__(&self) -> String {
        format!(
            "ChainResult(index={}, agrees={})",
            self.index, self.agrees_with_majority
        )
    }
}

impl From<&RustChainResult> for PyChainResult {
    fn from(c: &RustChainResult) -> Self {
        Self {
            index: c.index,
            answer: c.raw_answer.clone(),
            normalized_answer: c.normalized_answer.clone(),
            agrees_with_majority: c.agrees_with_majority,
        }
    }
}

/// Consensus pool from ensemble generation.
#[pyclass(name = "ConsensusPool")]
#[derive(Clone)]
pub struct PyConsensusPool {
    chains: Vec<PyChainResult>,
    selected: String,
}

#[pymethods]
impl PyConsensusPool {
    /// Get all chain results.
    #[getter]
    fn chains(&self) -> Vec<PyChainResult> {
        self.chains.clone()
    }

    /// Get the selected answer.
    #[getter]
    fn selected(&self) -> String {
        self.selected.clone()
    }

    /// Get the agreement ratio (0.0 to 1.0).
    fn agreement_ratio(&self) -> f64 {
        if self.chains.is_empty() {
            return 0.0;
        }
        let agreeing = self
            .chains
            .iter()
            .filter(|c| c.agrees_with_majority)
            .count();
        agreeing as f64 / self.chains.len() as f64
    }

    /// Check if all chains agree.
    fn has_unanimous_agreement(&self) -> bool {
        self.chains.iter().all(|c| c.agrees_with_majority)
    }

    /// Get dissenting chains.
    fn dissenting_chains(&self) -> Vec<PyChainResult> {
        self.chains
            .iter()
            .filter(|c| !c.agrees_with_majority)
            .cloned()
            .collect()
    }

    /// Get vote counts as a dict.
    fn vote_counts(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for chain in &self.chains {
            *counts.entry(chain.normalized_answer.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn __len__(&self) -> usize {
        self.chains.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ConsensusPool(chains={}, agreement={:.1}%)",
            self.chains.len(),
            self.agreement_ratio() * 100.0
        )
    }
}

/// A single step in the agent trajectory.
#[pyclass(name = "Step")]
#[derive(Clone)]
pub struct PyStep {
    /// The agent's reasoning.
    #[pyo3(get)]
    pub thought: String,
    /// The action taken (tool name or "Final Answer").
    #[pyo3(get)]
    pub action: String,
    /// The input to the action/tool.
    #[pyo3(get)]
    pub action_input: String,
    /// The observation/result from the action.
    #[pyo3(get)]
    pub observation: String,
}

#[pymethods]
impl PyStep {
    fn __repr__(&self) -> String {
        format!(
            "Step(action='{}', input='{}')",
            self.action,
            truncate_str(&self.action_input, 30)
        )
    }
}

impl From<&RustStep> for PyStep {
    fn from(s: &RustStep) -> Self {
        Self {
            thought: s.thought.clone(),
            action: s.action.clone(),
            action_input: s.action_input.clone(),
            observation: s.observation.clone(),
        }
    }
}

/// Result from agent execution.
#[pyclass(name = "AgentResult")]
#[derive(Clone)]
pub struct PyAgentResult {
    /// The final output/answer.
    #[pyo3(get)]
    pub output: String,
    /// Number of steps taken.
    #[pyo3(get)]
    pub steps: usize,
    /// Total tokens used.
    #[pyo3(get)]
    pub tokens: u32,
    /// Whether the agent succeeded.
    #[pyo3(get)]
    pub success: bool,
    /// Error if agent failed.
    #[pyo3(get)]
    pub error: Option<String>,
    /// The full trajectory.
    trajectory: Vec<PyStep>,
}

#[pymethods]
impl PyAgentResult {
    /// Get the full trajectory of steps.
    fn trajectory(&self) -> Vec<PyStep> {
        self.trajectory.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentResult(success={}, steps={}, output='{}')",
            self.success,
            self.steps,
            truncate_str(&self.output, 50)
        )
    }

    fn __str__(&self) -> String {
        self.output.clone()
    }
}

/// Result from Program of Thought execution.
#[pyclass(name = "ProgramResult")]
#[derive(Clone)]
pub struct PyProgramResult {
    /// The output from executing the code.
    #[pyo3(get)]
    pub output: String,
    /// The generated code.
    #[pyo3(get)]
    pub code: String,
    /// Number of attempts made.
    #[pyo3(get)]
    pub attempts: usize,
    /// Total tokens used.
    #[pyo3(get)]
    pub tokens: u32,
    /// Whether execution succeeded.
    #[pyo3(get)]
    pub success: bool,
    /// Error message if failed.
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl PyProgramResult {
    fn __repr__(&self) -> String {
        format!(
            "ProgramResult(success={}, attempts={})",
            self.success, self.attempts
        )
    }

    fn __str__(&self) -> String {
        self.output.clone()
    }
}

impl From<ProgramResult> for PyProgramResult {
    fn from(r: ProgramResult) -> Self {
        Self {
            output: r.output,
            code: r.code,
            attempts: r.attempts as usize,
            tokens: r.tokens,
            success: r.success,
            error: r.error,
        }
    }
}

/// Execution result from running code directly.
#[pyclass(name = "ExecutionResult")]
#[derive(Clone)]
pub struct PyExecutionResult {
    /// Standard output.
    #[pyo3(get)]
    pub stdout: String,
    /// Standard error.
    #[pyo3(get)]
    pub stderr: String,
    /// Whether execution succeeded.
    #[pyo3(get)]
    pub success: bool,
    /// Exit code if available.
    #[pyo3(get)]
    pub exit_code: Option<i32>,
    /// Execution duration in milliseconds.
    #[pyo3(get)]
    pub duration_ms: u64,
}

#[pymethods]
impl PyExecutionResult {
    /// Get the primary output.
    fn output(&self) -> &str {
        if !self.stdout.is_empty() {
            &self.stdout
        } else {
            &self.stderr
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionResult(success={}, exit_code={:?})",
            self.success, self.exit_code
        )
    }
}

// ============================================================================
// Tool definition for Python
// ============================================================================

/// A tool definition for use with agents.
///
/// Example:
/// ```python
/// tool = ToolDef("calculator", "Perform math", lambda x: str(eval(x)))
/// result = agent(llm, "What is 2+2?").tool(tool).go()
/// ```
#[pyclass(name = "ToolDef")]
pub struct PyToolDef {
    name: String,
    description: String,
    callable: PyObject,
}

impl Clone for PyToolDef {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            name: self.name.clone(),
            description: self.description.clone(),
            callable: self.callable.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyToolDef {
    /// Create a new tool definition.
    ///
    /// Args:
    ///     name: The tool name.
    ///     description: What the tool does.
    ///     execute: A callable `(input: str) -> str`.
    #[new]
    fn new(name: String, description: String, execute: PyObject) -> Self {
        Self {
            name,
            description,
            callable: execute,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolDef(name='{}', description='{}')",
            self.name, self.description
        )
    }
}

// ============================================================================
// Executor wrapper for Python
// ============================================================================

/// A code executor configuration.
///
/// Example:
/// ```python
/// executor = Executor.python()
/// executor = Executor.bash().timeout(10)
/// result = executor.execute("echo hello")
/// ```
#[pyclass(name = "Executor")]
#[derive(Clone)]
pub struct PyExecutor {
    kind: ExecutorKind,
    timeout_secs: u64,
}

#[derive(Clone)]
enum ExecutorKind {
    Python,
    Node,
    Bash,
    Ruby,
}

#[pymethods]
impl PyExecutor {
    /// Create a Python executor.
    #[staticmethod]
    fn python() -> Self {
        Self {
            kind: ExecutorKind::Python,
            timeout_secs: 30,
        }
    }

    /// Create a Node.js executor.
    #[staticmethod]
    fn node() -> Self {
        Self {
            kind: ExecutorKind::Node,
            timeout_secs: 30,
        }
    }

    /// Create a Bash executor.
    #[staticmethod]
    fn bash() -> Self {
        Self {
            kind: ExecutorKind::Bash,
            timeout_secs: 30,
        }
    }

    /// Create a Ruby executor.
    #[staticmethod]
    fn ruby() -> Self {
        Self {
            kind: ExecutorKind::Ruby,
            timeout_secs: 30,
        }
    }

    /// Set execution timeout in seconds.
    fn timeout(&self, secs: u64) -> Self {
        Self {
            kind: self.kind.clone(),
            timeout_secs: secs,
        }
    }

    /// Execute code directly and return the result.
    fn execute(&self, code: String) -> PyResult<PyExecutionResult> {
        let executor = self.build_executor();
        let result = futures::executor::block_on(executor.execute(&code));
        Ok(PyExecutionResult {
            stdout: result.stdout,
            stderr: result.stderr,
            success: result.success,
            exit_code: result.exit_code,
            duration_ms: result.duration.as_millis() as u64,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Executor(kind='{}', timeout={}s)",
            self.kind_name(),
            self.timeout_secs
        )
    }
}

impl PyExecutor {
    fn build_executor(&self) -> ProcessExecutor {
        let exec = match self.kind {
            ExecutorKind::Python => python_executor(),
            ExecutorKind::Node => node_executor(),
            ExecutorKind::Bash => bash_executor(),
            ExecutorKind::Ruby => ruby_executor(),
        };
        exec.timeout(std::time::Duration::from_secs(self.timeout_secs))
    }

    fn kind_name(&self) -> &str {
        match self.kind {
            ExecutorKind::Python => "python",
            ExecutorKind::Node => "node",
            ExecutorKind::Bash => "bash",
            ExecutorKind::Ruby => "ruby",
        }
    }
}

// ============================================================================
// Refine Builder (works with ApiLlm or callables)
// ============================================================================

/// Builder for iterative refinement (works with ApiLlm or callables).
///
/// Example:
/// ```python
/// from kkachi import refine, Checks, ApiLlm
///
/// llm = ApiLlm.from_env()
/// result = refine(llm, "Write a function") \\
///     .validate(Checks().require("def ")) \\
///     .max_iter(5) \\
///     .go()
/// ```
#[pyclass(name = "RefineBuilder")]
pub struct PyRefineBuilder {
    llm: PyObject,
    prompt: String,
    max_iter: u32,
    target: f64,
    require_patterns: Vec<String>,
    forbid_patterns: Vec<String>,
    validator: Option<ValidatorNode>,
}

impl Clone for PyRefineBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            prompt: self.prompt.clone(),
            max_iter: self.max_iter,
            target: self.target,
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            validator: self.validator.clone(),
        })
    }
}

#[pymethods]
impl PyRefineBuilder {
    #[new]
    fn new(llm: PyObject, prompt: String) -> Self {
        Self {
            llm,
            prompt,
            max_iter: 5,
            target: 1.0,
            require_patterns: Vec::new(),
            forbid_patterns: Vec::new(),
            validator: None,
        }
    }

    /// Set a composed validator.
    fn validate(&self, validator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut new = self.clone();
        new.validator = Some(extract_validator_node(validator)?);
        Ok(new)
    }

    /// Set maximum iterations.
    fn max_iter(&self, n: u32) -> Self {
        let mut new = self.clone();
        new.max_iter = n;
        new
    }

    /// Set target score threshold.
    fn target(&self, score: f64) -> Self {
        let mut new = self.clone();
        new.target = score;
        new
    }

    /// Add a required pattern for validation.
    fn require(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.require_patterns.push(pattern);
        new
    }

    /// Add a forbidden pattern for validation.
    fn forbid(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.forbid_patterns.push(pattern);
        new
    }

    /// Execute refinement with smart LLM detection.
    fn go(&self) -> PyResult<PyRefineResult> {
        Python::with_gil(|py| {
            // Try to extract PyApiLlm first (fast path)
            if let Ok(api_llm) = self.llm.extract::<PyRef<crate::llm::PyApiLlm>>(py) {
                return self.run_with_api_llm(api_llm.inner_ref());
            }

            // Fallback to PyCallableLlm (Python callable path)
            let callable_llm = PyCallableLlm::new_from_ref(py, &self.llm);
            self.run_with_callable(&callable_llm)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RefineBuilder(prompt='{}', max_iter={})",
            truncate_str(&self.prompt, 30),
            self.max_iter
        )
    }
}

impl PyRefineBuilder {
    /// Fast path: use LlmVariant directly (pure Rust async)
    fn run_with_api_llm(&self, llm: &crate::llm::LlmVariant) -> PyResult<PyRefineResult> {
        use kkachi::recursive::refine;
        use pyo3::exceptions::PyRuntimeError;

        let result = if let Some(ref node) = self.validator {
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            refine(llm, &self.prompt)
                .validate(dyn_v)
                .max_iter(self.max_iter)
                .target(self.target)
                .go()
        } else if !self.require_patterns.is_empty() || !self.forbid_patterns.is_empty() {
            let mut cb = checks();
            for pattern in &self.require_patterns {
                cb = cb.require(pattern);
            }
            for pattern in &self.forbid_patterns {
                cb = cb.forbid(pattern);
            }
            refine(llm, &self.prompt)
                .validate(cb)
                .max_iter(self.max_iter)
                .target(self.target)
                .go()
        } else {
            refine(llm, &self.prompt)
                .max_iter(self.max_iter)
                .target(self.target)
                .go()
        };

        result
            .map(|r| r.into())
            .map_err(|e| PyRuntimeError::new_err(format!("Refinement error: {}", e)))
    }

    /// Fallback: use PyCallableLlm (Python callable)
    fn run_with_callable(&self, llm: &PyCallableLlm) -> PyResult<PyRefineResult> {
        use kkachi::recursive::refine;
        use pyo3::exceptions::PyRuntimeError;

        let result = if let Some(ref node) = self.validator {
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            refine(llm, &self.prompt)
                .validate(dyn_v)
                .max_iter(self.max_iter)
                .target(self.target)
                .go()
        } else if !self.require_patterns.is_empty() || !self.forbid_patterns.is_empty() {
            let mut cb = checks();
            for pattern in &self.require_patterns {
                cb = cb.require(pattern);
            }
            for pattern in &self.forbid_patterns {
                cb = cb.forbid(pattern);
            }
            refine(llm, &self.prompt)
                .validate(cb)
                .max_iter(self.max_iter)
                .target(self.target)
                .go()
        } else {
            refine(llm, &self.prompt)
                .max_iter(self.max_iter)
                .target(self.target)
                .go()
        };

        result
            .map(|r| r.into())
            .map_err(|e| PyRuntimeError::new_err(format!("Refinement error: {}", e)))
    }
}

// ============================================================================
// Builder classes
// ============================================================================

/// Builder for Chain of Thought reasoning.
///
/// Example:
/// ```python
/// result = ReasonBuilder(llm, "What is 25 * 37?") \
///     .max_iter(5) \
///     .go()
/// ```
#[pyclass(name = "ReasonBuilder")]
pub struct PyReasonBuilder {
    llm: PyObject,
    prompt: String,
    max_iter: u32,
    target: f64,
    include_reasoning: bool,
    // Inline validation
    require_patterns: Vec<String>,
    forbid_patterns: Vec<String>,
    regex_pattern: Option<String>,
    // Composed validator (overrides inline patterns)
    validator: Option<ValidatorNode>,
}

impl Clone for PyReasonBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            prompt: self.prompt.clone(),
            max_iter: self.max_iter,
            target: self.target,
            include_reasoning: self.include_reasoning,
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            regex_pattern: self.regex_pattern.clone(),
            validator: self.validator.clone(),
        })
    }
}

#[pymethods]
impl PyReasonBuilder {
    /// Create a new Chain of Thought builder.
    ///
    /// Args:
    ///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
    ///     prompt: The question/problem to reason about.
    #[new]
    fn new(llm: PyObject, prompt: String) -> Self {
        Self {
            llm,
            prompt,
            max_iter: 5,
            target: 1.0,
            include_reasoning: true,
            require_patterns: Vec::new(),
            forbid_patterns: Vec::new(),
            regex_pattern: None,
            validator: None,
        }
    }

    /// Set a composed validator (Checks, Semantic, or Validator).
    ///
    /// Overrides inline .require()/.forbid()/.regex() patterns.
    fn validate(&self, validator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut new = self.clone();
        new.validator = Some(extract_validator_node(validator)?);
        Ok(new)
    }

    /// Set maximum refinement iterations.
    fn max_iter(&self, n: u32) -> Self {
        let mut new = self.clone();
        new.max_iter = n;
        new
    }

    /// Set target validation score.
    fn target(&self, score: f64) -> Self {
        let mut new = self.clone();
        new.target = score;
        new
    }

    /// Disable reasoning inclusion in result.
    fn no_reasoning(&self) -> Self {
        let mut new = self.clone();
        new.include_reasoning = false;
        new
    }

    /// Add a required pattern for validation.
    fn require(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.require_patterns.push(pattern);
        new
    }

    /// Add a forbidden pattern for validation.
    fn forbid(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.forbid_patterns.push(pattern);
        new
    }

    /// Add a regex pattern for validation.
    fn regex(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.regex_pattern = Some(pattern);
        new
    }

    /// Execute the Chain of Thought reasoning.
    fn go(&self) -> PyResult<PyReasonResult> {
        Python::with_gil(|py| {
            // Try ApiLlm first (fast path - no executor nesting)
            if let Ok(api_llm) = self.llm.extract::<PyRef<crate::llm::PyApiLlm>>(py) {
                return self.run_with_api_llm(api_llm.inner_ref());
            }

            // Fallback to PyCallableLlm (Python callable)
            let callable_llm = PyCallableLlm::new_from_ref(py, &self.llm);
            self.run_with_callable(&callable_llm)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ReasonBuilder(prompt='{}', max_iter={})",
            truncate_str(&self.prompt, 30),
            self.max_iter
        )
    }
}

impl PyReasonBuilder {
    fn run_with_api_llm(&self, llm: &crate::llm::LlmVariant) -> PyResult<PyReasonResult> {
        

        let result = if let Some(ref node) = self.validator {
            // Composed validator takes priority
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            let mut builder = reason(llm, &self.prompt)
                .validate(dyn_v)
                .max_iter(self.max_iter)
                .target(self.target);
            if !self.include_reasoning {
                builder = builder.no_reasoning();
            }
            builder.go()
        } else {
            let has_inline = !self.require_patterns.is_empty()
                || !self.forbid_patterns.is_empty()
                || self.regex_pattern.is_some();

            if has_inline {
                let mut cb = checks();
                for pattern in &self.require_patterns {
                    cb = cb.require(pattern);
                }
                for pattern in &self.forbid_patterns {
                    cb = cb.forbid(pattern);
                }
                if let Some(ref regex) = self.regex_pattern {
                    cb = cb.regex(regex);
                }
                let mut builder = reason(llm, &self.prompt)
                    .validate(cb)
                    .max_iter(self.max_iter)
                    .target(self.target);
                if !self.include_reasoning {
                    builder = builder.no_reasoning();
                }
                builder.go()
            } else {
                let mut builder = reason(llm, &self.prompt)
                    .max_iter(self.max_iter)
                    .target(self.target);
                if !self.include_reasoning {
                    builder = builder.no_reasoning();
                }
                builder.go()
            }
        };

        Ok(result.into())
    }

    fn run_with_callable(&self, llm: &PyCallableLlm) -> PyResult<PyReasonResult> {

        let result = if let Some(ref node) = self.validator {
            // Composed validator takes priority
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            let mut builder = reason(llm, &self.prompt)
                .validate(dyn_v)
                .max_iter(self.max_iter)
                .target(self.target);
            if !self.include_reasoning {
                builder = builder.no_reasoning();
            }
            builder.go()
        } else {
            let has_inline = !self.require_patterns.is_empty()
                || !self.forbid_patterns.is_empty()
                || self.regex_pattern.is_some();

            if has_inline {
                let mut cb = checks();
                for pattern in &self.require_patterns {
                    cb = cb.require(pattern);
                }
                for pattern in &self.forbid_patterns {
                    cb = cb.forbid(pattern);
                }
                if let Some(ref regex) = self.regex_pattern {
                    cb = cb.regex(regex);
                }
                let mut builder = reason(llm, &self.prompt)
                    .validate(cb)
                    .max_iter(self.max_iter)
                    .target(self.target);
                if !self.include_reasoning {
                    builder = builder.no_reasoning();
                }
                builder.go()
            } else {
                let mut builder = reason(llm, &self.prompt)
                    .max_iter(self.max_iter)
                    .target(self.target);
                if !self.include_reasoning {
                    builder = builder.no_reasoning();
                }
                builder.go()
            }
        };

        Ok(result.into())
    }
}

/// Builder for Best of N generation.
///
/// Example:
/// ```python
/// result = BestOfBuilder(llm, "Write a haiku", 5) \
///     .metric(lambda x: 1.0 if len(x.splitlines()) == 3 else 0.0) \
///     .go()
/// ```
#[pyclass(name = "BestOfBuilder")]
pub struct PyBestOfBuilder {
    llm: PyObject,
    prompt: String,
    n: usize,
    scorer: Option<PyObject>,
    scorer_weight: f64,
    with_reasoning: bool,
    // Inline validation
    require_patterns: Vec<String>,
    forbid_patterns: Vec<String>,
    // Composed validator (overrides inline patterns)
    validator: Option<ValidatorNode>,
}

impl Clone for PyBestOfBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            prompt: self.prompt.clone(),
            n: self.n,
            scorer: self.scorer.as_ref().map(|s| s.clone_ref(py)),
            scorer_weight: self.scorer_weight,
            with_reasoning: self.with_reasoning,
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            validator: self.validator.clone(),
        })
    }
}

#[pymethods]
impl PyBestOfBuilder {
    /// Create a new Best of N builder.
    ///
    /// Args:
    ///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
    ///     prompt: The prompt for generation.
    ///     n: Number of candidates to generate.
    #[new]
    fn new(llm: PyObject, prompt: String, n: usize) -> Self {
        Self {
            llm,
            prompt,
            n: n.max(1),
            scorer: None,
            scorer_weight: 0.5,
            with_reasoning: false,
            require_patterns: Vec::new(),
            forbid_patterns: Vec::new(),
            validator: None,
        }
    }

    /// Set a composed validator (Checks, Semantic, or Validator).
    fn validate(&self, validator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut new = self.clone();
        new.validator = Some(extract_validator_node(validator)?);
        Ok(new)
    }

    /// Set a custom scoring metric.
    ///
    /// Args:
    ///     scorer: A callable `(output: str) -> float` returning 0.0-1.0.
    fn metric(&self, scorer: PyObject) -> Self {
        let mut new = self.clone();
        new.scorer = Python::with_gil(|py| Some(scorer.clone_ref(py)));
        new
    }

    /// Set scorer weight vs validator weight.
    fn scorer_weight(&self, weight: f64) -> Self {
        let mut new = self.clone();
        new.scorer_weight = weight.clamp(0.0, 1.0);
        new
    }

    /// Enable Chain of Thought for each candidate.
    fn with_reasoning(&self) -> Self {
        let mut new = self.clone();
        new.with_reasoning = true;
        new
    }

    /// Add a required pattern for validation.
    fn require(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.require_patterns.push(pattern);
        new
    }

    /// Add a forbidden pattern for validation.
    fn forbid(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.forbid_patterns.push(pattern);
        new
    }

    /// Execute and return the best result.
    fn go(&self) -> PyResult<PyBestOfResult> {
        let (result, _) = self.run_internal()?;
        Ok(result)
    }

    /// Execute and return result with candidate pool.
    fn go_with_pool(&self) -> PyResult<(PyBestOfResult, PyCandidatePool)> {
        self.run_internal()
    }

    fn __repr__(&self) -> String {
        format!(
            "BestOfBuilder(n={}, prompt='{}')",
            self.n,
            truncate_str(&self.prompt, 30)
        )
    }
}

impl PyBestOfBuilder {
    fn run_internal(&self) -> PyResult<(PyBestOfResult, PyCandidatePool)> {
        Python::with_gil(|py| {
            // Try ApiLlm first (fast path - no executor nesting)
            if let Ok(api_llm) = self.llm.extract::<PyRef<crate::llm::PyApiLlm>>(py) {
                return self.run_internal_with_api_llm(api_llm.inner_ref());
            }

            // Fallback to PyCallableLlm (Python callable)
            let callable_llm = PyCallableLlm::new_from_ref(py, &self.llm);
            self.run_internal_with_callable(&callable_llm)
        })
    }

    fn run_internal_with_api_llm(&self, llm: &crate::llm::LlmVariant) -> PyResult<(PyBestOfResult, PyCandidatePool)> {
        let scorer_obj = Python::with_gil(|py| self.scorer.as_ref().map(|s| s.clone_ref(py)));

        // Build scorer closure if provided
        let py_scorer = scorer_obj.map(|scorer_py| {
            move |output: &str| -> f64 {
                Python::with_gil(|py| {
                    scorer_py
                        .call1(py, (output.to_string(),))
                        .and_then(|r| r.extract::<f64>(py))
                        .unwrap_or(0.0)
                })
            }
        });

        let (result, pool) = if let Some(ref node) = self.validator {
            // Composed validator takes priority
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            if let Some(scorer) = py_scorer {
                let mut builder = best_of(llm, &self.prompt)
                    .n(self.n)
                    .validate(dyn_v)
                    .metric(scorer)
                    .scorer_weight(self.scorer_weight);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                builder.go_with_pool()
            } else {
                let mut builder = best_of(llm, &self.prompt)
                    .n(self.n)
                    .validate(dyn_v)
                    .scorer_weight(self.scorer_weight);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                builder.go_with_pool()
            }
        } else {
            let has_inline = !self.require_patterns.is_empty() || !self.forbid_patterns.is_empty();

            if has_inline {
                let mut cb = checks();
                for pattern in &self.require_patterns {
                    cb = cb.require(pattern);
                }
                for pattern in &self.forbid_patterns {
                    cb = cb.forbid(pattern);
                }

                if let Some(scorer) = py_scorer {
                    let mut builder = best_of(llm, &self.prompt)
                        .n(self.n)
                        .validate(cb)
                        .metric(scorer)
                        .scorer_weight(self.scorer_weight);
                    if self.with_reasoning {
                        builder = builder.with_reasoning();
                    }
                    builder.go_with_pool()
                } else {
                    let mut builder = best_of(llm, &self.prompt)
                        .n(self.n)
                        .validate(cb)
                        .scorer_weight(self.scorer_weight);
                    if self.with_reasoning {
                        builder = builder.with_reasoning();
                    }
                    builder.go_with_pool()
                }
            } else if let Some(scorer) = py_scorer {
                let mut builder = best_of(llm, &self.prompt)
                    .n(self.n)
                    .metric(scorer)
                    .scorer_weight(self.scorer_weight);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                builder.go_with_pool()
            } else {
                let mut builder = best_of(llm, &self.prompt)
                    .n(self.n)
                    .scorer_weight(self.scorer_weight);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                builder.go_with_pool()
            }
        };

        let py_result = PyBestOfResult::from(result);
        let py_pool = PyCandidatePool {
            candidates: pool
                .candidates()
                .iter()
                .map(PyScoredCandidate::from)
                .collect(),
            total_tokens: pool.total_tokens(),
        };

        Ok((py_result, py_pool))
    }

    fn run_internal_with_callable(&self, llm: &PyCallableLlm) -> PyResult<(PyBestOfResult, PyCandidatePool)> {
        let scorer_obj = Python::with_gil(|py| self.scorer.as_ref().map(|s| s.clone_ref(py)));

        // Build scorer closure if provided
        let py_scorer = scorer_obj.map(|scorer_py| {
            move |output: &str| -> f64 {
                Python::with_gil(|py| {
                    scorer_py
                        .call1(py, (output.to_string(),))
                        .and_then(|r| r.extract::<f64>(py))
                        .unwrap_or(0.0)
                })
            }
        });

        let (result, pool) = if let Some(ref node) = self.validator {
            // Composed validator takes priority
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            if let Some(scorer) = py_scorer {
                let mut builder = best_of(llm, &self.prompt)
                    .n(self.n)
                    .validate(dyn_v)
                    .metric(scorer)
                    .scorer_weight(self.scorer_weight);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                builder.go_with_pool()
            } else {
                let mut builder = best_of(llm, &self.prompt)
                    .n(self.n)
                    .validate(dyn_v)
                    .scorer_weight(self.scorer_weight);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                builder.go_with_pool()
            }
        } else {
            let has_inline = !self.require_patterns.is_empty() || !self.forbid_patterns.is_empty();

            if has_inline {
                let mut cb = checks();
                for pattern in &self.require_patterns {
                    cb = cb.require(pattern);
                }
                for pattern in &self.forbid_patterns {
                    cb = cb.forbid(pattern);
                }

                if let Some(scorer) = py_scorer {
                    let mut builder = best_of(llm, &self.prompt)
                        .n(self.n)
                        .validate(cb)
                        .metric(scorer)
                        .scorer_weight(self.scorer_weight);
                    if self.with_reasoning {
                        builder = builder.with_reasoning();
                    }
                    builder.go_with_pool()
                } else {
                    let mut builder = best_of(llm, &self.prompt)
                        .n(self.n)
                        .validate(cb)
                        .scorer_weight(self.scorer_weight);
                    if self.with_reasoning {
                        builder = builder.with_reasoning();
                    }
                    builder.go_with_pool()
                }
            } else if let Some(scorer) = py_scorer {
                let mut builder = best_of(llm, &self.prompt)
                    .n(self.n)
                    .metric(scorer)
                    .scorer_weight(self.scorer_weight);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                builder.go_with_pool()
            } else {
                let mut builder = best_of(llm, &self.prompt)
                    .n(self.n)
                    .scorer_weight(self.scorer_weight);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                builder.go_with_pool()
            }
        };

        let py_result = PyBestOfResult::from(result);
        let py_pool = PyCandidatePool {
            candidates: pool
                .candidates()
                .iter()
                .map(PyScoredCandidate::from)
                .collect(),
            total_tokens: pool.total_tokens(),
        };

        Ok((py_result, py_pool))
    }
}

/// Builder for ensemble (multi-chain) generation.
///
/// Example:
/// ```python
/// result = EnsembleBuilder(llm, "What is the capital of France?", 5) \
///     .aggregate("majority_vote") \
///     .go()
/// ```
#[pyclass(name = "EnsembleBuilder")]
pub struct PyEnsembleBuilder {
    llm: PyObject,
    prompt: String,
    n: usize,
    aggregate: String,
    with_reasoning: bool,
    normalize: bool,
    // Inline validation
    require_patterns: Vec<String>,
    forbid_patterns: Vec<String>,
    // Composed validator
    validator: Option<ValidatorNode>,
}

impl Clone for PyEnsembleBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            prompt: self.prompt.clone(),
            n: self.n,
            aggregate: self.aggregate.clone(),
            with_reasoning: self.with_reasoning,
            normalize: self.normalize,
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            validator: self.validator.clone(),
        })
    }
}

#[pymethods]
impl PyEnsembleBuilder {
    /// Create a new ensemble builder.
    ///
    /// Args:
    ///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
    ///     prompt: The prompt/question.
    ///     n: Number of chains to generate.
    #[new]
    fn new(llm: PyObject, prompt: String, n: usize) -> Self {
        Self {
            llm,
            prompt,
            n: n.max(1),
            aggregate: "majority_vote".to_string(),
            with_reasoning: false,
            normalize: true,
            require_patterns: Vec::new(),
            forbid_patterns: Vec::new(),
            validator: None,
        }
    }

    /// Set a composed validator (Checks, Semantic, or Validator).
    fn validate(&self, validator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut new = self.clone();
        new.validator = Some(extract_validator_node(validator)?);
        Ok(new)
    }

    /// Set aggregation strategy.
    ///
    /// Options: "majority_vote", "longest", "shortest", "first_success", "unanimous"
    fn aggregate(&self, strategy: String) -> Self {
        let mut new = self.clone();
        new.aggregate = strategy;
        new
    }

    /// Enable Chain of Thought for each chain.
    fn with_reasoning(&self) -> Self {
        let mut new = self.clone();
        new.with_reasoning = true;
        new
    }

    /// Disable answer normalization (trim + lowercase).
    fn no_normalize(&self) -> Self {
        let mut new = self.clone();
        new.normalize = false;
        new
    }

    /// Add a required pattern for validation.
    fn require(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.require_patterns.push(pattern);
        new
    }

    /// Add a forbidden pattern for validation.
    fn forbid(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.forbid_patterns.push(pattern);
        new
    }

    /// Execute and return the selected result.
    fn go(&self) -> PyResult<PyEnsembleResult> {
        let (result, _) = self.run_internal()?;
        Ok(result)
    }

    /// Execute and return result with consensus pool.
    fn go_with_consensus(&self) -> PyResult<(PyEnsembleResult, PyConsensusPool)> {
        self.run_internal()
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleBuilder(n={}, aggregate='{}')",
            self.n, self.aggregate
        )
    }
}

impl PyEnsembleBuilder {
    fn parse_aggregate(&self) -> Aggregate {
        match self.aggregate.as_str() {
            "longest" => Aggregate::LongestAnswer,
            "shortest" => Aggregate::ShortestAnswer,
            "first_success" => Aggregate::FirstSuccess,
            "unanimous" => Aggregate::Unanimous,
            _ => Aggregate::MajorityVote,
        }
    }

    fn run_internal(&self) -> PyResult<(PyEnsembleResult, PyConsensusPool)> {
        Python::with_gil(|py| {
            // Try ApiLlm first (fast path - no executor nesting)
            if let Ok(api_llm) = self.llm.extract::<PyRef<crate::llm::PyApiLlm>>(py) {
                return self.run_internal_with_api_llm(api_llm.inner_ref());
            }

            // Fallback to PyCallableLlm (Python callable)
            let callable_llm = PyCallableLlm::new_from_ref(py, &self.llm);
            self.run_internal_with_callable(&callable_llm)
        })
    }

    fn run_internal_with_api_llm(&self, llm: &crate::llm::LlmVariant) -> PyResult<(PyEnsembleResult, PyConsensusPool)> {
        let aggregate = self.parse_aggregate();

        let (result, consensus) = if let Some(ref node) = self.validator {
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            let mut builder = ensemble(llm, &self.prompt)
                .n(self.n)
                .validate(dyn_v)
                .aggregate(aggregate);
            if self.with_reasoning {
                builder = builder.with_reasoning();
            }
            if !self.normalize {
                builder = builder.no_normalize();
            }
            builder.go_with_consensus()
        } else {
            let has_inline = !self.require_patterns.is_empty() || !self.forbid_patterns.is_empty();

            if has_inline {
                let mut cb = checks();
                for pattern in &self.require_patterns {
                    cb = cb.require(pattern);
                }
                for pattern in &self.forbid_patterns {
                    cb = cb.forbid(pattern);
                }
                let mut builder = ensemble(llm, &self.prompt)
                    .n(self.n)
                    .validate(cb)
                    .aggregate(aggregate);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                if !self.normalize {
                    builder = builder.no_normalize();
                }
                builder.go_with_consensus()
            } else {
                let mut builder = ensemble(llm, &self.prompt).n(self.n).aggregate(aggregate);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                if !self.normalize {
                    builder = builder.no_normalize();
                }
                builder.go_with_consensus()
            }
        };

        let py_result = PyEnsembleResult::from(result);
        let py_consensus = PyConsensusPool {
            chains: consensus.chains().iter().map(PyChainResult::from).collect(),
            selected: consensus.selected().to_string(),
        };

        Ok((py_result, py_consensus))
    }

    fn run_internal_with_callable(&self, llm: &PyCallableLlm) -> PyResult<(PyEnsembleResult, PyConsensusPool)> {
        let aggregate = self.parse_aggregate();

        let (result, consensus) = if let Some(ref node) = self.validator {
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            let mut builder = ensemble(llm, &self.prompt)
                .n(self.n)
                .validate(dyn_v)
                .aggregate(aggregate);
            if self.with_reasoning {
                builder = builder.with_reasoning();
            }
            if !self.normalize {
                builder = builder.no_normalize();
            }
            builder.go_with_consensus()
        } else {
            let has_inline = !self.require_patterns.is_empty() || !self.forbid_patterns.is_empty();

            if has_inline {
                let mut cb = checks();
                for pattern in &self.require_patterns {
                    cb = cb.require(pattern);
                }
                for pattern in &self.forbid_patterns {
                    cb = cb.forbid(pattern);
                }
                let mut builder = ensemble(llm, &self.prompt)
                    .n(self.n)
                    .validate(cb)
                    .aggregate(aggregate);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                if !self.normalize {
                    builder = builder.no_normalize();
                }
                builder.go_with_consensus()
            } else {
                let mut builder = ensemble(llm, &self.prompt).n(self.n).aggregate(aggregate);
                if self.with_reasoning {
                    builder = builder.with_reasoning();
                }
                if !self.normalize {
                    builder = builder.no_normalize();
                }
                builder.go_with_consensus()
            }
        };

        let py_result = PyEnsembleResult::from(result);
        let py_consensus = PyConsensusPool {
            chains: consensus.chains().iter().map(PyChainResult::from).collect(),
            selected: consensus.selected().to_string(),
        };

        Ok((py_result, py_consensus))
    }
}

/// Builder for ReAct agent.
///
/// Example:
/// ```python
/// calc = ToolDef("calculator", "Do math", lambda x: str(eval(x)))
/// result = AgentBuilder(llm, "What is 2+2?") \
///     .tool(calc) \
///     .max_steps(10) \
///     .go()
/// ```
#[pyclass(name = "AgentBuilder")]
pub struct PyAgentBuilder {
    llm: PyObject,
    goal: String,
    tools: Vec<PyToolDef>,
    max_steps: usize,
}

impl Clone for PyAgentBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            goal: self.goal.clone(),
            tools: self.tools.clone(),
            max_steps: self.max_steps,
        })
    }
}

#[pymethods]
impl PyAgentBuilder {
    /// Create a new agent builder.
    ///
    /// Args:
    ///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
    ///     goal: The goal for the agent to accomplish.
    #[new]
    fn new(llm: PyObject, goal: String) -> Self {
        Self {
            llm,
            goal,
            tools: Vec::new(),
            max_steps: 10,
        }
    }

    /// Add a tool to the agent.
    fn tool(&self, tool_def: PyToolDef) -> Self {
        let mut new = self.clone();
        new.tools.push(tool_def);
        new
    }

    /// Set maximum reasoning steps.
    fn max_steps(&self, n: usize) -> Self {
        let mut new = self.clone();
        new.max_steps = n;
        new
    }

    /// Execute the agent.
    fn go(&self) -> PyResult<PyAgentResult> {
        Python::with_gil(|py| {
            // Try ApiLlm first (fast path - no executor nesting)
            if let Ok(api_llm) = self.llm.extract::<PyRef<crate::llm::PyApiLlm>>(py) {
                return self.run_with_api_llm(api_llm.inner_ref());
            }

            // Fallback to PyCallableLlm (Python callable)
            let callable_llm = PyCallableLlm::new_from_ref(py, &self.llm);
            self.run_with_callable(&callable_llm)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentBuilder(goal='{}', tools={}, max_steps={})",
            truncate_str(&self.goal, 30),
            self.tools.len(),
            self.max_steps
        )
    }
}

impl PyAgentBuilder {
    fn run_with_api_llm(&self, llm: &crate::llm::LlmVariant) -> PyResult<PyAgentResult> {
        // Create Rust tools from PyToolDefs
        let rust_tools: Vec<PyCallableTool> = Python::with_gil(|py| {
            self.tools
                .iter()
                .map(|t| PyCallableTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    callable: t.callable.clone_ref(py),
                })
                .collect()
        });

        // Build the agent with tool references
        let mut builder = agent(llm, &self.goal).max_steps(self.max_steps);
        for tool in &rust_tools {
            builder = builder.tool(tool);
        }

        let result = builder.go();

        Ok(PyAgentResult {
            output: result.output,
            steps: result.steps,
            tokens: result.tokens,
            success: result.success,
            error: result.error,
            trajectory: result.trajectory.iter().map(PyStep::from).collect(),
        })
    }

    fn run_with_callable(&self, llm: &PyCallableLlm) -> PyResult<PyAgentResult> {
        // Create Rust tools from PyToolDefs
        let rust_tools: Vec<PyCallableTool> = Python::with_gil(|py| {
            self.tools
                .iter()
                .map(|t| PyCallableTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    callable: t.callable.clone_ref(py),
                })
                .collect()
        });

        // Build the agent with tool references
        let mut builder = agent(llm, &self.goal).max_steps(self.max_steps);
        for tool in &rust_tools {
            builder = builder.tool(tool);
        }

        let result = builder.go();

        Ok(PyAgentResult {
            output: result.output,
            steps: result.steps,
            tokens: result.tokens,
            success: result.success,
            error: result.error,
            trajectory: result.trajectory.iter().map(PyStep::from).collect(),
        })
    }
}

/// Builder for Program of Thought.
///
/// Example:
/// ```python
/// result = ProgramBuilder(llm, "Calculate Fibonacci(50)") \
///     .executor(Executor.python()) \
///     .max_iter(3) \
///     .go()
/// ```
#[pyclass(name = "ProgramBuilder")]
pub struct PyProgramBuilder {
    llm: PyObject,
    problem: String,
    executor: Option<PyExecutor>,
    max_iter: usize,
    include_code: bool,
    language: Option<String>,
    // Inline validation
    require_patterns: Vec<String>,
    forbid_patterns: Vec<String>,
    regex_pattern: Option<String>,
    // Composed validator
    validator: Option<ValidatorNode>,
}

impl Clone for PyProgramBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            problem: self.problem.clone(),
            executor: self.executor.clone(),
            max_iter: self.max_iter,
            include_code: self.include_code,
            language: self.language.clone(),
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            regex_pattern: self.regex_pattern.clone(),
            validator: self.validator.clone(),
        })
    }
}

#[pymethods]
impl PyProgramBuilder {
    /// Create a new Program of Thought builder.
    ///
    /// Args:
    ///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
    ///     problem: The problem to solve with code.
    #[new]
    fn new(llm: PyObject, problem: String) -> Self {
        Self {
            llm,
            problem,
            executor: None,
            max_iter: 3,
            include_code: true,
            language: None,
            require_patterns: Vec::new(),
            forbid_patterns: Vec::new(),
            regex_pattern: None,
            validator: None,
        }
    }

    /// Set a composed validator (Checks, Semantic, or Validator).
    fn validate(&self, validator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut new = self.clone();
        new.validator = Some(extract_validator_node(validator)?);
        Ok(new)
    }

    /// Set the code executor.
    fn executor(&self, executor: PyExecutor) -> Self {
        let mut new = self.clone();
        new.executor = Some(executor);
        new
    }

    /// Set maximum code generation attempts.
    fn max_iter(&self, n: usize) -> Self {
        let mut new = self.clone();
        new.max_iter = n;
        new
    }

    /// Disable code inclusion in result.
    fn no_code(&self) -> Self {
        let mut new = self.clone();
        new.include_code = false;
        new
    }

    /// Override the target programming language.
    fn language(&self, lang: String) -> Self {
        let mut new = self.clone();
        new.language = Some(lang);
        new
    }

    /// Add a required pattern for output validation.
    fn require(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.require_patterns.push(pattern);
        new
    }

    /// Add a forbidden pattern for output validation.
    fn forbid(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.forbid_patterns.push(pattern);
        new
    }

    /// Add a regex pattern for output validation.
    fn regex(&self, pattern: String) -> Self {
        let mut new = self.clone();
        new.regex_pattern = Some(pattern);
        new
    }

    /// Execute the Program of Thought.
    fn go(&self) -> PyResult<PyProgramResult> {
        let executor = self.executor.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("No executor set. Use .executor(Executor.python()) or similar.")
        })?;

        Python::with_gil(|py| {
            // Try ApiLlm first (fast path - no executor nesting)
            if let Ok(api_llm) = self.llm.extract::<PyRef<crate::llm::PyApiLlm>>(py) {
                return self.run_with_api_llm(api_llm.inner_ref(), executor);
            }

            // Fallback to PyCallableLlm (Python callable)
            let callable_llm = PyCallableLlm::new_from_ref(py, &self.llm);
            self.run_with_callable(&callable_llm, executor)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgramBuilder(problem='{}', max_iter={})",
            truncate_str(&self.problem, 30),
            self.max_iter
        )
    }
}

impl PyProgramBuilder {
    fn run_with_api_llm(&self, llm: &crate::llm::LlmVariant, executor: &PyExecutor) -> PyResult<PyProgramResult> {
        

        let rust_executor = executor.build_executor();

        let result = if let Some(ref node) = self.validator {
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            let mut builder = program(llm, &self.problem)
                .executor(rust_executor)
                .validate(dyn_v)
                .max_iter(self.max_iter as u32);
            if !self.include_code {
                builder = builder.no_code();
            }
            if let Some(ref lang) = self.language {
                builder = builder.language(lang);
            }
            builder.go()
        } else {
            let has_inline = !self.require_patterns.is_empty()
                || !self.forbid_patterns.is_empty()
                || self.regex_pattern.is_some();

            if has_inline {
                let mut cb = checks();
                for pattern in &self.require_patterns {
                    cb = cb.require(pattern);
                }
                for pattern in &self.forbid_patterns {
                    cb = cb.forbid(pattern);
                }
                if let Some(ref regex) = self.regex_pattern {
                    cb = cb.regex(regex);
                }
                let mut builder = program(llm, &self.problem)
                    .executor(rust_executor)
                    .validate(cb)
                    .max_iter(self.max_iter as u32);
                if !self.include_code {
                    builder = builder.no_code();
                }
                if let Some(ref lang) = self.language {
                    builder = builder.language(lang);
                }
                builder.go()
            } else {
                let mut builder = program(llm, &self.problem)
                    .executor(rust_executor)
                    .max_iter(self.max_iter as u32);
                if !self.include_code {
                    builder = builder.no_code();
                }
                if let Some(ref lang) = self.language {
                    builder = builder.language(lang);
                }
                builder.go()
            }
        };

        Ok(result.into())
    }

    fn run_with_callable(&self, llm: &PyCallableLlm, executor: &PyExecutor) -> PyResult<PyProgramResult> {

        let rust_executor = executor.build_executor();

        let result = if let Some(ref node) = self.validator {
            let dyn_v = Python::with_gil(|py| DynValidator(node.materialize(py)));
            let mut builder = program(llm, &self.problem)
                .executor(rust_executor)
                .validate(dyn_v)
                .max_iter(self.max_iter as u32);
            if !self.include_code {
                builder = builder.no_code();
            }
            if let Some(ref lang) = self.language {
                builder = builder.language(lang);
            }
            builder.go()
        } else {
            let has_inline = !self.require_patterns.is_empty()
                || !self.forbid_patterns.is_empty()
                || self.regex_pattern.is_some();

            if has_inline {
                let mut cb = checks();
                for pattern in &self.require_patterns {
                    cb = cb.require(pattern);
                }
                for pattern in &self.forbid_patterns {
                    cb = cb.forbid(pattern);
                }
                if let Some(ref regex) = self.regex_pattern {
                    cb = cb.regex(regex);
                }
                let mut builder = program(llm, &self.problem)
                    .executor(rust_executor)
                    .validate(cb)
                    .max_iter(self.max_iter as u32);
                if !self.include_code {
                    builder = builder.no_code();
                }
                if let Some(ref lang) = self.language {
                    builder = builder.language(lang);
                }
                builder.go()
            } else {
                let mut builder = program(llm, &self.problem)
                    .executor(rust_executor)
                    .max_iter(self.max_iter as u32);
                if !self.include_code {
                    builder = builder.no_code();
                }
                if let Some(ref lang) = self.language {
                    builder = builder.language(lang);
                }
                builder.go()
            }
        };

        Ok(result.into())
    }
}

// ============================================================================
// Module-level functions (convenience entry points)
// ============================================================================

/// Iterative refinement with validation.
///
/// Args:
///     llm: An ApiLlm or callable `(prompt: str, feedback: Optional[str]) -> str`
///     prompt: The prompt/task to refine.
///
/// Returns:
///     RefineBuilderV2: A builder to configure and execute refinement.
///
/// Example:
///     ```python
///     from kkachi import refine, Checks, ApiLlm
///
///     llm = ApiLlm.from_env()
///     result = refine(llm, "Write a function") \\
///         .validate(Checks().require("def ")) \\
///         .max_iter(5) \\
///         .go()
///     ```
#[pyfunction]
#[pyo3(name = "refine")]
pub fn py_refine(llm: PyObject, prompt: String) -> PyRefineBuilder {
    PyRefineBuilder::new(llm, prompt)
}

/// Chain of Thought reasoning.
///
/// Args:
///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
///     prompt: The question/problem.
///
/// Returns:
///     ReasonBuilder: A builder to configure and execute reasoning.
#[pyfunction]
#[pyo3(name = "reason")]
pub fn py_reason(llm: PyObject, prompt: String) -> PyReasonBuilder {
    PyReasonBuilder::new(llm, prompt)
}

/// Best of N candidate generation.
///
/// Args:
///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
///     prompt: The prompt for generation.
///     n: Number of candidates to generate.
///
/// Returns:
///     BestOfBuilder: A builder to configure and execute generation.
#[pyfunction]
#[pyo3(name = "best_of")]
pub fn py_best_of(llm: PyObject, prompt: String, n: usize) -> PyBestOfBuilder {
    PyBestOfBuilder::new(llm, prompt, n)
}

/// Multi-chain ensemble with voting.
///
/// Args:
///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
///     prompt: The prompt/question.
///     n: Number of chains to generate.
///
/// Returns:
///     EnsembleBuilder: A builder to configure and execute ensemble.
#[pyfunction]
#[pyo3(name = "ensemble")]
pub fn py_ensemble(llm: PyObject, prompt: String, n: usize) -> PyEnsembleBuilder {
    PyEnsembleBuilder::new(llm, prompt, n)
}

/// ReAct agent with tool calling.
///
/// Args:
///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
///     goal: The goal for the agent.
///
/// Returns:
///     AgentBuilder: A builder to configure and execute the agent.
#[pyfunction]
#[pyo3(name = "agent")]
pub fn py_agent(llm: PyObject, goal: String) -> PyAgentBuilder {
    PyAgentBuilder::new(llm, goal)
}

/// Program of Thought - code generation and execution.
///
/// Args:
///     llm: A callable `(prompt: str, feedback: Optional[str]) -> str`
///     problem: The problem to solve with code.
///
/// Returns:
///     ProgramBuilder: A builder to configure and execute.
#[pyfunction]
#[pyo3(name = "program")]
pub fn py_program(llm: PyObject, problem: String) -> PyProgramBuilder {
    PyProgramBuilder::new(llm, problem)
}

// ============================================================================
// Helpers
// ============================================================================

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}
