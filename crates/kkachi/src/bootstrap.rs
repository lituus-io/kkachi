// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! BootstrapFewShot optimizer implementation
//!
//! Generates predictions via a teacher LLM, evaluates each with a [`Metric`],
//! and selects the top-K scoring examples as demonstrations.
//!
//! ## Algorithm
//!
//! 1. For each training example, build a prompt from the example's input fields
//! 2. Call the teacher LLM to generate a prediction
//! 3. Score the prediction against the expected output using the metric
//! 4. Collect all (index, score) pairs
//! 5. Sort by score descending, take the top `max_demos` as demonstrations
//!
//! ## Design
//!
//! - Uses GATs for zero-cost async (via the [`Llm`] trait)
//! - Returns indices into the training buffer, not copies
//! - Returns a [`CompiledProgram`] from the full bootstrap method
//! - Configurable via [`OptimizerConfig`]

use crate::compiled::CompiledProgram;
use crate::error::Result;
use crate::evaluate::Evaluate;
use crate::intern::Sym;
use crate::metric::Metric;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig, Rng};
use crate::recursive::llm::Llm;
use smallvec::SmallVec;

/// BootstrapFewShot optimizer.
///
/// Bootstraps demonstrations by running a teacher LLM on training examples
/// and selecting the top-scoring predictions (according to a metric) as
/// demonstrations for future prompts.
///
/// ## Zero-Copy Design
///
/// - Returns indices into the training buffer, not copies
/// - Uses GATs for zero-cost async
/// - Configurable via `OptimizerConfig`
#[derive(Clone, Copy)]
pub struct BootstrapFewShot {
    config: OptimizerConfig,
}

impl BootstrapFewShot {
    /// Create a new BootstrapFewShot optimizer.
    pub const fn new(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub const fn default() -> Self {
        Self::new(OptimizerConfig::new())
    }

    /// Set maximum number of demonstrations.
    pub const fn with_max_demos(mut self, n: u8) -> Self {
        self.config.max_demos = n;
        self
    }

    /// Set metric threshold.
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.config.metric_threshold = threshold;
        self
    }

    /// Get the configuration.
    pub const fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Bootstrap demos using a teacher LLM and metric.
    ///
    /// For each training example (up to `batch_size`), calls the teacher LLM
    /// with the example's input fields as the prompt and scores the prediction
    /// against the expected output using the metric. All examples scoring
    /// above the threshold are collected, sorted by score descending, and the
    /// top `max_demos` are returned as demo indices.
    ///
    /// Returns an `OptimizationResult` with demo indices and aggregate score.
    pub async fn bootstrap<'a, L, M>(
        &self,
        trainset: &ExampleSet<'_>,
        llm: &'a L,
        metric: &M,
        output_field: Option<Sym>,
    ) -> Result<OptimizationResult>
    where
        L: Llm,
        M: Metric,
    {
        let max_examples = (self.config.batch_size as usize).min(trainset.len());
        let max_demos = self.config.max_demos as usize;

        // Collect (index, score) for every example that passes
        let mut scored: Vec<(u32, f64)> = Vec::with_capacity(max_examples);

        for (idx, view) in trainset.iter().enumerate() {
            if idx >= max_examples {
                break;
            }

            // Build prompt from input fields
            let prompt: String = view
                .inputs()
                .map(|(sym, val)| format!("{}: {}", sym.as_str(), val))
                .collect::<Vec<_>>()
                .join("\n");

            // Get expected output
            let expected = if let Some(out_sym) = output_field {
                view.get_output(out_sym).unwrap_or("").to_string()
            } else {
                view.outputs()
                    .next()
                    .map(|(_, val)| val.to_string())
                    .unwrap_or_default()
            };

            // Call teacher LLM
            let prediction = match llm.generate(&prompt, "", None).await {
                Ok(output) => output.text,
                Err(_) => continue, // skip failed predictions
            };

            // Score with metric
            let score = metric.evaluate(&prediction, &expected);

            if score >= self.config.metric_threshold as f64 {
                scored.push((idx as u32, score));
            }
        }

        // Sort by score descending to pick the best demonstrations
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_demos);

        let avg_score = if scored.is_empty() {
            0.0
        } else {
            scored.iter().map(|(_, s)| s).sum::<f64>() / scored.len() as f64
        };

        let demo_indices: SmallVec<[u32; 8]> = scored.iter().map(|(idx, _)| *idx).collect();
        let iterations = max_examples.max(1) as u16;

        Ok(OptimizationResult {
            demo_indices,
            score: avg_score,
            iterations,
        })
    }

    /// Full bootstrap returning a [`CompiledProgram`].
    ///
    /// Runs [`bootstrap`](Self::bootstrap) and wraps the result in a
    /// `CompiledProgram` with the given instruction. This is the primary
    /// entry point for optimization pipelines that persist their output.
    pub async fn compile<'a, L, M>(
        &self,
        trainset: &ExampleSet<'_>,
        llm: &'a L,
        metric: &M,
        instruction: &str,
        output_field: Option<Sym>,
    ) -> Result<CompiledProgram>
    where
        L: Llm,
        M: Metric,
    {
        let result = self.bootstrap(trainset, llm, metric, output_field).await?;

        Ok(CompiledProgram::new(
            instruction.to_string(),
            result.demo_indices,
            result.score,
            "BootstrapFewShot".to_string(),
        )
        .with_meta("max_demos", self.config.max_demos.to_string())
        .with_meta("threshold", self.config.metric_threshold.to_string())
        .with_meta("metric", metric.name().to_string()))
    }
}

impl Optimizer for BootstrapFewShot {
    type Output<'a> = OptimizationResult;
    type OptimizeFut<'a> = std::future::Ready<Result<OptimizationResult>>;

    fn optimize<'a>(&'a self, trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a> {
        // Basic implementation that just selects first N examples
        // Full implementation requires LM client passed separately via bootstrap()
        let n = (self.config.max_demos as usize).min(trainset.len());
        let indices: SmallVec<[u32; 8]> = (0..n as u32).collect();

        std::future::ready(Ok(OptimizationResult {
            demo_indices: indices,
            score: 0.0,
            iterations: 0,
        }))
    }

    fn name(&self) -> &'static str {
        "BootstrapFewShot"
    }
}

/// BootstrapFewShot with random search over demo combinations.
///
/// After bootstrapping a pool of candidate demos, this variant randomly
/// samples `num_candidates` subsets of size `max_demos`, evaluates each
/// combination on the trainset, and returns the best-scoring set.
#[derive(Clone, Copy)]
pub struct BootstrapFewShotWithRandomSearch {
    config: OptimizerConfig,
    num_candidates: u16,
}

impl BootstrapFewShotWithRandomSearch {
    /// Create new optimizer.
    pub const fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            num_candidates: 10,
        }
    }

    /// Set number of candidate combinations to try.
    pub const fn with_num_candidates(mut self, n: u16) -> Self {
        self.num_candidates = n;
        self
    }

    /// Bootstrap a pool of candidate demos, then evaluate random subsets.
    ///
    /// 1. Run the teacher LLM on all training examples and score them.
    /// 2. Collect all examples that pass the metric threshold into a pool.
    /// 3. Randomly sample `num_candidates` subsets of size `max_demos`.
    /// 4. Evaluate each subset by running the LLM with those demos and
    ///    scoring on the remainder of the trainset.
    /// 5. Return the best-scoring subset.
    pub async fn bootstrap_search<'a, L, M>(
        &self,
        trainset: &ExampleSet<'_>,
        llm: &'a L,
        metric: &M,
        instruction: Option<&str>,
        output_field: Option<Sym>,
    ) -> Result<CompiledProgram>
    where
        L: Llm,
        M: Metric,
    {
        let max_demos = self.config.max_demos as usize;
        let max_examples = (self.config.batch_size as usize).min(trainset.len());

        // Phase 1: Score all examples with teacher LLM
        let mut scored: Vec<(u32, f64)> = Vec::with_capacity(max_examples);

        for (idx, view) in trainset.iter().enumerate() {
            if idx >= max_examples {
                break;
            }

            let prompt: String = view
                .inputs()
                .map(|(sym, val)| format!("{}: {}", sym.as_str(), val))
                .collect::<Vec<_>>()
                .join("\n");

            let expected = if let Some(out_sym) = output_field {
                view.get_output(out_sym).unwrap_or("").to_string()
            } else {
                view.outputs()
                    .next()
                    .map(|(_, val)| val.to_string())
                    .unwrap_or_default()
            };

            let prediction = match llm.generate(&prompt, "", None).await {
                Ok(output) => output.text,
                Err(_) => continue,
            };

            let score = metric.evaluate(&prediction, &expected);
            if score >= self.config.metric_threshold as f64 {
                scored.push((idx as u32, score));
            }
        }

        // Phase 2: Random search over demo subsets
        let pool_size = scored.len();
        if pool_size == 0 {
            let inst = instruction.unwrap_or("").to_string();
            return Ok(CompiledProgram::new(
                inst,
                SmallVec::new(),
                0.0,
                "BootstrapFewShotWithRandomSearch".to_string(),
            ));
        }

        let mut rng = Rng::new(self.config.seed);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_indices: SmallVec<[u32; 8]> = SmallVec::new();

        let demo_count = max_demos.min(pool_size);

        for _ in 0..self.num_candidates {
            // Randomly sample `demo_count` indices from the pool
            let mut pool_indices: Vec<usize> = (0..pool_size).collect();
            rng.shuffle(&mut pool_indices);
            pool_indices.truncate(demo_count);

            let candidate_indices: SmallVec<[u32; 8]> =
                pool_indices.iter().map(|&i| scored[i].0).collect();

            // Evaluate this candidate set using the Evaluate harness
            let eval = if let Some(inst) = instruction {
                Evaluate::new(llm, metric)
                    .instruction(inst)
                    .demos(&candidate_indices)
            } else {
                Evaluate::new(llm, metric).demos(&candidate_indices)
            };

            let eval = if let Some(out_sym) = output_field {
                eval.output_field(out_sym)
            } else {
                eval
            };

            let eval_result = eval.run_async(trainset).await;

            if eval_result.mean > best_score {
                best_score = eval_result.mean;
                best_indices = candidate_indices;
            }
        }

        let inst = instruction.unwrap_or("").to_string();
        Ok(CompiledProgram::new(
            inst,
            best_indices,
            best_score,
            "BootstrapFewShotWithRandomSearch".to_string(),
        )
        .with_meta("num_candidates", self.num_candidates.to_string())
        .with_meta("max_demos", self.config.max_demos.to_string())
        .with_meta("metric", metric.name().to_string()))
    }
}

impl Optimizer for BootstrapFewShotWithRandomSearch {
    type Output<'a> = OptimizationResult;
    type OptimizeFut<'a> = std::future::Ready<Result<OptimizationResult>>;

    fn optimize<'a>(&'a self, trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a> {
        let mut rng = Rng::new(self.config.seed);
        let n = trainset.len();

        if n == 0 {
            return std::future::ready(Ok(OptimizationResult {
                demo_indices: SmallVec::new(),
                score: 0.0,
                iterations: 0,
            }));
        }

        // Generate random demo indices
        let demo_count = (self.config.max_demos as usize).min(n);
        let mut indices: SmallVec<[u32; 8]> = (0..n as u32).collect();
        rng.shuffle(&mut indices);
        indices.truncate(demo_count);

        std::future::ready(Ok(OptimizationResult {
            demo_indices: indices,
            score: 0.0,
            iterations: 1,
        }))
    }

    fn name(&self) -> &'static str {
        "BootstrapFewShotWithRandomSearch"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::intern::sym;
    use crate::metric::ExactMatch;
    use crate::optimizer::ExampleMeta;
    use crate::predict::FieldRange;
    use crate::recursive::llm::MockLlm;

    /// Helper to build a simple dataset from (input, expected) pairs.
    fn build_dataset(pairs: &[(&str, &str)]) -> (Buffer, Vec<ExampleMeta>, Sym, Sym) {
        let input_sym = sym("question");
        let output_sym = sym("answer");

        let mut buf = Vec::new();
        let mut metas = Vec::new();

        for (input, expected) in pairs {
            let input_start = buf.len() as u32;
            buf.extend_from_slice(input.as_bytes());
            let input_end = buf.len() as u32;

            let output_start = buf.len() as u32;
            buf.extend_from_slice(expected.as_bytes());
            let output_end = buf.len() as u32;

            let meta = ExampleMeta {
                input_ranges: [
                    (input_sym, FieldRange::new(input_start, input_end)),
                    (Sym::EMPTY, FieldRange::new(0, 0)),
                    (Sym::EMPTY, FieldRange::new(0, 0)),
                    (Sym::EMPTY, FieldRange::new(0, 0)),
                ],
                input_count: 1,
                output_ranges: [
                    (output_sym, FieldRange::new(output_start, output_end)),
                    (Sym::EMPTY, FieldRange::new(0, 0)),
                ],
                output_count: 1,
            };
            metas.push(meta);
        }

        let buffer = Buffer::from_bytes(buf);
        (buffer, metas, input_sym, output_sym)
    }

    #[test]
    fn test_bootstrap_creation() {
        let config = OptimizerConfig::default();
        let optimizer = BootstrapFewShot::new(config).with_max_demos(5);

        assert_eq!(optimizer.name(), "BootstrapFewShot");
        assert_eq!(optimizer.config.max_demos, 5);
    }

    #[tokio::test]
    async fn test_optimize_empty() {
        let optimizer = BootstrapFewShot::default();
        static BUFFER: Buffer = Buffer::Static(b"");
        let trainset = ExampleSet::new(&BUFFER, &[]);

        let result = optimizer.optimize(&trainset).await;
        assert!(result.is_ok());
        assert!(result.unwrap().demo_indices.is_empty());
    }

    #[test]
    fn test_random_search() {
        let optimizer = BootstrapFewShotWithRandomSearch::new(OptimizerConfig::default())
            .with_num_candidates(5);

        assert_eq!(optimizer.name(), "BootstrapFewShotWithRandomSearch");
        assert_eq!(optimizer.num_candidates, 5);
    }

    #[test]
    fn test_config_const() {
        // Ensure config can be created in const context
        const CONFIG: OptimizerConfig = OptimizerConfig::new().with_max_demos(8);
        assert_eq!(CONFIG.max_demos, 8);

        const OPT: BootstrapFewShot = BootstrapFewShot::new(CONFIG);
        assert_eq!(OPT.config.max_demos, 8);
    }

    #[tokio::test]
    async fn test_bootstrap_with_metric() {
        // LLM that echoes the expected answer for known questions
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("2+2") {
                "4".to_string()
            } else if prompt.contains("3+3") {
                "6".to_string()
            } else {
                "wrong".to_string()
            }
        });

        let (buffer, metas, _input_sym, output_sym) = build_dataset(&[
            ("What is 2+2?", "4"),
            ("What is 3+3?", "6"),
            ("What is 5+5?", "10"),
        ]);

        let dataset = ExampleSet::new(&buffer, &metas);
        let optimizer = BootstrapFewShot::default().with_max_demos(4);

        let result = optimizer
            .bootstrap(&dataset, &llm, &ExactMatch, Some(output_sym))
            .await
            .unwrap();

        // First two examples should pass (LLM gives correct answers), third should fail
        assert_eq!(result.demo_indices.len(), 2);
        assert!(result.score > 0.0);
    }

    #[tokio::test]
    async fn test_compile_returns_compiled_program() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("2+2") {
                "4".to_string()
            } else {
                "wrong".to_string()
            }
        });

        let (buffer, metas, _input_sym, output_sym) =
            build_dataset(&[("What is 2+2?", "4"), ("What is 1+1?", "2")]);

        let dataset = ExampleSet::new(&buffer, &metas);
        let optimizer = BootstrapFewShot::default().with_max_demos(4);

        let program = optimizer
            .compile(
                &dataset,
                &llm,
                &ExactMatch,
                "Answer the math question.",
                Some(output_sym),
            )
            .await
            .unwrap();

        assert_eq!(program.optimizer, "BootstrapFewShot");
        assert_eq!(program.instruction, "Answer the math question.");
        assert!(program.metadata.contains_key("metric"));
        // At least the first example (2+2=4) should be selected
        assert!(!program.demo_indices.is_empty());
    }
}
