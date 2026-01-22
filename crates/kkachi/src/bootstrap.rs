// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! BootstrapFewShot optimizer implementation
//!
//! Uses GATs for zero-cost async and returns indices into the training buffer
//! rather than copying demonstrations.

use crate::error::Result;
use crate::intern::Sym;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig, Rng};
use crate::predict::{LMClient, Predict, PredictOutput};
use smallvec::SmallVec;

/// BootstrapFewShot optimizer.
///
/// Bootstraps demonstrations by running the module on training examples
/// and selecting successful predictions as demonstrations.
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

    /// Bootstrap demos using an LM client and metric function.
    ///
    /// Returns indices of successful examples to use as demonstrations.
    /// This is an async function that iterates through examples and evaluates them.
    pub async fn bootstrap<'a, L, M>(
        &self,
        trainset: &ExampleSet<'_>,
        predict: &Predict<'_, '_>,
        lm: &'a L,
        metric: M,
        prompt_buffer: &mut Vec<u8>,
    ) -> Result<OptimizationResult>
    where
        L: LMClient,
        M: Fn(&PredictOutput<'_>, Sym) -> f64,
    {
        let mut successful: SmallVec<[u32; 8]> = SmallVec::new();
        let max_examples = self.config.batch_size as usize;
        let max_demos = self.config.max_demos as usize;

        for (idx, example) in trainset.iter().enumerate() {
            // Stop if we have enough or processed enough
            if idx >= max_examples || successful.len() >= max_demos {
                break;
            }

            // Build inputs from example
            let mut inputs = crate::types::Inputs::new();
            for (sym, value) in example.inputs() {
                inputs.insert(sym.as_str(), value);
            }

            // Build prompt and call LM
            let prompt = predict.build_prompt_into(&inputs, prompt_buffer);
            let output = lm.generate(prompt).await?;

            // Parse response
            let text = output.text()?;
            let ranges = predict.parse_response_ranges(text);
            let pred_output = PredictOutput {
                buffer: output.buffer,
                field_ranges: ranges,
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.completion_tokens,
            };

            // Get first output field symbol
            let output_sym = predict
                .signature()
                .output_fields
                .first()
                .map(|f| crate::intern::sym(&f.name))
                .unwrap_or(crate::intern::ANSWER);

            // Evaluate with metric
            let score = metric(&pred_output, output_sym);

            // If good enough, add as demo
            if score >= self.config.metric_threshold as f64 {
                successful.push(idx as u32);
            }
        }

        let iterations = successful.len().max(1);
        Ok(OptimizationResult {
            demo_indices: successful.clone(),
            score: successful.len() as f64 / iterations as f64,
            iterations: iterations as u16,
        })
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
    use crate::predict::LMOutput;
    use crate::str_view::StrView;

    // MockLM defined for potential future LM integration tests
    #[allow(dead_code)]
    struct MockLM;

    #[allow(dead_code)]
    impl LMClient for MockLM {
        type GenerateFut<'a> = std::future::Ready<Result<LMOutput<'a>>>;

        fn generate<'a>(&'a self, _prompt: StrView<'a>) -> Self::GenerateFut<'a> {
            static BUFFER: Buffer = Buffer::Static(b"Answer: 42");
            std::future::ready(Ok(LMOutput {
                buffer: BUFFER.view_all(),
                prompt_tokens: 10,
                completion_tokens: 5,
            }))
        }
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
}
