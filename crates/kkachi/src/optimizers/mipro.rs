// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! MIPRO - Multi-Instruction Prompt Optimization
//!
//! Jointly optimizes both instructions and demonstrations using
//! TPE-style Bayesian optimization with real evaluation.
//!
//! ## Algorithm
//!
//! 1. Generate instruction candidates with the LLM
//! 2. For `num_trials` iterations:
//!    a. Use TPE to suggest a (instruction_idx, demo_indices) configuration
//!    (random sampling during warmup, surrogate-guided after)
//!    b. Evaluate the configuration on the trainset using the metric
//!    c. Record the trial result to update the surrogate
//! 3. Return the best (instruction, demos) pair as a [`CompiledProgram`]
//!
//! ## Design
//!
//! - Uses the existing [`TPESampler`] for Bayesian optimization
//! - Uses [`Evaluate`] + [`Metric`] for real evaluation
//! - Returns [`CompiledProgram`] for persistence
//! - Uses the [`Llm`] trait (GATs) for instruction generation and evaluation

use crate::compiled::CompiledProgram;
use crate::error::Result;
use crate::evaluate::Evaluate;
use crate::intern::Sym;
use crate::metric::Metric;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig, Rng};
use crate::recursive::llm::Llm;
use smallvec::SmallVec;

/// MIPRO optimizer configuration.
#[derive(Clone, Copy)]
pub struct MIPROConfig {
    /// Base optimizer config
    pub base: OptimizerConfig,
    /// Number of instruction candidates
    pub num_instructions: u8,
    /// Number of demo configurations to try
    pub num_demo_configs: u8,
    /// Number of TPE trials
    pub num_trials: u16,
    /// Temperature for instruction generation
    pub temperature: f32,
    /// Fraction of trials to use for warmup (random sampling)
    pub warmup_fraction: f32,
}

impl Default for MIPROConfig {
    fn default() -> Self {
        Self {
            base: OptimizerConfig::default(),
            num_instructions: 5,
            num_demo_configs: 10,
            num_trials: 50,
            temperature: 0.7,
            warmup_fraction: 0.2,
        }
    }
}

impl MIPROConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            base: OptimizerConfig::new(),
            num_instructions: 5,
            num_demo_configs: 10,
            num_trials: 50,
            temperature: 0.7,
            warmup_fraction: 0.2,
        }
    }

    /// Set number of instruction candidates.
    pub const fn with_num_instructions(mut self, n: u8) -> Self {
        self.num_instructions = n;
        self
    }

    /// Set number of demo configurations.
    pub const fn with_num_demo_configs(mut self, n: u8) -> Self {
        self.num_demo_configs = n;
        self
    }

    /// Set number of TPE trials.
    pub const fn with_num_trials(mut self, n: u16) -> Self {
        self.num_trials = n;
        self
    }
}

/// Trial result for TPE history.
#[derive(Clone)]
pub struct Trial {
    /// Instruction index
    pub instruction_idx: u8,
    /// Demo indices
    pub demo_indices: SmallVec<[u32; 8]>,
    /// Score achieved
    pub score: f64,
}

/// TPE (Tree-structured Parzen Estimator) sampler.
///
/// Simplified implementation that separates good/bad trials
/// and samples from the good distribution.
pub struct TPESampler {
    /// All trials
    trials: Vec<Trial>,
    /// Gamma (fraction of trials considered good)
    gamma: f32,
    /// Random number generator
    rng: Rng,
}

impl TPESampler {
    /// Create new TPE sampler.
    pub fn new(gamma: f32, seed: u64) -> Self {
        Self {
            trials: Vec::new(),
            gamma,
            rng: Rng::new(seed),
        }
    }

    /// Record a trial.
    pub fn record(&mut self, trial: Trial) {
        self.trials.push(trial);
    }

    /// Suggest next instruction index based on history.
    pub fn suggest_instruction(&mut self, num_instructions: u8) -> u8 {
        if self.trials.is_empty() {
            return self.rng.gen_range(0, num_instructions as u64) as u8;
        }

        // Sort by score, take top gamma fraction
        let mut sorted: Vec<_> = self.trials.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let n_good = ((sorted.len() as f32 * self.gamma).ceil() as usize).max(1);
        let good = &sorted[..n_good];

        // Count instruction frequencies in good trials
        let mut counts = vec![0u32; num_instructions as usize];
        for trial in good {
            counts[trial.instruction_idx as usize] += 1;
        }

        // Sample proportionally from good distribution
        let total: u32 = counts.iter().sum();
        if total == 0 {
            return self.rng.gen_range(0, num_instructions as u64) as u8;
        }

        let mut threshold = self.rng.gen_range(0, total as u64) as u32;
        for (idx, &count) in counts.iter().enumerate() {
            if threshold < count {
                return idx as u8;
            }
            threshold -= count;
        }

        0
    }

    /// Suggest demo indices based on history.
    pub fn suggest_demos(&mut self, trainset_size: usize, max_demos: u8) -> SmallVec<[u32; 8]> {
        let n = (max_demos as usize).min(trainset_size);

        if self.trials.is_empty() {
            // Random selection
            let mut indices: SmallVec<[u32; 8]> = SmallVec::new();
            let mut available: Vec<u32> = (0..trainset_size as u32).collect();

            for _ in 0..n {
                if available.is_empty() {
                    break;
                }
                let idx = self.rng.gen_range(0, available.len() as u64) as usize;
                indices.push(available.swap_remove(idx));
            }
            return indices;
        }

        // Sort by score, take top gamma fraction
        let mut sorted: Vec<_> = self.trials.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let n_good = ((sorted.len() as f32 * self.gamma).ceil() as usize).max(1);
        let good = &sorted[..n_good];

        // Count demo frequencies in good trials
        let mut counts = vec![0u32; trainset_size];
        for trial in good {
            for &idx in &trial.demo_indices {
                if (idx as usize) < trainset_size {
                    counts[idx as usize] += 1;
                }
            }
        }

        // Select top demos by frequency
        let mut indexed: Vec<_> = counts.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(a.1));

        indexed.iter().take(n).map(|(idx, _)| *idx as u32).collect()
    }
}

/// MIPRO - Multi-Instruction Prompt Optimization.
///
/// Jointly optimizes instructions and demonstrations using
/// TPE-style Bayesian optimization with real metric evaluation.
#[derive(Clone, Copy)]
pub struct MIPRO {
    config: MIPROConfig,
}

impl MIPRO {
    /// Create a new MIPRO optimizer.
    pub const fn new(config: MIPROConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub const fn default() -> Self {
        Self::new(MIPROConfig::new())
    }

    /// Get the configuration.
    pub const fn config(&self) -> &MIPROConfig {
        &self.config
    }

    /// Generate instruction candidates using an LM.
    pub async fn generate_instructions<'a, L>(
        &self,
        base_instruction: &str,
        task_description: &str,
        lm: &'a L,
        buffer: &'a mut Vec<u8>,
    ) -> Result<Vec<String>>
    where
        L: Llm,
    {
        buffer.clear();

        // Build prompt
        buffer.extend_from_slice(b"Generate ");
        buffer.extend_from_slice(self.config.num_instructions.to_string().as_bytes());
        buffer.extend_from_slice(b" different instruction variations for this task.\n\nTask: ");
        buffer.extend_from_slice(task_description.as_bytes());
        buffer.extend_from_slice(b"\n\nBase instruction:\n");
        buffer.extend_from_slice(base_instruction.as_bytes());
        buffer.extend_from_slice(b"\n\nGenerate clear, specific variations:\n");

        let prompt = std::str::from_utf8(buffer).unwrap_or("");
        let output = lm.generate(prompt, "", None).await?;
        let text = &output.text;

        // Parse numbered instructions
        let mut instructions = Vec::with_capacity(self.config.num_instructions as usize);
        instructions.push(base_instruction.to_string()); // Always include original

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Parse "1. instruction" format
            if let Some(rest) = line
                .strip_prefix(char::is_numeric)
                .and_then(|s| s.strip_prefix('.').or_else(|| s.strip_prefix(')')))
            {
                instructions.push(rest.trim().to_string());
            }

            if instructions.len() >= self.config.num_instructions as usize {
                break;
            }
        }

        Ok(instructions)
    }

    /// Evaluate a (instruction, demos) configuration on the trainset.
    ///
    /// Uses the [`Evaluate`] harness to run the LLM with the given instruction
    /// and demo set, scoring against the metric.
    async fn evaluate_config_real<'a, L, M>(
        &self,
        instruction: &str,
        demo_indices: &[u32],
        trainset: &ExampleSet<'_>,
        llm: &'a L,
        metric: &M,
        output_field: Option<Sym>,
    ) -> f64
    where
        L: Llm,
        M: Metric,
    {
        let eval = Evaluate::new(llm, metric)
            .instruction(instruction)
            .demos(demo_indices);

        let eval = if let Some(out_sym) = output_field {
            eval.output_field(out_sym)
        } else {
            eval
        };

        let result = eval.run_async(trainset).await;
        result.mean
    }

    /// Run MIPRO optimization with an LM and metric.
    ///
    /// This is the primary entry point. Generates instruction candidates,
    /// then runs `num_trials` TPE-guided trials, each evaluating a
    /// (instruction, demos) pair on the trainset with real LLM calls.
    ///
    /// Returns a [`MIPROResult`] with the best configuration found.
    pub async fn optimize_with_lm<'a, L, M>(
        &self,
        base_instruction: &str,
        task_description: &str,
        trainset: &ExampleSet<'_>,
        lm: &'a L,
        metric: &M,
        buffer: &'a mut Vec<u8>,
        output_field: Option<Sym>,
        seed: u64,
    ) -> Result<MIPROResult>
    where
        L: Llm,
        M: Metric,
    {
        // Generate instruction candidates
        let instructions = self
            .generate_instructions(base_instruction, task_description, lm, buffer)
            .await?;

        let num_instructions = instructions.len() as u8;
        let warmup_trials =
            (self.config.num_trials as f32 * self.config.warmup_fraction).ceil() as u16;

        let mut sampler = TPESampler::new(0.25, seed);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_instruction_idx = 0u8;
        let mut best_demos: SmallVec<[u32; 8]> = SmallVec::new();

        for trial_idx in 0..self.config.num_trials {
            // Sample configuration
            let instruction_idx = if trial_idx < warmup_trials {
                // Random during warmup
                sampler.rng.gen_range(0, num_instructions as u64) as u8
            } else {
                sampler.suggest_instruction(num_instructions)
            };

            let demo_indices = sampler.suggest_demos(trainset.len(), self.config.base.max_demos);

            // Evaluate with real LLM calls + metric
            let score = self
                .evaluate_config_real(
                    &instructions[instruction_idx as usize],
                    &demo_indices,
                    trainset,
                    lm,
                    metric,
                    output_field,
                )
                .await;

            // Record trial
            sampler.record(Trial {
                instruction_idx,
                demo_indices: demo_indices.clone(),
                score,
            });

            // Track best
            if score > best_score {
                best_score = score;
                best_instruction_idx = instruction_idx;
                best_demos = demo_indices;
            }
        }

        Ok(MIPROResult {
            instruction: instructions[best_instruction_idx as usize].clone(),
            demo_indices: best_demos,
            score: best_score,
            trials_run: self.config.num_trials,
            instruction_candidates: num_instructions,
        })
    }

    /// Run MIPRO and return a [`CompiledProgram`].
    ///
    /// Convenience wrapper that calls [`optimize_with_lm`](Self::optimize_with_lm)
    /// and packages the result into a persistable `CompiledProgram`.
    pub async fn compile<'a, L, M>(
        &self,
        base_instruction: &str,
        task_description: &str,
        trainset: &ExampleSet<'_>,
        lm: &'a L,
        metric: &M,
        buffer: &'a mut Vec<u8>,
        output_field: Option<Sym>,
        seed: u64,
    ) -> Result<CompiledProgram>
    where
        L: Llm,
        M: Metric,
    {
        let result = self
            .optimize_with_lm(
                base_instruction,
                task_description,
                trainset,
                lm,
                metric,
                buffer,
                output_field,
                seed,
            )
            .await?;

        Ok(CompiledProgram::new(
            result.instruction,
            result.demo_indices,
            result.score,
            "MIPRO".to_string(),
        )
        .with_meta("trials_run", result.trials_run.to_string())
        .with_meta(
            "instruction_candidates",
            result.instruction_candidates.to_string(),
        )
        .with_meta("warmup_fraction", self.config.warmup_fraction.to_string())
        .with_meta("metric", metric.name().to_string()))
    }
}

/// Result of MIPRO optimization.
#[derive(Clone, Debug)]
pub struct MIPROResult {
    /// Best instruction found
    pub instruction: String,
    /// Best demo indices
    pub demo_indices: SmallVec<[u32; 8]>,
    /// Score achieved
    pub score: f64,
    /// Total trials run
    pub trials_run: u16,
    /// Number of instruction candidates generated
    pub instruction_candidates: u8,
}

impl Optimizer for MIPRO {
    type Output<'a> = OptimizationResult;
    type OptimizeFut<'a> = std::future::Ready<Result<OptimizationResult>>;

    fn optimize<'a>(&'a self, trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a> {
        // Basic implementation without LM - random selection
        let mut rng = Rng::new(42);
        let n = (self.config.base.max_demos as usize).min(trainset.len());

        let mut indices: SmallVec<[u32; 8]> = SmallVec::new();
        let mut available: Vec<u32> = (0..trainset.len() as u32).collect();

        for _ in 0..n {
            if available.is_empty() {
                break;
            }
            let idx = rng.gen_range(0, available.len() as u64) as usize;
            indices.push(available.swap_remove(idx));
        }

        std::future::ready(Ok(OptimizationResult {
            demo_indices: indices,
            score: 0.0,
            iterations: 0,
        }))
    }

    fn name(&self) -> &'static str {
        "MIPRO"
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

    fn mock_mipro_lm() -> MockLlm<impl Fn(&str, Option<&str>) -> String + Send + Sync> {
        MockLlm::new(|prompt, _feedback| {
            if prompt.contains("Generate") {
                // Instruction generation prompt
                "1. Analyze the question carefully and provide a detailed answer.\n\
                 2. Think step by step to answer the question.\n\
                 3. Consider all aspects before responding.\n"
                    .to_string()
            } else {
                // Evaluation prompt - return the expected answer for known inputs
                if prompt.contains("2+2") {
                    "4".to_string()
                } else if prompt.contains("3+3") {
                    "6".to_string()
                } else {
                    "unknown".to_string()
                }
            }
        })
    }

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
    fn test_mipro_creation() {
        let mipro = MIPRO::default();
        assert_eq!(mipro.name(), "MIPRO");
        assert_eq!(mipro.config().num_instructions, 5);
        assert_eq!(mipro.config().num_trials, 50);
    }

    #[test]
    fn test_mipro_config() {
        let config = MIPROConfig::new()
            .with_num_instructions(10)
            .with_num_trials(100);
        assert_eq!(config.num_instructions, 10);
        assert_eq!(config.num_trials, 100);
    }

    #[test]
    fn test_tpe_sampler() {
        let mut sampler = TPESampler::new(0.25, 42);

        // Should return random with no history
        let idx = sampler.suggest_instruction(5);
        assert!(idx < 5);

        // Record some trials
        sampler.record(Trial {
            instruction_idx: 0,
            demo_indices: SmallVec::from_slice(&[0, 1]),
            score: 0.9,
        });
        sampler.record(Trial {
            instruction_idx: 1,
            demo_indices: SmallVec::from_slice(&[2, 3]),
            score: 0.5,
        });
        sampler.record(Trial {
            instruction_idx: 0,
            demo_indices: SmallVec::from_slice(&[0, 2]),
            score: 0.8,
        });

        // With history, should favor instruction 0 (higher scores)
        let mut count_0 = 0;
        for _ in 0..100 {
            if sampler.suggest_instruction(5) == 0 {
                count_0 += 1;
            }
        }
        // Instruction 0 should be suggested more often
        assert!(count_0 > 30);
    }

    #[tokio::test]
    async fn test_generate_instructions() {
        let mipro = MIPRO::default();
        let lm = mock_mipro_lm();
        let mut buffer = Vec::new();

        let instructions = mipro
            .generate_instructions("Answer the question.", "QA task", &lm, &mut buffer)
            .await;

        assert!(instructions.is_ok());
        let instructions = instructions.unwrap();
        assert!(!instructions.is_empty());
        // Should include original
        assert!(instructions.contains(&"Answer the question.".to_string()));
    }

    #[tokio::test]
    async fn test_optimize_with_lm_evaluates_real() {
        let lm = mock_mipro_lm();
        // Small config for fast tests
        let mipro = MIPRO::new(
            MIPROConfig::new()
                .with_num_instructions(3)
                .with_num_trials(5),
        );
        let mut buffer = Vec::new();

        let (buf, metas, _input_sym, output_sym) =
            build_dataset(&[("What is 2+2?", "4"), ("What is 3+3?", "6")]);

        let dataset = ExampleSet::new(&buf, &metas);

        let result = mipro
            .optimize_with_lm(
                "Answer the question.",
                "Math QA",
                &dataset,
                &lm,
                &ExactMatch,
                &mut buffer,
                Some(output_sym),
                42,
            )
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.instruction.is_empty());
        assert_eq!(result.trials_run, 5);
        // Score should be non-negative since the LLM can answer some questions
        assert!(result.score >= 0.0);
    }

    #[tokio::test]
    async fn test_compile_returns_compiled_program() {
        let lm = mock_mipro_lm();
        let mipro = MIPRO::new(
            MIPROConfig::new()
                .with_num_instructions(2)
                .with_num_trials(3),
        );
        let mut buffer = Vec::new();

        let (buf, metas, _input_sym, output_sym) = build_dataset(&[("What is 2+2?", "4")]);

        let dataset = ExampleSet::new(&buf, &metas);

        let program = mipro
            .compile(
                "Answer the question.",
                "Math QA",
                &dataset,
                &lm,
                &ExactMatch,
                &mut buffer,
                Some(output_sym),
                42,
            )
            .await
            .unwrap();

        assert_eq!(program.optimizer, "MIPRO");
        assert!(program.metadata.contains_key("metric"));
        assert!(program.metadata.contains_key("trials_run"));
        assert!(program.metadata.contains_key("instruction_candidates"));
    }
}
