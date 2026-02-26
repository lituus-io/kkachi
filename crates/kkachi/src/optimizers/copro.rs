// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! COPRO - Coordinate Prompt Optimization
//!
//! Hill-climbing instruction optimizer that uses an LLM to generate candidate
//! instructions and the [`Evaluate`] harness to score them on a training set.
//!
//! ## Algorithm
//!
//! 1. Start with base instruction from the caller
//! 2. For each depth iteration:
//!    a. Use the LLM to generate `breadth` instruction variations from the
//!       current best instruction
//!    b. Evaluate each candidate on the training set using the metric
//!    c. Keep the best-scoring instruction
//! 3. Return a [`CompiledProgram`] with the best instruction found
//!
//! ## Design
//!
//! - Uses the [`Llm`] trait (GATs) for both instruction generation and
//!   candidate evaluation
//! - Uses [`Evaluate`] + [`Metric`] for real evaluation on the trainset
//! - Returns [`CompiledProgram`] for persistence

use crate::compiled::CompiledProgram;
use crate::error::Result;
use crate::evaluate::Evaluate;
use crate::intern::Sym;
use crate::metric::Metric;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig};
use crate::recursive::llm::Llm;
use smallvec::SmallVec;

/// COPRO optimizer configuration.
#[derive(Clone, Copy)]
pub struct COPROConfig {
    /// Base optimizer config
    pub base: OptimizerConfig,
    /// Number of instruction candidates to generate per iteration
    pub breadth: u8,
    /// Number of refinement iterations
    pub depth: u8,
    /// Temperature for instruction generation
    pub temperature: f32,
}

impl Default for COPROConfig {
    fn default() -> Self {
        Self {
            base: OptimizerConfig::default(),
            breadth: 5,
            depth: 3,
            temperature: 0.7,
        }
    }
}

impl COPROConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            base: OptimizerConfig::new(),
            breadth: 5,
            depth: 3,
            temperature: 0.7,
        }
    }

    /// Set breadth.
    pub const fn with_breadth(mut self, n: u8) -> Self {
        self.breadth = n;
        self
    }

    /// Set depth.
    pub const fn with_depth(mut self, n: u8) -> Self {
        self.depth = n;
        self
    }
}

/// COPRO - Coordinate Prompt Optimization.
///
/// Optimizes the instruction portion of a prompt by generating candidate
/// variations with the LLM and evaluating them on the training set using
/// a metric. Hill-climbing: the best instruction from each iteration
/// seeds the next round of generation.
#[derive(Clone, Copy)]
pub struct COPRO {
    config: COPROConfig,
}

impl COPRO {
    /// Create a new COPRO optimizer.
    pub const fn new(config: COPROConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub const fn default() -> Self {
        Self::new(COPROConfig::new())
    }

    /// Get the configuration.
    pub const fn config(&self) -> &COPROConfig {
        &self.config
    }

    /// Generate instruction candidates using an LM.
    ///
    /// Builds a meta-prompt asking the LLM to generate `breadth` variations
    /// of the given instruction, then parses the numbered response.
    pub async fn generate_candidates<'a, L>(
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

        // Build meta-prompt for instruction generation
        buffer.extend_from_slice(b"Generate ");
        buffer.extend_from_slice(self.config.breadth.to_string().as_bytes());
        buffer.extend_from_slice(
            b" different versions of this instruction for a language model task.\n\n",
        );
        buffer.extend_from_slice(b"Task: ");
        buffer.extend_from_slice(task_description.as_bytes());
        buffer.extend_from_slice(b"\n\nOriginal instruction:\n");
        buffer.extend_from_slice(base_instruction.as_bytes());
        buffer.extend_from_slice(b"\n\nGenerate variations that are:\n");
        buffer.extend_from_slice(b"- Clear and specific\n");
        buffer.extend_from_slice(b"- Different in phrasing but same intent\n");
        buffer.extend_from_slice(b"- Potentially more effective\n\n");
        buffer.extend_from_slice(b"Output each instruction on a new line, numbered 1-");
        buffer.extend_from_slice(self.config.breadth.to_string().as_bytes());
        buffer.extend_from_slice(b":\n");

        let prompt = std::str::from_utf8(buffer).unwrap_or("");
        let output = lm.generate(prompt, "", None).await?;
        let text = &output.text;

        // Parse numbered instructions
        let mut candidates = Vec::with_capacity(self.config.breadth as usize);
        for line in text.lines() {
            let line = line.trim();
            // Skip empty lines
            if line.is_empty() {
                continue;
            }
            // Parse "1. instruction" or "1) instruction" format
            if let Some(rest) = line
                .strip_prefix(char::is_numeric)
                .and_then(|s| s.strip_prefix('.').or_else(|| s.strip_prefix(')')))
            {
                candidates.push(rest.trim().to_string());
            } else if !line.starts_with(char::is_numeric) {
                // Also accept lines without numbers
                candidates.push(line.to_string());
            }

            if candidates.len() >= self.config.breadth as usize {
                break;
            }
        }

        // Always include the original
        if candidates.is_empty() {
            candidates.push(base_instruction.to_string());
        }

        Ok(candidates)
    }

    /// Evaluate an instruction candidate on the trainset using the Evaluate harness.
    ///
    /// Runs the LLM with the given instruction on the trainset and returns
    /// the mean metric score.
    async fn evaluate_instruction_real<'a, L, M>(
        &self,
        instruction: &str,
        trainset: &ExampleSet<'_>,
        llm: &'a L,
        metric: &M,
        output_field: Option<Sym>,
    ) -> f64
    where
        L: Llm,
        M: Metric,
    {
        let eval = Evaluate::new(llm, metric).instruction(instruction);

        let eval = if let Some(out_sym) = output_field {
            eval.output_field(out_sym)
        } else {
            eval
        };

        let result = eval.run_async(trainset).await;
        result.mean
    }

    /// Run COPRO optimization with an LM and metric.
    ///
    /// This is the primary entry point. For each depth iteration, generates
    /// `breadth` candidate instructions, evaluates each on the trainset
    /// with real LLM calls + metric scoring, and keeps the best.
    ///
    /// Returns a [`COPROResult`] with the best instruction and its score.
    pub async fn optimize_with_lm<'a, L, M>(
        &self,
        base_instruction: &str,
        task_description: &str,
        trainset: &ExampleSet<'_>,
        lm: &'a L,
        metric: &M,
        buffer: &'a mut Vec<u8>,
        output_field: Option<Sym>,
    ) -> Result<COPROResult>
    where
        L: Llm,
        M: Metric,
    {
        let mut best_instruction = base_instruction.to_string();
        let mut best_score = self
            .evaluate_instruction_real(&best_instruction, trainset, lm, metric, output_field)
            .await;
        let mut total_candidates = 1u16; // count the base instruction

        for _depth in 0..self.config.depth {
            // Generate candidates based on current best
            let candidates = self
                .generate_candidates(&best_instruction, task_description, lm, buffer)
                .await?;

            total_candidates += candidates.len() as u16;

            // Evaluate each candidate on the trainset
            for candidate in candidates {
                let score = self
                    .evaluate_instruction_real(&candidate, trainset, lm, metric, output_field)
                    .await;
                if score > best_score {
                    best_score = score;
                    best_instruction = candidate;
                }
            }
        }

        Ok(COPROResult {
            instruction: best_instruction,
            score: best_score,
            candidates_evaluated: total_candidates,
            depth_iterations: self.config.depth,
        })
    }

    /// Run COPRO and return a [`CompiledProgram`].
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
            )
            .await?;

        Ok(CompiledProgram::new(
            result.instruction,
            SmallVec::new(), // COPRO optimizes instructions, not demos
            result.score,
            "COPRO".to_string(),
        )
        .with_meta("breadth", self.config.breadth.to_string())
        .with_meta("depth", self.config.depth.to_string())
        .with_meta(
            "candidates_evaluated",
            result.candidates_evaluated.to_string(),
        )
        .with_meta("metric", metric.name().to_string()))
    }
}

/// Result of COPRO optimization.
#[derive(Clone, Debug)]
pub struct COPROResult {
    /// Best instruction found
    pub instruction: String,
    /// Score achieved
    pub score: f64,
    /// Total candidates evaluated
    pub candidates_evaluated: u16,
    /// Depth iterations run
    pub depth_iterations: u8,
}

impl Optimizer for COPRO {
    type Output<'a> = OptimizationResult;
    type OptimizeFut<'a> = std::future::Ready<Result<OptimizationResult>>;

    fn optimize<'a>(&'a self, trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a> {
        // Basic implementation without LM - just selects demos
        let n = (self.config.base.max_demos as usize).min(trainset.len());
        let indices: SmallVec<[u32; 8]> = (0..n as u32).collect();

        std::future::ready(Ok(OptimizationResult {
            demo_indices: indices,
            score: 0.0,
            iterations: 0,
        }))
    }

    fn name(&self) -> &'static str {
        "COPRO"
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

    fn mock_instruction_lm() -> MockLlm<impl Fn(&str, Option<&str>) -> String + Send + Sync> {
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
    fn test_copro_creation() {
        let copro = COPRO::default();
        assert_eq!(copro.name(), "COPRO");
        assert_eq!(copro.config().breadth, 5);
        assert_eq!(copro.config().depth, 3);
    }

    #[test]
    fn test_copro_config() {
        let config = COPROConfig::new().with_breadth(10).with_depth(5);
        assert_eq!(config.breadth, 10);
        assert_eq!(config.depth, 5);
    }

    #[tokio::test]
    async fn test_generate_candidates() {
        let lm = mock_instruction_lm();
        let copro = COPRO::default();
        let mut buffer = Vec::new();

        let candidates = copro
            .generate_candidates("Answer the question.", "QA task", &lm, &mut buffer)
            .await;

        assert!(candidates.is_ok());
        let candidates = candidates.unwrap();
        assert!(!candidates.is_empty());
    }

    #[tokio::test]
    async fn test_optimize_with_lm_evaluates_on_trainset() {
        let lm = mock_instruction_lm();
        let copro = COPRO::new(COPROConfig::new().with_breadth(3).with_depth(1));
        let mut buffer = Vec::new();

        let (buf, metas, _input_sym, output_sym) =
            build_dataset(&[("What is 2+2?", "4"), ("What is 3+3?", "6")]);

        let dataset = ExampleSet::new(&buf, &metas);

        let result = copro
            .optimize_with_lm(
                "Answer the question.",
                "Math QA",
                &dataset,
                &lm,
                &ExactMatch,
                &mut buffer,
                Some(output_sym),
            )
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.instruction.is_empty());
        assert!(result.candidates_evaluated >= 1);
    }

    #[tokio::test]
    async fn test_compile_returns_compiled_program() {
        let lm = mock_instruction_lm();
        let copro = COPRO::new(COPROConfig::new().with_breadth(2).with_depth(1));
        let mut buffer = Vec::new();

        let (buf, metas, _input_sym, output_sym) =
            build_dataset(&[("What is 2+2?", "4")]);

        let dataset = ExampleSet::new(&buf, &metas);

        let program = copro
            .compile(
                "Answer the question.",
                "Math QA",
                &dataset,
                &lm,
                &ExactMatch,
                &mut buffer,
                Some(output_sym),
            )
            .await
            .unwrap();

        assert_eq!(program.optimizer, "COPRO");
        assert!(program.metadata.contains_key("metric"));
        assert!(program.metadata.contains_key("breadth"));
        assert!(program.metadata.contains_key("depth"));
    }
}
