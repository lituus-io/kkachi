// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! COPRO - Coordinate Prompt Optimization
//!
//! Generates instruction variations using an LM and evaluates them
//! to find the best performing instruction for a signature.
//!
//! ## Algorithm
//!
//! 1. Start with base instruction from signature
//! 2. Use LM to generate N instruction variations
//! 3. Evaluate each variation on training set
//! 4. Keep best performing instructions
//! 5. Repeat with depth (refining previous best)

use crate::error::Result;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig};
use crate::predict::LMClient;
use crate::str_view::StrView;
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
/// Optimizes the instruction portion of a signature by generating
/// variations and evaluating them.
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
    pub async fn generate_candidates<'a, L>(
        &self,
        base_instruction: &str,
        task_description: &str,
        lm: &'a L,
        buffer: &'a mut Vec<u8>,
    ) -> Result<Vec<String>>
    where
        L: LMClient,
    {
        buffer.clear();

        // Build prompt for instruction generation
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

        let prompt = unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) };
        let output = lm.generate(prompt).await?;
        let text = output.text()?.as_str();

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

    /// Evaluate an instruction candidate.
    pub fn evaluate_instruction<'a>(&self, _instruction: &str, _trainset: &ExampleSet<'a>) -> f64 {
        // Simplified evaluation - in full impl would run predictions
        // and compute metrics
        0.5
    }

    /// Run COPRO optimization with an LM.
    pub async fn optimize_with_lm<'a, L>(
        &self,
        base_instruction: &str,
        task_description: &str,
        trainset: &ExampleSet<'a>,
        lm: &'a L,
        buffer: &'a mut Vec<u8>,
    ) -> Result<COPROResult>
    where
        L: LMClient,
    {
        let mut best_instruction = base_instruction.to_string();
        let mut best_score = self.evaluate_instruction(&best_instruction, trainset);
        let mut total_candidates = 0u16;

        for _depth in 0..self.config.depth {
            // Generate candidates based on current best
            let candidates = self
                .generate_candidates(&best_instruction, task_description, lm, buffer)
                .await?;

            total_candidates += candidates.len() as u16;

            // Evaluate each candidate
            for candidate in candidates {
                let score = self.evaluate_instruction(&candidate, trainset);
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
    use crate::predict::LMOutput;

    struct MockLM;

    impl LMClient for MockLM {
        type GenerateFut<'a>
            = std::future::Ready<Result<LMOutput<'a>>>
        where
            Self: 'a;

        fn generate<'a>(&'a self, _prompt: StrView<'a>) -> Self::GenerateFut<'a> {
            static BUFFER: Buffer = Buffer::Static(
                b"1. Analyze the question carefully and provide a detailed answer.\n\
                  2. Think step by step to answer the question.\n\
                  3. Consider all aspects before responding.\n",
            );
            std::future::ready(Ok(LMOutput {
                buffer: BUFFER.view_all(),
                prompt_tokens: 50,
                completion_tokens: 30,
            }))
        }
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
        let copro = COPRO::default();
        let lm = MockLM;
        let mut buffer = Vec::new();

        let candidates = copro
            .generate_candidates("Answer the question.", "QA task", &lm, &mut buffer)
            .await;

        assert!(candidates.is_ok());
        let candidates = candidates.unwrap();
        assert!(!candidates.is_empty());
    }
}
