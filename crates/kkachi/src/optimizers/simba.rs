// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! SIMBA - Self-Improving Modular Boosting Approach
//!
//! An iterative optimization approach that analyzes failures,
//! generates improvements, and refines the module.
//!
//! ## Algorithm
//!
//! 1. Evaluate module on training set
//! 2. Analyze failures using LM reflection
//! 3. Generate improvement suggestions
//! 4. Apply improvements (better instructions/demos)
//! 5. Repeat until convergence or max iterations

use crate::error::Result;
use crate::optimizer::{ExampleSet, OptimizationResult, Optimizer, OptimizerConfig};
use crate::predict::LMClient;
use crate::str_view::StrView;
use smallvec::SmallVec;

/// SIMBA optimizer configuration.
#[derive(Clone, Copy)]
pub struct SIMBAConfig {
    /// Base optimizer config
    pub base: OptimizerConfig,
    /// Maximum improvement iterations
    pub max_iterations: u8,
    /// Number of failures to analyze per iteration
    pub failures_per_iteration: u8,
    /// Temperature for reflection
    pub temperature: f32,
    /// Minimum improvement required to continue
    pub min_improvement: f64,
}

impl Default for SIMBAConfig {
    fn default() -> Self {
        Self {
            base: OptimizerConfig::default(),
            max_iterations: 5,
            failures_per_iteration: 10,
            temperature: 0.7,
            min_improvement: 0.01,
        }
    }
}

impl SIMBAConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            base: OptimizerConfig::new(),
            max_iterations: 5,
            failures_per_iteration: 10,
            temperature: 0.7,
            min_improvement: 0.01,
        }
    }

    /// Set max iterations.
    pub const fn with_max_iterations(mut self, n: u8) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set failures per iteration.
    pub const fn with_failures_per_iteration(mut self, n: u8) -> Self {
        self.failures_per_iteration = n;
        self
    }

    /// Set minimum improvement threshold.
    pub const fn with_min_improvement(mut self, threshold: f64) -> Self {
        self.min_improvement = threshold;
        self
    }
}

/// A failure case for analysis.
#[derive(Clone)]
pub struct FailureCase {
    /// Index in training set
    pub example_idx: u32,
    /// Expected output
    pub expected: String,
    /// Actual output
    pub actual: String,
    /// Error description
    pub error: String,
}

/// Improvement suggestion from LM reflection.
#[derive(Clone)]
pub struct Improvement {
    /// Type of improvement
    pub kind: ImprovementKind,
    /// Description
    pub description: String,
    /// Confidence (0-1)
    pub confidence: f64,
}

/// Types of improvements.
#[derive(Clone, Copy, Debug)]
pub enum ImprovementKind {
    /// Improve instruction clarity
    InstructionClarity,
    /// Add more specific examples
    MoreExamples,
    /// Better example selection
    BetterExamples,
    /// Handle edge cases
    EdgeCases,
    /// Fix formatting issues
    Formatting,
}

/// SIMBA optimizer.
///
/// Self-improving optimization through failure analysis and reflection.
#[derive(Clone, Copy)]
pub struct SIMBA {
    config: SIMBAConfig,
}

impl SIMBA {
    /// Create a new SIMBA optimizer.
    pub const fn new(config: SIMBAConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub const fn default() -> Self {
        Self::new(SIMBAConfig::new())
    }

    /// Get the configuration.
    pub const fn config(&self) -> &SIMBAConfig {
        &self.config
    }

    /// Analyze failures using LM reflection.
    pub async fn analyze_failures<'a, L>(
        &self,
        failures: &[FailureCase],
        lm: &'a L,
        buffer: &'a mut Vec<u8>,
    ) -> Result<Vec<Improvement>>
    where
        L: LMClient,
    {
        if failures.is_empty() {
            return Ok(Vec::new());
        }

        buffer.clear();

        // Build reflection prompt
        buffer.extend_from_slice(b"Analyze these failure cases and suggest improvements:\n\n");

        for (i, failure) in failures
            .iter()
            .take(self.config.failures_per_iteration as usize)
            .enumerate()
        {
            buffer.extend_from_slice(b"Failure ");
            buffer.extend_from_slice((i + 1).to_string().as_bytes());
            buffer.extend_from_slice(b":\n");
            buffer.extend_from_slice(b"  Expected: ");
            buffer.extend_from_slice(failure.expected.as_bytes());
            buffer.extend_from_slice(b"\n  Actual: ");
            buffer.extend_from_slice(failure.actual.as_bytes());
            buffer.extend_from_slice(b"\n  Error: ");
            buffer.extend_from_slice(failure.error.as_bytes());
            buffer.extend_from_slice(b"\n\n");
        }

        buffer.extend_from_slice(
            b"Based on these failures, suggest specific improvements. For each suggestion:\n\
              - Type: (InstructionClarity, MoreExamples, BetterExamples, EdgeCases, Formatting)\n\
              - Description: specific improvement\n\
              - Confidence: 0-1\n\n",
        );

        let prompt = unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) };
        let output = lm.generate(prompt).await?;
        let text = output.text()?.as_str();

        // Parse improvements (simplified parsing)
        let mut improvements = Vec::new();

        for line in text.lines() {
            let line = line.trim();

            if line.starts_with("Type:") || line.starts_with("- Type:") {
                let kind = if line.contains("InstructionClarity") {
                    ImprovementKind::InstructionClarity
                } else if line.contains("MoreExamples") {
                    ImprovementKind::MoreExamples
                } else if line.contains("BetterExamples") {
                    ImprovementKind::BetterExamples
                } else if line.contains("EdgeCases") {
                    ImprovementKind::EdgeCases
                } else if line.contains("Formatting") {
                    ImprovementKind::Formatting
                } else {
                    continue;
                };

                improvements.push(Improvement {
                    kind,
                    description: String::new(),
                    confidence: 0.5,
                });
            } else if line.starts_with("Description:") || line.starts_with("- Description:") {
                if let Some(imp) = improvements.last_mut() {
                    imp.description = line
                        .trim_start_matches("Description:")
                        .trim_start_matches("- Description:")
                        .trim()
                        .to_string();
                }
            } else if line.starts_with("Confidence:") || line.starts_with("- Confidence:") {
                if let Some(imp) = improvements.last_mut() {
                    let conf_str = line
                        .trim_start_matches("Confidence:")
                        .trim_start_matches("- Confidence:")
                        .trim();
                    if let Ok(conf) = conf_str.parse::<f64>() {
                        imp.confidence = conf.clamp(0.0, 1.0);
                    }
                }
            }
        }

        Ok(improvements)
    }

    /// Generate improved instruction based on improvements.
    pub async fn generate_improved_instruction<'a, L>(
        &self,
        current_instruction: &str,
        improvements: &[Improvement],
        lm: &'a L,
        buffer: &'a mut Vec<u8>,
    ) -> Result<String>
    where
        L: LMClient,
    {
        if improvements.is_empty() {
            return Ok(current_instruction.to_string());
        }

        buffer.clear();

        buffer
            .extend_from_slice(b"Improve this instruction based on the following suggestions:\n\n");
        buffer.extend_from_slice(b"Current instruction:\n");
        buffer.extend_from_slice(current_instruction.as_bytes());
        buffer.extend_from_slice(b"\n\nSuggested improvements:\n");

        for imp in improvements {
            buffer.extend_from_slice(b"- ");
            buffer.extend_from_slice(format!("{:?}: ", imp.kind).as_bytes());
            buffer.extend_from_slice(imp.description.as_bytes());
            buffer.push(b'\n');
        }

        buffer.extend_from_slice(b"\nGenerate an improved instruction:\n");

        let prompt = unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) };
        let output = lm.generate(prompt).await?;
        let text = output.text()?.as_str();

        // Extract first non-empty line as new instruction
        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with("Improved") && !line.starts_with("Here") {
                return Ok(line.to_string());
            }
        }

        Ok(current_instruction.to_string())
    }

    /// Run SIMBA optimization with an LM.
    pub async fn optimize_with_lm<'a, L>(
        &self,
        base_instruction: &str,
        trainset: &ExampleSet<'a>,
        lm: &'a L,
        buffer: &'a mut Vec<u8>,
    ) -> Result<SIMBAResult>
    where
        L: LMClient,
    {
        let mut current_instruction = base_instruction.to_string();
        let mut best_score = 0.0f64;
        let mut best_instruction = current_instruction.clone();
        let best_demos: SmallVec<[u32; 8]> =
            (0..(self.config.base.max_demos as usize).min(trainset.len()) as u32).collect();
        let mut total_improvements = 0u16;

        for _iteration in 0..self.config.max_iterations {
            // Evaluate current configuration (simplified)
            let (score, failures) = self.evaluate(&current_instruction, trainset);

            // Track best
            if score > best_score {
                best_score = score;
                best_instruction = current_instruction.clone();
            }

            // Check for convergence
            if failures.is_empty() {
                break;
            }

            // Analyze failures
            let improvements = self.analyze_failures(&failures, lm, buffer).await?;
            total_improvements += improvements.len() as u16;

            if improvements.is_empty() {
                break;
            }

            // Generate improved instruction
            current_instruction = self
                .generate_improved_instruction(&current_instruction, &improvements, lm, buffer)
                .await?;

            // Check if improvement is significant enough
            let (new_score, _) = self.evaluate(&current_instruction, trainset);
            if new_score - score < self.config.min_improvement {
                break;
            }
        }

        Ok(SIMBAResult {
            instruction: best_instruction,
            demo_indices: best_demos,
            score: best_score,
            iterations: self.config.max_iterations,
            improvements_applied: total_improvements,
        })
    }

    /// Evaluate a configuration (simplified).
    fn evaluate(&self, _instruction: &str, trainset: &ExampleSet<'_>) -> (f64, Vec<FailureCase>) {
        // Simplified evaluation - real impl would run predictions
        // Return mock score and some mock failures
        let failures = if trainset.len() > 3 {
            vec![FailureCase {
                example_idx: 0,
                expected: "expected answer".to_string(),
                actual: "wrong answer".to_string(),
                error: "Mismatch".to_string(),
            }]
        } else {
            Vec::new()
        };

        (0.6, failures)
    }
}

/// Result of SIMBA optimization.
#[derive(Clone, Debug)]
pub struct SIMBAResult {
    /// Best instruction found
    pub instruction: String,
    /// Best demo indices
    pub demo_indices: SmallVec<[u32; 8]>,
    /// Score achieved
    pub score: f64,
    /// Iterations run
    pub iterations: u8,
    /// Total improvements applied
    pub improvements_applied: u16,
}

impl Optimizer for SIMBA {
    type Output<'a> = OptimizationResult;
    type OptimizeFut<'a> = std::future::Ready<Result<OptimizationResult>>;

    fn optimize<'a>(&'a self, trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a> {
        // Basic implementation without LM
        let n = (self.config.base.max_demos as usize).min(trainset.len());
        let indices: SmallVec<[u32; 8]> = (0..n as u32).collect();

        std::future::ready(Ok(OptimizationResult {
            demo_indices: indices,
            score: 0.0,
            iterations: 0,
        }))
    }

    fn name(&self) -> &'static str {
        "SIMBA"
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
                b"Type: InstructionClarity\n\
                  Description: Make the instruction more specific\n\
                  Confidence: 0.8\n\n\
                  Type: EdgeCases\n\
                  Description: Handle empty inputs\n\
                  Confidence: 0.7\n",
            );
            std::future::ready(Ok(LMOutput {
                buffer: BUFFER.view_all(),
                prompt_tokens: 100,
                completion_tokens: 50,
            }))
        }
    }

    #[test]
    fn test_simba_creation() {
        let simba = SIMBA::default();
        assert_eq!(simba.name(), "SIMBA");
        assert_eq!(simba.config().max_iterations, 5);
    }

    #[test]
    fn test_simba_config() {
        let config = SIMBAConfig::new()
            .with_max_iterations(10)
            .with_failures_per_iteration(20)
            .with_min_improvement(0.05);
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.failures_per_iteration, 20);
        assert!((config.min_improvement - 0.05).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_analyze_failures() {
        let simba = SIMBA::default();
        let lm = MockLM;
        let mut buffer = Vec::new();

        let failures = vec![FailureCase {
            example_idx: 0,
            expected: "yes".to_string(),
            actual: "no".to_string(),
            error: "Wrong answer".to_string(),
        }];

        let improvements = simba.analyze_failures(&failures, &lm, &mut buffer).await;

        assert!(improvements.is_ok());
        let improvements = improvements.unwrap();
        assert!(!improvements.is_empty());
    }

    #[test]
    fn test_improvement_kind() {
        let imp = Improvement {
            kind: ImprovementKind::InstructionClarity,
            description: "Be more specific".to_string(),
            confidence: 0.8,
        };

        assert!(matches!(imp.kind, ImprovementKind::InstructionClarity));
    }
}
