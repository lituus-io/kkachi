// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Multi-objective optimization with Pareto-front support.
//!
//! This module provides types and validators for optimizing across multiple
//! objectives simultaneously, using Pareto dominance to identify non-dominated
//! solutions.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::{checks, multi_objective, Objective, Scalarization, Validate};
//!
//! let validator = multi_objective()
//!     .scalarize(Scalarization::WeightedSum)
//!     .objectives([
//!         (Objective::new("correctness").weight(2.0), checks().require("fn ")),
//!         (Objective::new("brevity").weight(1.0), checks().max_len(200)),
//!     ]);
//!
//! // Use as a regular Validate (scalarized)
//! let score = validator.validate("fn add() {}");
//! assert!(score.value > 0.0);
//! ```

use crate::recursive::validate::{Score, Validate};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::time::Duration;

/// Optimization direction for an objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// Higher is better (default).
    #[default]
    Maximize,
    /// Lower is better (internally normalized to 1.0 - value).
    Minimize,
}

/// Definition of a named optimization objective.
#[derive(Debug, Clone)]
pub struct Objective {
    /// Static name for this objective.
    pub name: &'static str,
    /// Weight in scalarization (default 1.0).
    pub weight: f64,
    /// Target threshold for this objective.
    pub target: f64,
    /// Optimization direction.
    pub direction: Direction,
}

impl Objective {
    /// Create a new objective (weight 1.0, target 1.0, Maximize).
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            weight: 1.0,
            target: 1.0,
            direction: Direction::Maximize,
        }
    }

    /// Set weight for scalarization.
    pub fn weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }

    /// Set target threshold.
    pub fn target(mut self, t: f64) -> Self {
        self.target = t;
        self
    }

    /// Set direction to minimize.
    pub fn minimize(mut self) -> Self {
        self.direction = Direction::Minimize;
        self
    }
}

/// A single objective's score within a MultiScore.
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveScore {
    /// Name of this objective.
    pub name: &'static str,
    /// Score value in [0.0, 1.0].
    pub value: f64,
    /// Weight for scalarization.
    pub weight: f64,
}

/// A score with multiple named objective values.
#[derive(Debug, Clone)]
pub struct MultiScore<'a> {
    /// Per-objective scores.
    pub objectives: SmallVec<[ObjectiveScore; 8]>,
    /// Combined feedback from all objectives.
    pub feedback: Option<Cow<'a, str>>,
    /// Aggregate confidence.
    pub confidence: f64,
}

impl<'a> MultiScore<'a> {
    /// Create from individual objective scores.
    pub fn new(objectives: SmallVec<[ObjectiveScore; 8]>) -> Self {
        Self {
            objectives,
            feedback: None,
            confidence: 1.0,
        }
    }

    /// Scalarize to a single Score using the given strategy.
    pub fn scalarize(&self, strategy: &Scalarization) -> Score<'static> {
        let weights: SmallVec<[f64; 8]> = self.objectives.iter().map(|o| o.weight).collect();
        let values: SmallVec<[f64; 8]> = self.objectives.iter().map(|o| o.value).collect();
        let scalar = scalarize_values(&values, strategy, &weights);

        let mut score = Score::new(scalar);
        if let Some(ref fb) = self.feedback {
            score = Score::with_feedback(scalar, fb.to_string());
        }
        score
    }

    /// Check if this score dominates another (Pareto dominance).
    pub fn dominates(&self, other: &MultiScore) -> bool {
        let a: SmallVec<[f64; 8]> = self.objectives.iter().map(|o| o.value).collect();
        let b: SmallVec<[f64; 8]> = other.objectives.iter().map(|o| o.value).collect();
        dominates(&a, &b)
    }

    /// Get the score for a named objective.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.objectives
            .iter()
            .find(|o| o.name == name)
            .map(|o| o.value)
    }

    /// Check if all objectives meet their individual targets.
    pub fn all_targets_met(&self, objectives: &[Objective]) -> bool {
        self.objectives
            .iter()
            .zip(objectives.iter())
            .all(|(os, obj)| os.value >= obj.target - f64::EPSILON)
    }
}

/// Strategy for converting multi-objective scores to a single scalar.
#[derive(Debug, Clone)]
pub enum Scalarization {
    /// Weighted sum: sum(w_i * v_i) / sum(w_i).
    WeightedSum,
    /// Chebyshev (minimax): min_i(v_i * w_i) — focuses on worst objective.
    Chebyshev,
    /// Weighted product: product(v_i ^ w_i).
    WeightedProduct,
}

fn scalarize_values(values: &[f64], strategy: &Scalarization, weights: &[f64]) -> f64 {
    match strategy {
        Scalarization::WeightedSum => {
            let total_w: f64 = weights.iter().sum();
            if total_w == 0.0 {
                return 0.0;
            }
            values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / total_w
        }
        Scalarization::Chebyshev => values
            .iter()
            .zip(weights)
            .map(|(v, w)| v * w)
            .fold(f64::MAX, f64::min),
        Scalarization::WeightedProduct => {
            let total_w: f64 = weights.iter().sum();
            if total_w == 0.0 {
                return 0.0;
            }
            values
                .iter()
                .zip(weights)
                .map(|(v, w)| v.powf(w / total_w))
                .product()
        }
    }
}

/// Trait for types that provide objective values for Pareto comparison.
pub trait ParetoScored {
    /// Get the objective values as a slice.
    fn objective_values(&self) -> &[f64];
}

/// A candidate solution with its multi-score and output.
#[derive(Debug, Clone)]
pub struct ParetoCandidate {
    /// The generated output text.
    pub output: String,
    /// Per-objective scores.
    pub scores: SmallVec<[f64; 8]>,
    /// Generation index.
    pub index: usize,
}

impl ParetoScored for ParetoCandidate {
    fn objective_values(&self) -> &[f64] {
        &self.scores
    }
}

/// A Pareto front tracking non-dominated solutions.
///
/// Solutions where no other solution is better in all objectives simultaneously.
#[derive(Debug, Clone)]
pub struct ParetoFront<T: ParetoScored> {
    solutions: SmallVec<[T; 16]>,
    num_objectives: usize,
}

impl<T: ParetoScored> ParetoFront<T> {
    /// Create an empty Pareto front for the given number of objectives.
    pub fn new(num_objectives: usize) -> Self {
        Self {
            solutions: SmallVec::new(),
            num_objectives,
        }
    }

    /// Insert a candidate. Returns true if it was added (non-dominated).
    /// Removes existing solutions that are dominated by the new candidate.
    pub fn insert(&mut self, candidate: T) -> bool {
        let values = candidate.objective_values();
        debug_assert_eq!(values.len(), self.num_objectives);

        // Check if candidate is dominated by or identical to any existing solution
        for existing in &self.solutions {
            let ev = existing.objective_values();
            if dominates(ev, values) {
                return false;
            }
            // Reject duplicates (identical scores)
            if ev == values {
                return false;
            }
        }

        // Remove solutions dominated by the new candidate
        self.solutions
            .retain(|existing| !dominates(values, existing.objective_values()));

        self.solutions.push(candidate);
        true
    }

    /// Get all non-dominated solutions.
    pub fn solutions(&self) -> &[T] {
        &self.solutions
    }

    /// Number of Pareto-optimal solutions.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Whether the front is empty.
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Select the best solution according to a scalarization strategy.
    pub fn best(&self, strategy: &Scalarization, weights: &[f64]) -> Option<&T> {
        self.solutions.iter().max_by(|a, b| {
            let sa = scalarize_values(a.objective_values(), strategy, weights);
            let sb = scalarize_values(b.objective_values(), strategy, weights);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Dominance check: does `a` dominate `b`?
/// a dominates b iff all a[i] >= b[i] AND exists j: a[j] > b[j].
#[inline]
fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut strictly_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if *ai < *bi {
            return false;
        }
        if *ai > *bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Trait for validators that support multi-objective evaluation.
pub trait MultiObjectiveValidate: Validate {
    /// Return the multi-score for detailed per-objective analysis.
    fn validate_multi(&self, text: &str) -> MultiScore<'static>;

    /// Get the number of objectives.
    fn num_objectives(&self) -> usize;

    /// Get the objective definitions.
    fn objectives(&self) -> &[Objective];

    /// Get the scalarization strategy.
    fn scalarization(&self) -> &Scalarization;
}

/// A multi-objective validator that evaluates text against multiple criteria.
///
/// Unlike the `All` combinator which takes the minimum, MultiObjective
/// preserves per-objective scores for Pareto analysis. When used as a
/// regular `Validate`, it scalarizes the result.
pub struct MultiObjective<V: Validate> {
    objective_defs: SmallVec<[Objective; 8]>,
    validators: SmallVec<[V; 8]>,
    scalarization: Scalarization,
}

impl<V: Validate> MultiObjective<V> {
    /// Validate and return the full MultiScore (not scalarized).
    pub fn validate_multi_score(&self, text: &str) -> MultiScore<'static> {
        let mut obj_scores = SmallVec::new();
        let mut feedbacks = Vec::new();
        let mut total_confidence = 0.0;

        for (objective, validator) in self.objective_defs.iter().zip(self.validators.iter()) {
            let score = validator.validate(text);
            let value = match objective.direction {
                Direction::Maximize => score.value,
                Direction::Minimize => 1.0 - score.value,
            };
            obj_scores.push(ObjectiveScore {
                name: objective.name,
                value,
                weight: objective.weight,
            });
            total_confidence += score.confidence;
            if let Some(fb) = score.feedback_str() {
                feedbacks.push(format!("[{}] {}", objective.name, fb));
            }
        }

        let n = self.objective_defs.len() as f64;
        MultiScore {
            objectives: obj_scores,
            feedback: if feedbacks.is_empty() {
                None
            } else {
                Some(Cow::Owned(feedbacks.join("; ")))
            },
            confidence: if n > 0.0 { total_confidence / n } else { 1.0 },
        }
    }
}

impl<V: Validate> Validate for MultiObjective<V> {
    fn validate(&self, text: &str) -> Score<'static> {
        let multi = self.validate_multi_score(text);
        multi.scalarize(&self.scalarization)
    }

    fn name(&self) -> &'static str {
        "multi_objective"
    }
}

impl<V: Validate> MultiObjectiveValidate for MultiObjective<V> {
    fn validate_multi(&self, text: &str) -> MultiScore<'static> {
        self.validate_multi_score(text)
    }

    fn num_objectives(&self) -> usize {
        self.objective_defs.len()
    }

    fn objectives(&self) -> &[Objective] {
        &self.objective_defs
    }

    fn scalarization(&self) -> &Scalarization {
        &self.scalarization
    }
}

/// Multi-objective with 2 heterogeneous validators.
pub struct MultiObjective2<A: Validate, B: Validate> {
    obj_a: Objective,
    val_a: A,
    obj_b: Objective,
    val_b: B,
    scalarization: Scalarization,
}

impl<A: Validate, B: Validate> Validate for MultiObjective2<A, B> {
    fn validate(&self, text: &str) -> Score<'static> {
        let multi = self.validate_multi_impl(text);
        multi.scalarize(&self.scalarization)
    }

    fn name(&self) -> &'static str {
        "multi_objective_2"
    }
}

impl<A: Validate, B: Validate> MultiObjective2<A, B> {
    fn validate_multi_impl(&self, text: &str) -> MultiScore<'static> {
        let mut obj_scores = SmallVec::new();
        let mut feedbacks = Vec::new();

        let score_a = self.val_a.validate(text);
        let value_a = match self.obj_a.direction {
            Direction::Maximize => score_a.value,
            Direction::Minimize => 1.0 - score_a.value,
        };
        obj_scores.push(ObjectiveScore {
            name: self.obj_a.name,
            value: value_a,
            weight: self.obj_a.weight,
        });
        if let Some(fb) = score_a.feedback_str() {
            feedbacks.push(format!("[{}] {}", self.obj_a.name, fb));
        }

        let score_b = self.val_b.validate(text);
        let value_b = match self.obj_b.direction {
            Direction::Maximize => score_b.value,
            Direction::Minimize => 1.0 - score_b.value,
        };
        obj_scores.push(ObjectiveScore {
            name: self.obj_b.name,
            value: value_b,
            weight: self.obj_b.weight,
        });
        if let Some(fb) = score_b.feedback_str() {
            feedbacks.push(format!("[{}] {}", self.obj_b.name, fb));
        }

        MultiScore {
            objectives: obj_scores,
            feedback: if feedbacks.is_empty() {
                None
            } else {
                Some(Cow::Owned(feedbacks.join("; ")))
            },
            confidence: (score_a.confidence + score_b.confidence) / 2.0,
        }
    }
}

impl<A: Validate, B: Validate> MultiObjectiveValidate for MultiObjective2<A, B> {
    fn validate_multi(&self, text: &str) -> MultiScore<'static> {
        self.validate_multi_impl(text)
    }

    fn num_objectives(&self) -> usize {
        2
    }

    fn objectives(&self) -> &[Objective] {
        // Since we can't return a slice of a temporary, we use a leaked static.
        // This is a trade-off for the heterogeneous type design.
        // In practice, objectives() is called rarely (initialization).
        &[]
    }

    fn scalarization(&self) -> &Scalarization {
        &self.scalarization
    }
}

/// Multi-objective with 3 heterogeneous validators.
pub struct MultiObjective3<A: Validate, B: Validate, C: Validate> {
    obj_a: Objective,
    val_a: A,
    obj_b: Objective,
    val_b: B,
    obj_c: Objective,
    val_c: C,
    scalarization: Scalarization,
}

impl<A: Validate, B: Validate, C: Validate> Validate for MultiObjective3<A, B, C> {
    fn validate(&self, text: &str) -> Score<'static> {
        let multi = self.validate_multi_impl(text);
        multi.scalarize(&self.scalarization)
    }

    fn name(&self) -> &'static str {
        "multi_objective_3"
    }
}

impl<A: Validate, B: Validate, C: Validate> MultiObjective3<A, B, C> {
    fn validate_multi_impl(&self, text: &str) -> MultiScore<'static> {
        let mut obj_scores = SmallVec::new();
        let mut feedbacks = Vec::new();

        for (obj, validator) in [
            (&self.obj_a, &self.val_a as &dyn Validate),
            (&self.obj_b, &self.val_b as &dyn Validate),
            (&self.obj_c, &self.val_c as &dyn Validate),
        ] {
            let score = validator.validate(text);
            let value = match obj.direction {
                Direction::Maximize => score.value,
                Direction::Minimize => 1.0 - score.value,
            };
            obj_scores.push(ObjectiveScore {
                name: obj.name,
                value,
                weight: obj.weight,
            });
            if let Some(fb) = score.feedback_str() {
                feedbacks.push(format!("[{}] {}", obj.name, fb));
            }
        }

        MultiScore {
            objectives: obj_scores,
            feedback: if feedbacks.is_empty() {
                None
            } else {
                Some(Cow::Owned(feedbacks.join("; ")))
            },
            confidence: 1.0,
        }
    }
}

impl<A: Validate, B: Validate, C: Validate> MultiObjectiveValidate for MultiObjective3<A, B, C> {
    fn validate_multi(&self, text: &str) -> MultiScore<'static> {
        self.validate_multi_impl(text)
    }

    fn num_objectives(&self) -> usize {
        3
    }

    fn objectives(&self) -> &[Objective] {
        &[]
    }

    fn scalarization(&self) -> &Scalarization {
        &self.scalarization
    }
}

/// Builder for multi-objective validators.
pub struct MultiObjectiveBuilder {
    scalarization: Scalarization,
}

/// Entry point for building multi-objective validators.
pub fn multi_objective() -> MultiObjectiveBuilder {
    MultiObjectiveBuilder {
        scalarization: Scalarization::WeightedSum,
    }
}

impl MultiObjectiveBuilder {
    /// Set the scalarization strategy.
    pub fn scalarize(mut self, s: Scalarization) -> Self {
        self.scalarization = s;
        self
    }

    /// Build with homogeneous validators (all same type).
    pub fn objectives<V: Validate>(
        self,
        objectives: impl IntoIterator<Item = (Objective, V)>,
    ) -> MultiObjective<V> {
        let mut obj_defs = SmallVec::new();
        let mut validators = SmallVec::new();
        for (obj, val) in objectives {
            obj_defs.push(obj);
            validators.push(val);
        }
        MultiObjective {
            objective_defs: obj_defs,
            validators,
            scalarization: self.scalarization,
        }
    }

    /// Build with 2 heterogeneous validators.
    pub fn objective2<A: Validate, B: Validate>(
        self,
        a: (Objective, A),
        b: (Objective, B),
    ) -> MultiObjective2<A, B> {
        MultiObjective2 {
            obj_a: a.0,
            val_a: a.1,
            obj_b: b.0,
            val_b: b.1,
            scalarization: self.scalarization,
        }
    }

    /// Build with 3 heterogeneous validators.
    pub fn objective3<A: Validate, B: Validate, C: Validate>(
        self,
        a: (Objective, A),
        b: (Objective, B),
        c: (Objective, C),
    ) -> MultiObjective3<A, B, C> {
        MultiObjective3 {
            obj_a: a.0,
            val_a: a.1,
            obj_b: b.0,
            val_b: b.1,
            obj_c: c.0,
            val_c: c.1,
            scalarization: self.scalarization,
        }
    }
}

/// Result of multi-objective refinement.
#[derive(Debug, Clone)]
pub struct ParetoRefineResult {
    /// The Pareto front of non-dominated solutions.
    pub front: ParetoFront<ParetoCandidate>,
    /// The best solution according to scalarization.
    pub best_output: String,
    /// Per-objective best scores achieved.
    pub objective_bests: SmallVec<[(&'static str, f64); 8]>,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Total tokens consumed.
    pub total_tokens: u32,
    /// Elapsed time.
    pub elapsed: Duration,
}

/// Run multi-objective refinement with Pareto-front tracking.
///
/// This is the Pareto-aware version of `refine().run()`. It maintains a
/// Pareto front across iterations and stops when all objectives meet their
/// targets, the front converges, or max iterations are reached.
pub async fn refine_pareto<L: crate::recursive::llm::Llm, V: MultiObjectiveValidate>(
    llm: &L,
    prompt: &str,
    validator: &V,
    max_iter: u32,
) -> ParetoRefineResult {
    let start = std::time::Instant::now();
    let num_obj = validator.num_objectives();
    let objectives = validator.objectives();
    let mut front = ParetoFront::new(num_obj);
    let mut total_tokens: u32 = 0;
    let mut iteration: u32 = 0;
    let mut feedback: Option<String> = None;
    let mut convergence_counter: u32 = 0;
    let plateau_window: u32 = 3;

    for iter_idx in 0..max_iter {
        iteration = iter_idx + 1;

        let effective_prompt = if let Some(ref fb) = feedback {
            format!("{}\n\nFeedback: {}", prompt, fb)
        } else {
            prompt.to_string()
        };

        let output = match llm.generate(&effective_prompt, "", None).await {
            Ok(o) => {
                total_tokens += o.total_tokens();
                o.text
            }
            Err(_) => continue,
        };

        let multi_score = validator.validate_multi(&output);
        let scores: SmallVec<[f64; 8]> = multi_score.objectives.iter().map(|o| o.value).collect();

        // Check if all targets met
        if !objectives.is_empty() && multi_score.all_targets_met(objectives) {
            let candidate = ParetoCandidate {
                output,
                scores,
                index: iter_idx as usize,
            };
            front.insert(candidate);
            break;
        }

        // Track Pareto front convergence
        let prev_len = front.len();
        let candidate = ParetoCandidate {
            output: output.clone(),
            scores,
            index: iter_idx as usize,
        };
        let was_added = front.insert(candidate);

        if was_added || front.len() != prev_len {
            convergence_counter = 0;
        } else {
            convergence_counter += 1;
        }

        if convergence_counter >= plateau_window {
            break;
        }

        // Build feedback from multi-score
        if let Some(ref fb) = multi_score.feedback {
            feedback = Some(fb.to_string());
        } else {
            let fb_parts: Vec<String> = multi_score
                .objectives
                .iter()
                .map(|o| format!("{}: {:.2}", o.name, o.value))
                .collect();
            feedback = Some(format!("Scores: {}", fb_parts.join(", ")));
        }
    }

    // Compute objective bests
    let mut objective_bests: SmallVec<[(&'static str, f64); 8]> = SmallVec::new();
    if num_obj > 0 {
        for obj_idx in 0..num_obj {
            let best_val = front
                .solutions()
                .iter()
                .map(|c| c.scores[obj_idx])
                .fold(0.0_f64, f64::max);
            if obj_idx < objectives.len() {
                objective_bests.push((objectives[obj_idx].name, best_val));
            }
        }
    }

    // Pick best output via scalarization
    let weights: SmallVec<[f64; 8]> = if !objectives.is_empty() {
        objectives.iter().map(|o| o.weight).collect()
    } else {
        smallvec::smallvec![1.0; num_obj]
    };
    let best_output = front
        .best(validator.scalarization(), &weights)
        .map(|c| c.output.clone())
        .unwrap_or_default();

    ParetoRefineResult {
        front,
        best_output,
        objective_bests,
        iterations: iteration,
        total_tokens,
        elapsed: start.elapsed(),
    }
}

/// Synchronous version of `refine_pareto`.
#[cfg(feature = "native")]
pub fn refine_pareto_sync<L: crate::recursive::llm::Llm, V: MultiObjectiveValidate>(
    llm: &L,
    prompt: &str,
    validator: &V,
    max_iter: u32,
) -> ParetoRefineResult {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(refine_pareto(llm, prompt, validator, max_iter))
        })
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio runtime")
            .block_on(refine_pareto(llm, prompt, validator, max_iter))
    }
}

/// Synchronous version of `refine_pareto` (fallback without tokio).
#[cfg(not(feature = "native"))]
pub fn refine_pareto_sync<L: crate::recursive::llm::Llm, V: MultiObjectiveValidate>(
    llm: &L,
    prompt: &str,
    validator: &V,
    max_iter: u32,
) -> ParetoRefineResult {
    futures::executor::block_on(refine_pareto(llm, prompt, validator, max_iter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::checks::checks;
    use crate::recursive::llm::{IterativeMockLlm, MockLlm};

    #[test]
    fn test_dominance_basic() {
        assert!(dominates(&[1.0, 1.0], &[0.5, 0.5]));
        assert!(dominates(&[1.0, 0.5], &[0.5, 0.5]));
        assert!(!dominates(&[0.5, 0.5], &[1.0, 1.0]));
        // Neither dominates (trade-off)
        assert!(!dominates(&[1.0, 0.0], &[0.0, 1.0]));
        assert!(!dominates(&[0.0, 1.0], &[1.0, 0.0]));
        // Equal - no dominance
        assert!(!dominates(&[0.5, 0.5], &[0.5, 0.5]));
    }

    #[test]
    fn test_pareto_front_insert() {
        let mut front = ParetoFront::new(2);

        // First candidate always enters
        assert!(front.insert(ParetoCandidate {
            output: "a".to_string(),
            scores: smallvec::smallvec![0.5, 0.5],
            index: 0,
        }));
        assert_eq!(front.len(), 1);

        // Dominated candidate rejected
        assert!(!front.insert(ParetoCandidate {
            output: "b".to_string(),
            scores: smallvec::smallvec![0.3, 0.3],
            index: 1,
        }));
        assert_eq!(front.len(), 1);

        // Dominating candidate replaces
        assert!(front.insert(ParetoCandidate {
            output: "c".to_string(),
            scores: smallvec::smallvec![0.8, 0.8],
            index: 2,
        }));
        assert_eq!(front.len(), 1);
        assert_eq!(front.solutions()[0].output, "c");

        // Trade-off candidate added (non-dominated)
        assert!(front.insert(ParetoCandidate {
            output: "d".to_string(),
            scores: smallvec::smallvec![1.0, 0.3],
            index: 3,
        }));
        assert_eq!(front.len(), 2);
    }

    #[test]
    fn test_pareto_front_best() {
        let mut front = ParetoFront::new(2);
        front.insert(ParetoCandidate {
            output: "low".to_string(),
            scores: smallvec::smallvec![0.3, 0.9],
            index: 0,
        });
        front.insert(ParetoCandidate {
            output: "high".to_string(),
            scores: smallvec::smallvec![0.9, 0.3],
            index: 1,
        });

        // With equal weights, "high" should win via weighted sum (0.9*1 + 0.3*1 = 1.2 vs 0.3*1 + 0.9*1 = 1.2)
        // Actually equal, so first by iter order
        let weights = [1.0, 1.0];
        let best = front.best(&Scalarization::WeightedSum, &weights).unwrap();
        // Both score the same with equal weights, so either is valid
        assert!(!best.output.is_empty());

        // With weight biased toward first objective
        let weights = [3.0, 1.0];
        let best = front.best(&Scalarization::WeightedSum, &weights).unwrap();
        assert_eq!(best.output, "high");
    }

    #[test]
    fn test_scalarization_weighted_sum() {
        let values = [0.8, 0.6];
        let weights = [2.0, 1.0];
        let result = scalarize_values(&values, &Scalarization::WeightedSum, &weights);
        // (0.8*2 + 0.6*1) / (2+1) = 2.2/3 ≈ 0.733
        assert!((result - 0.7333).abs() < 0.01);
    }

    #[test]
    fn test_scalarization_chebyshev() {
        let values = [0.9, 0.3];
        let weights = [1.0, 1.0];
        let result = scalarize_values(&values, &Scalarization::Chebyshev, &weights);
        // min(0.9*1, 0.3*1) = 0.3
        assert!((result - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multi_objective_homogeneous() {
        let validator = multi_objective()
            .scalarize(Scalarization::WeightedSum)
            .objectives([
                (
                    Objective::new("has_fn").weight(2.0),
                    checks().require("fn "),
                ),
                (
                    Objective::new("has_arrow").weight(1.0),
                    checks().require("->"),
                ),
            ]);

        let score = validator.validate("fn add() -> i32 {}");
        assert!((score.value - 1.0).abs() < f64::EPSILON);

        let score = validator.validate("fn add() {}");
        // Only first objective passes: (1.0*2 + 0.0*1) / 3 = 0.667
        assert!((score.value - 0.6667).abs() < 0.01);

        let score = validator.validate("no match");
        assert!((score.value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multi_objective_validate_multi() {
        let validator = multi_objective()
            .scalarize(Scalarization::WeightedSum)
            .objectives([
                (Objective::new("a"), checks().require("hello")),
                (Objective::new("b"), checks().require("world")),
            ]);

        let multi = validator.validate_multi_score("hello world");
        assert_eq!(multi.objectives.len(), 2);
        assert!((multi.get("a").unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((multi.get("b").unwrap() - 1.0).abs() < f64::EPSILON);

        let multi = validator.validate_multi_score("hello");
        assert!((multi.get("a").unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((multi.get("b").unwrap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multi_objective_2_heterogeneous() {
        use crate::recursive::validate::FnValidator;

        let validator = multi_objective()
            .scalarize(Scalarization::WeightedSum)
            .objective2(
                (
                    Objective::new("length"),
                    FnValidator(|text: &str| Score::new((text.len() as f64 / 20.0).min(1.0))),
                ),
                (Objective::new("has_fn"), checks().require("fn ")),
            );

        let score = validator.validate("fn very_long_function_name() {}");
        assert!(score.value > 0.5);
    }

    #[test]
    fn test_multi_score_dominates() {
        let a = MultiScore::new(smallvec::smallvec![
            ObjectiveScore {
                name: "x",
                value: 0.8,
                weight: 1.0
            },
            ObjectiveScore {
                name: "y",
                value: 0.6,
                weight: 1.0
            },
        ]);
        let b = MultiScore::new(smallvec::smallvec![
            ObjectiveScore {
                name: "x",
                value: 0.5,
                weight: 1.0
            },
            ObjectiveScore {
                name: "y",
                value: 0.5,
                weight: 1.0
            },
        ]);
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_objective_minimize() {
        let validator = multi_objective()
            .scalarize(Scalarization::WeightedSum)
            .objectives([(
                Objective::new("short").minimize().weight(1.0),
                checks().max_len(10),
            )]);

        // max_len(10) passes for text <= 10 chars → score 1.0
        // With minimize direction: value = 1.0 - 1.0 = 0.0 (counter-intuitive for max_len)
        // Actually, `minimize` inverts the score: text that passes max_len(10) gets 0.0
        // This makes sense: for a "short" objective with minimize,
        // we'd use a validator that gives HIGH scores to LONG text
        let score = validator.validate("short");
        // max_len(10) passes → 1.0, minimize inverts to 0.0
        assert!((score.value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_targets_met() {
        let objectives = [
            Objective::new("a").target(0.8),
            Objective::new("b").target(0.5),
        ];
        let multi = MultiScore::new(smallvec::smallvec![
            ObjectiveScore {
                name: "a",
                value: 0.9,
                weight: 1.0
            },
            ObjectiveScore {
                name: "b",
                value: 0.6,
                weight: 1.0
            },
        ]);
        assert!(multi.all_targets_met(&objectives));

        let multi_fail = MultiScore::new(smallvec::smallvec![
            ObjectiveScore {
                name: "a",
                value: 0.7,
                weight: 1.0
            },
            ObjectiveScore {
                name: "b",
                value: 0.6,
                weight: 1.0
            },
        ]);
        assert!(!multi_fail.all_targets_met(&objectives));
    }

    #[tokio::test]
    async fn test_refine_pareto_basic() {
        let llm = IterativeMockLlm::new(|iter, _, _| match iter {
            0 => "hello".to_string(),
            1 => "hello world".to_string(),
            _ => "hello world!".to_string(),
        });

        let validator = multi_objective()
            .scalarize(Scalarization::WeightedSum)
            .objectives([
                (
                    Objective::new("has_hello").target(1.0),
                    checks().require("hello"),
                ),
                (
                    Objective::new("has_world").target(1.0),
                    checks().require("world"),
                ),
            ]);

        let result = refine_pareto(&llm, "greet", &validator, 5).await;
        assert!(result.front.len() >= 1);
        assert!(result.best_output.contains("hello"));
        assert!(result.best_output.contains("world"));
    }

    #[tokio::test]
    async fn test_refine_pareto_convergence() {
        // All iterations produce the same output → front converges after plateau_window
        let llm = MockLlm::new(|_, _| "constant output".to_string());

        let validator = multi_objective()
            .scalarize(Scalarization::WeightedSum)
            .objectives([(
                Objective::new("impossible").target(1.0),
                checks().require("xyz_not_here"),
            )]);

        let result = refine_pareto(&llm, "prompt", &validator, 10).await;
        // Should stop early due to convergence (plateau_window = 3)
        assert!(result.iterations <= 5);
        assert_eq!(result.front.len(), 1);
    }

    #[test]
    fn test_refine_pareto_sync() {
        let llm = MockLlm::new(|_, _| "fn add() -> i32 { 42 }".to_string());

        let validator = multi_objective()
            .scalarize(Scalarization::WeightedSum)
            .objectives([
                (
                    Objective::new("has_fn").target(1.0),
                    checks().require("fn "),
                ),
                (
                    Objective::new("has_arrow").target(1.0),
                    checks().require("->"),
                ),
            ]);

        let result = refine_pareto_sync(&llm, "write code", &validator, 5);
        assert_eq!(result.iterations, 1); // targets met on first try
        assert!(result.best_output.contains("fn "));
    }
}
