// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Convergence criteria for recursive refinement
//!
//! Defines when to stop iterating based on various conditions.

use super::state::RecursiveState;
use crate::str_view::StrView;

/// Convergence criterion for stopping iteration.
///
/// Implementations decide when the recursive loop should terminate.
pub trait ConvergenceCriterion: Send + Sync {
    /// Check if we should stop iterating.
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool;

    /// Optional: provide feedback if not converged.
    fn feedback<'a>(&self, _state: &RecursiveState<'a>) -> Option<StrView<'a>> {
        None
    }
}

/// Stop after N iterations.
#[derive(Clone, Copy, Debug)]
pub struct MaxIterations(pub u32);

impl ConvergenceCriterion for MaxIterations {
    #[inline]
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool {
        state.iteration >= self.0
    }
}

impl Default for MaxIterations {
    fn default() -> Self {
        Self(10)
    }
}

/// Stop when score exceeds threshold.
#[derive(Clone, Copy, Debug)]
pub struct ScoreThreshold(pub f64);

impl ConvergenceCriterion for ScoreThreshold {
    #[inline]
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool {
        state.best_score() >= self.0
    }
}

impl Default for ScoreThreshold {
    fn default() -> Self {
        Self(0.9)
    }
}

/// Stop when score improvement drops below threshold.
#[derive(Clone, Copy, Debug)]
pub struct ScorePlateau {
    /// Minimum improvement to continue (e.g., 0.01 = 1%)
    pub min_improvement: f64,
    /// Window of iterations to compare
    pub window: usize,
}

impl ConvergenceCriterion for ScorePlateau {
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool {
        if state.scores.len() < self.window + 1 {
            return false;
        }

        let len = state.scores.len();
        let recent = &state.scores[len - self.window..];
        let previous = state.scores[len - self.window - 1];

        let best_recent = recent
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let improvement = best_recent - previous;
        improvement < self.min_improvement
    }
}

impl Default for ScorePlateau {
    fn default() -> Self {
        Self {
            min_improvement: 0.01,
            window: 3,
        }
    }
}

/// Stop when no feedback is generated (critic is satisfied).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoFeedback;

impl ConvergenceCriterion for NoFeedback {
    #[inline]
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool {
        // Check if the last iteration had no feedback
        state
            .history
            .last()
            .map(|r| r.feedback.is_none())
            .unwrap_or(false)
    }
}

/// Composite: stop when ANY criterion is met.
#[derive(Clone, Copy, Debug)]
pub struct Any<A, B>(pub A, pub B);

impl<A: ConvergenceCriterion, B: ConvergenceCriterion> ConvergenceCriterion for Any<A, B> {
    #[inline]
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool {
        self.0.should_stop(state) || self.1.should_stop(state)
    }

    fn feedback<'a>(&self, state: &RecursiveState<'a>) -> Option<StrView<'a>> {
        self.0.feedback(state).or_else(|| self.1.feedback(state))
    }
}

/// Composite: stop when ALL criteria are met.
#[derive(Clone, Copy, Debug)]
pub struct All<A, B>(pub A, pub B);

impl<A: ConvergenceCriterion, B: ConvergenceCriterion> ConvergenceCriterion for All<A, B> {
    #[inline]
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool {
        self.0.should_stop(state) && self.1.should_stop(state)
    }

    fn feedback<'a>(&self, state: &RecursiveState<'a>) -> Option<StrView<'a>> {
        // Return feedback from whichever hasn't converged
        if !self.0.should_stop(state) {
            self.0.feedback(state)
        } else {
            self.1.feedback(state)
        }
    }
}

/// Combined criterion: max iterations OR score threshold.
///
/// The most common pattern - stop when we hit max iterations or achieve target score.
#[derive(Clone, Copy, Debug)]
pub struct StandardConvergence {
    /// Maximum number of iterations before stopping.
    pub max_iterations: u32,
    /// Target score threshold to achieve.
    pub score_threshold: f64,
}

impl ConvergenceCriterion for StandardConvergence {
    #[inline]
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool {
        state.iteration >= self.max_iterations || state.best_score() >= self.score_threshold
    }
}

impl Default for StandardConvergence {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            score_threshold: 0.9,
        }
    }
}

/// Smart convergence: combines plateau detection with max iterations and score threshold.
#[derive(Clone, Copy, Debug)]
pub struct SmartConvergence {
    /// Maximum number of iterations before stopping.
    pub max_iterations: u32,
    /// Target score threshold to achieve.
    pub score_threshold: f64,
    /// Number of iterations to consider for plateau detection.
    pub plateau_window: usize,
    /// Minimum improvement required to not be considered a plateau.
    pub plateau_threshold: f64,
}

impl ConvergenceCriterion for SmartConvergence {
    fn should_stop(&self, state: &RecursiveState<'_>) -> bool {
        // Always stop at max iterations
        if state.iteration >= self.max_iterations {
            return true;
        }

        // Stop if we achieved target score
        if state.best_score() >= self.score_threshold {
            return true;
        }

        // Stop if we've plateaued
        if state.scores.len() >= self.plateau_window + 1 {
            let len = state.scores.len();
            let recent_avg = state.scores[len - self.plateau_window..]
                .iter()
                .sum::<f64>()
                / self.plateau_window as f64;
            let prev_avg = state.scores[len - self.plateau_window - 1];

            if (recent_avg - prev_avg).abs() < self.plateau_threshold {
                return true;
            }
        }

        false
    }
}

impl Default for SmartConvergence {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            score_threshold: 0.9,
            plateau_window: 3,
            plateau_threshold: 0.01,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::state::RecursiveState;

    #[test]
    fn test_max_iterations() {
        let criterion = MaxIterations(3);
        let mut state = RecursiveState::new();

        assert!(!criterion.should_stop(&state));
        state.iteration = 2;
        assert!(!criterion.should_stop(&state));
        state.iteration = 3;
        assert!(criterion.should_stop(&state));
    }

    #[test]
    fn test_score_threshold() {
        let criterion = ScoreThreshold(0.8);
        let mut state = RecursiveState::new();

        state.record_score(0.5);
        assert!(!criterion.should_stop(&state));

        state.record_score(0.85);
        assert!(criterion.should_stop(&state));
    }

    #[test]
    fn test_score_plateau() {
        let criterion = ScorePlateau {
            min_improvement: 0.05,
            window: 2,
        };
        let mut state = RecursiveState::new();

        // Need at least window + 1 scores
        state.record_score(0.5);
        state.record_score(0.6);
        assert!(!criterion.should_stop(&state));

        state.record_score(0.7);
        assert!(!criterion.should_stop(&state)); // Still improving

        state.record_score(0.71);
        state.record_score(0.72);
        assert!(criterion.should_stop(&state)); // Plateau detected
    }

    #[test]
    fn test_any_composite() {
        let criterion = Any(MaxIterations(5), ScoreThreshold(0.9));
        let mut state = RecursiveState::new();

        assert!(!criterion.should_stop(&state));

        state.record_score(0.95);
        assert!(criterion.should_stop(&state)); // Score triggered

        state = RecursiveState::new();
        state.iteration = 5;
        assert!(criterion.should_stop(&state)); // Max iterations triggered
    }

    #[test]
    fn test_all_composite() {
        let criterion = All(MaxIterations(5), ScoreThreshold(0.9));
        let mut state = RecursiveState::new();

        state.record_score(0.95);
        assert!(!criterion.should_stop(&state)); // Score met, but not max iterations

        state.iteration = 5;
        assert!(criterion.should_stop(&state)); // Both met

        state = RecursiveState::new();
        state.iteration = 5;
        state.record_score(0.5);
        assert!(!criterion.should_stop(&state)); // Max iterations met, but not score
    }

    #[test]
    fn test_standard_convergence() {
        let criterion = StandardConvergence::default();
        let mut state = RecursiveState::new();

        assert!(!criterion.should_stop(&state));

        state.iteration = 10;
        assert!(criterion.should_stop(&state));
    }
}
