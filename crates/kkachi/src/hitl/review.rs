// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Review context and decisions for HITL.

use crate::diff::ModuleDiff;
use crate::recursive::IterationRecord;

/// Reason why a review was triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewTrigger {
    /// Regular interval review.
    Interval,
    /// Score dropped from previous iteration.
    ScoreDrop,
    /// Process is about to converge/finish.
    Convergence,
    /// First iteration.
    FirstIteration,
    /// Manually requested by user.
    Manual,
    /// Keyword in feedback triggered review.
    Keyword,
}

impl ReviewTrigger {
    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Interval => "Scheduled review interval",
            Self::ScoreDrop => "Score decreased",
            Self::Convergence => "About to accept result",
            Self::FirstIteration => "First iteration",
            Self::Manual => "Manual review requested",
            Self::Keyword => "Keyword triggered",
        }
    }

    /// Whether this trigger indicates a potential problem.
    pub fn is_warning(&self) -> bool {
        matches!(self, Self::ScoreDrop)
    }
}

/// Context provided to the human reviewer.
///
/// Contains all information needed to make an informed review decision.
#[derive(Debug)]
pub struct ReviewContext<'a> {
    /// Current iteration number.
    pub iteration: u32,
    /// Maximum configured iterations.
    pub max_iterations: u32,
    /// Diff from previous iteration.
    pub diff: ModuleDiff<'a>,
    /// Current score.
    pub score: f64,
    /// Previous iteration score.
    pub prev_score: f64,
    /// Critic feedback (if any).
    pub feedback: Option<&'a str>,
    /// Current output/answer.
    pub output: &'a str,
    /// Previous output (for comparison).
    pub prev_output: Option<&'a str>,
    /// Full iteration history.
    pub history: &'a [IterationRecord<'a>],
    /// Reason this review was triggered.
    pub trigger: ReviewTrigger,
    /// Domain/category.
    pub domain: &'a str,
    /// Original question/input.
    pub question: &'a str,
}

impl<'a> ReviewContext<'a> {
    /// Create a new review context.
    pub fn new(
        iteration: u32,
        max_iterations: u32,
        score: f64,
        output: &'a str,
        trigger: ReviewTrigger,
    ) -> Self {
        Self {
            iteration,
            max_iterations,
            diff: ModuleDiff::new(),
            score,
            prev_score: 0.0,
            feedback: None,
            output,
            prev_output: None,
            history: &[],
            trigger,
            domain: "",
            question: "",
        }
    }

    /// Set the diff.
    pub fn with_diff(mut self, diff: ModuleDiff<'a>) -> Self {
        self.diff = diff;
        self
    }

    /// Set the previous score.
    pub fn with_prev_score(mut self, score: f64) -> Self {
        self.prev_score = score;
        self
    }

    /// Set the feedback.
    pub fn with_feedback(mut self, feedback: &'a str) -> Self {
        self.feedback = Some(feedback);
        self
    }

    /// Set the previous output.
    pub fn with_prev_output(mut self, output: &'a str) -> Self {
        self.prev_output = Some(output);
        self
    }

    /// Set the iteration history.
    pub fn with_history(mut self, history: &'a [IterationRecord<'a>]) -> Self {
        self.history = history;
        self
    }

    /// Set the domain.
    pub fn with_domain(mut self, domain: &'a str) -> Self {
        self.domain = domain;
        self
    }

    /// Set the question.
    pub fn with_question(mut self, question: &'a str) -> Self {
        self.question = question;
        self
    }

    /// Get the score change.
    pub fn score_change(&self) -> f64 {
        self.score - self.prev_score
    }

    /// Check if score improved.
    pub fn score_improved(&self) -> bool {
        self.score > self.prev_score
    }

    /// Get progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.max_iterations == 0 {
            0.0
        } else {
            self.iteration as f64 / self.max_iterations as f64
        }
    }

    /// Check if this is the final iteration.
    pub fn is_final(&self) -> bool {
        self.iteration >= self.max_iterations || matches!(self.trigger, ReviewTrigger::Convergence)
    }
}

/// Human's decision after reviewing an iteration.
#[derive(Debug, Clone)]
pub enum ReviewDecision {
    /// Accept this iteration and continue.
    Accept,
    /// Reject this iteration and try an alternative.
    Reject,
    /// Manually edit the content.
    Edit {
        /// Edited instruction (if changed).
        instruction: Option<String>,
        /// Edited output (if changed).
        output: Option<String>,
        /// Additional guidance for next iteration.
        guidance: Option<String>,
    },
    /// Stop and return the current best result.
    Stop,
    /// Accept this iteration as final (stop iteration).
    AcceptFinal,
    /// Go back to a previous iteration.
    Rollback {
        /// Iteration to rollback to.
        to_iteration: u32,
    },
    /// Skip the next N iterations without review.
    SkipNext {
        /// Number of iterations to skip.
        count: u32,
    },
    /// Pause for a period and resume.
    Pause {
        /// Duration to pause.
        duration: std::time::Duration,
    },
    /// Request more information.
    RequestInfo {
        /// What information is needed.
        query: String,
    },
}

impl ReviewDecision {
    /// Create an edit decision with just output changes.
    pub fn edit_output(output: String) -> Self {
        Self::Edit {
            instruction: None,
            output: Some(output),
            guidance: None,
        }
    }

    /// Create an edit decision with guidance.
    pub fn with_guidance(guidance: String) -> Self {
        Self::Edit {
            instruction: None,
            output: None,
            guidance: Some(guidance),
        }
    }

    /// Check if this decision continues iteration.
    pub fn continues_iteration(&self) -> bool {
        matches!(
            self,
            Self::Accept | Self::Edit { .. } | Self::SkipNext { .. } | Self::Rollback { .. }
        )
    }

    /// Check if this decision stops iteration.
    pub fn stops_iteration(&self) -> bool {
        matches!(self, Self::Stop | Self::AcceptFinal)
    }

    /// Get the number of iterations to skip (0 if not skipping).
    pub fn skip_count(&self) -> u32 {
        match self {
            Self::SkipNext { count } => *count,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_review_trigger_description() {
        assert!(!ReviewTrigger::Interval.description().is_empty());
        assert!(ReviewTrigger::ScoreDrop.is_warning());
        assert!(!ReviewTrigger::Interval.is_warning());
    }

    #[test]
    fn test_review_context_basic() {
        let ctx = ReviewContext::new(3, 10, 0.75, "output", ReviewTrigger::Interval);

        assert_eq!(ctx.iteration, 3);
        assert_eq!(ctx.max_iterations, 10);
        assert_eq!(ctx.score, 0.75);
        assert_eq!(ctx.output, "output");
    }

    #[test]
    fn test_review_context_score_change() {
        let ctx =
            ReviewContext::new(1, 10, 0.75, "output", ReviewTrigger::Interval).with_prev_score(0.5);

        assert_eq!(ctx.score_change(), 0.25);
        assert!(ctx.score_improved());
    }

    #[test]
    fn test_review_context_progress() {
        let ctx = ReviewContext::new(5, 10, 0.5, "output", ReviewTrigger::Interval);
        assert_eq!(ctx.progress(), 0.5);

        let ctx2 = ReviewContext::new(10, 10, 0.9, "output", ReviewTrigger::Convergence);
        assert!(ctx2.is_final());
    }

    #[test]
    fn test_review_decision_continues() {
        assert!(ReviewDecision::Accept.continues_iteration());
        assert!(ReviewDecision::SkipNext { count: 5 }.continues_iteration());
        assert!(!ReviewDecision::Stop.continues_iteration());
        assert!(!ReviewDecision::AcceptFinal.continues_iteration());
    }

    #[test]
    fn test_review_decision_stops() {
        assert!(ReviewDecision::Stop.stops_iteration());
        assert!(ReviewDecision::AcceptFinal.stops_iteration());
        assert!(!ReviewDecision::Accept.stops_iteration());
    }

    #[test]
    fn test_review_decision_skip_count() {
        assert_eq!(ReviewDecision::SkipNext { count: 3 }.skip_count(), 3);
        assert_eq!(ReviewDecision::Accept.skip_count(), 0);
    }

    #[test]
    fn test_edit_decision_helpers() {
        let decision = ReviewDecision::edit_output("new output".to_string());
        if let ReviewDecision::Edit { output, .. } = decision {
            assert_eq!(output, Some("new output".to_string()));
        } else {
            panic!("Expected Edit decision");
        }
    }
}
