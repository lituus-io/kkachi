// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Human reviewer traits.

use super::{ReviewContext, ReviewDecision};

/// Synchronous human reviewer trait.
///
/// Implement this trait to create custom review interfaces
/// (terminal, web, GUI, etc.).
pub trait HumanReviewer: Send + Sync {
    /// Review an iteration and return a decision.
    ///
    /// This method blocks until the human makes a decision.
    fn review(&self, ctx: ReviewContext<'_>) -> ReviewDecision;

    /// Called to display iteration progress (optional).
    ///
    /// Default implementation does nothing.
    fn on_progress(&self, _iteration: u32, _score: f64) {}

    /// Called when iteration starts (optional).
    fn on_iteration_start(&self, _iteration: u32) {}

    /// Called when iteration completes (optional).
    fn on_iteration_complete(&self, _iteration: u32, _score: f64) {}

    /// Called when process completes (optional).
    fn on_complete(&self, _final_score: f64, _total_iterations: u32) {}

    /// Called when an error occurs (optional).
    fn on_error(&self, _error: &str) {}
}

/// Asynchronous human reviewer trait.
///
/// Use this for non-blocking review interfaces (web APIs, etc.).
#[async_trait::async_trait]
pub trait AsyncHumanReviewer: Send + Sync {
    /// Review an iteration and return a decision asynchronously.
    async fn review(&self, ctx: ReviewContext<'_>) -> ReviewDecision;

    /// Called to display iteration progress (optional).
    async fn on_progress(&self, _iteration: u32, _score: f64) {}

    /// Called when iteration starts (optional).
    async fn on_iteration_start(&self, _iteration: u32) {}

    /// Called when iteration completes (optional).
    async fn on_iteration_complete(&self, _iteration: u32, _score: f64) {}

    /// Called when process completes (optional).
    async fn on_complete(&self, _final_score: f64, _total_iterations: u32) {}

    /// Called when an error occurs (optional).
    async fn on_error(&self, _error: &str) {}
}

/// A reviewer that always accepts without human input.
///
/// Useful for testing or automated pipelines.
#[derive(Debug, Clone, Default)]
pub struct AutoAcceptReviewer;

impl HumanReviewer for AutoAcceptReviewer {
    fn review(&self, _ctx: ReviewContext<'_>) -> ReviewDecision {
        ReviewDecision::Accept
    }
}

/// A reviewer that accepts based on score threshold.
///
/// Accepts if score is above threshold, otherwise continues.
#[derive(Debug, Clone)]
pub struct ThresholdReviewer {
    /// Score threshold for auto-accept.
    pub threshold: f64,
}

impl ThresholdReviewer {
    /// Create a new threshold reviewer.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl HumanReviewer for ThresholdReviewer {
    fn review(&self, ctx: ReviewContext<'_>) -> ReviewDecision {
        if ctx.score >= self.threshold {
            ReviewDecision::AcceptFinal
        } else {
            ReviewDecision::Accept
        }
    }
}

/// A reviewer that records decisions for later playback.
///
/// Useful for testing and debugging.
#[derive(Debug)]
pub struct RecordingReviewer {
    /// Recorded contexts.
    pub contexts: std::sync::Mutex<Vec<RecordedReview>>,
    /// Default decision.
    pub default_decision: ReviewDecision,
}

impl Default for RecordingReviewer {
    fn default() -> Self {
        Self {
            contexts: std::sync::Mutex::new(Vec::new()),
            default_decision: ReviewDecision::Accept,
        }
    }
}

/// A recorded review for playback.
#[derive(Debug, Clone)]
pub struct RecordedReview {
    /// Iteration number.
    pub iteration: u32,
    /// Score at time of review.
    pub score: f64,
    /// Trigger reason.
    pub trigger: super::ReviewTrigger,
    /// Output at time of review.
    pub output: String,
}

impl RecordingReviewer {
    /// Create a new recording reviewer with default accept.
    pub fn new() -> Self {
        Self {
            contexts: std::sync::Mutex::new(Vec::new()),
            default_decision: ReviewDecision::Accept,
        }
    }

    /// Set the default decision.
    pub fn with_default(mut self, decision: ReviewDecision) -> Self {
        self.default_decision = decision;
        self
    }

    /// Get all recorded reviews.
    pub fn recordings(&self) -> Vec<RecordedReview> {
        self.contexts.lock().unwrap().clone()
    }
}

impl HumanReviewer for RecordingReviewer {
    fn review(&self, ctx: ReviewContext<'_>) -> ReviewDecision {
        let recorded = RecordedReview {
            iteration: ctx.iteration,
            score: ctx.score,
            trigger: ctx.trigger,
            output: ctx.output.to_string(),
        };

        self.contexts.lock().unwrap().push(recorded);
        self.default_decision.clone()
    }
}

/// A reviewer that uses a callback function.
pub struct CallbackReviewer<F>
where
    F: Fn(ReviewContext<'_>) -> ReviewDecision + Send + Sync,
{
    callback: F,
}

impl<F> CallbackReviewer<F>
where
    F: Fn(ReviewContext<'_>) -> ReviewDecision + Send + Sync,
{
    /// Create a new callback reviewer.
    pub fn new(callback: F) -> Self {
        Self { callback }
    }
}

impl<F> HumanReviewer for CallbackReviewer<F>
where
    F: Fn(ReviewContext<'_>) -> ReviewDecision + Send + Sync,
{
    fn review(&self, ctx: ReviewContext<'_>) -> ReviewDecision {
        (self.callback)(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_accept_reviewer() {
        let reviewer = AutoAcceptReviewer;
        let ctx = ReviewContext::new(0, 10, 0.5, "output", super::super::ReviewTrigger::Interval);
        let decision = reviewer.review(ctx);
        assert!(matches!(decision, ReviewDecision::Accept));
    }

    #[test]
    fn test_threshold_reviewer_accept() {
        let reviewer = ThresholdReviewer::new(0.8);
        let ctx = ReviewContext::new(0, 10, 0.9, "output", super::super::ReviewTrigger::Interval);
        let decision = reviewer.review(ctx);
        assert!(matches!(decision, ReviewDecision::AcceptFinal));
    }

    #[test]
    fn test_threshold_reviewer_continue() {
        let reviewer = ThresholdReviewer::new(0.8);
        let ctx = ReviewContext::new(0, 10, 0.5, "output", super::super::ReviewTrigger::Interval);
        let decision = reviewer.review(ctx);
        assert!(matches!(decision, ReviewDecision::Accept));
    }

    #[test]
    fn test_recording_reviewer() {
        let reviewer = RecordingReviewer::new();

        let ctx1 = ReviewContext::new(0, 10, 0.5, "output1", super::super::ReviewTrigger::Interval);
        reviewer.review(ctx1);

        let ctx2 = ReviewContext::new(1, 10, 0.7, "output2", super::super::ReviewTrigger::Interval);
        reviewer.review(ctx2);

        let recordings = reviewer.recordings();
        assert_eq!(recordings.len(), 2);
        assert_eq!(recordings[0].iteration, 0);
        assert_eq!(recordings[1].iteration, 1);
    }

    #[test]
    fn test_callback_reviewer() {
        let reviewer = CallbackReviewer::new(|ctx| {
            if ctx.score >= 0.9 {
                ReviewDecision::AcceptFinal
            } else {
                ReviewDecision::Accept
            }
        });

        let ctx = ReviewContext::new(0, 10, 0.95, "output", super::super::ReviewTrigger::Interval);
        let decision = reviewer.review(ctx);
        assert!(matches!(decision, ReviewDecision::AcceptFinal));
    }
}
