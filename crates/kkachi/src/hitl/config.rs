// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! HITL configuration.

use crate::diff::DiffStyle;
use std::time::Duration;

/// Configuration for Human-in-the-Loop review.
///
/// Controls when and how human review is triggered during
/// optimization or refinement processes.
#[derive(Debug, Clone)]
pub struct HITLConfig {
    /// Whether HITL is enabled.
    pub enabled: bool,
    /// Review every N iterations (0 = disabled).
    pub interval: u32,
    /// Review when score drops from previous iteration.
    pub on_score_drop: bool,
    /// Review before accepting convergence/final result.
    pub on_convergence: bool,
    /// Review on first iteration.
    pub on_first: bool,
    /// Review when feedback contains certain keywords.
    pub on_keywords: Vec<String>,
    /// Timeout waiting for human response (None = wait forever).
    pub timeout: Option<Duration>,
    /// Show diffs during review.
    pub show_diff: bool,
    /// Diff rendering style.
    pub diff_style: DiffStyle,
    /// Auto-accept after N seconds of no response (None = never).
    pub auto_accept_timeout: Option<Duration>,
    /// Minimum score threshold to skip review.
    pub skip_above_score: Option<f64>,
}

impl Default for HITLConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl HITLConfig {
    /// Create a disabled HITL config.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            interval: 0,
            on_score_drop: false,
            on_convergence: false,
            on_first: false,
            on_keywords: Vec::new(),
            timeout: None,
            show_diff: true,
            diff_style: DiffStyle::Unified,
            auto_accept_timeout: None,
            skip_above_score: None,
        }
    }

    /// Create a config that reviews every iteration.
    pub fn every_iteration() -> Self {
        Self {
            enabled: true,
            interval: 1,
            on_score_drop: true,
            on_convergence: true,
            on_first: true,
            on_keywords: Vec::new(),
            timeout: None,
            show_diff: true,
            diff_style: DiffStyle::Unified,
            auto_accept_timeout: None,
            skip_above_score: None,
        }
    }

    /// Create a config that reviews every N iterations.
    pub fn every(n: u32) -> Self {
        Self {
            enabled: true,
            interval: n,
            on_score_drop: false,
            on_convergence: true,
            on_first: false,
            on_keywords: Vec::new(),
            timeout: None,
            show_diff: true,
            diff_style: DiffStyle::Unified,
            auto_accept_timeout: None,
            skip_above_score: None,
        }
    }

    /// Create a config that reviews only at completion/convergence.
    pub fn on_completion() -> Self {
        Self {
            enabled: true,
            interval: 0,
            on_score_drop: false,
            on_convergence: true,
            on_first: false,
            on_keywords: Vec::new(),
            timeout: None,
            show_diff: true,
            diff_style: DiffStyle::Unified,
            auto_accept_timeout: None,
            skip_above_score: None,
        }
    }

    /// Create a config that reviews on score drops.
    pub fn on_regression() -> Self {
        Self {
            enabled: true,
            interval: 0,
            on_score_drop: true,
            on_convergence: true,
            on_first: false,
            on_keywords: Vec::new(),
            timeout: None,
            show_diff: true,
            diff_style: DiffStyle::Unified,
            auto_accept_timeout: None,
            skip_above_score: None,
        }
    }

    /// Enable HITL.
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Set review interval.
    pub fn with_interval(mut self, n: u32) -> Self {
        self.interval = n;
        self
    }

    /// Enable review on score drop.
    pub fn with_score_drop_review(mut self, enabled: bool) -> Self {
        self.on_score_drop = enabled;
        self
    }

    /// Enable review on convergence.
    pub fn with_convergence_review(mut self, enabled: bool) -> Self {
        self.on_convergence = enabled;
        self
    }

    /// Enable review on first iteration.
    pub fn with_first_review(mut self, enabled: bool) -> Self {
        self.on_first = enabled;
        self
    }

    /// Set keywords that trigger review.
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.on_keywords = keywords;
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set auto-accept timeout.
    pub fn with_auto_accept(mut self, timeout: Duration) -> Self {
        self.auto_accept_timeout = Some(timeout);
        self
    }

    /// Set diff style.
    pub fn with_diff_style(mut self, style: DiffStyle) -> Self {
        self.diff_style = style;
        self
    }

    /// Disable diff display.
    pub fn without_diff(mut self) -> Self {
        self.show_diff = false;
        self
    }

    /// Skip review if score is above threshold.
    pub fn skip_above(mut self, score: f64) -> Self {
        self.skip_above_score = Some(score);
        self
    }

    /// Check if review should be triggered for given conditions.
    pub fn should_review(
        &self,
        iteration: u32,
        score: f64,
        prev_score: Option<f64>,
        is_converged: bool,
        feedback: Option<&str>,
    ) -> Option<super::ReviewTrigger> {
        if !self.enabled {
            return None;
        }

        // Check skip threshold
        if let Some(threshold) = self.skip_above_score {
            if score >= threshold {
                return None;
            }
        }

        // First iteration
        if iteration == 0 && self.on_first {
            return Some(super::ReviewTrigger::FirstIteration);
        }

        // Score drop
        if self.on_score_drop {
            if let Some(prev) = prev_score {
                if score < prev {
                    return Some(super::ReviewTrigger::ScoreDrop);
                }
            }
        }

        // Convergence
        if is_converged && self.on_convergence {
            return Some(super::ReviewTrigger::Convergence);
        }

        // Regular interval
        if self.interval > 0 && iteration > 0 && iteration % self.interval == 0 {
            return Some(super::ReviewTrigger::Interval);
        }

        // Keywords in feedback
        if !self.on_keywords.is_empty() {
            if let Some(fb) = feedback {
                let fb_lower = fb.to_lowercase();
                for keyword in &self.on_keywords {
                    if fb_lower.contains(&keyword.to_lowercase()) {
                        return Some(super::ReviewTrigger::Keyword);
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_config() {
        let config = HITLConfig::disabled();
        assert!(!config.enabled);
        assert!(config.should_review(0, 0.5, None, false, None).is_none());
    }

    #[test]
    fn test_every_iteration_config() {
        let config = HITLConfig::every_iteration();
        assert!(config.enabled);
        assert_eq!(config.interval, 1);
        assert!(config.on_first);
    }

    #[test]
    fn test_should_review_first() {
        let config = HITLConfig::every_iteration();
        let trigger = config.should_review(0, 0.5, None, false, None);
        assert!(matches!(
            trigger,
            Some(super::super::ReviewTrigger::FirstIteration)
        ));
    }

    #[test]
    fn test_should_review_interval() {
        let config = HITLConfig::every(3);

        // Not on iteration 1
        assert!(config
            .should_review(1, 0.5, Some(0.4), false, None)
            .is_none());

        // On iteration 3
        let trigger = config.should_review(3, 0.5, Some(0.4), false, None);
        assert!(matches!(
            trigger,
            Some(super::super::ReviewTrigger::Interval)
        ));
    }

    #[test]
    fn test_should_review_score_drop() {
        let config = HITLConfig::on_regression();

        // No drop
        assert!(config
            .should_review(1, 0.6, Some(0.5), false, None)
            .is_none());

        // Score dropped
        let trigger = config.should_review(1, 0.4, Some(0.5), false, None);
        assert!(matches!(
            trigger,
            Some(super::super::ReviewTrigger::ScoreDrop)
        ));
    }

    #[test]
    fn test_should_review_convergence() {
        let config = HITLConfig::on_completion();

        // Not converged
        assert!(config
            .should_review(5, 0.8, Some(0.7), false, None)
            .is_none());

        // Converged
        let trigger = config.should_review(5, 0.9, Some(0.85), true, None);
        assert!(matches!(
            trigger,
            Some(super::super::ReviewTrigger::Convergence)
        ));
    }

    #[test]
    fn test_skip_above_score() {
        let config = HITLConfig::every_iteration().skip_above(0.9);

        // Below threshold
        let trigger = config.should_review(0, 0.5, None, false, None);
        assert!(trigger.is_some());

        // Above threshold
        let trigger = config.should_review(0, 0.95, None, false, None);
        assert!(trigger.is_none());
    }

    #[test]
    fn test_builder_methods() {
        let config = HITLConfig::disabled()
            .enable()
            .with_interval(5)
            .with_score_drop_review(true)
            .with_diff_style(DiffStyle::Compact)
            .with_timeout(Duration::from_secs(30));

        assert!(config.enabled);
        assert_eq!(config.interval, 5);
        assert!(config.on_score_drop);
        assert_eq!(config.diff_style, DiffStyle::Compact);
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
    }
}
