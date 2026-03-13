// Copyright 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Recall/Precision mode configuration for tuning LLM output thresholds.
//!
//! This module provides a high-level abstraction for configuring the tradeoff
//! between recall (catching all relevant results) and precision (only returning
//! confident results).
//!
//! # Example
//!
//! ```rust
//! use kkachi::RecallPrecisionMode;
//!
//! // High recall: permissive, catches more results
//! let mode = RecallPrecisionMode::high_recall(0.6);
//! assert!(mode.favors_recall());
//!
//! // High precision: strict, only confident results
//! let mode = RecallPrecisionMode::high_precision(0.95);
//! assert!(!mode.favors_recall());
//!
//! // Get the effective threshold
//! assert_eq!(RecallPrecisionMode::Balanced.threshold(), 0.8);
//! ```

use serde::{Deserialize, Serialize};

/// High-level abstraction for tuning recall/precision tradeoffs.
///
/// This enum provides semantic modes that translate to concrete threshold
/// and behavior configurations across the system.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "threshold")]
pub enum RecallPrecisionMode {
    /// High recall mode - permissive thresholds, more candidates pass.
    /// Optimizes for not missing relevant results (fewer false negatives).
    /// Typical threshold: 0.5-0.7
    HighRecall(f64),

    /// Balanced mode - default settings with 0.8 threshold.
    Balanced,

    /// High precision mode - strict thresholds, only high-quality candidates.
    /// Optimizes for accuracy of returned results (fewer false positives).
    /// Typical threshold: 0.9+
    HighPrecision(f64),

    /// Custom mode with explicit threshold.
    Custom(f64),
}

impl Default for RecallPrecisionMode {
    #[inline]
    fn default() -> Self {
        Self::Balanced
    }
}

impl RecallPrecisionMode {
    /// Default threshold for high-recall mode.
    pub const HIGH_RECALL_DEFAULT: f64 = 0.6;

    /// Default threshold for balanced mode.
    pub const BALANCED_DEFAULT: f64 = 0.8;

    /// Default threshold for high-precision mode.
    pub const HIGH_PRECISION_DEFAULT: f64 = 0.9;

    /// Create a high-recall mode with the given threshold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kkachi::RecallPrecisionMode;
    ///
    /// let mode = RecallPrecisionMode::high_recall(0.5);
    /// assert_eq!(mode.threshold(), 0.5);
    /// assert!(mode.favors_recall());
    /// ```
    #[inline]
    pub const fn high_recall(threshold: f64) -> Self {
        Self::HighRecall(threshold)
    }

    /// Create a high-precision mode with the given threshold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kkachi::RecallPrecisionMode;
    ///
    /// let mode = RecallPrecisionMode::high_precision(0.95);
    /// assert_eq!(mode.threshold(), 0.95);
    /// assert!(!mode.favors_recall());
    /// ```
    #[inline]
    pub const fn high_precision(threshold: f64) -> Self {
        Self::HighPrecision(threshold)
    }

    /// Create a custom mode with an explicit threshold.
    #[inline]
    pub const fn custom(threshold: f64) -> Self {
        Self::Custom(threshold)
    }

    /// Get the effective score threshold for this mode.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kkachi::RecallPrecisionMode;
    ///
    /// assert_eq!(RecallPrecisionMode::Balanced.threshold(), 0.8);
    /// assert_eq!(RecallPrecisionMode::high_recall(0.6).threshold(), 0.6);
    /// ```
    #[inline]
    pub fn threshold(&self) -> f64 {
        match self {
            Self::HighRecall(t) => *t,
            Self::Balanced => Self::BALANCED_DEFAULT,
            Self::HighPrecision(t) => *t,
            Self::Custom(t) => *t,
        }
    }

    /// Get the permissiveness factor (0.0 = strict, 1.0 = permissive).
    ///
    /// This is useful for scaling other parameters based on the mode.
    #[inline]
    pub fn permissiveness(&self) -> f64 {
        match self {
            Self::HighRecall(_) => 0.8,
            Self::Balanced => 0.5,
            Self::HighPrecision(_) => 0.2,
            Self::Custom(t) => 1.0 - t,
        }
    }

    /// Whether this mode favors recall over precision.
    ///
    /// Returns `true` for `HighRecall` mode, `false` otherwise.
    #[inline]
    pub fn favors_recall(&self) -> bool {
        matches!(self, Self::HighRecall(_))
    }

    /// Whether this mode favors precision over recall.
    ///
    /// Returns `true` for `HighPrecision` mode, `false` otherwise.
    #[inline]
    pub fn favors_precision(&self) -> bool {
        matches!(self, Self::HighPrecision(_))
    }

    /// Clamp a score based on this mode's threshold.
    ///
    /// Returns `true` if the score meets or exceeds the threshold.
    #[inline]
    pub fn accepts(&self, score: f64) -> bool {
        score >= self.threshold()
    }

    /// Create mode from a raw threshold value, inferring the appropriate variant.
    ///
    /// - threshold < 0.7 -> HighRecall
    /// - 0.7 <= threshold < 0.85 -> Balanced (uses actual threshold via Custom)
    /// - threshold >= 0.85 -> HighPrecision
    pub fn from_threshold(threshold: f64) -> Self {
        if threshold < 0.7 {
            Self::HighRecall(threshold)
        } else if threshold < 0.85 {
            if (threshold - Self::BALANCED_DEFAULT).abs() < 0.01 {
                Self::Balanced
            } else {
                Self::Custom(threshold)
            }
        } else {
            Self::HighPrecision(threshold)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_balanced() {
        assert_eq!(
            RecallPrecisionMode::default(),
            RecallPrecisionMode::Balanced
        );
    }

    #[test]
    fn test_threshold_values() {
        assert!((RecallPrecisionMode::Balanced.threshold() - 0.8).abs() < 0.001);
        assert!((RecallPrecisionMode::high_recall(0.5).threshold() - 0.5).abs() < 0.001);
        assert!((RecallPrecisionMode::high_precision(0.95).threshold() - 0.95).abs() < 0.001);
        assert!((RecallPrecisionMode::custom(0.75).threshold() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_favors_recall() {
        assert!(RecallPrecisionMode::high_recall(0.5).favors_recall());
        assert!(!RecallPrecisionMode::Balanced.favors_recall());
        assert!(!RecallPrecisionMode::high_precision(0.9).favors_recall());
    }

    #[test]
    fn test_favors_precision() {
        assert!(!RecallPrecisionMode::high_recall(0.5).favors_precision());
        assert!(!RecallPrecisionMode::Balanced.favors_precision());
        assert!(RecallPrecisionMode::high_precision(0.9).favors_precision());
    }

    #[test]
    fn test_accepts() {
        let mode = RecallPrecisionMode::high_recall(0.6);
        assert!(mode.accepts(0.6));
        assert!(mode.accepts(0.8));
        assert!(!mode.accepts(0.5));

        let mode = RecallPrecisionMode::high_precision(0.9);
        assert!(mode.accepts(0.9));
        assert!(mode.accepts(1.0));
        assert!(!mode.accepts(0.89));
    }

    #[test]
    fn test_from_threshold() {
        assert!(matches!(
            RecallPrecisionMode::from_threshold(0.5),
            RecallPrecisionMode::HighRecall(_)
        ));
        assert!(matches!(
            RecallPrecisionMode::from_threshold(0.8),
            RecallPrecisionMode::Balanced
        ));
        assert!(matches!(
            RecallPrecisionMode::from_threshold(0.9),
            RecallPrecisionMode::HighPrecision(_)
        ));
    }

    #[test]
    fn test_permissiveness() {
        assert!(RecallPrecisionMode::high_recall(0.5).permissiveness() > 0.5);
        assert!((RecallPrecisionMode::Balanced.permissiveness() - 0.5).abs() < 0.001);
        assert!(RecallPrecisionMode::high_precision(0.9).permissiveness() < 0.5);
    }

    #[test]
    fn test_serde_roundtrip() {
        let modes = vec![
            RecallPrecisionMode::HighRecall(0.6),
            RecallPrecisionMode::Balanced,
            RecallPrecisionMode::HighPrecision(0.95),
            RecallPrecisionMode::Custom(0.75),
        ];

        for mode in modes {
            let json = serde_json::to_string(&mode).unwrap();
            let parsed: RecallPrecisionMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, parsed);
        }
    }
}
