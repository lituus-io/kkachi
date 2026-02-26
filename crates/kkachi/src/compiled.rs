// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Compiled program representing optimized LLM configurations.
//!
//! A `CompiledProgram` captures the result of an optimization process
//! (instruction tuning, demo selection) in a serializable format.
//!
//! # Examples
//!
//! ```
//! use kkachi::compiled::CompiledProgram;
//! use smallvec::smallvec;
//!
//! let prog = CompiledProgram::new(
//!     "Translate English to French concisely.".to_string(),
//!     smallvec![0, 3, 7],
//!     0.92,
//!     "MIPRO".to_string(),
//! );
//!
//! assert_eq!(prog.score, 0.92);
//! assert_eq!(prog.demo_indices.len(), 3);
//! ```

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A compiled program from optimization, serializable to disk.
///
/// This captures the artifacts produced by an optimizer run:
/// - The tuned instruction/system prompt
/// - Indices into a demo pool for few-shot examples
/// - The score achieved during optimization
/// - Which optimizer produced the result
/// - Arbitrary string metadata
///
/// # Serialization
///
/// `CompiledProgram` supports JSON serialization via [`save`](CompiledProgram::save) /
/// [`load`](CompiledProgram::load), and a compact format via
/// [`save_compact`](CompiledProgram::save_compact).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledProgram {
    /// The optimized instruction/prompt.
    pub instruction: String,
    /// Indices of selected demonstration examples.
    pub demo_indices: SmallVec<[u32; 8]>,
    /// Final optimization score achieved.
    pub score: f64,
    /// Name of the optimizer that produced this.
    pub optimizer: String,
    /// Optional metadata (e.g., hyperparameters, dataset info).
    #[serde(default)]
    pub metadata: std::collections::BTreeMap<String, String>,
}

impl CompiledProgram {
    /// Create a new compiled program.
    ///
    /// # Arguments
    ///
    /// * `instruction` - The optimized instruction/system prompt
    /// * `demo_indices` - Indices into the training set for selected demos
    /// * `score` - Final optimization score (typically 0.0..=1.0)
    /// * `optimizer` - Name of the optimizer that produced this result
    pub fn new(
        instruction: String,
        demo_indices: SmallVec<[u32; 8]>,
        score: f64,
        optimizer: String,
    ) -> Self {
        Self {
            instruction,
            demo_indices,
            score,
            optimizer,
            metadata: std::collections::BTreeMap::new(),
        }
    }

    /// Add a metadata entry, returning `self` for chaining.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Save to a pretty-printed JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or serialization fails.
    #[cfg(feature = "std")]
    pub fn save(&self, path: &str) -> crate::error::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or deserialization fails.
    #[cfg(feature = "std")]
    pub fn load(path: &str) -> crate::error::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let prog: Self = serde_json::from_str(&data)?;
        Ok(prog)
    }

    /// Save to a compact (non-pretty) JSON file.
    ///
    /// This produces a smaller file than [`save`](Self::save) by omitting
    /// whitespace formatting. Uses `serde_json` for broad compatibility;
    /// a binary format (e.g., bincode) could be swapped in without
    /// changing the public API.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or serialization fails.
    #[cfg(feature = "std")]
    pub fn save_compact(&self, path: &str) -> crate::error::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Deserialize from a JSON string (useful for embedding in configs).
    pub fn from_json(json: &str) -> crate::error::Result<Self> {
        let prog: Self = serde_json::from_str(json)?;
        Ok(prog)
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string_pretty(self)?;
        Ok(json)
    }

    /// Serialize to a compact JSON string (no whitespace).
    pub fn to_json_compact(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string(self)?;
        Ok(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn sample_program() -> CompiledProgram {
        CompiledProgram::new(
            "Translate the following sentence to French.".to_string(),
            smallvec![0, 2, 5],
            0.87,
            "MIPRO".to_string(),
        )
        .with_meta("dataset", "wmt14")
        .with_meta("iterations", "25")
    }

    #[test]
    fn test_new() {
        let prog = CompiledProgram::new(
            "Summarize.".to_string(),
            smallvec![1, 4],
            0.95,
            "COPRO".to_string(),
        );
        assert_eq!(prog.instruction, "Summarize.");
        assert_eq!(prog.demo_indices.as_slice(), &[1, 4]);
        assert!((prog.score - 0.95).abs() < f64::EPSILON);
        assert_eq!(prog.optimizer, "COPRO");
        assert!(prog.metadata.is_empty());
    }

    #[test]
    fn test_with_meta() {
        let prog = sample_program();
        assert_eq!(prog.metadata.get("dataset").unwrap(), "wmt14");
        assert_eq!(prog.metadata.get("iterations").unwrap(), "25");
    }

    #[test]
    fn test_json_roundtrip() {
        let prog = sample_program();
        let json = prog.to_json().unwrap();
        let loaded = CompiledProgram::from_json(&json).unwrap();

        assert_eq!(loaded.instruction, prog.instruction);
        assert_eq!(loaded.demo_indices.as_slice(), prog.demo_indices.as_slice());
        assert!((loaded.score - prog.score).abs() < f64::EPSILON);
        assert_eq!(loaded.optimizer, prog.optimizer);
        assert_eq!(loaded.metadata, prog.metadata);
    }

    #[test]
    fn test_compact_json_roundtrip() {
        let prog = sample_program();
        let compact = prog.to_json_compact().unwrap();

        // Compact should not contain newlines
        assert!(!compact.contains('\n'));

        let loaded = CompiledProgram::from_json(&compact).unwrap();
        assert_eq!(loaded.instruction, prog.instruction);
        assert_eq!(loaded.demo_indices.as_slice(), prog.demo_indices.as_slice());
    }

    #[test]
    fn test_empty_demo_indices() {
        let prog = CompiledProgram::new(
            "Zero-shot prompt.".to_string(),
            SmallVec::new(),
            0.60,
            "SIMBA".to_string(),
        );

        let json = prog.to_json().unwrap();
        let loaded = CompiledProgram::from_json(&json).unwrap();
        assert!(loaded.demo_indices.is_empty());
    }

    #[test]
    fn test_metadata_default_empty() {
        // Deserializing JSON without a "metadata" key should yield an empty map
        let json = r#"{
            "instruction": "test",
            "demo_indices": [1],
            "score": 0.5,
            "optimizer": "manual"
        }"#;
        let prog = CompiledProgram::from_json(json).unwrap();
        assert!(prog.metadata.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compiled.json");
        let path_str = path.to_str().unwrap();

        let prog = sample_program();
        prog.save(path_str).unwrap();

        let loaded = CompiledProgram::load(path_str).unwrap();
        assert_eq!(loaded.instruction, prog.instruction);
        assert_eq!(loaded.optimizer, prog.optimizer);
        assert_eq!(loaded.metadata, prog.metadata);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_save_compact_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compiled_compact.json");
        let path_str = path.to_str().unwrap();

        let prog = sample_program();
        prog.save_compact(path_str).unwrap();

        // Verify it's compact (single line)
        let raw = std::fs::read_to_string(path_str).unwrap();
        assert!(!raw.contains('\n'));

        let loaded = CompiledProgram::load(path_str).unwrap();
        assert_eq!(loaded.instruction, prog.instruction);
    }

    #[test]
    fn test_serde_clone() {
        let prog = sample_program();
        let cloned = prog.clone();
        assert_eq!(cloned.instruction, prog.instruction);
        assert_eq!(cloned.demo_indices.as_slice(), prog.demo_indices.as_slice());
    }

    #[test]
    fn test_debug_impl() {
        let prog = sample_program();
        let debug = format!("{:?}", prog);
        assert!(debug.contains("MIPRO"));
        assert!(debug.contains("Translate"));
    }
}
