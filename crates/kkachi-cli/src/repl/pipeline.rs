// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Pipeline loading and execution for the REPL.
//!
//! Pipelines are defined in JSON/YAML files and loaded at runtime.
//! This allows code-defined pipelines to be executed interactively.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A loaded pipeline ready for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    /// Pipeline name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Pipeline stages in execution order.
    pub stages: Vec<PipelineStage>,
    /// Default configuration.
    #[serde(default)]
    pub defaults: PipelineDefaults,
}

/// Default configuration for the pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineDefaults {
    /// Default model.
    pub model: Option<String>,
    /// Default temperature.
    pub temperature: Option<f32>,
    /// Default max tokens.
    pub max_tokens: Option<u32>,
}

/// A single stage in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage name (for display and breakpoints).
    pub name: String,
    /// Stage type.
    #[serde(rename = "type")]
    pub stage_type: StageType,
    /// Stage-specific configuration.
    #[serde(default)]
    pub config: StageConfig,
    /// Whether to pause for HITL review after this stage.
    #[serde(default)]
    pub breakpoint: bool,
}

/// Types of pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StageType {
    /// Retrieve documents from vector store.
    Retriever,
    /// Chain of thought reasoning.
    ChainOfThought,
    /// Simple prediction.
    Predict,
    /// Code generation with execution.
    ProgramOfThought,
    /// ReAct agent with tools.
    ReAct,
    /// Refinement loop.
    Refine,
    /// Validation/assertion stage.
    Validator,
    /// Custom stage (for extensibility).
    Custom(String),
}

/// Stage-specific configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageConfig {
    /// Signature (for prediction stages).
    pub signature: Option<String>,
    /// Instruction/system prompt.
    pub instruction: Option<String>,
    /// Number of examples to retrieve (for retriever).
    pub k: Option<usize>,
    /// Similarity threshold (for retriever).
    pub threshold: Option<f32>,
    /// Max iterations (for refine).
    pub max_iterations: Option<u32>,
    /// Score threshold (for refine/validator).
    pub score_threshold: Option<f64>,
    /// Tools (for ReAct).
    pub tools: Option<Vec<String>>,
    /// Custom parameters.
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
}

/// Result of executing a pipeline stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage name.
    pub stage_name: String,
    /// Stage type.
    pub stage_type: StageType,
    /// Output from this stage.
    pub output: String,
    /// Score (if applicable).
    pub score: Option<f64>,
    /// Execution time in milliseconds.
    pub duration_ms: u64,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Result of executing a full pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Pipeline name.
    pub pipeline_name: String,
    /// Results from each stage.
    pub stage_results: Vec<StageResult>,
    /// Final output.
    pub output: String,
    /// Total execution time in milliseconds.
    pub total_duration_ms: u64,
    /// Whether execution was interrupted (e.g., by HITL).
    pub interrupted: bool,
    /// Stage where execution stopped (if interrupted).
    pub stopped_at: Option<String>,
}

/// Pipeline execution state (for resuming after HITL).
#[derive(Debug, Clone)]
pub struct PipelineExecutionState {
    /// Current stage index.
    pub current_stage: usize,
    /// Results so far.
    pub results: Vec<StageResult>,
    /// Current input for next stage.
    pub current_input: String,
    /// Original input.
    pub original_input: String,
}

impl Pipeline {
    /// Load a pipeline from a JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, PipelineError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| PipelineError::IoError(e.to_string()))?;

        Self::from_json(&content)
    }

    /// Parse pipeline from JSON string.
    pub fn from_json(json: &str) -> Result<Self, PipelineError> {
        serde_json::from_str(json).map_err(|e| PipelineError::ParseError(e.to_string()))
    }

    /// Parse pipeline from YAML string.
    #[cfg(feature = "yaml")]
    pub fn from_yaml(yaml: &str) -> Result<Self, PipelineError> {
        serde_yaml::from_str(yaml).map_err(|e| PipelineError::ParseError(e.to_string()))
    }

    /// Get stage by name.
    pub fn get_stage(&self, name: &str) -> Option<&PipelineStage> {
        self.stages.iter().find(|s| s.name == name)
    }

    /// Get stage index by name.
    pub fn stage_index(&self, name: &str) -> Option<usize> {
        self.stages.iter().position(|s| s.name == name)
    }

    /// Set breakpoint on a stage.
    pub fn set_breakpoint(&mut self, stage_name: &str, enabled: bool) -> bool {
        if let Some(stage) = self.stages.iter_mut().find(|s| s.name == stage_name) {
            stage.breakpoint = enabled;
            true
        } else {
            false
        }
    }

    /// Get all breakpoint stage names.
    pub fn breakpoints(&self) -> Vec<&str> {
        self.stages
            .iter()
            .filter(|s| s.breakpoint)
            .map(|s| s.name.as_str())
            .collect()
    }

    /// Validate pipeline configuration.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.name.is_empty() {
            return Err(PipelineError::ValidationError(
                "Pipeline name is required".into(),
            ));
        }
        if self.stages.is_empty() {
            return Err(PipelineError::ValidationError(
                "Pipeline must have at least one stage".into(),
            ));
        }

        // Check for duplicate stage names
        let mut seen = std::collections::HashSet::new();
        for stage in &self.stages {
            if !seen.insert(&stage.name) {
                return Err(PipelineError::ValidationError(format!(
                    "Duplicate stage name: {}",
                    stage.name
                )));
            }
        }

        // Validate stage configs
        for stage in &self.stages {
            stage.validate()?;
        }

        Ok(())
    }

    /// Export pipeline to JSON.
    pub fn to_json(&self) -> Result<String, PipelineError> {
        serde_json::to_string_pretty(self).map_err(|e| PipelineError::ParseError(e.to_string()))
    }
}

impl PipelineStage {
    /// Validate stage configuration.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.name.is_empty() {
            return Err(PipelineError::ValidationError(
                "Stage name is required".into(),
            ));
        }

        // Check required config based on stage type
        match &self.stage_type {
            StageType::ChainOfThought | StageType::Predict => {
                if self.config.signature.is_none() {
                    return Err(PipelineError::ValidationError(format!(
                        "Stage '{}' requires a signature",
                        self.name
                    )));
                }
            }
            StageType::Retriever => {
                // k is optional, defaults to 5
            }
            StageType::ReAct => {
                // tools are optional
            }
            _ => {}
        }

        Ok(())
    }

    /// Get display string for stage type.
    pub fn type_display(&self) -> &str {
        match &self.stage_type {
            StageType::Retriever => "Retriever",
            StageType::ChainOfThought => "ChainOfThought",
            StageType::Predict => "Predict",
            StageType::ProgramOfThought => "ProgramOfThought",
            StageType::ReAct => "ReAct",
            StageType::Refine => "Refine",
            StageType::Validator => "Validator",
            StageType::Custom(name) => name,
        }
    }
}

impl PipelineResult {
    /// Get the final score (from last stage with a score).
    pub fn final_score(&self) -> Option<f64> {
        self.stage_results.iter().rev().find_map(|r| r.score)
    }

    /// Check if pipeline completed successfully.
    pub fn is_success(&self) -> bool {
        !self.interrupted && !self.output.is_empty()
    }
}

/// Pipeline-related errors.
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// IO error (file not found, etc.).
    IoError(String),
    /// Parse error (invalid JSON/YAML).
    ParseError(String),
    /// Validation error.
    ValidationError(String),
    /// Execution error.
    ExecutionError(String),
    /// Stage not found.
    StageNotFound(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::IoError(msg) => write!(f, "IO error: {}", msg),
            PipelineError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            PipelineError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            PipelineError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            PipelineError::StageNotFound(name) => write!(f, "Stage not found: {}", name),
        }
    }
}

impl std::error::Error for PipelineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_parse_json() {
        let json = r#"{
            "name": "test_pipeline",
            "description": "A test pipeline",
            "stages": [
                {
                    "name": "retriever",
                    "type": "retriever",
                    "config": {
                        "k": 5,
                        "threshold": 0.7
                    }
                },
                {
                    "name": "generator",
                    "type": "chain_of_thought",
                    "config": {
                        "signature": "context, question -> answer"
                    }
                }
            ]
        }"#;

        let pipeline = Pipeline::from_json(json).unwrap();
        assert_eq!(pipeline.name, "test_pipeline");
        assert_eq!(pipeline.stages.len(), 2);
        assert_eq!(pipeline.stages[0].stage_type, StageType::Retriever);
        assert_eq!(pipeline.stages[1].stage_type, StageType::ChainOfThought);
    }

    #[test]
    fn test_pipeline_validation() {
        let json = r#"{
            "name": "test",
            "description": "test",
            "stages": [
                {
                    "name": "gen",
                    "type": "chain_of_thought",
                    "config": {
                        "signature": "q -> a"
                    }
                }
            ]
        }"#;

        let pipeline = Pipeline::from_json(json).unwrap();
        assert!(pipeline.validate().is_ok());
    }

    #[test]
    fn test_pipeline_validation_missing_signature() {
        let json = r#"{
            "name": "test",
            "description": "test",
            "stages": [
                {
                    "name": "gen",
                    "type": "chain_of_thought",
                    "config": {}
                }
            ]
        }"#;

        let pipeline = Pipeline::from_json(json).unwrap();
        assert!(pipeline.validate().is_err());
    }

    #[test]
    fn test_pipeline_breakpoints() {
        let json = r#"{
            "name": "test",
            "description": "test",
            "stages": [
                {"name": "a", "type": "retriever", "breakpoint": true},
                {"name": "b", "type": "predict", "config": {"signature": "q -> a"}},
                {"name": "c", "type": "validator", "breakpoint": true}
            ]
        }"#;

        let pipeline = Pipeline::from_json(json).unwrap();
        let breakpoints = pipeline.breakpoints();
        assert_eq!(breakpoints, vec!["a", "c"]);
    }

    #[test]
    fn test_set_breakpoint() {
        let json = r#"{
            "name": "test",
            "description": "test",
            "stages": [
                {"name": "a", "type": "retriever"},
                {"name": "b", "type": "predict", "config": {"signature": "q -> a"}}
            ]
        }"#;

        let mut pipeline = Pipeline::from_json(json).unwrap();
        assert!(pipeline.set_breakpoint("a", true));
        assert!(pipeline.stages[0].breakpoint);
        assert!(!pipeline.set_breakpoint("nonexistent", true));
    }

    #[test]
    fn test_stage_type_display() {
        let stage = PipelineStage {
            name: "test".into(),
            stage_type: StageType::ChainOfThought,
            config: StageConfig::default(),
            breakpoint: false,
        };
        assert_eq!(stage.type_display(), "ChainOfThought");
    }
}
