// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! REPL session state management.

use kkachi::recursive::{memory, Memory, Recall};
use kkachi::HITLConfig;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use super::pipeline::{Pipeline, PipelineExecutionState};

/// Type alias for the REPL memory store.
pub type ReplMemory = Arc<RwLock<Memory>>;

/// Demo data for sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoData {
    /// Input text.
    pub input: String,
    /// Output text.
    pub output: String,
}

/// Signature data for sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureData {
    /// Raw signature string.
    pub raw: String,
    /// Input field names.
    pub inputs: Vec<String>,
    /// Output field names.
    pub outputs: Vec<String>,
}

impl SignatureData {
    /// Parse a signature string (e.g., "question -> answer").
    pub fn parse(sig: &str) -> Option<Self> {
        let parts: Vec<&str> = sig.split("->").collect();
        if parts.len() != 2 {
            return None;
        }

        let inputs: Vec<String> = parts[0]
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let outputs: Vec<String> = parts[1]
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if inputs.is_empty() || outputs.is_empty() {
            return None;
        }

        Some(Self {
            raw: sig.to_string(),
            inputs,
            outputs,
        })
    }
}

/// LLM provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProviderType {
    #[default]
    OpenAI,
    Anthropic,
    Google,
    Local,
}

impl ProviderType {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(Self::OpenAI),
            "anthropic" => Some(Self::Anthropic),
            "google" | "gemini" => Some(Self::Google),
            "local" | "ollama" | "llama" => Some(Self::Local),
            _ => None,
        }
    }

    /// Get display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::OpenAI => "OpenAI",
            Self::Anthropic => "Anthropic",
            Self::Google => "Google",
            Self::Local => "Local",
        }
    }
}

/// LLM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMConfig {
    /// Model name.
    pub model: String,
    /// Temperature (0.0 - 2.0).
    pub temperature: f32,
    /// Max tokens.
    pub max_tokens: Option<u32>,
    /// Top P.
    pub top_p: Option<f32>,
    /// API key (optional, uses env if not set).
    #[serde(skip)]
    pub api_key: Option<String>,
    /// Base URL (for local/custom endpoints).
    pub base_url: Option<String>,
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_string(),
            temperature: 0.7,
            max_tokens: None,
            top_p: None,
            api_key: None,
            base_url: None,
        }
    }
}

/// Create a default memory store.
fn default_memory() -> ReplMemory {
    Arc::new(RwLock::new(memory()))
}

/// Complete REPL session state.
#[derive(Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// Current signature.
    pub signature: Option<SignatureData>,
    /// Current instruction/system prompt.
    pub instruction: String,
    /// Few-shot demonstrations.
    pub demos: Vec<DemoData>,
    /// LLM configuration.
    pub lm_config: LMConfig,
    /// Current provider.
    pub provider: ProviderType,
    /// HITL configuration.
    #[serde(skip)]
    pub hitl: HITLConfig,
    /// Domain for storage.
    pub domain: String,
    /// Iteration snapshots from last operation.
    #[serde(skip)]
    pub iterations: Vec<IterationSnapshot>,
    /// Current working question/input.
    pub current_input: Option<String>,
    /// Last output/answer.
    pub last_output: Option<String>,
    /// Last score.
    pub last_score: Option<f64>,
    /// Loaded pipeline (if any).
    #[serde(skip)]
    pub pipeline: Option<Pipeline>,
    /// Pipeline execution state (for resuming after HITL).
    #[serde(skip)]
    pub pipeline_execution: Option<PipelineExecutionState>,
    /// In-memory store for document retrieval.
    #[serde(skip, default = "default_memory")]
    pub memory_store: ReplMemory,
}

impl std::fmt::Debug for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionState")
            .field("signature", &self.signature)
            .field("instruction", &self.instruction)
            .field("demos", &self.demos)
            .field("lm_config", &self.lm_config)
            .field("provider", &self.provider)
            .field("hitl", &self.hitl)
            .field("domain", &self.domain)
            .field("iterations", &self.iterations)
            .field("current_input", &self.current_input)
            .field("last_output", &self.last_output)
            .field("last_score", &self.last_score)
            .field("pipeline", &self.pipeline)
            .field("pipeline_execution", &self.pipeline_execution)
            .field(
                "memory_store",
                &format!("<Memory: {} docs>", self.store_len()),
            )
            .finish()
    }
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            signature: None,
            instruction: String::new(),
            demos: Vec::new(),
            lm_config: LMConfig::default(),
            provider: ProviderType::default(),
            hitl: HITLConfig::disabled(),
            domain: "general".to_string(),
            iterations: Vec::new(),
            current_input: None,
            last_output: None,
            last_score: None,
            pipeline: None,
            pipeline_execution: None,
            memory_store: Arc::new(RwLock::new(memory())),
        }
    }
}

impl SessionState {
    /// Create a new session state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the signature.
    pub fn set_signature(&mut self, sig: &str) -> bool {
        if let Some(data) = SignatureData::parse(sig) {
            self.signature = Some(data);
            true
        } else {
            false
        }
    }

    /// Add a demo.
    pub fn add_demo(&mut self, input: String, output: String) {
        self.demos.push(DemoData { input, output });
    }

    /// Remove a demo by index.
    pub fn remove_demo(&mut self, index: usize) -> bool {
        if index < self.demos.len() {
            self.demos.remove(index);
            true
        } else {
            false
        }
    }

    /// Clear all demos.
    pub fn clear_demos(&mut self) {
        self.demos.clear();
    }

    /// Get demos as tuples for diff computation.
    pub fn demos_as_tuples(&self) -> Vec<(String, String)> {
        self.demos
            .iter()
            .map(|d| (d.input.clone(), d.output.clone()))
            .collect()
    }

    /// Check if the session is configured enough to run.
    pub fn is_ready(&self) -> bool {
        self.signature.is_some() && !self.instruction.is_empty()
    }

    /// Reset the session.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Create a snapshot of the current state.
    pub fn snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            instruction: self.instruction.clone(),
            demos: self.demos.clone(),
            timestamp: Instant::now(),
        }
    }

    /// Restore from a snapshot.
    pub fn restore(&mut self, snapshot: &StateSnapshot) {
        self.instruction = snapshot.instruction.clone();
        self.demos = snapshot.demos.clone();
    }

    /// Load a pipeline.
    pub fn load_pipeline(&mut self, pipeline: Pipeline) {
        self.pipeline = Some(pipeline);
        self.pipeline_execution = None;
    }

    /// Unload the current pipeline.
    pub fn unload_pipeline(&mut self) {
        self.pipeline = None;
        self.pipeline_execution = None;
    }

    /// Check if a pipeline is loaded.
    pub fn has_pipeline(&self) -> bool {
        self.pipeline.is_some()
    }

    /// Get the loaded pipeline.
    pub fn get_pipeline(&self) -> Option<&Pipeline> {
        self.pipeline.as_ref()
    }

    /// Get mutable reference to the loaded pipeline.
    pub fn get_pipeline_mut(&mut self) -> Option<&mut Pipeline> {
        self.pipeline.as_mut()
    }

    /// Start pipeline execution.
    pub fn start_pipeline_execution(&mut self, input: String) {
        self.pipeline_execution = Some(PipelineExecutionState {
            current_stage: 0,
            results: Vec::new(),
            current_input: input.clone(),
            original_input: input,
        });
    }

    /// Get current pipeline execution state.
    pub fn get_execution_state(&self) -> Option<&PipelineExecutionState> {
        self.pipeline_execution.as_ref()
    }

    /// Get mutable execution state.
    pub fn get_execution_state_mut(&mut self) -> Option<&mut PipelineExecutionState> {
        self.pipeline_execution.as_mut()
    }

    /// Clear pipeline execution state.
    pub fn clear_execution_state(&mut self) {
        self.pipeline_execution = None;
    }

    // Memory store methods

    /// Add a document to the memory store.
    pub fn store_add(&self, id: impl Into<String>, content: impl Into<String>) {
        self.memory_store.write().add_with_id(id, &content.into());
    }

    /// Remove a document from the memory store.
    pub fn store_remove(&self, id: &str) -> bool {
        self.memory_store.write().remove(id)
    }

    /// Search the memory store.
    pub fn store_search(&self, query: &str, k: usize) -> Vec<Recall> {
        self.memory_store.read().search(query, k)
    }

    /// Get a document by ID.
    pub fn store_get(&self, id: &str) -> Option<String> {
        self.memory_store.read().get(id)
    }

    /// Get the number of documents in the store.
    pub fn store_len(&self) -> usize {
        self.memory_store.read().len()
    }

    /// Check if the store is empty.
    pub fn store_is_empty(&self) -> bool {
        self.memory_store.read().is_empty()
    }

    /// Clear the memory store by replacing it with a new one.
    pub fn store_clear(&self) {
        *self.memory_store.write() = memory();
    }

    /// List all document IDs in the store.
    pub fn store_list_ids(&self) -> Vec<String> {
        // Memory doesn't expose an iterator, so return empty for now.
        Vec::new()
    }
}

/// Snapshot of state at a point in time.
#[derive(Clone)]
pub struct StateSnapshot {
    /// Instruction at this point.
    pub instruction: String,
    /// Demos at this point.
    pub demos: Vec<DemoData>,
    /// When this snapshot was taken.
    pub timestamp: Instant,
}

/// Undo/redo history for state.
#[derive(Default)]
pub struct StateHistory {
    /// Stack of snapshots.
    snapshots: Vec<StateSnapshot>,
    /// Current position in the stack.
    position: usize,
    /// Maximum history size.
    max_size: usize,
}

impl StateHistory {
    /// Create a new history with default max size.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            position: 0,
            max_size: 50,
        }
    }

    /// Create with custom max size.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            position: 0,
            max_size,
        }
    }

    /// Push a new snapshot.
    pub fn push(&mut self, snapshot: StateSnapshot) {
        // Truncate any future history
        self.snapshots.truncate(self.position);

        // Add new snapshot
        self.snapshots.push(snapshot);

        // Trim if over max size
        if self.snapshots.len() > self.max_size {
            self.snapshots.remove(0);
        } else {
            self.position = self.snapshots.len();
        }
    }

    /// Undo - get previous snapshot.
    pub fn undo(&mut self) -> Option<&StateSnapshot> {
        if self.position > 1 {
            self.position -= 1;
            self.snapshots.get(self.position - 1)
        } else {
            None
        }
    }

    /// Redo - get next snapshot.
    pub fn redo(&mut self) -> Option<&StateSnapshot> {
        if self.position < self.snapshots.len() {
            self.position += 1;
            self.snapshots.get(self.position - 1)
        } else {
            None
        }
    }

    /// Check if undo is available.
    pub fn can_undo(&self) -> bool {
        self.position > 1
    }

    /// Check if redo is available.
    pub fn can_redo(&self) -> bool {
        self.position < self.snapshots.len()
    }

    /// Get the current position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get the total number of snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.position = 0;
    }
}

/// Snapshot of a single iteration during refinement/optimization.
#[derive(Debug, Clone)]
pub struct IterationSnapshot {
    /// Iteration number.
    pub iteration: u32,
    /// Instruction at this iteration.
    pub instruction: String,
    /// Demos at this iteration.
    pub demos: Vec<DemoData>,
    /// Output/answer at this iteration.
    pub output: String,
    /// Score at this iteration.
    pub score: f64,
    /// Feedback (if any).
    pub feedback: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_parse() {
        let sig = SignatureData::parse("question -> answer").unwrap();
        assert_eq!(sig.inputs, vec!["question"]);
        assert_eq!(sig.outputs, vec!["answer"]);

        let sig2 = SignatureData::parse("context, question -> answer, reasoning").unwrap();
        assert_eq!(sig2.inputs, vec!["context", "question"]);
        assert_eq!(sig2.outputs, vec!["answer", "reasoning"]);

        assert!(SignatureData::parse("invalid").is_none());
        assert!(SignatureData::parse("->").is_none());
    }

    #[test]
    fn test_provider_type() {
        assert_eq!(ProviderType::from_str("openai"), Some(ProviderType::OpenAI));
        assert_eq!(
            ProviderType::from_str("ANTHROPIC"),
            Some(ProviderType::Anthropic)
        );
        assert_eq!(ProviderType::from_str("google"), Some(ProviderType::Google));
        assert!(ProviderType::from_str("unknown").is_none());
    }

    #[test]
    fn test_session_state_demos() {
        let mut state = SessionState::new();

        state.add_demo("Q1".to_string(), "A1".to_string());
        assert_eq!(state.demos.len(), 1);

        state.add_demo("Q2".to_string(), "A2".to_string());
        assert_eq!(state.demos.len(), 2);

        assert!(state.remove_demo(0));
        assert_eq!(state.demos.len(), 1);
        assert_eq!(state.demos[0].input, "Q2");

        assert!(!state.remove_demo(10)); // Out of bounds
    }

    #[test]
    fn test_session_state_ready() {
        let mut state = SessionState::new();
        assert!(!state.is_ready());

        state.set_signature("q -> a");
        assert!(!state.is_ready()); // Still missing instruction

        state.instruction = "You are helpful.".to_string();
        assert!(state.is_ready());
    }

    #[test]
    fn test_state_history() {
        let mut history = StateHistory::new();
        assert!(history.is_empty());
        assert!(!history.can_undo());
        assert!(!history.can_redo());

        // Add snapshots
        history.push(StateSnapshot {
            instruction: "v1".to_string(),
            demos: vec![],
            timestamp: Instant::now(),
        });

        history.push(StateSnapshot {
            instruction: "v2".to_string(),
            demos: vec![],
            timestamp: Instant::now(),
        });

        assert_eq!(history.len(), 2);
        assert!(history.can_undo());
        assert!(!history.can_redo());

        // Undo
        let prev = history.undo().unwrap();
        assert_eq!(prev.instruction, "v1");
        assert!(history.can_redo());

        // Redo
        let next = history.redo().unwrap();
        assert_eq!(next.instruction, "v2");
    }
}
