// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Training mode for recursive refinement with RAG database updates.
//!
//! This module provides a training runner that:
//! 1. Uses CLI critics to validate against real tools (cargo, terraform, etc.)
//! 2. Updates the context database during training iterations
//! 3. Updates the RAG vector store upon convergence
//!
//! The training mode creates a feedback loop where successful refinements
//! become future few-shot examples for RAG retrieval.

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use super::retrieve::VectorSearchResult;
use super::retrieve::{Embedder, VectorStore};
use super::runner::RunnerConfig;

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use super::critic::Critic;
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use super::runner::{NoProgress, ProgressCallback, RefinementResult};
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use super::state::RecursiveState;
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use super::storage::{ContextId, ContextStore, ContextUpdate};
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use crate::error::Result;
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use crate::str_view::StrView;
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use smallvec::SmallVec;

/// RAG example stored during training.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Unique identifier.
    pub id: String,
    /// Original question/input.
    pub question: String,
    /// Refined answer/output.
    pub answer: String,
    /// Quality score from critic.
    pub score: f64,
    /// Number of iterations to converge.
    pub iterations: usize,
    /// Domain/category.
    pub domain: String,
    /// Error corrections made during refinement.
    pub error_corrections: Vec<(String, String)>,
}

impl TrainingExample {
    /// Format as few-shot example for prompts.
    pub fn as_few_shot(&self) -> String {
        format!(
            "Q: {}\nA: {}\n[Score: {:.2}, Iterations: {}]",
            self.question, self.answer, self.score, self.iterations
        )
    }

    /// Format with error corrections for learning.
    pub fn as_learning_example(&self) -> String {
        let mut result = format!(
            "Question: {}\n\nFinal Answer:\n{}\n",
            self.question, self.answer
        );

        if !self.error_corrections.is_empty() {
            result.push_str("\nError Corrections Made:\n");
            for (i, (error, correction)) in self.error_corrections.iter().enumerate() {
                result.push_str(&format!("{}. Error: {}\n", i + 1, error));
                if !correction.is_empty() {
                    result.push_str(&format!("   Fix: {}\n", correction));
                }
            }
        }

        result
    }
}

/// Mutable vector store for training updates.
pub trait MutableVectorStore: VectorStore {
    /// Add a document to the store.
    fn add(&mut self, id: String, content: String);

    /// Add multiple documents.
    fn add_batch(&mut self, docs: Vec<(String, String)>);

    /// Remove a document by ID.
    fn remove(&mut self, id: &str) -> bool;

    /// Clear all documents.
    fn clear(&mut self);

    /// Update a document (remove + add).
    fn update(&mut self, id: String, content: String) {
        self.remove(&id);
        self.add(id, content);
    }
}

/// Configuration for training mode.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Base runner configuration.
    pub runner: RunnerConfig,
    /// Minimum score to add to RAG database.
    pub min_rag_score: f64,
    /// Whether to store intermediate iterations (not just final).
    pub store_iterations: bool,
    /// Maximum examples per domain in RAG.
    pub max_examples_per_domain: usize,
    /// Whether to update RAG on every improvement or just convergence.
    pub update_on_improvement: bool,
    /// Similarity threshold for deduplication.
    pub dedup_threshold: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            runner: RunnerConfig::default(),
            min_rag_score: 0.8,
            store_iterations: false,
            max_examples_per_domain: 1000,
            update_on_improvement: false,
            dedup_threshold: 0.95,
        }
    }
}

impl TrainingConfig {
    /// Create with domain.
    pub fn with_domain(domain: impl Into<String>) -> Self {
        Self {
            runner: RunnerConfig::with_domain(domain),
            ..Default::default()
        }
    }

    /// Set minimum RAG score.
    pub fn with_min_rag_score(mut self, score: f64) -> Self {
        self.min_rag_score = score;
        self
    }

    /// Enable storing intermediate iterations.
    pub fn with_store_iterations(mut self, store: bool) -> Self {
        self.store_iterations = store;
        self
    }

    /// Set update behavior.
    pub fn with_update_on_improvement(mut self, update: bool) -> Self {
        self.update_on_improvement = update;
        self
    }
}

/// Training statistics.
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    /// Total refinements run.
    pub total_refinements: usize,
    /// Successful refinements (met score threshold).
    pub successful_refinements: usize,
    /// Cache hits.
    pub cache_hits: usize,
    /// Total iterations across all refinements.
    pub total_iterations: usize,
    /// Examples added to RAG.
    pub rag_additions: usize,
    /// Examples updated in RAG.
    pub rag_updates: usize,
    /// Examples deduplicated (not added due to similarity).
    pub rag_deduplicated: usize,
}

/// Training runner that integrates refinement with RAG database updates.
///
/// This runner:
/// 1. Uses CLI critics to validate against real tooling
/// 2. Updates context database with each improvement
/// 3. Updates RAG vector store upon convergence for future retrieval
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub struct TrainingRunner<'a, C, E, V, P = NoProgress>
where
    C: Critic,
    E: Embedder,
    V: MutableVectorStore,
    P: ProgressCallback,
{
    /// Critic for CLI validation.
    critic: &'a C,
    /// Context storage (DuckDB).
    context_store: &'a ContextStore,
    /// RAG vector store.
    rag_store: &'a mut V,
    /// Embedder for RAG.
    embedder: &'a E,
    /// Configuration.
    config: TrainingConfig,
    /// Progress callback.
    progress: P,
    /// Training statistics.
    stats: TrainingStats,
}

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl<'a, C, E, V> TrainingRunner<'a, C, E, V, NoProgress>
where
    C: Critic,
    E: Embedder,
    V: MutableVectorStore,
{
    /// Create a new training runner.
    pub fn new(
        critic: &'a C,
        context_store: &'a ContextStore,
        rag_store: &'a mut V,
        embedder: &'a E,
        config: TrainingConfig,
    ) -> Self {
        Self {
            critic,
            context_store,
            rag_store,
            embedder,
            config,
            progress: NoProgress,
            stats: TrainingStats::default(),
        }
    }
}

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl<'a, C, E, V, P> TrainingRunner<'a, C, E, V, P>
where
    C: Critic,
    E: Embedder,
    V: MutableVectorStore,
    P: ProgressCallback,
{
    /// Add progress callback.
    pub fn with_progress<P2: ProgressCallback>(
        self,
        progress: P2,
    ) -> TrainingRunner<'a, C, E, V, P2> {
        TrainingRunner {
            critic: self.critic,
            context_store: self.context_store,
            rag_store: self.rag_store,
            embedder: self.embedder,
            config: self.config,
            progress,
            stats: self.stats,
        }
    }

    /// Get current training statistics.
    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// Run training refinement with RAG updates.
    ///
    /// This method:
    /// 1. Checks cache for existing high-quality answers
    /// 2. Runs refinement loop with CLI critic validation
    /// 3. Updates context database on improvements
    /// 4. Updates RAG database upon convergence
    pub fn train<G>(&mut self, question: &str, mut generate: G) -> Result<RefinementResult>
    where
        G: FnMut(u32, Option<&str>) -> Result<String>,
    {
        let context_id = ContextId::from_question(question, &self.config.runner.domain);
        self.stats.total_refinements += 1;

        // Check cache
        if self.config.runner.use_cache {
            if let Some(cached) = self
                .context_store
                .lookup(question, &self.config.runner.domain)
            {
                if cached.score as f64 >= self.config.runner.cache_threshold {
                    self.progress.on_cache_hit(&context_id);
                    self.stats.cache_hits += 1;
                    return Ok(RefinementResult {
                        answer: cached.answer,
                        summary: cached.summary,
                        score: cached.score as f64,
                        iterations: 0,
                        from_cache: true,
                        context_id,
                        error_corrections: vec![],
                    });
                }
            }
        }

        // Run refinement loop
        let mut iteration = 0u32;
        let mut scores: SmallVec<[f64; 8]> = SmallVec::new();
        let mut last_feedback: Option<String> = None;
        let mut error_corrections: Vec<(String, String)> = Vec::new();
        let mut best_output = String::new();
        let mut best_score = 0.0f64;
        let mut last_stored_score = 0.0f64;

        loop {
            // Generate output (LLM call with feedback from previous iteration)
            let output = generate(iteration, last_feedback.as_deref())?;

            // Evaluate with CLI critic (runs real tools)
            let (score, feedback, is_satisfactory) = {
                let temp_state = RecursiveState::with_scores(&scores);
                let eval = self.critic.evaluate(StrView::new(&output), &temp_state);
                let is_sat = eval.is_satisfactory();
                (eval.score, eval.feedback, is_sat)
            };
            scores.push(score);
            self.stats.total_iterations += 1;

            // Track best
            let improved = score > best_score;
            if improved {
                best_score = score;
                best_output = output.clone();

                // Update context database on improvement
                if self.config.update_on_improvement && score >= self.config.runner.min_store_score
                {
                    let _ = self.context_store.upsert(&ContextUpdate {
                        question,
                        domain: &self.config.runner.domain,
                        answer: &best_output,
                        summary: "",
                        score: best_score as f32,
                        iterations: iteration as u16,
                        error_corrections: &error_corrections,
                    });
                    last_stored_score = best_score;
                }
            }

            // Report progress
            self.progress
                .on_iteration(iteration, score, feedback.as_deref());

            // Check convergence
            if score >= self.config.runner.recursive.score_threshold || is_satisfactory {
                best_output = output;
                best_score = score;
                break;
            }

            // Safety check
            if iteration >= self.config.runner.recursive.max_iterations {
                break;
            }

            // Record error correction if there was feedback
            if let Some(ref fb) = feedback {
                error_corrections.push((fb.clone(), String::new()));
            }

            // Store feedback for next iteration
            last_feedback = feedback;
            iteration += 1;
        }

        // Final storage update
        if best_score >= self.config.runner.min_store_score && best_score > last_stored_score {
            let _ = self.context_store.upsert(&ContextUpdate {
                question,
                domain: &self.config.runner.domain,
                answer: &best_output,
                summary: "",
                score: best_score as f32,
                iterations: iteration as u16,
                error_corrections: &error_corrections,
            });
        }

        // Update RAG database upon convergence if score meets threshold
        if best_score >= self.config.min_rag_score {
            self.update_rag_database(
                question,
                &best_output,
                best_score,
                iteration as usize,
                &error_corrections,
            );
            self.stats.successful_refinements += 1;
        }

        let result = RefinementResult {
            answer: best_output,
            summary: String::new(),
            score: best_score,
            iterations: iteration as usize,
            from_cache: false,
            context_id,
            error_corrections,
        };

        self.progress.on_complete(&result);

        Ok(result)
    }

    /// Update the RAG database with a successful refinement.
    fn update_rag_database(
        &mut self,
        question: &str,
        answer: &str,
        score: f64,
        iterations: usize,
        error_corrections: &[(String, String)],
    ) {
        let domain = &self.config.runner.domain;
        let doc_id = format!("{}:{}", domain, ContextId::from_question(question, domain));

        // Create the training example
        let example = TrainingExample {
            id: doc_id.clone(),
            question: question.to_string(),
            answer: answer.to_string(),
            score,
            iterations,
            domain: domain.to_string(),
            error_corrections: error_corrections.to_vec(),
        };

        // Check for similar existing examples (deduplication)
        let query_embedding = self.embedder.embed(question);
        let similar = self.rag_store.search(&query_embedding, 1);

        if let Some(top) = similar.first() {
            if top.score >= self.config.dedup_threshold {
                // Very similar example exists - update if better score
                if score > 0.0 {
                    // Parse existing score from content if possible, otherwise update
                    self.rag_store.update(doc_id, example.as_learning_example());
                    self.stats.rag_updates += 1;
                } else {
                    self.stats.rag_deduplicated += 1;
                }
                return;
            }
        }

        // Add new example
        self.rag_store.add(doc_id, example.as_learning_example());
        self.stats.rag_additions += 1;
    }

    /// Retrieve similar examples from RAG for few-shot prompting.
    pub fn retrieve_few_shot(&self, question: &str, k: usize) -> Vec<VectorSearchResult> {
        self.rag_store.search_text(question, k)
    }

    /// Train on a batch of questions.
    pub fn train_batch<G, F>(
        &mut self,
        questions: &[&str],
        mut generate_factory: F,
    ) -> Vec<Result<RefinementResult>>
    where
        G: FnMut(u32, Option<&str>) -> Result<String>,
        F: FnMut(&str) -> G,
    {
        questions
            .iter()
            .map(|q| {
                let generate = generate_factory(q);
                self.train(q, generate)
            })
            .collect()
    }
}

/// Implement MutableVectorStore for InMemoryVectorStore
impl<E: Embedder> MutableVectorStore for super::retrieve::InMemoryVectorStore<E> {
    fn add(&mut self, id: String, content: String) {
        super::retrieve::InMemoryVectorStore::add(self, id, content);
    }

    fn add_batch(&mut self, docs: Vec<(String, String)>) {
        super::retrieve::InMemoryVectorStore::add_batch(self, docs);
    }

    fn remove(&mut self, id: &str) -> bool {
        super::retrieve::InMemoryVectorStore::remove(self, id)
    }

    fn clear(&mut self) {
        super::retrieve::InMemoryVectorStore::clear(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_example_as_few_shot() {
        let example = TrainingExample {
            id: "test".to_string(),
            question: "How to parse JSON?".to_string(),
            answer: "use serde_json::from_str".to_string(),
            score: 0.95,
            iterations: 2,
            domain: "rust".to_string(),
            error_corrections: vec![],
        };

        let few_shot = example.as_few_shot();
        assert!(few_shot.contains("How to parse JSON?"));
        assert!(few_shot.contains("serde_json"));
        assert!(few_shot.contains("0.95"));
    }

    #[test]
    fn test_training_example_with_corrections() {
        let example = TrainingExample {
            id: "test".to_string(),
            question: "Write a function".to_string(),
            answer: "fn foo() {}".to_string(),
            score: 1.0,
            iterations: 3,
            domain: "rust".to_string(),
            error_corrections: vec![
                (
                    "Missing semicolon".to_string(),
                    "Added semicolon".to_string(),
                ),
                ("Type error".to_string(), "Fixed type".to_string()),
            ],
        };

        let learning = example.as_learning_example();
        assert!(learning.contains("Error Corrections Made"));
        assert!(learning.contains("Missing semicolon"));
        assert!(learning.contains("Type error"));
    }

    #[test]
    fn test_training_config_defaults() {
        let config = TrainingConfig::default();
        assert_eq!(config.min_rag_score, 0.8);
        assert!(!config.store_iterations);
        assert_eq!(config.max_examples_per_domain, 1000);
    }

    #[test]
    fn test_training_stats_default() {
        let stats = TrainingStats::default();
        assert_eq!(stats.total_refinements, 0);
        assert_eq!(stats.rag_additions, 0);
    }
}
