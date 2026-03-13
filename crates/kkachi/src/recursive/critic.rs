// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! LM-as-Critic system for providing structured feedback on LLM outputs.
//!
//! Provides the [`Critic`] trait for analyzing outputs and generating feedback,
//! with implementations ranging from zero-cost pass-through to LLM-powered analysis.
//!
//! # Architecture
//!
//! The critic system sits between validation and refinement: after a validator
//! produces a [`Score`], the critic can analyze *why* the output scored as it
//! did and suggest improvements. This is distinct from semantic validation
//! (which produces scores) — the critic consumes scores and produces feedback.
//!
//! # Implementations
//!
//! - [`NoCritic`]: Zero-cost pass-through that adds no feedback overhead.
//! - [`LlmCritic`]: Uses an LLM (potentially a cheaper model) to analyze outputs.
//! - [`FnCritic`]: Wraps a closure for custom synchronous critique logic.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::llm::MockLlm;
//! use kkachi::recursive::validate::Score;
//! use kkachi::recursive::critic::{Critic, NoCritic};
//!
//! # futures::executor::block_on(async {
//! let critic = NoCritic;
//! let score = Score::partial(0.6, "Missing error handling");
//! let feedback = critic.critique("Write a parser", "fn parse() {}", score).await.unwrap();
//!
//! assert_eq!(feedback.analysis.as_ref(), "No critic feedback");
//! assert!(feedback.suggestions.is_none());
//! # });
//! ```

use crate::error::Result;
use crate::recursive::llm::Llm;
use crate::recursive::validate::Score;
use std::borrow::Cow;
use std::future::Future;

// ============================================================================
// CriticFeedback
// ============================================================================

/// Structured feedback from a critic about an LLM output.
///
/// This is a zero-copy type: `analysis` and `suggestions` borrow from the
/// critic's response when possible, falling back to owned strings for
/// LLM-generated feedback.
///
/// # Lifetime
///
/// The `'a` lifetime ties the feedback to the [`Score`] and any borrowed
/// string data from the critic. Use [`CriticFeedback::into_owned`] to
/// obtain a `'static` version when needed.
#[derive(Debug, Clone)]
pub struct CriticFeedback<'a> {
    /// The validation score, potentially augmented by critic analysis.
    pub score: Score<'a>,
    /// LLM-generated explanation of the output's strengths and weaknesses.
    pub analysis: Cow<'a, str>,
    /// Optional suggestions for improvement.
    pub suggestions: Option<Cow<'a, str>>,
}

impl<'a> CriticFeedback<'a> {
    /// Create feedback with just a score and analysis.
    #[inline]
    pub fn new(score: Score<'a>, analysis: impl Into<Cow<'a, str>>) -> Self {
        Self {
            score,
            analysis: analysis.into(),
            suggestions: None,
        }
    }

    /// Add suggestions for improvement.
    #[inline]
    pub fn with_suggestions(mut self, suggestions: impl Into<Cow<'a, str>>) -> Self {
        self.suggestions = Some(suggestions.into());
        self
    }

    /// Get the analysis as a string slice.
    #[inline]
    pub fn analysis_str(&self) -> &str {
        self.analysis.as_ref()
    }

    /// Get the suggestions as a string slice, if present.
    #[inline]
    pub fn suggestions_str(&self) -> Option<&str> {
        self.suggestions.as_deref()
    }

    /// Check if the critic provided any suggestions.
    #[inline]
    pub fn has_suggestions(&self) -> bool {
        self.suggestions.is_some()
    }

    /// Convert to an owned version with `'static` lifetime.
    pub fn into_owned(self) -> CriticFeedback<'static> {
        CriticFeedback {
            score: self.score.into_owned(),
            analysis: Cow::Owned(self.analysis.into_owned()),
            suggestions: self.suggestions.map(|s| Cow::Owned(s.into_owned())),
        }
    }
}

// ============================================================================
// Critic trait (GAT-based)
// ============================================================================

/// Trait for analyzing LLM outputs and providing structured feedback.
///
/// This trait uses Generic Associated Types (GATs) for the critique future,
/// allowing zero-cost async for synchronous critics (e.g., [`NoCritic`]
/// returns [`std::future::Ready`]) while supporting true async for
/// LLM-powered critics.
///
/// # Design
///
/// The `Critic` trait is intentionally separate from [`Validate`](crate::recursive::validate::Validate):
/// - Validators produce scores (quantitative assessment).
/// - Critics consume scores and produce feedback (qualitative analysis).
///
/// This separation allows mixing cheap validators with expensive critics,
/// or using no critic at all for maximum performance.
pub trait Critic: Send + Sync {
    /// The future type returned by [`critique`](Critic::critique).
    ///
    /// Using GATs allows each implementation to define its own future type:
    /// - [`NoCritic`] uses `std::future::Ready` (zero allocation).
    /// - [`LlmCritic`] uses a named future backed by the LLM's generate call.
    /// - [`FnCritic`] uses `std::future::Ready` (synchronous closure).
    type CritiqueFut<'a>: Future<Output = Result<CriticFeedback<'a>>> + Send + 'a
    where
        Self: 'a;

    /// Analyze an LLM output and produce structured feedback.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The original prompt that produced the output.
    /// * `output` - The LLM-generated output to critique.
    /// * `score` - The validation score from the validator.
    ///
    /// # Returns
    ///
    /// A [`CriticFeedback`] containing analysis and optional suggestions.
    fn critique<'a>(
        &'a self,
        prompt: &'a str,
        output: &'a str,
        score: Score<'a>,
    ) -> Self::CritiqueFut<'a>;

    /// Get the name of this critic for logging and debugging.
    fn name(&self) -> &'static str;
}

// ============================================================================
// NoCritic (zero-cost pass-through)
// ============================================================================

/// A zero-cost critic that passes through the validator score without analysis.
///
/// This is the default critic for refinement loops that do not need
/// LLM-powered feedback. It returns [`std::future::Ready`], which the
/// compiler can inline completely — no allocation, no async overhead.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::validate::Score;
/// use kkachi::recursive::critic::{Critic, NoCritic};
///
/// # futures::executor::block_on(async {
/// let critic = NoCritic;
/// let score = Score::pass();
/// let feedback = critic.critique("prompt", "output", score).await.unwrap();
///
/// assert!(feedback.score.is_perfect());
/// assert_eq!(feedback.analysis.as_ref(), "No critic feedback");
/// # });
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCritic;

impl Critic for NoCritic {
    type CritiqueFut<'a> = std::future::Ready<Result<CriticFeedback<'a>>>;

    #[inline]
    fn critique<'a>(
        &'a self,
        _prompt: &'a str,
        _output: &'a str,
        score: Score<'a>,
    ) -> Self::CritiqueFut<'a> {
        std::future::ready(Ok(CriticFeedback {
            score,
            analysis: Cow::Borrowed("No critic feedback"),
            suggestions: None,
        }))
    }

    #[inline]
    fn name(&self) -> &'static str {
        "no_critic"
    }
}

// ============================================================================
// LlmCritic (LLM-powered analysis)
// ============================================================================

/// An LLM-powered critic that analyzes outputs and generates feedback.
///
/// This critic sends a structured prompt to an LLM (which can be a cheaper
/// model than the one generating outputs) asking it to analyze the output's
/// strengths, weaknesses, and suggest improvements.
///
/// # Lifetime
///
/// The `'c` lifetime ties the critic to the LLM reference it borrows.
/// This avoids `Arc` overhead while ensuring the LLM outlives the critic.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::llm::MockLlm;
/// use kkachi::recursive::validate::Score;
/// use kkachi::recursive::critic::{Critic, LlmCritic};
///
/// let llm = MockLlm::new(|_, _| {
///     "ANALYSIS: The output is reasonable.\nSUGGESTIONS: Add error handling.".to_string()
/// });
///
/// let critic = LlmCritic::new(&llm);
///
/// # futures::executor::block_on(async {
/// let score = Score::partial(0.6, "Missing error handling");
/// let feedback = critic.critique("Write a parser", "fn parse() {}", score).await.unwrap();
///
/// assert!(feedback.analysis_str().contains("reasonable"));
/// assert!(feedback.suggestions_str().unwrap().contains("error handling"));
/// # });
/// ```
pub struct LlmCritic<'c, L: Llm> {
    llm: &'c L,
}

impl<'c, L: Llm> LlmCritic<'c, L> {
    /// Create a new LLM-powered critic.
    ///
    /// # Arguments
    ///
    /// * `llm` - Reference to the LLM to use for critique. This can be a
    ///   cheaper/faster model than the one generating outputs.
    #[inline]
    pub fn new(llm: &'c L) -> Self {
        Self { llm }
    }
}

/// Build the critique prompt sent to the LLM.
///
/// This is a free function (not an impl method) because it does not
/// depend on any type parameters — it purely formats strings.
fn build_critique_prompt(prompt: &str, output: &str, score: &Score<'_>) -> String {
    let feedback_line = match score.feedback_str() {
        Some(fb) => format!("Validator feedback: {}\n", fb),
        None => String::new(),
    };

    format!(
        "Analyze this output for the given task. What's good? What needs improvement? \
         Score: {score_val:.1}/1.0\n\n\
         TASK:\n{prompt}\n\n\
         OUTPUT:\n{output}\n\n\
         {feedback}\
         Respond in this format:\n\
         ANALYSIS: <what's good and what's wrong>\n\
         SUGGESTIONS: <specific improvements to make>",
        score_val = score.value,
        prompt = prompt,
        output = output,
        feedback = feedback_line,
    )
}

/// Parse the LLM's critique response into analysis and suggestions.
///
/// Looks for structured `ANALYSIS:` and `SUGGESTIONS:` markers in the
/// response. If no markers are found, the entire response is used as
/// the analysis with no suggestions.
fn parse_critique_response(response: &str) -> (String, Option<String>) {
    // Look for structured ANALYSIS/SUGGESTIONS markers
    if let Some(analysis_start) = response.find("ANALYSIS:") {
        let after_marker = &response[analysis_start + "ANALYSIS:".len()..];
        let analysis_end = after_marker
            .find("SUGGESTIONS:")
            .unwrap_or(after_marker.len());
        let analysis = after_marker[..analysis_end].trim().to_string();

        let suggestions = response.find("SUGGESTIONS:").and_then(|sugg_start| {
            let sugg_text = &response[sugg_start + "SUGGESTIONS:".len()..];
            let trimmed = sugg_text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });

        (analysis, suggestions)
    } else {
        // No structured markers — use the entire response as analysis
        (response.trim().to_string(), None)
    }
}

impl<'c, L: Llm> Critic for LlmCritic<'c, L> {
    type CritiqueFut<'a>
        = LlmCritiqueFut<'a, L>
    where
        Self: 'a;

    fn critique<'a>(
        &'a self,
        prompt: &'a str,
        output: &'a str,
        score: Score<'a>,
    ) -> Self::CritiqueFut<'a> {
        // Build the critique prompt eagerly so the future owns it from the start.
        let critique_prompt = build_critique_prompt(prompt, output, &score);

        LlmCritiqueFut {
            llm: self.llm,
            score: Some(score),
            critique_prompt,
            state: LlmCritiqueFutState::Init,
        }
    }

    #[inline]
    fn name(&self) -> &'static str {
        "llm_critic"
    }
}

/// Future state for [`LlmCritic::critique`].
///
/// This is a hand-rolled future to avoid boxing. It drives the LLM's
/// `generate` call and then parses the response into [`CriticFeedback`].
enum LlmCritiqueFutState<'a, L: Llm + 'a> {
    /// Initial state — prompt is ready, generate has not been called.
    Init,
    /// Awaiting the LLM's generate future.
    Generating(L::GenerateFut<'a>),
    /// Terminal state — future has been consumed.
    Done,
}

/// Named future for [`LlmCritic`] critique operations.
///
/// This avoids boxing by carrying the LLM's generate future inline.
/// The critique prompt is stored as an owned `String` and the inner
/// generate future borrows from it via `Pin` guarantees.
pub struct LlmCritiqueFut<'a, L: Llm + 'a> {
    llm: &'a L,
    score: Option<Score<'a>>,
    /// The critique prompt, built eagerly in [`LlmCritic::critique`].
    /// This must be stored here (not in the state enum) so that the
    /// generate future can borrow from it without self-referential issues.
    critique_prompt: String,
    state: LlmCritiqueFutState<'a, L>,
}

impl<'a, L: Llm + 'a> Future for LlmCritiqueFut<'a, L> {
    type Output = Result<CriticFeedback<'a>>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We only access fields through pinned projection. The inner
        // future (`L::GenerateFut`) is pinned in place within the enum variant
        // and never moved after creation. The `critique_prompt` field is stored
        // in the struct (not the enum) and is never moved after pinning.
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match this.state {
                LlmCritiqueFutState::Init => {
                    // SAFETY: `this.critique_prompt` is owned by the pinned struct and
                    // will not be moved or dropped while the generate future is alive.
                    // The generate future borrows `&'a str` where `'a` is the struct's
                    // lifetime. Since the struct is pinned and owns the String, the
                    // reference is valid for the duration of the future.
                    let prompt_ref: &str = &this.critique_prompt;
                    let prompt_ref: &'a str = unsafe { &*(prompt_ref as *const str) };

                    let fut = this.llm.generate(prompt_ref, "", None);
                    this.state = LlmCritiqueFutState::Generating(fut);
                    // Fall through to poll the new future
                }
                LlmCritiqueFutState::Generating(ref mut fut) => {
                    // SAFETY: The future is structurally pinned within the enum
                    // variant and is never moved after being placed here.
                    let pinned_fut = unsafe { std::pin::Pin::new_unchecked(fut) };

                    match pinned_fut.poll(cx) {
                        std::task::Poll::Ready(result) => {
                            let score =
                                this.score.take().expect("score consumed before Generating");

                            this.state = LlmCritiqueFutState::Done;

                            return std::task::Poll::Ready(match result {
                                Ok(lm_output) => {
                                    let (analysis, suggestions) =
                                        parse_critique_response(&lm_output.text);

                                    Ok(CriticFeedback {
                                        score,
                                        analysis: Cow::Owned(analysis),
                                        suggestions: suggestions.map(Cow::Owned),
                                    })
                                }
                                Err(e) => {
                                    // On LLM failure, return a degraded feedback with the
                                    // original score intact — do not lose the validation data.
                                    Ok(CriticFeedback {
                                        score,
                                        analysis: Cow::Owned(format!("Critic LLM error: {}", e)),
                                        suggestions: None,
                                    })
                                }
                            });
                        }
                        std::task::Poll::Pending => return std::task::Poll::Pending,
                    }
                }
                LlmCritiqueFutState::Done => {
                    panic!("LlmCritiqueFut polled after completion");
                }
            }
        }
    }
}

// SAFETY: The future is Send because all its fields are Send:
// - `&'a L` is Send because L: Llm requires Send + Sync
// - `Score<'a>` contains Cow<'a, str> and f64 which are Send
// - `String` is Send
// - `L::GenerateFut<'a>` is Send per the Llm trait bound
unsafe impl<'a, L: Llm + 'a> Send for LlmCritiqueFut<'a, L> {}

// ============================================================================
// FnCritic (closure wrapper)
// ============================================================================

/// A synchronous critic backed by a closure.
///
/// This wraps a `Fn(&str, &str, f64) -> CriticFeedback<'static>` closure,
/// returning [`std::future::Ready`] for zero async overhead.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::validate::Score;
/// use kkachi::recursive::critic::{Critic, CriticFeedback, FnCritic};
/// use std::borrow::Cow;
///
/// let critic = FnCritic::new(|_prompt, output, score_val| {
///     let analysis = if score_val >= 0.8 {
///         "Output is good"
///     } else {
///         "Output needs improvement"
///     };
///     CriticFeedback::new(
///         Score::new(score_val),
///         Cow::Borrowed(analysis),
///     )
/// });
///
/// # futures::executor::block_on(async {
/// let feedback = critic.critique("task", "result", Score::new(0.5)).await.unwrap();
/// assert_eq!(feedback.analysis_str(), "Output needs improvement");
/// # });
/// ```
pub struct FnCritic<F>
where
    F: Fn(&str, &str, f64) -> CriticFeedback<'static> + Send + Sync,
{
    func: F,
}

impl<F> FnCritic<F>
where
    F: Fn(&str, &str, f64) -> CriticFeedback<'static> + Send + Sync,
{
    /// Create a new closure-backed critic.
    ///
    /// The closure receives:
    /// - `prompt`: the original task prompt
    /// - `output`: the LLM-generated output
    /// - `score`: the validation score as `f64` (0.0 to 1.0)
    #[inline]
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> Critic for FnCritic<F>
where
    F: Fn(&str, &str, f64) -> CriticFeedback<'static> + Send + Sync,
{
    type CritiqueFut<'a>
        = std::future::Ready<Result<CriticFeedback<'a>>>
    where
        Self: 'a;

    #[inline]
    fn critique<'a>(
        &'a self,
        prompt: &'a str,
        output: &'a str,
        score: Score<'a>,
    ) -> Self::CritiqueFut<'a> {
        let score_val = score.value;
        let feedback = (self.func)(prompt, output, score_val);
        std::future::ready(Ok(feedback))
    }

    #[inline]
    fn name(&self) -> &'static str {
        "fn_critic"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::MockLlm;
    use crate::recursive::validate::Score;

    // ========================================================================
    // CriticFeedback tests
    // ========================================================================

    #[test]
    fn test_critic_feedback_new() {
        let score = Score::pass();
        let feedback = CriticFeedback::new(score, "Looks good");

        assert!(feedback.score.is_perfect());
        assert_eq!(feedback.analysis_str(), "Looks good");
        assert!(!feedback.has_suggestions());
        assert!(feedback.suggestions_str().is_none());
    }

    #[test]
    fn test_critic_feedback_with_suggestions() {
        let score = Score::partial(0.6, "Missing tests");
        let feedback =
            CriticFeedback::new(score, "Output lacks test coverage").with_suggestions("Add tests");

        assert!((feedback.score.value - 0.6).abs() < f64::EPSILON);
        assert_eq!(feedback.analysis_str(), "Output lacks test coverage");
        assert!(feedback.has_suggestions());
        assert_eq!(feedback.suggestions_str(), Some("Add tests"));
    }

    #[test]
    fn test_critic_feedback_into_owned() {
        let score = Score::partial(0.7, "Partial");
        let feedback =
            CriticFeedback::new(score, "Analysis text").with_suggestions("Suggestion text");

        let owned: CriticFeedback<'static> = feedback.into_owned();

        assert!((owned.score.value - 0.7).abs() < f64::EPSILON);
        assert_eq!(owned.analysis_str(), "Analysis text");
        assert_eq!(owned.suggestions_str(), Some("Suggestion text"));
    }

    #[test]
    fn test_critic_feedback_borrowed_cow() {
        let score = Score::pass();
        let feedback = CriticFeedback {
            score,
            analysis: Cow::Borrowed("borrowed analysis"),
            suggestions: Some(Cow::Borrowed("borrowed suggestion")),
        };

        assert_eq!(feedback.analysis_str(), "borrowed analysis");
        assert_eq!(feedback.suggestions_str(), Some("borrowed suggestion"));

        // into_owned converts borrowed to owned
        let owned = feedback.into_owned();
        assert_eq!(owned.analysis_str(), "borrowed analysis");
    }

    // ========================================================================
    // NoCritic tests
    // ========================================================================

    #[tokio::test]
    async fn test_no_critic_pass_through() {
        let critic = NoCritic;
        let score = Score::pass();

        let feedback = critic
            .critique("Write code", "fn main() {}", score)
            .await
            .unwrap();

        assert!(feedback.score.is_perfect());
        assert_eq!(feedback.analysis_str(), "No critic feedback");
        assert!(!feedback.has_suggestions());
    }

    #[tokio::test]
    async fn test_no_critic_preserves_score() {
        let critic = NoCritic;
        let score = Score::partial(0.42, "Custom feedback");

        let feedback = critic.critique("task", "output", score).await.unwrap();

        assert!((feedback.score.value - 0.42).abs() < f64::EPSILON);
        assert_eq!(feedback.score.feedback_str(), Some("Custom feedback"));
    }

    #[test]
    fn test_no_critic_name() {
        let critic = NoCritic;
        assert_eq!(critic.name(), "no_critic");
    }

    #[test]
    fn test_no_critic_is_default() {
        let critic = NoCritic::default();
        assert_eq!(critic.name(), "no_critic");
    }

    #[test]
    fn test_no_critic_is_copy() {
        let critic = NoCritic;
        let copy = critic;
        assert_eq!(copy.name(), "no_critic");
        // Original is still usable (Copy)
        assert_eq!(critic.name(), "no_critic");
    }

    // ========================================================================
    // LlmCritic tests
    // ========================================================================

    #[tokio::test]
    async fn test_llm_critic_basic() {
        let llm = MockLlm::new(|_, _| {
            "ANALYSIS: The output is well-structured.\n\
             SUGGESTIONS: Consider adding documentation."
                .to_string()
        });

        let critic = LlmCritic::new(&llm);
        let score = Score::partial(0.7, "Missing docs");

        let feedback = critic
            .critique(
                "Write a Rust function",
                "fn add(a: i32, b: i32) -> i32 { a + b }",
                score,
            )
            .await
            .unwrap();

        assert!((feedback.score.value - 0.7).abs() < f64::EPSILON);
        assert!(feedback.analysis_str().contains("well-structured"));
        assert!(feedback.has_suggestions());
        assert!(feedback
            .suggestions_str()
            .unwrap()
            .contains("documentation"));
    }

    #[tokio::test]
    async fn test_llm_critic_no_suggestions_marker() {
        let llm =
            MockLlm::new(|_, _| "ANALYSIS: The output is perfect. No issues found.".to_string());

        let critic = LlmCritic::new(&llm);
        let score = Score::pass();

        let feedback = critic
            .critique("task", "great output", score)
            .await
            .unwrap();

        assert!(feedback.analysis_str().contains("perfect"));
        // No SUGGESTIONS marker means no suggestions
        assert!(!feedback.has_suggestions());
    }

    #[tokio::test]
    async fn test_llm_critic_unstructured_response() {
        let llm = MockLlm::new(|_, _| {
            "The output looks reasonable but could use some error handling.".to_string()
        });

        let critic = LlmCritic::new(&llm);
        let score = Score::new(0.6);

        let feedback = critic.critique("task", "output", score).await.unwrap();

        // Without ANALYSIS marker, full response becomes the analysis
        assert!(feedback.analysis_str().contains("reasonable"));
        assert!(!feedback.has_suggestions());
    }

    #[tokio::test]
    async fn test_llm_critic_llm_failure_graceful() {
        use crate::recursive::llm::FailingLlm;

        let llm = FailingLlm::new("LLM is down");
        let critic = LlmCritic::new(&llm);
        let score = Score::partial(0.5, "Some feedback");

        // Should not return Err — degrades gracefully
        let feedback = critic.critique("task", "output", score).await.unwrap();

        // Score is preserved even when LLM fails
        assert!((feedback.score.value - 0.5).abs() < f64::EPSILON);
        assert!(feedback.analysis_str().contains("Critic LLM error"));
        assert!(!feedback.has_suggestions());
    }

    #[tokio::test]
    async fn test_llm_critic_name() {
        let llm = MockLlm::new(|_, _| String::new());
        let critic = LlmCritic::new(&llm);
        assert_eq!(critic.name(), "llm_critic");
    }

    #[test]
    fn test_llm_critic_build_critique_prompt_with_feedback() {
        let score = Score::partial(0.6, "Missing error handling");
        let prompt = build_critique_prompt("Write a parser", "fn parse() {}", &score);

        assert!(prompt.contains("Score: 0.6/1.0"));
        assert!(prompt.contains("Write a parser"));
        assert!(prompt.contains("fn parse() {}"));
        assert!(prompt.contains("Missing error handling"));
        assert!(prompt.contains("ANALYSIS:"));
        assert!(prompt.contains("SUGGESTIONS:"));
    }

    #[test]
    fn test_llm_critic_build_critique_prompt_no_feedback() {
        let score = Score::new(0.8);
        let prompt = build_critique_prompt("task", "output", &score);

        assert!(prompt.contains("Score: 0.8/1.0"));
        assert!(prompt.contains("task"));
        assert!(prompt.contains("output"));
        // No "Validator feedback:" line when score has no feedback
        assert!(!prompt.contains("Validator feedback:"));
    }

    #[test]
    fn test_llm_critic_parse_response_structured() {
        let response =
            "ANALYSIS: Good code quality.\nSUGGESTIONS: Add more tests and documentation.";
        let (analysis, suggestions) = parse_critique_response(response);

        assert_eq!(analysis, "Good code quality.");
        assert_eq!(
            suggestions,
            Some("Add more tests and documentation.".to_string())
        );
    }

    #[test]
    fn test_llm_critic_parse_response_analysis_only() {
        let response = "ANALYSIS: Everything looks great, no issues.";
        let (analysis, suggestions) = parse_critique_response(response);

        assert_eq!(analysis, "Everything looks great, no issues.");
        assert!(suggestions.is_none());
    }

    #[test]
    fn test_llm_critic_parse_response_unstructured() {
        let response = "This is just a plain response without markers.";
        let (analysis, suggestions) = parse_critique_response(response);

        assert_eq!(analysis, "This is just a plain response without markers.");
        assert!(suggestions.is_none());
    }

    #[test]
    fn test_llm_critic_parse_response_empty_suggestions() {
        let response = "ANALYSIS: Good output.\nSUGGESTIONS: ";
        let (analysis, suggestions) = parse_critique_response(response);

        assert_eq!(analysis, "Good output.");
        // Empty suggestions are treated as None
        assert!(suggestions.is_none());
    }

    #[test]
    fn test_llm_critic_parse_response_multiline() {
        let response = "ANALYSIS: The code has several issues:\n\
                         1. No error handling\n\
                         2. Missing documentation\n\
                         SUGGESTIONS: Add Result return type.\n\
                         Use /// doc comments.";
        let (analysis, suggestions) = parse_critique_response(response);

        assert!(analysis.contains("No error handling"));
        assert!(analysis.contains("Missing documentation"));
        assert!(suggestions.as_ref().unwrap().contains("Result return type"));
        assert!(suggestions.as_ref().unwrap().contains("doc comments"));
    }

    // ========================================================================
    // FnCritic tests
    // ========================================================================

    #[tokio::test]
    async fn test_fn_critic_basic() {
        let critic = FnCritic::new(|_prompt, _output, score_val| {
            let analysis = if score_val >= 0.8 {
                "Output is good"
            } else {
                "Output needs improvement"
            };
            CriticFeedback::new(Score::new(score_val), Cow::Borrowed(analysis))
        });

        let feedback = critic
            .critique("task", "output", Score::new(0.5))
            .await
            .unwrap();

        assert_eq!(feedback.analysis_str(), "Output needs improvement");
        assert!((feedback.score.value - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_fn_critic_with_suggestions() {
        let critic = FnCritic::new(|_prompt, output, _score_val| {
            let has_fn = output.contains("fn ");
            if has_fn {
                CriticFeedback::new(Score::pass(), Cow::Borrowed("Contains a function"))
            } else {
                CriticFeedback::new(
                    Score::fail("No function"),
                    Cow::Borrowed("Missing function"),
                )
                .with_suggestions(Cow::Borrowed("Add a fn declaration"))
            }
        });

        let feedback = critic
            .critique("Write code", "let x = 1;", Score::new(0.0))
            .await
            .unwrap();

        assert_eq!(feedback.analysis_str(), "Missing function");
        assert_eq!(feedback.suggestions_str(), Some("Add a fn declaration"));

        let feedback = critic
            .critique("Write code", "fn add() {}", Score::new(1.0))
            .await
            .unwrap();

        assert_eq!(feedback.analysis_str(), "Contains a function");
        assert!(!feedback.has_suggestions());
    }

    #[tokio::test]
    async fn test_fn_critic_receives_score_value() {
        let critic = FnCritic::new(|_prompt, _output, score_val| {
            CriticFeedback::new(
                Score::new(score_val),
                Cow::Owned(format!("Score was {:.1}", score_val)),
            )
        });

        let feedback = critic
            .critique("task", "output", Score::new(0.75))
            .await
            .unwrap();

        assert_eq!(feedback.analysis_str(), "Score was 0.8"); // 0.75 rounds to 0.8 at 1 decimal
                                                              // Actually let's just check it contains "Score was"
        assert!(feedback.analysis_str().starts_with("Score was"));
    }

    #[test]
    fn test_fn_critic_name() {
        let critic = FnCritic::new(|_, _, _| CriticFeedback::new(Score::pass(), "ok"));
        assert_eq!(critic.name(), "fn_critic");
    }

    // ========================================================================
    // Integration / cross-type tests
    // ========================================================================

    #[tokio::test]
    async fn test_critic_with_fail_score() {
        let critic = NoCritic;
        let score = Score::fail("Completely wrong");

        let feedback = critic
            .critique("Write Rust", "print('hello')", score)
            .await
            .unwrap();

        assert!((feedback.score.value - 0.0).abs() < f64::EPSILON);
        assert_eq!(feedback.score.feedback_str(), Some("Completely wrong"));
    }

    #[tokio::test]
    async fn test_critic_with_breakdown_score() {
        use smallvec::smallvec;

        let critic = NoCritic;
        let score = Score::partial(0.75, "Partial pass")
            .with_breakdown(smallvec![("syntax", 1.0), ("style", 0.5)]);

        let feedback = critic.critique("task", "output", score).await.unwrap();

        assert!((feedback.score.value - 0.75).abs() < f64::EPSILON);
        let breakdown = feedback.score.breakdown.as_ref().unwrap();
        assert_eq!(breakdown.len(), 2);
        assert_eq!(breakdown[0].0, "syntax");
        assert!((breakdown[0].1 - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_llm_critic_prompt_includes_score() {
        let llm = MockLlm::new(|prompt, _| {
            // Verify the critique prompt contains the score
            assert!(prompt.contains("Score: 0.3/1.0"));
            assert!(prompt.contains("Fix the bugs"));
            "ANALYSIS: Needs work.\nSUGGESTIONS: Fix bugs.".to_string()
        });

        let critic = LlmCritic::new(&llm);
        let score = Score::partial(0.3, "Has bugs");

        let feedback = critic
            .critique("Fix the bugs", "buggy code", score)
            .await
            .unwrap();

        assert!(feedback.analysis_str().contains("Needs work"));
    }

    /// Verify that critics can be used generically through trait bounds.
    async fn use_critic_generic<'a, C: Critic>(
        critic: &'a C,
        output: &'a str,
    ) -> CriticFeedback<'a> {
        let score = Score::new(0.5);
        critic.critique("test prompt", output, score).await.unwrap()
    }

    #[tokio::test]
    async fn test_critic_generic_usage_no_critic() {
        let critic = NoCritic;
        let feedback = use_critic_generic(&critic, "some output").await;
        assert_eq!(feedback.analysis_str(), "No critic feedback");
    }

    #[tokio::test]
    async fn test_critic_generic_usage_fn_critic() {
        let critic = FnCritic::new(|_, _, _| {
            CriticFeedback::new(Score::new(0.5), Cow::Borrowed("From closure"))
        });
        let feedback = use_critic_generic(&critic, "some output").await;
        assert_eq!(feedback.analysis_str(), "From closure");
    }

    #[tokio::test]
    async fn test_critic_generic_usage_llm_critic() {
        let llm = MockLlm::new(|_, _| "ANALYSIS: From LLM.".to_string());
        let critic = LlmCritic::new(&llm);
        let feedback = use_critic_generic(&critic, "some output").await;
        assert!(feedback.analysis_str().contains("From LLM"));
    }
}
