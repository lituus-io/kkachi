// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Universal component model: the [`Step`] trait.
//!
//! Every building block in kkachi implements `Step` — a unit that takes text
//! in and produces text out with a score. Combinators snap steps together
//! without dynamic dispatch or allocation.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::step::{StepOutput, Step, StepExt};
//! use kkachi::recursive::{MockLlm, reason, best_of};
//!
//! // Chain steps: reason → best_of
//! // Race steps: fast.race(thorough) — pick the best score
//! // Retry: step.retry(3, 0.9) — retry until score >= target
//! ```

use crate::error::Result;
use smallvec::SmallVec;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Result of executing a step.
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// The produced text.
    pub text: String,
    /// Quality score (0.0–1.0).
    pub score: f64,
    /// Total tokens consumed.
    pub tokens: u32,
    /// Extensible key-value metadata.
    pub metadata: SmallVec<[(&'static str, String); 4]>,
}

impl StepOutput {
    /// Create a new step output.
    #[inline]
    pub fn new(text: String, score: f64, tokens: u32) -> Self {
        Self {
            text,
            score,
            tokens,
            metadata: SmallVec::new(),
        }
    }

    /// Attach a metadata key-value pair.
    #[inline]
    pub fn with_meta(mut self, key: &'static str, value: String) -> Self {
        self.metadata.push((key, value));
        self
    }
}

/// The universal building block. Every component implements this.
///
/// A `Step` takes text in and produces text + score out. Steps compose
/// via the [`StepExt`] combinator trait.
///
/// # Design
///
/// Uses GATs for the future type — no boxing, no dynamic dispatch.
/// Each implementation defines its own zero-cost future.
pub trait Step: Send + Sync {
    /// The future type returned by [`run`](Step::run).
    type Fut<'a>: Future<Output = Result<StepOutput>> + Send + 'a
    where
        Self: 'a;

    /// Execute this step with the given input text.
    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a>;

    /// Get the step name for logging and debugging.
    fn name(&self) -> &'static str;
}

/// Extension trait for composing steps.
///
/// Provides combinator methods that snap steps together with zero
/// dynamic dispatch. All combinators return concrete types.
pub trait StepExt: Step + Sized {
    /// Sequential: run self, then feed output to next.
    #[inline]
    fn then<S: Step>(self, next: S) -> Chain<Self, S> {
        Chain {
            first: self,
            second: next,
        }
    }

    /// Parallel: run self and other concurrently, pick best score.
    #[inline]
    fn race<S: Step>(self, other: S) -> Race<Self, S> {
        Race { a: self, b: other }
    }

    /// Parallel: run self and other concurrently, concatenate outputs.
    #[inline]
    fn par<S: Step>(self, other: S) -> Par<Self, S> {
        Par { a: self, b: other }
    }

    /// Retry self up to N times until score >= target.
    #[inline]
    fn retry(self, n: u32, target: f64) -> Retry<Self> {
        Retry {
            inner: self,
            max_attempts: n,
            target,
        }
    }

    /// Apply a transform to the output text.
    #[inline]
    fn map<F: Fn(&str) -> String + Send + Sync>(self, f: F) -> Map<Self, F> {
        Map {
            inner: self,
            func: f,
        }
    }

    /// Gate: only run if condition returns true, otherwise pass through.
    #[inline]
    fn when<F: Fn(&str) -> bool + Send + Sync>(self, cond: F) -> When<Self, F> {
        When { inner: self, cond }
    }

    /// Fallback: if self fails (score == 0), try other.
    #[inline]
    fn fallback<S: Step>(self, other: S) -> Fallback<Self, S> {
        Fallback {
            primary: self,
            backup: other,
        }
    }
}

impl<T: Step> StepExt for T {}

// ============================================================================
// Chain combinator — sequential composition
// ============================================================================

/// Sequential composition: runs first, then feeds output to second.
pub struct Chain<A, B> {
    first: A,
    second: B,
}

impl<A: Step, B: Step> Step for Chain<A, B> {
    type Fut<'a>
        = ChainFut<'a, A, B>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        ChainFut {
            chain: self,
            input,
            state: ChainState::First(None),
        }
    }

    fn name(&self) -> &'static str {
        "chain"
    }
}

enum ChainState<'a, A: Step + 'a, B: Step + 'a> {
    First(Option<A::Fut<'a>>),
    Between(StepOutput),
    Second(StepOutput, Option<B::Fut<'a>>),
    Done,
}

/// Future for [`Chain`] — sequential composition.
pub struct ChainFut<'a, A: Step + 'a, B: Step + 'a> {
    chain: &'a Chain<A, B>,
    input: &'a str,
    state: ChainState<'a, A, B>,
}

impl<'a, A: Step, B: Step> Future for ChainFut<'a, A, B> {
    type Output = Result<StepOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match &mut this.state {
                ChainState::First(ref mut slot) => {
                    if slot.is_none() {
                        *slot = Some(this.chain.first.run(this.input));
                    }
                    let fut = slot.as_mut().unwrap();
                    let pinned = unsafe { Pin::new_unchecked(fut) };
                    match pinned.poll(cx) {
                        Poll::Ready(Ok(output)) => {
                            this.state = ChainState::Between(output);
                        }
                        Poll::Ready(Err(e)) => {
                            this.state = ChainState::Done;
                            return Poll::Ready(Err(e));
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ChainState::Between(_) => {
                    // Move output out, transition to Second
                    let output = match std::mem::replace(&mut this.state, ChainState::Done) {
                        ChainState::Between(o) => o,
                        _ => unreachable!(),
                    };
                    this.state = ChainState::Second(output, None);
                }
                ChainState::Second(ref first_output, ref mut slot) => {
                    if slot.is_none() {
                        // SAFETY: first_output.text is stored in this pinned future
                        // and will not be moved. The borrow is valid for the lifetime
                        // of the future.
                        let text_ref: &str = &first_output.text;
                        let text_ref: &'a str = unsafe { &*(text_ref as *const str) };
                        *slot = Some(this.chain.second.run(text_ref));
                    }
                    let fut = slot.as_mut().unwrap();
                    let pinned = unsafe { Pin::new_unchecked(fut) };
                    match pinned.poll(cx) {
                        Poll::Ready(Ok(mut output)) => {
                            // Accumulate tokens from first step
                            if let ChainState::Second(ref first, _) =
                                std::mem::replace(&mut this.state, ChainState::Done)
                            {
                                output.tokens += first.tokens;
                            }
                            return Poll::Ready(Ok(output));
                        }
                        Poll::Ready(Err(e)) => {
                            this.state = ChainState::Done;
                            return Poll::Ready(Err(e));
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ChainState::Done => {
                    panic!("ChainFut polled after completion");
                }
            }
        }
    }
}

unsafe impl<'a, A: Step, B: Step> Send for ChainFut<'a, A, B> {}

// ============================================================================
// Race combinator — parallel, pick best
// ============================================================================

/// Parallel composition: runs both concurrently, picks the highest score.
pub struct Race<A, B> {
    a: A,
    b: B,
}

impl<A: Step, B: Step> Step for Race<A, B> {
    type Fut<'a>
        = RaceFut<'a, A, B>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        RaceFut {
            fut_a: self.a.run(input),
            fut_b: self.b.run(input),
            result_a: None,
            result_b: None,
        }
    }

    fn name(&self) -> &'static str {
        "race"
    }
}

/// Future for [`Race`] — parallel composition, pick best score.
pub struct RaceFut<'a, A: Step + 'a, B: Step + 'a> {
    fut_a: A::Fut<'a>,
    fut_b: B::Fut<'a>,
    result_a: Option<Result<StepOutput>>,
    result_b: Option<Result<StepOutput>>,
}

impl<'a, A: Step, B: Step> Future for RaceFut<'a, A, B> {
    type Output = Result<StepOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        if this.result_a.is_none() {
            let pinned = unsafe { Pin::new_unchecked(&mut this.fut_a) };
            if let Poll::Ready(result) = pinned.poll(cx) {
                this.result_a = Some(result);
            }
        }

        if this.result_b.is_none() {
            let pinned = unsafe { Pin::new_unchecked(&mut this.fut_b) };
            if let Poll::Ready(result) = pinned.poll(cx) {
                this.result_b = Some(result);
            }
        }

        if this.result_a.is_some() && this.result_b.is_some() {
            let a = this.result_a.take().unwrap();
            let b = this.result_b.take().unwrap();

            Poll::Ready(match (a, b) {
                (Ok(out_a), Ok(out_b)) => {
                    if out_a.score >= out_b.score {
                        Ok(out_a)
                    } else {
                        Ok(out_b)
                    }
                }
                (Ok(out), Err(_)) | (Err(_), Ok(out)) => Ok(out),
                (Err(e), Err(_)) => Err(e),
            })
        } else {
            Poll::Pending
        }
    }
}

unsafe impl<'a, A: Step, B: Step> Send for RaceFut<'a, A, B> {}

// ============================================================================
// Par combinator — parallel, concatenate
// ============================================================================

/// Parallel composition: runs both concurrently, concatenates outputs.
pub struct Par<A, B> {
    a: A,
    b: B,
}

impl<A: Step, B: Step> Step for Par<A, B> {
    type Fut<'a>
        = ParFut<'a, A, B>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        ParFut {
            fut_a: self.a.run(input),
            fut_b: self.b.run(input),
            result_a: None,
            result_b: None,
        }
    }

    fn name(&self) -> &'static str {
        "par"
    }
}

/// Future for [`Par`] — parallel composition, concatenate outputs.
pub struct ParFut<'a, A: Step + 'a, B: Step + 'a> {
    fut_a: A::Fut<'a>,
    fut_b: B::Fut<'a>,
    result_a: Option<Result<StepOutput>>,
    result_b: Option<Result<StepOutput>>,
}

impl<'a, A: Step, B: Step> Future for ParFut<'a, A, B> {
    type Output = Result<StepOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        if this.result_a.is_none() {
            let pinned = unsafe { Pin::new_unchecked(&mut this.fut_a) };
            if let Poll::Ready(result) = pinned.poll(cx) {
                this.result_a = Some(result);
            }
        }

        if this.result_b.is_none() {
            let pinned = unsafe { Pin::new_unchecked(&mut this.fut_b) };
            if let Poll::Ready(result) = pinned.poll(cx) {
                this.result_b = Some(result);
            }
        }

        if this.result_a.is_some() && this.result_b.is_some() {
            let a = this.result_a.take().unwrap();
            let b = this.result_b.take().unwrap();

            Poll::Ready(match (a, b) {
                (Ok(out_a), Ok(out_b)) => {
                    let mut text = out_a.text;
                    text.push_str("\n\n");
                    text.push_str(&out_b.text);
                    let score = (out_a.score + out_b.score) / 2.0;
                    let tokens = out_a.tokens + out_b.tokens;
                    Ok(StepOutput::new(text, score, tokens))
                }
                (Ok(out), Err(_)) | (Err(_), Ok(out)) => Ok(out),
                (Err(e), Err(_)) => Err(e),
            })
        } else {
            Poll::Pending
        }
    }
}

unsafe impl<'a, A: Step, B: Step> Send for ParFut<'a, A, B> {}

// ============================================================================
// Retry combinator
// ============================================================================

/// Retry a step up to N times until score >= target.
pub struct Retry<S> {
    inner: S,
    max_attempts: u32,
    target: f64,
}

impl<S: Step> Step for Retry<S> {
    type Fut<'a>
        = RetryFut<'a, S>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        RetryFut {
            inner: &self.inner,
            input,
            max_attempts: self.max_attempts,
            target: self.target,
            attempt: 0,
            best: None,
            pending: None,
        }
    }

    fn name(&self) -> &'static str {
        "retry"
    }
}

/// Future for [`Retry`] — retry until score >= target.
pub struct RetryFut<'a, S: Step + 'a> {
    inner: &'a S,
    input: &'a str,
    max_attempts: u32,
    target: f64,
    attempt: u32,
    best: Option<StepOutput>,
    pending: Option<S::Fut<'a>>,
}

impl<'a, S: Step> Future for RetryFut<'a, S> {
    type Output = Result<StepOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            if let Some(ref mut fut) = this.pending {
                let pinned = unsafe { Pin::new_unchecked(fut) };
                match pinned.poll(cx) {
                    Poll::Ready(Ok(output)) => {
                        this.pending = None;
                        if output.score >= this.target {
                            return Poll::Ready(Ok(output));
                        }
                        this.best = Some(match this.best.take() {
                            Some(prev) if prev.score >= output.score => prev,
                            _ => output,
                        });
                    }
                    Poll::Ready(Err(e)) => {
                        this.pending = None;
                        return Poll::Ready(Err(e));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }

            if this.attempt < this.max_attempts {
                this.attempt += 1;
                this.pending = Some(this.inner.run(this.input));
            } else {
                return Poll::Ready(Ok(this
                    .best
                    .take()
                    .unwrap_or_else(|| StepOutput::new(String::new(), 0.0, 0))));
            }
        }
    }
}

unsafe impl<'a, S: Step> Send for RetryFut<'a, S> {}

// ============================================================================
// Map combinator
// ============================================================================

/// Transform the output text of a step.
pub struct Map<S, F> {
    inner: S,
    func: F,
}

impl<S: Step, F: Fn(&str) -> String + Send + Sync> Step for Map<S, F> {
    type Fut<'a>
        = MapFut<'a, S, F>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        MapFut {
            func: &self.func,
            inner_fut: self.inner.run(input),
        }
    }

    fn name(&self) -> &'static str {
        "map"
    }
}

/// Future for [`Map`] — transform output text.
pub struct MapFut<'a, S: Step + 'a, F> {
    func: &'a F,
    inner_fut: S::Fut<'a>,
}

impl<'a, S: Step, F: Fn(&str) -> String> Future for MapFut<'a, S, F> {
    type Output = Result<StepOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        let pinned = unsafe { Pin::new_unchecked(&mut this.inner_fut) };
        match pinned.poll(cx) {
            Poll::Ready(Ok(mut output)) => {
                output.text = (this.func)(&output.text);
                Poll::Ready(Ok(output))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

unsafe impl<'a, S: Step, F: Fn(&str) -> String + Send + Sync> Send for MapFut<'a, S, F> {}

// ============================================================================
// When combinator — conditional gate
// ============================================================================

/// Conditional step: only runs inner if condition holds, otherwise passes input through.
pub struct When<S, F> {
    inner: S,
    cond: F,
}

impl<S: Step, F: Fn(&str) -> bool + Send + Sync> Step for When<S, F> {
    type Fut<'a>
        = WhenFut<'a, S>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        if (self.cond)(input) {
            WhenFut::Run(self.inner.run(input))
        } else {
            WhenFut::Pass(Some(StepOutput::new(input.to_string(), 1.0, 0)))
        }
    }

    fn name(&self) -> &'static str {
        "when"
    }
}

/// Future for [`When`] — conditional gate.
#[allow(clippy::large_enum_variant)]
pub enum WhenFut<'a, S: Step + 'a> {
    /// Condition was true — run the inner step.
    Run(S::Fut<'a>),
    /// Condition was false — pass input through.
    Pass(Option<StepOutput>),
}

impl<'a, S: Step> Future for WhenFut<'a, S> {
    type Output = Result<StepOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        match this {
            WhenFut::Run(ref mut fut) => {
                let pinned = unsafe { Pin::new_unchecked(fut) };
                pinned.poll(cx)
            }
            WhenFut::Pass(ref mut output) => Poll::Ready(Ok(output.take().unwrap())),
        }
    }
}

unsafe impl<'a, S: Step> Send for WhenFut<'a, S> {}

// ============================================================================
// Fallback combinator
// ============================================================================

/// Fallback: if primary produces score == 0, try backup.
pub struct Fallback<A, B> {
    primary: A,
    backup: B,
}

impl<A: Step, B: Step> Step for Fallback<A, B> {
    type Fut<'a>
        = FallbackFut<'a, A, B>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        FallbackFut {
            fallback: self,
            input,
            state: FallbackState::Primary(None),
        }
    }

    fn name(&self) -> &'static str {
        "fallback"
    }
}

enum FallbackState<'a, A: Step + 'a, B: Step + 'a> {
    Primary(Option<A::Fut<'a>>),
    Backup(Option<B::Fut<'a>>),
    Done,
}

/// Future for [`Fallback`] — try primary, fall back on failure.
pub struct FallbackFut<'a, A: Step + 'a, B: Step + 'a> {
    fallback: &'a Fallback<A, B>,
    input: &'a str,
    state: FallbackState<'a, A, B>,
}

impl<'a, A: Step, B: Step> Future for FallbackFut<'a, A, B> {
    type Output = Result<StepOutput>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match &mut this.state {
                FallbackState::Primary(ref mut slot) => {
                    if slot.is_none() {
                        *slot = Some(this.fallback.primary.run(this.input));
                    }
                    let pinned = unsafe { Pin::new_unchecked(slot.as_mut().unwrap()) };
                    match pinned.poll(cx) {
                        Poll::Ready(Ok(output)) if output.score > 0.0 => {
                            this.state = FallbackState::Done;
                            return Poll::Ready(Ok(output));
                        }
                        Poll::Ready(Ok(_)) | Poll::Ready(Err(_)) => {
                            this.state = FallbackState::Backup(None);
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                FallbackState::Backup(ref mut slot) => {
                    if slot.is_none() {
                        *slot = Some(this.fallback.backup.run(this.input));
                    }
                    let pinned = unsafe { Pin::new_unchecked(slot.as_mut().unwrap()) };
                    match pinned.poll(cx) {
                        Poll::Ready(result) => {
                            this.state = FallbackState::Done;
                            return Poll::Ready(result);
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                FallbackState::Done => {
                    panic!("FallbackFut polled after completion");
                }
            }
        }
    }
}

unsafe impl<'a, A: Step, B: Step> Send for FallbackFut<'a, A, B> {}

// ============================================================================
// Batch execution — run_all
// ============================================================================

/// Run multiple steps concurrently on the same input, returning all results.
///
/// Uses FuturesUnordered for maximum concurrency without thread spawning.
///
/// # Examples
///
/// ```ignore
/// let results = run_all("input text", &[
///     &reason_step,
///     &best_of_step,
///     &ensemble_step,
/// ]).await;
/// ```
pub async fn run_all<'a>(input: &'a str, steps: &'a [&'a dyn DynStep]) -> Vec<Result<StepOutput>> {
    use futures::stream::{FuturesUnordered, StreamExt};

    let mut futs = FuturesUnordered::new();
    for (i, step) in steps.iter().enumerate() {
        let fut = step.run_dyn(input);
        futs.push(async move { (i, fut.await) });
    }

    let mut results: Vec<Option<Result<StepOutput>>> = (0..steps.len()).map(|_| None).collect();
    while let Some((i, result)) = futs.next().await {
        results[i] = Some(result);
    }

    results.into_iter().map(|r| r.unwrap()).collect()
}

/// Object-safe version of Step for use in run_all and collections.
///
/// This is the ONLY place we use dynamic dispatch — at the batch boundary.
/// Individual step composition remains fully monomorphic.
pub trait DynStep: Send + Sync {
    /// Execute this step (boxed future for object safety).
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<StepOutput>> + Send + 'a>>;

    /// Get the step name.
    fn dyn_name(&self) -> &'static str;
}

impl<S: Step> DynStep for S {
    fn run_dyn<'a>(
        &'a self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<StepOutput>> + Send + 'a>> {
        Box::pin(self.run(input))
    }

    fn dyn_name(&self) -> &'static str {
        self.name()
    }
}

// ============================================================================
// FnStep — closure as step
// ============================================================================

/// A step backed by a synchronous closure.
pub struct FnStep<F>
where
    F: Fn(&str) -> Result<StepOutput> + Send + Sync,
{
    func: F,
    step_name: &'static str,
}

impl<F> FnStep<F>
where
    F: Fn(&str) -> Result<StepOutput> + Send + Sync,
{
    /// Create a new closure-backed step.
    pub fn new(name: &'static str, func: F) -> Self {
        Self {
            func,
            step_name: name,
        }
    }
}

impl<F> Step for FnStep<F>
where
    F: Fn(&str) -> Result<StepOutput> + Send + Sync,
{
    type Fut<'a>
        = std::future::Ready<Result<StepOutput>>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        std::future::ready((self.func)(input))
    }

    fn name(&self) -> &'static str {
        self.step_name
    }
}

// ============================================================================
// ValidateStep — validator as step
// ============================================================================

/// Wraps a [`Validate`](crate::recursive::validate::Validate) implementation as a [`Step`].
///
/// Passes input through if validation passes; returns the input text
/// with the validation score.
pub struct ValidateStep<V> {
    validator: V,
}

impl<V> ValidateStep<V> {
    /// Create a new validate step.
    pub fn new(validator: V) -> Self {
        Self { validator }
    }
}

impl<V: crate::recursive::validate::Validate> Step for ValidateStep<V> {
    type Fut<'a>
        = std::future::Ready<Result<StepOutput>>
    where
        Self: 'a;

    fn run<'a>(&'a self, input: &'a str) -> Self::Fut<'a> {
        let score = self.validator.validate(input);
        std::future::ready(Ok(StepOutput::new(input.to_string(), score.value, 0)))
    }

    fn name(&self) -> &'static str {
        "validate"
    }
}

/// Create a validate step from a validator.
#[inline]
pub fn validate_step<V: crate::recursive::validate::Validate>(validator: V) -> ValidateStep<V> {
    ValidateStep::new(validator)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_step() -> FnStep<impl Fn(&str) -> Result<StepOutput> + Send + Sync> {
        FnStep::new("echo", |input| {
            Ok(StepOutput::new(input.to_string(), 1.0, 0))
        })
    }

    fn upper_step() -> FnStep<impl Fn(&str) -> Result<StepOutput> + Send + Sync> {
        FnStep::new("upper", |input| {
            Ok(StepOutput::new(input.to_uppercase(), 1.0, 0))
        })
    }

    fn score_step(s: f64) -> FnStep<impl Fn(&str) -> Result<StepOutput> + Send + Sync> {
        FnStep::new("scored", move |input| {
            Ok(StepOutput::new(input.to_string(), s, 0))
        })
    }

    fn fail_step() -> FnStep<impl Fn(&str) -> Result<StepOutput> + Send + Sync> {
        FnStep::new("fail", |input| {
            Ok(StepOutput::new(input.to_string(), 0.0, 0))
        })
    }

    #[tokio::test]
    async fn test_fn_step() {
        let step = echo_step();
        let output = step.run("hello").await.unwrap();
        assert_eq!(output.text, "hello");
        assert!((output.score - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_chain() {
        let step = echo_step().then(upper_step());
        let output = step.run("hello").await.unwrap();
        assert_eq!(output.text, "HELLO");
    }

    #[tokio::test]
    async fn test_race_picks_better() {
        let a = score_step(0.3);
        let b = score_step(0.9);
        let step = a.race(b);
        let output = step.run("test").await.unwrap();
        assert!((output.score - 0.9).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_par_concatenates() {
        let a = echo_step();
        let b = upper_step();
        let step = a.par(b);
        let output = step.run("hello").await.unwrap();
        assert!(output.text.contains("hello"));
        assert!(output.text.contains("HELLO"));
    }

    #[tokio::test]
    async fn test_retry_until_target() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let counter = AtomicU32::new(0);
        let step = FnStep::new("improving", move |input| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            let score = if n >= 2 { 1.0 } else { 0.3 };
            Ok(StepOutput::new(input.to_string(), score, 0))
        });

        let retried = step.retry(5, 0.9);
        let output = retried.run("test").await.unwrap();
        assert!(output.score >= 0.9);
    }

    #[tokio::test]
    async fn test_map() {
        let step = echo_step().map(|s| format!("[{}]", s));
        let output = step.run("hello").await.unwrap();
        assert_eq!(output.text, "[hello]");
    }

    #[tokio::test]
    async fn test_when_true() {
        let step = upper_step().when(|s| s.len() > 3);
        let output = step.run("hello").await.unwrap();
        assert_eq!(output.text, "HELLO");
    }

    #[tokio::test]
    async fn test_when_false_passthrough() {
        let step = upper_step().when(|s| s.len() > 10);
        let output = step.run("hi").await.unwrap();
        assert_eq!(output.text, "hi"); // passed through unchanged
    }

    #[tokio::test]
    async fn test_fallback_primary_ok() {
        let step = score_step(0.8).fallback(echo_step());
        let output = step.run("test").await.unwrap();
        assert!((output.score - 0.8).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_fallback_primary_fails() {
        let step = fail_step().fallback(score_step(0.9));
        let output = step.run("test").await.unwrap();
        assert!((output.score - 0.9).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_validate_step() {
        use crate::recursive::checks::checks;

        let v = validate_step(checks().require("fn "));
        let output = v.run("fn main() {}").await.unwrap();
        assert!(output.score >= 1.0);

        let output = v.run("let x = 1").await.unwrap();
        assert!(output.score < 1.0);
    }

    #[tokio::test]
    async fn test_complex_composition() {
        // echo → upper → validate(requires "HELLO")
        let step = echo_step().then(upper_step()).then(validate_step(
            crate::recursive::checks::checks().require("HELLO"),
        ));

        let output = step.run("hello world").await.unwrap();
        assert!(output.text.contains("HELLO"));
        assert!(output.score >= 1.0);
    }

    #[tokio::test]
    async fn test_run_all() {
        let a = echo_step();
        let b = upper_step();
        let c = score_step(0.5);

        let steps: Vec<&dyn DynStep> = vec![&a, &b, &c];
        let results = run_all("hello", &steps).await;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap().text, "hello");
        assert_eq!(results[1].as_ref().unwrap().text, "HELLO");
        assert!((results[2].as_ref().unwrap().score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_step_output_metadata() {
        let output = StepOutput::new("test".to_string(), 1.0, 42)
            .with_meta("model", "gpt-4".to_string())
            .with_meta("latency", "100ms".to_string());

        assert_eq!(output.metadata.len(), 2);
        assert_eq!(output.metadata[0].0, "model");
        assert_eq!(output.metadata[0].1, "gpt-4");
    }
}
