// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Node combinators for building complex graphs

use super::node::Node;
use crate::error::Result;
use crate::intern::{sym, Sym};
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

/// Sequential composition: executes A, then B with A's output.
///
/// `Chain<A, B>` represents the pipeline `A >> B` where the output
/// of A becomes the input to B.
pub struct Chain<A, B> {
    first: A,
    second: B,
}

impl<A, B> Chain<A, B> {
    /// Create a new chain.
    #[inline]
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A, B> Clone for Chain<A, B>
where
    A: Clone,
    B: Clone,
{
    fn clone(&self) -> Self {
        Self {
            first: self.first.clone(),
            second: self.second.clone(),
        }
    }
}

impl<A, B> Node for Chain<A, B>
where
    A: Node,
    B: Node,
    for<'a> B::Input<'a>: From<A::Output<'a>>,
{
    type Input<'a>
        = A::Input<'a>
    where
        Self: 'a;
    type Output<'a>
        = B::Output<'a>
    where
        Self: 'a;
    type Fut<'a>
        = ChainFut<'a, A, B>
    where
        Self: 'a;

    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
        ChainFut {
            state: ChainState::First {
                first: &self.first,
                second: &self.second,
                fut: self.first.call(input),
            },
        }
    }

    fn id(&self) -> Sym {
        sym("chain")
    }
}

/// Future for Chain combinator.
pub struct ChainFut<'a, A: Node + 'a, B: Node + 'a> {
    state: ChainState<'a, A, B>,
}

enum ChainState<'a, A: Node + 'a, B: Node + 'a> {
    First {
        #[allow(dead_code)]
        first: &'a A,
        second: &'a B,
        fut: A::Fut<'a>,
    },
    Second {
        fut: B::Fut<'a>,
    },
    Done,
}

impl<'a, A, B> Future for ChainFut<'a, A, B>
where
    A: Node,
    B: Node,
    for<'b> B::Input<'b>: From<A::Output<'b>>,
{
    type Output = Result<B::Output<'a>>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We never move out of the pinned state
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match &mut this.state {
                ChainState::First { second, fut, .. } => {
                    // SAFETY: fut is pinned as part of self
                    let fut = unsafe { Pin::new_unchecked(fut) };
                    match fut.poll(cx) {
                        std::task::Poll::Ready(Ok(output)) => {
                            let input = B::Input::from(output);
                            let second_fut = second.call(input);
                            this.state = ChainState::Second { fut: second_fut };
                        }
                        std::task::Poll::Ready(Err(e)) => {
                            this.state = ChainState::Done;
                            return std::task::Poll::Ready(Err(e));
                        }
                        std::task::Poll::Pending => return std::task::Poll::Pending,
                    }
                }
                ChainState::Second { fut } => {
                    // SAFETY: fut is pinned as part of self
                    let fut = unsafe { Pin::new_unchecked(fut) };
                    let result = fut.poll(cx);
                    if result.is_ready() {
                        this.state = ChainState::Done;
                    }
                    return result;
                }
                ChainState::Done => panic!("ChainFut polled after completion"),
            }
        }
    }
}

// SAFETY: ChainFut is Send if both inner futures are Send
unsafe impl<'a, A, B> Send for ChainFut<'a, A, B>
where
    A: Node,
    B: Node,
    A::Fut<'a>: Send,
    B::Fut<'a>: Send,
{
}

/// Parallel composition: executes A and B concurrently.
///
/// Both nodes receive the same input (which must be Clone or Copy),
/// and their outputs are combined into a tuple.
pub struct Par<A, B> {
    left: A,
    right: B,
}

impl<A, B> Par<A, B> {
    /// Create a new parallel combinator.
    #[inline]
    pub fn new(left: A, right: B) -> Self {
        Self { left, right }
    }
}

impl<A, B> Clone for Par<A, B>
where
    A: Clone,
    B: Clone,
{
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
        }
    }
}

impl<A, B> Node for Par<A, B>
where
    A: Node,
    B: Node,
    for<'a> A::Input<'a>: Clone,
    for<'a> B::Input<'a>: From<A::Input<'a>>,
    for<'a> A::Output<'a>: Send,
    for<'a> B::Output<'a>: Send,
{
    type Input<'a>
        = A::Input<'a>
    where
        Self: 'a;
    type Output<'a>
        = (A::Output<'a>, B::Output<'a>)
    where
        Self: 'a;
    type Fut<'a>
        = ParFut<'a, A, B>
    where
        Self: 'a;

    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
        let left_input = input.clone();
        let right_input = B::Input::from(input);
        ParFut {
            left: self.left.call(left_input),
            right: self.right.call(right_input),
            left_done: None,
            right_done: None,
        }
    }

    fn id(&self) -> Sym {
        sym("par")
    }
}

/// Future for Par combinator.
pub struct ParFut<'a, A: Node + 'a, B: Node + 'a> {
    left: A::Fut<'a>,
    right: B::Fut<'a>,
    left_done: Option<A::Output<'a>>,
    right_done: Option<B::Output<'a>>,
}

impl<'a, A, B> Future for ParFut<'a, A, B>
where
    A: Node,
    B: Node,
{
    type Output = Result<(A::Output<'a>, B::Output<'a>)>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: We don't move out of pinned fields
        let this = unsafe { self.get_unchecked_mut() };

        // Poll left if not done
        if this.left_done.is_none() {
            let left = unsafe { Pin::new_unchecked(&mut this.left) };
            match left.poll(cx) {
                std::task::Poll::Ready(Ok(output)) => {
                    this.left_done = Some(output);
                }
                std::task::Poll::Ready(Err(e)) => return std::task::Poll::Ready(Err(e)),
                std::task::Poll::Pending => {}
            }
        }

        // Poll right if not done
        if this.right_done.is_none() {
            let right = unsafe { Pin::new_unchecked(&mut this.right) };
            match right.poll(cx) {
                std::task::Poll::Ready(Ok(output)) => {
                    this.right_done = Some(output);
                }
                std::task::Poll::Ready(Err(e)) => return std::task::Poll::Ready(Err(e)),
                std::task::Poll::Pending => {}
            }
        }

        // Both done?
        if this.left_done.is_some() && this.right_done.is_some() {
            let left = this.left_done.take().unwrap();
            let right = this.right_done.take().unwrap();
            std::task::Poll::Ready(Ok((left, right)))
        } else {
            std::task::Poll::Pending
        }
    }
}

// SAFETY: ParFut is Send if both inner futures are Send
unsafe impl<'a, A, B> Send for ParFut<'a, A, B>
where
    A: Node,
    B: Node,
    A::Fut<'a>: Send,
    B::Fut<'a>: Send,
    A::Output<'a>: Send,
    B::Output<'a>: Send,
{
}

/// Conditional branching: if condition then T else F.
pub struct Branch<S, C, T, F> {
    source: S,
    cond: C,
    then_node: T,
    else_node: F,
}

impl<S, C, T, F> Branch<S, C, T, F> {
    /// Create a new branch combinator.
    #[inline]
    pub fn new(source: S, cond: C, then_node: T, else_node: F) -> Self {
        Self {
            source,
            cond,
            then_node,
            else_node,
        }
    }
}

impl<S, C, T, F> Clone for Branch<S, C, T, F>
where
    S: Clone,
    C: Clone,
    T: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            cond: self.cond.clone(),
            then_node: self.then_node.clone(),
            else_node: self.else_node.clone(),
        }
    }
}

impl<S, C, T, F> Node for Branch<S, C, T, F>
where
    S: Node,
    C: Fn(&S::Output<'_>) -> bool + Send + Sync,
    T: Node,
    F: Node,
    for<'a> T::Input<'a>: From<S::Output<'a>>,
    for<'a> F::Input<'a>: From<S::Output<'a>>,
    for<'a> T::Output<'a>: Into<F::Output<'a>>,
{
    type Input<'a>
        = S::Input<'a>
    where
        Self: 'a;
    type Output<'a>
        = F::Output<'a>
    where
        Self: 'a;
    type Fut<'a>
        = BranchFut<'a, S, C, T, F>
    where
        Self: 'a;

    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
        BranchFut {
            state: BranchState::Source {
                branch: self,
                fut: self.source.call(input),
            },
        }
    }

    fn id(&self) -> Sym {
        sym("branch")
    }
}

/// Future for Branch combinator.
pub struct BranchFut<'a, S: Node + 'a, C, T: Node + 'a, F: Node + 'a> {
    state: BranchState<'a, S, C, T, F>,
}

enum BranchState<'a, S: Node + 'a, C, T: Node + 'a, F: Node + 'a> {
    Source {
        branch: &'a Branch<S, C, T, F>,
        fut: S::Fut<'a>,
    },
    Then {
        fut: T::Fut<'a>,
    },
    Else {
        fut: F::Fut<'a>,
    },
    Done,
}

impl<'a, S, C, T, F> Future for BranchFut<'a, S, C, T, F>
where
    S: Node,
    C: Fn(&S::Output<'_>) -> bool + Send + Sync,
    T: Node,
    F: Node,
    for<'b> T::Input<'b>: From<S::Output<'b>>,
    for<'b> F::Input<'b>: From<S::Output<'b>>,
    for<'b> T::Output<'b>: Into<F::Output<'b>>,
{
    type Output = Result<F::Output<'a>>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match &mut this.state {
                BranchState::Source { branch, fut } => {
                    let fut = unsafe { Pin::new_unchecked(fut) };
                    match fut.poll(cx) {
                        std::task::Poll::Ready(Ok(output)) => {
                            if (branch.cond)(&output) {
                                let input = T::Input::from(output);
                                this.state = BranchState::Then {
                                    fut: branch.then_node.call(input),
                                };
                            } else {
                                let input = F::Input::from(output);
                                this.state = BranchState::Else {
                                    fut: branch.else_node.call(input),
                                };
                            }
                        }
                        std::task::Poll::Ready(Err(e)) => {
                            this.state = BranchState::Done;
                            return std::task::Poll::Ready(Err(e));
                        }
                        std::task::Poll::Pending => return std::task::Poll::Pending,
                    }
                }
                BranchState::Then { fut } => {
                    let fut = unsafe { Pin::new_unchecked(fut) };
                    match fut.poll(cx) {
                        std::task::Poll::Ready(Ok(output)) => {
                            this.state = BranchState::Done;
                            return std::task::Poll::Ready(Ok(output.into()));
                        }
                        std::task::Poll::Ready(Err(e)) => {
                            this.state = BranchState::Done;
                            return std::task::Poll::Ready(Err(e));
                        }
                        std::task::Poll::Pending => return std::task::Poll::Pending,
                    }
                }
                BranchState::Else { fut } => {
                    let fut = unsafe { Pin::new_unchecked(fut) };
                    let result = fut.poll(cx);
                    if result.is_ready() {
                        this.state = BranchState::Done;
                    }
                    return result;
                }
                BranchState::Done => panic!("BranchFut polled after completion"),
            }
        }
    }
}

// SAFETY: BranchFut is Send if inner futures are Send
unsafe impl<'a, S, C, T, F> Send for BranchFut<'a, S, C, T, F>
where
    S: Node,
    C: Fn(&S::Output<'_>) -> bool + Send + Sync,
    T: Node,
    F: Node,
    S::Fut<'a>: Send,
    T::Fut<'a>: Send,
    F::Fut<'a>: Send,
{
}

/// Fan-out: sends the same input to multiple targets.
pub struct Fanout<S, T> {
    source: S,
    targets: T,
}

impl<S, T> Fanout<S, T> {
    /// Create a new fanout combinator.
    #[inline]
    pub fn new(source: S, targets: T) -> Self {
        Self { source, targets }
    }
}

impl<S, T> Clone for Fanout<S, T>
where
    S: Clone,
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            targets: self.targets.clone(),
        }
    }
}

impl<S, T> Node for Fanout<S, T>
where
    S: Node,
    T: Node,
    for<'a> T::Input<'a>: From<S::Output<'a>>,
{
    type Input<'a>
        = S::Input<'a>
    where
        Self: 'a;
    type Output<'a>
        = T::Output<'a>
    where
        Self: 'a;
    type Fut<'a>
        = FanoutFut<'a, S, T>
    where
        Self: 'a;

    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
        FanoutFut {
            state: FanoutState::Source {
                fanout: self,
                fut: self.source.call(input),
            },
        }
    }

    fn id(&self) -> Sym {
        sym("fanout")
    }
}

/// Future for Fanout combinator.
pub struct FanoutFut<'a, S: Node + 'a, T: Node + 'a> {
    state: FanoutState<'a, S, T>,
}

enum FanoutState<'a, S: Node + 'a, T: Node + 'a> {
    Source {
        fanout: &'a Fanout<S, T>,
        fut: S::Fut<'a>,
    },
    Targets {
        fut: T::Fut<'a>,
    },
    Done,
}

impl<'a, S, T> Future for FanoutFut<'a, S, T>
where
    S: Node,
    T: Node,
    for<'b> T::Input<'b>: From<S::Output<'b>>,
{
    type Output = Result<T::Output<'a>>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match &mut this.state {
                FanoutState::Source { fanout, fut } => {
                    let fut = unsafe { Pin::new_unchecked(fut) };
                    match fut.poll(cx) {
                        std::task::Poll::Ready(Ok(output)) => {
                            let input = T::Input::from(output);
                            this.state = FanoutState::Targets {
                                fut: fanout.targets.call(input),
                            };
                        }
                        std::task::Poll::Ready(Err(e)) => {
                            this.state = FanoutState::Done;
                            return std::task::Poll::Ready(Err(e));
                        }
                        std::task::Poll::Pending => return std::task::Poll::Pending,
                    }
                }
                FanoutState::Targets { fut } => {
                    let fut = unsafe { Pin::new_unchecked(fut) };
                    let result = fut.poll(cx);
                    if result.is_ready() {
                        this.state = FanoutState::Done;
                    }
                    return result;
                }
                FanoutState::Done => panic!("FanoutFut polled after completion"),
            }
        }
    }
}

// SAFETY: FanoutFut is Send if inner futures are Send
unsafe impl<'a, S, T> Send for FanoutFut<'a, S, T>
where
    S: Node,
    T: Node,
    S::Fut<'a>: Send,
    T::Fut<'a>: Send,
{
}

/// Feedback loop: iterates until condition is met.
pub struct Feedback<N, F> {
    inner: N,
    should_continue: F,
    max_iters: usize,
}

impl<N, F> Feedback<N, F> {
    /// Create a new feedback loop.
    #[inline]
    pub fn new(inner: N, should_continue: F, max_iters: usize) -> Self {
        Self {
            inner,
            should_continue,
            max_iters,
        }
    }
}

impl<N, F> Clone for Feedback<N, F>
where
    N: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            should_continue: self.should_continue.clone(),
            max_iters: self.max_iters,
        }
    }
}

impl<N, F> Node for Feedback<N, F>
where
    N: Node,
    F: Fn(&N::Output<'_>, usize) -> bool + Send + Sync,
    for<'a> N::Input<'a>: From<N::Output<'a>> + Clone,
{
    type Input<'a>
        = N::Input<'a>
    where
        Self: 'a;
    type Output<'a>
        = N::Output<'a>
    where
        Self: 'a;
    type Fut<'a>
        = FeedbackFut<'a, N, F>
    where
        Self: 'a;

    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
        FeedbackFut {
            feedback: self,
            state: FeedbackState::Running {
                fut: self.inner.call(input),
                iter: 0,
            },
        }
    }

    fn id(&self) -> Sym {
        sym("feedback")
    }
}

/// Future for Feedback combinator.
pub struct FeedbackFut<'a, N: Node + 'a, F> {
    feedback: &'a Feedback<N, F>,
    state: FeedbackState<'a, N>,
}

enum FeedbackState<'a, N: Node + 'a> {
    Running { fut: N::Fut<'a>, iter: usize },
    Done,
}

impl<'a, N, F> Future for FeedbackFut<'a, N, F>
where
    N: Node,
    F: Fn(&N::Output<'_>, usize) -> bool + Send + Sync,
    for<'b> N::Input<'b>: From<N::Output<'b>> + Clone,
{
    type Output = Result<N::Output<'a>>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        loop {
            match &mut this.state {
                FeedbackState::Running { fut, iter } => {
                    let fut = unsafe { Pin::new_unchecked(fut) };
                    match fut.poll(cx) {
                        std::task::Poll::Ready(Ok(output)) => {
                            let current_iter = *iter;
                            if current_iter < this.feedback.max_iters
                                && (this.feedback.should_continue)(&output, current_iter)
                            {
                                let input = N::Input::from(output);
                                this.state = FeedbackState::Running {
                                    fut: this.feedback.inner.call(input),
                                    iter: current_iter + 1,
                                };
                            } else {
                                this.state = FeedbackState::Done;
                                return std::task::Poll::Ready(Ok(output));
                            }
                        }
                        std::task::Poll::Ready(Err(e)) => {
                            this.state = FeedbackState::Done;
                            return std::task::Poll::Ready(Err(e));
                        }
                        std::task::Poll::Pending => return std::task::Poll::Pending,
                    }
                }
                FeedbackState::Done => panic!("FeedbackFut polled after completion"),
            }
        }
    }
}

// SAFETY: FeedbackFut is Send if inner future is Send
unsafe impl<'a, N, F> Send for FeedbackFut<'a, N, F>
where
    N: Node,
    F: Fn(&N::Output<'_>, usize) -> bool + Send + Sync,
    N::Fut<'a>: Send,
{
}

/// Map combinator: transforms output with a function.
pub struct Map<N, F, O> {
    inner: N,
    f: F,
    _marker: PhantomData<fn() -> O>,
}

impl<N, F, O> Map<N, F, O> {
    /// Create a new map combinator.
    #[inline]
    pub fn new(inner: N, f: F) -> Self {
        Self {
            inner,
            f,
            _marker: PhantomData,
        }
    }
}

impl<N, F, O> Clone for Map<N, F, O>
where
    N: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            f: self.f.clone(),
            _marker: PhantomData,
        }
    }
}

impl<N, F, O> Node for Map<N, F, O>
where
    N: Node,
    F: Fn(N::Output<'_>) -> O + Send + Sync,
    O: Send,
{
    type Input<'a>
        = N::Input<'a>
    where
        Self: 'a;
    type Output<'a>
        = O
    where
        Self: 'a;
    type Fut<'a>
        = MapFut<'a, N, F, O>
    where
        Self: 'a;

    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
        MapFut {
            fut: self.inner.call(input),
            f: &self.f,
            _marker: PhantomData,
        }
    }

    fn id(&self) -> Sym {
        sym("map")
    }
}

/// Future for Map combinator.
pub struct MapFut<'a, N: Node + 'a, F, O> {
    fut: N::Fut<'a>,
    f: &'a F,
    _marker: PhantomData<fn() -> O>,
}

impl<'a, N, F, O> Future for MapFut<'a, N, F, O>
where
    N: Node,
    F: Fn(N::Output<'a>) -> O,
{
    type Output = Result<O>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        let fut = unsafe { Pin::new_unchecked(&mut this.fut) };
        match fut.poll(cx) {
            std::task::Poll::Ready(Ok(output)) => std::task::Poll::Ready(Ok((this.f)(output))),
            std::task::Poll::Ready(Err(e)) => std::task::Poll::Ready(Err(e)),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

// SAFETY: MapFut is Send if inner future is Send
unsafe impl<'a, N, F, O> Send for MapFut<'a, N, F, O>
where
    N: Node,
    N::Fut<'a>: Send,
    F: Send + Sync,
    O: Send,
{
}

/// Filter combinator: only passes output if predicate is true.
pub struct Filter<N, P> {
    inner: N,
    predicate: P,
}

impl<N, P> Filter<N, P> {
    /// Create a new filter combinator.
    #[inline]
    pub fn new(inner: N, predicate: P) -> Self {
        Self { inner, predicate }
    }
}

impl<N, P> Clone for Filter<N, P>
where
    N: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            predicate: self.predicate.clone(),
        }
    }
}

impl<N, P> Node for Filter<N, P>
where
    N: Node,
    P: Fn(&N::Output<'_>) -> bool + Send + Sync,
{
    type Input<'a>
        = N::Input<'a>
    where
        Self: 'a;
    type Output<'a>
        = Option<N::Output<'a>>
    where
        Self: 'a;
    type Fut<'a>
        = FilterFut<'a, N, P>
    where
        Self: 'a;

    fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
        FilterFut {
            fut: self.inner.call(input),
            predicate: &self.predicate,
        }
    }

    fn id(&self) -> Sym {
        sym("filter")
    }
}

/// Future for Filter combinator.
pub struct FilterFut<'a, N: Node + 'a, P> {
    fut: N::Fut<'a>,
    predicate: &'a P,
}

impl<'a, N, P> Future for FilterFut<'a, N, P>
where
    N: Node,
    P: Fn(&N::Output<'a>) -> bool,
{
    type Output = Result<Option<N::Output<'a>>>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        let fut = unsafe { Pin::new_unchecked(&mut this.fut) };
        match fut.poll(cx) {
            std::task::Poll::Ready(Ok(output)) => {
                if (this.predicate)(&output) {
                    std::task::Poll::Ready(Ok(Some(output)))
                } else {
                    std::task::Poll::Ready(Ok(None))
                }
            }
            std::task::Poll::Ready(Err(e)) => std::task::Poll::Ready(Err(e)),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

// SAFETY: FilterFut is Send if inner future is Send
unsafe impl<'a, N, P> Send for FilterFut<'a, N, P>
where
    N: Node,
    N::Fut<'a>: Send,
    P: Send + Sync,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Increment;

    impl Node for Increment {
        type Input<'a> = i32;
        type Output<'a> = i32;
        type Fut<'a> = std::future::Ready<Result<i32>>;

        fn call<'a>(&'a self, input: Self::Input<'a>) -> Self::Fut<'a> {
            std::future::ready(Ok(input + 1))
        }

        fn id(&self) -> Sym {
            sym("increment")
        }
    }

    #[tokio::test]
    async fn test_chain() {
        let chain = Chain::new(Increment, Increment);
        let result = chain.call(0).await.unwrap();
        assert_eq!(result, 2);
    }

    #[tokio::test]
    async fn test_par() {
        let par = Par::new(Increment, Increment);
        let result = par.call(5).await.unwrap();
        assert_eq!(result, (6, 6));
    }

    #[tokio::test]
    async fn test_feedback() {
        let feedback = Feedback::new(Increment, |output: &i32, _| *output < 5, 10);
        let result = feedback.call(0).await.unwrap();
        assert_eq!(result, 5);
    }

    #[tokio::test]
    async fn test_feedback_max_iters() {
        // With max_iters=3, iterations 0,1,2 are allowed to continue
        // Start: 0 -> 1 (iter 0), 1 -> 2 (iter 1), 2 -> 3 (iter 2), 3 -> 4 (iter 3, stops)
        let feedback = Feedback::new(Increment, |_: &i32, _| true, 3);
        let result = feedback.call(0).await.unwrap();
        assert_eq!(result, 4);
    }

    #[tokio::test]
    async fn test_map() {
        let map = Map::new(Increment, |x| x * 2);
        let result = map.call(5).await.unwrap();
        assert_eq!(result, 12); // (5 + 1) * 2
    }

    #[tokio::test]
    async fn test_filter_pass() {
        let filter = Filter::new(Increment, |x: &i32| *x > 5);
        let result = filter.call(5).await.unwrap();
        assert_eq!(result, Some(6));
    }

    #[tokio::test]
    async fn test_filter_reject() {
        let filter = Filter::new(Increment, |x: &i32| *x > 10);
        let result = filter.call(5).await.unwrap();
        assert_eq!(result, None);
    }
}
