// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Optimizer traits and base implementations
//!
//! Uses GATs for zero-cost async and lifetimes over Arc for zero-copy.

use crate::buffer::Buffer;
use crate::error::Result;
use crate::intern::Sym;
use crate::predict::FieldRange;
use smallvec::SmallVec;
use std::future::Future;

/// Zero-copy example set - indices into a shared buffer.
pub struct ExampleSet<'a> {
    /// Shared buffer containing all example data
    buffer: &'a Buffer,
    /// Example metadata
    examples: &'a [ExampleMeta],
}

/// Example metadata - indices into shared buffer.
/// Uses fixed-size arrays for Copy semantics.
#[derive(Clone, Copy, Debug)]
pub struct ExampleMeta {
    /// Input field ranges [(symbol, range)]
    pub input_ranges: [(Sym, FieldRange); 4],
    /// Number of valid input ranges
    pub input_count: u8,
    /// Output field ranges [(symbol, range)]
    pub output_ranges: [(Sym, FieldRange); 2],
    /// Number of valid output ranges
    pub output_count: u8,
}

impl ExampleMeta {
    /// Create empty example metadata.
    pub const fn empty() -> Self {
        Self {
            input_ranges: [(Sym::EMPTY, FieldRange::new(0, 0)); 4],
            input_count: 0,
            output_ranges: [(Sym::EMPTY, FieldRange::new(0, 0)); 2],
            output_count: 0,
        }
    }

    /// Get input fields as iterator.
    pub fn inputs(&self) -> impl Iterator<Item = (Sym, FieldRange)> + '_ {
        self.input_ranges[..self.input_count as usize]
            .iter()
            .copied()
    }

    /// Get output fields as iterator.
    pub fn outputs(&self) -> impl Iterator<Item = (Sym, FieldRange)> + '_ {
        self.output_ranges[..self.output_count as usize]
            .iter()
            .copied()
    }
}

impl<'a> ExampleSet<'a> {
    /// Create a new example set.
    pub const fn new(buffer: &'a Buffer, examples: &'a [ExampleMeta]) -> Self {
        Self { buffer, examples }
    }

    /// Create from buffer with a specified count (for testing).
    /// This creates an example set with empty metadata but correct count.
    pub fn from_buffer(buffer: Buffer, count: usize) -> ExampleSet<'static> {
        // Leak the buffer for 'static lifetime (for testing only)
        let buffer_ref: &'static Buffer = Box::leak(Box::new(buffer));
        let examples: &'static [ExampleMeta] =
            Box::leak(vec![ExampleMeta::empty(); count].into_boxed_slice());
        ExampleSet {
            buffer: buffer_ref,
            examples,
        }
    }

    /// Get the number of examples.
    #[inline]
    pub const fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get the underlying buffer.
    #[inline]
    pub const fn buffer(&self) -> &'a Buffer {
        self.buffer
    }

    /// Get example metadata by index.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&ExampleMeta> {
        self.examples.get(idx)
    }

    /// Iterate over examples.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ExampleView<'a>> + '_ {
        self.examples.iter().map(|meta| ExampleView {
            buffer: self.buffer,
            meta,
        })
    }

    /// Get field value by example index and symbol.
    pub fn get_input(&self, idx: usize, sym: Sym) -> Option<&'a str> {
        let meta = self.examples.get(idx)?;
        for (s, fr) in meta.inputs() {
            if s == sym {
                let bytes = &self.buffer.as_slice()[fr.as_range()];
                return std::str::from_utf8(bytes).ok();
            }
        }
        None
    }

    /// Get output field value by example index and symbol.
    pub fn get_output(&self, idx: usize, sym: Sym) -> Option<&'a str> {
        let meta = self.examples.get(idx)?;
        for (s, fr) in meta.outputs() {
            if s == sym {
                let bytes = &self.buffer.as_slice()[fr.as_range()];
                return std::str::from_utf8(bytes).ok();
            }
        }
        None
    }
}

/// Zero-copy view into a single example.
#[derive(Clone, Copy)]
pub struct ExampleView<'a> {
    buffer: &'a Buffer,
    meta: &'a ExampleMeta,
}

impl<'a> ExampleView<'a> {
    /// Get concatenated input text (first input field).
    pub fn input_text(&self) -> crate::str_view::StrView<'a> {
        // Return first input field's value
        if let Some((_, fr)) = self.meta.inputs().next() {
            let bytes = &self.buffer.as_slice()[fr.as_range()];
            if let Ok(s) = std::str::from_utf8(bytes) {
                return crate::str_view::StrView::new(s);
            }
        }
        crate::str_view::StrView::new("")
    }

    /// Get input field value by symbol.
    pub fn get_input(&self, sym: Sym) -> Option<&'a str> {
        for (s, fr) in self.meta.inputs() {
            if s == sym {
                let bytes = &self.buffer.as_slice()[fr.as_range()];
                return std::str::from_utf8(bytes).ok();
            }
        }
        None
    }

    /// Get output field value by symbol.
    pub fn get_output(&self, sym: Sym) -> Option<&'a str> {
        for (s, fr) in self.meta.outputs() {
            if s == sym {
                let bytes = &self.buffer.as_slice()[fr.as_range()];
                return std::str::from_utf8(bytes).ok();
            }
        }
        None
    }

    /// Iterate over input fields.
    pub fn inputs(&self) -> impl Iterator<Item = (Sym, &'a str)> + '_ {
        self.meta.inputs().filter_map(|(sym, fr)| {
            let bytes = &self.buffer.as_slice()[fr.as_range()];
            std::str::from_utf8(bytes).ok().map(|s| (sym, s))
        })
    }

    /// Iterate over output fields.
    pub fn outputs(&self) -> impl Iterator<Item = (Sym, &'a str)> + '_ {
        self.meta.outputs().filter_map(|(sym, fr)| {
            let bytes = &self.buffer.as_slice()[fr.as_range()];
            std::str::from_utf8(bytes).ok().map(|s| (sym, s))
        })
    }
}

/// Trait for optimizers that improve modules.
///
/// Uses GATs for zero-cost async execution. No Arc, no dynamic dispatch.
pub trait Optimizer: Send + Sync {
    /// Output type - the optimized module or optimization result
    type Output<'a>
    where
        Self: 'a;

    /// Future type for optimization
    type OptimizeFut<'a>: Future<Output = Result<Self::Output<'a>>> + Send + 'a
    where
        Self: 'a;

    /// Optimize using the training set.
    fn optimize<'a>(&'a self, trainset: &'a ExampleSet<'a>) -> Self::OptimizeFut<'a>;

    /// Get optimizer name.
    fn name(&self) -> &'static str;

    /// Get optimizer ID.
    fn id(&self) -> Sym {
        crate::intern::sym(self.name())
    }
}

/// Configuration for optimization.
#[derive(Debug, Clone, Copy)]
pub struct OptimizerConfig {
    /// Maximum number of optimization iterations
    pub max_iterations: u16,
    /// Number of examples per iteration
    pub batch_size: u16,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Metric threshold to achieve (0.0 - 1.0)
    pub metric_threshold: f32,
    /// Maximum number of demonstrations
    pub max_demos: u8,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            batch_size: 50,
            seed: 42,
            metric_threshold: 0.5,
            max_demos: 4,
        }
    }
}

impl OptimizerConfig {
    /// Create a new configuration.
    pub const fn new() -> Self {
        Self {
            max_iterations: 10,
            batch_size: 50,
            seed: 42,
            metric_threshold: 0.5,
            max_demos: 4,
        }
    }

    /// Set max iterations.
    pub const fn with_max_iterations(mut self, n: u16) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set batch size.
    pub const fn with_batch_size(mut self, n: u16) -> Self {
        self.batch_size = n;
        self
    }

    /// Set seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set metric threshold.
    pub const fn with_metric_threshold(mut self, threshold: f32) -> Self {
        self.metric_threshold = threshold;
        self
    }

    /// Set max demos.
    pub const fn with_max_demos(mut self, n: u8) -> Self {
        self.max_demos = n;
        self
    }
}

/// Simple random number generator (LCG).
#[derive(Clone, Copy)]
pub struct Rng(u64);

impl Rng {
    /// Create from seed.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Get next random u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }

    /// Get random float in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Get random usize in [0, max).
    #[inline]
    pub fn next_usize(&mut self, max: usize) -> usize {
        ((self.next_f64() * max as f64) as usize).min(max.saturating_sub(1))
    }

    /// Shuffle a mutable slice.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }

    /// Get random value in range [0, max).
    #[inline]
    pub fn gen_range(&mut self, min: u64, max: u64) -> u64 {
        if max <= min {
            return min;
        }
        min + (self.next_u64() % (max - min))
    }

    /// Get random float in [0, 1).
    #[inline]
    pub fn gen_float(&mut self) -> f32 {
        self.next_f64() as f32
    }
}

/// Optimization result containing demo indices.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Indices of selected demonstrations
    pub demo_indices: SmallVec<[u32; 8]>,
    /// Final score achieved
    pub score: f64,
    /// Number of iterations run
    pub iterations: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OptimizerConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_config_builder() {
        let config = OptimizerConfig::new()
            .with_max_iterations(20)
            .with_batch_size(100)
            .with_seed(123);

        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_rng() {
        let mut rng = Rng::new(42);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_example_set_empty() {
        static BUFFER: Buffer = Buffer::Static(b"");
        let set = ExampleSet::new(&BUFFER, &[]);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }
}
