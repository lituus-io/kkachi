// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! WASM bindings for Kkachi
//!
//! This module provides WebAssembly bindings for using Kkachi in browsers
//! and edge runtimes like Cloudflare Workers, Fastly Compute@Edge, etc.
//!
//! ## Usage (JavaScript/TypeScript)
//!
//! ```typescript
//! import init, { KkachiOptimizer } from 'kkachi-wasm';
//!
//! await init();
//!
//! const optimizer = new KkachiOptimizer();
//! const result = await optimizer.optimize(JSON.stringify(examples));
//! console.log(result);
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Configuration for the WASM optimizer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Maximum number of iterations for refinement.
    pub max_iterations: u32,
    /// Score threshold for convergence (0.0 - 1.0).
    pub score_threshold: f64,
    /// Domain namespace for context.
    pub domain: Option<String>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            score_threshold: 0.9,
            domain: None,
        }
    }
}

/// Result from optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// The optimized output.
    pub output: String,
    /// Final score achieved.
    pub score: f64,
    /// Number of iterations taken.
    pub iterations: u32,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Main WASM optimizer class.
#[wasm_bindgen]
pub struct KkachiOptimizer {
    config: OptimizerConfig,
}

#[wasm_bindgen]
impl KkachiOptimizer {
    /// Create a new optimizer with default settings.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
        }
    }

    /// Create an optimizer with custom configuration (JSON string).
    #[wasm_bindgen]
    pub fn with_config(config_json: &str) -> Result<KkachiOptimizer, JsValue> {
        let config: OptimizerConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
        Ok(Self { config })
    }

    /// Set the maximum number of iterations.
    #[wasm_bindgen]
    pub fn set_max_iterations(&mut self, max: u32) {
        self.config.max_iterations = max;
    }

    /// Set the score threshold for convergence.
    #[wasm_bindgen]
    pub fn set_score_threshold(&mut self, threshold: f64) {
        self.config.score_threshold = threshold;
    }

    /// Set the domain namespace.
    #[wasm_bindgen]
    pub fn set_domain(&mut self, domain: &str) {
        self.config.domain = Some(domain.to_string());
    }

    /// Get the current configuration as JSON.
    #[wasm_bindgen]
    pub fn get_config(&self) -> String {
        serde_json::to_string(&self.config).unwrap_or_default()
    }

    /// Run optimization on the provided examples (JSON string).
    ///
    /// Returns a JSON string with the optimization result.
    #[wasm_bindgen]
    pub async fn optimize(&self, _examples_json: &str) -> Result<JsValue, JsValue> {
        // This is a placeholder for the actual implementation.
        // Full implementation would:
        // 1. Parse examples from JSON
        // 2. Run the optimization loop
        // 3. Return the optimized result

        let result = OptimizationResult {
            output: "Optimized output".to_string(),
            score: 0.95,
            iterations: 3,
            converged: true,
        };

        let result_json = serde_json::to_string(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))?;

        Ok(JsValue::from_str(&result_json))
    }

    /// Evaluate a single prediction against expected output.
    #[wasm_bindgen]
    pub fn evaluate(&self, prediction: &str, expected: &str) -> f64 {
        // Simple exact match for now
        if prediction == expected {
            1.0
        } else {
            // Calculate simple similarity
            let pred_words: std::collections::HashSet<&str> =
                prediction.split_whitespace().collect();
            let exp_words: std::collections::HashSet<&str> = expected.split_whitespace().collect();

            if exp_words.is_empty() {
                return 0.0;
            }

            let intersection = pred_words.intersection(&exp_words).count();
            intersection as f64 / exp_words.len() as f64
        }
    }
}

impl Default for KkachiOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize the WASM module.
#[wasm_bindgen(start)]
pub fn main() {
    // Set up panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get the library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
