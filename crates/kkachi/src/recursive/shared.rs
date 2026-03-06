// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Shared utilities extracted from best_of, ensemble, and reason modules.
//!
//! Eliminates code duplication for prompt assembly, output transformation,
//! diversity context generation, and synchronous execution wrappers.

use crate::recursive::defaults::Defaults;
use crate::recursive::rewrite::extract_code;

/// Style hints used for diversity in candidate generation.
pub(crate) const STYLE_HINTS: [&str; 8] = [
    "concise and minimal",
    "explicit and well-documented",
    "using a different algorithm or technique",
    "optimized for readability",
    "optimized for performance",
    "using standard library idioms",
    "using a creative or unconventional approach",
    "with extensive error handling",
];

/// Assemble a prompt from optional skill text, user prompt, and optional CoT suffix.
pub(crate) fn assemble_prompt(
    skill_text: Option<&str>,
    prompt: &str,
    with_reasoning: bool,
) -> String {
    let mut p = String::new();
    if let Some(s) = skill_text {
        p.push_str(s);
        p.push('\n');
    }
    p.push_str(prompt);
    if with_reasoning {
        p.push_str("\n\nLet's think step by step.");
    }
    p
}

/// Apply extract + defaults transform to raw LLM output.
///
/// 1. If `extract_lang` is set, extract code from markdown fences.
/// 2. If `defaults` is set, apply regex substitutions.
///
/// Returns the transformed text (may be the original if no transforms apply).
pub(crate) fn transform_output(
    text: &str,
    extract_lang: Option<&str>,
    defaults: Option<&Defaults>,
) -> String {
    let extracted = match extract_lang {
        Some(lang) => extract_code(text, lang)
            .map(|s| s.to_string())
            .unwrap_or_else(|| text.to_string()),
        None => text.to_string(),
    };
    match defaults {
        Some(d) => d.apply(&extracted),
        None => extracted,
    }
}

/// Execute a future synchronously.
///
/// When the `native` feature is enabled, creates a Tokio current-thread
/// runtime with the time driver so that `tokio::time::sleep` (used by
/// retry and rate-limit wrappers) works correctly. Falls back to
/// `futures::executor::block_on` when already inside a Tokio runtime
/// (e.g. `#[tokio::test]`) to avoid the "cannot start a runtime from
/// within a runtime" panic.
#[cfg(feature = "native")]
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    if tokio::runtime::Handle::try_current().is_ok() {
        // Inside an existing Tokio runtime (e.g. #[tokio::test]).
        // Cannot nest runtimes — callers should use .run().await instead.
        futures::executor::block_on(f)
    } else {
        // No runtime (Python/CLI path) — create one with time driver.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("failed to create tokio runtime for block_on");
        rt.block_on(f)
    }
}

/// Execute a future synchronously (non-native fallback).
#[cfg(not(feature = "native"))]
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    futures::executor::block_on(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_on_ready_future() {
        let val = block_on(std::future::ready(42));
        assert_eq!(val, 42);
    }

    #[test]
    fn test_block_on_with_tokio_sleep() {
        // This is the exact failing scenario: tokio::time::sleep requires
        // a Tokio runtime with the time driver enabled.
        let val = block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            "slept"
        });
        assert_eq!(val, "slept");
    }

    #[test]
    fn test_block_on_multiple_sleeps() {
        let val = block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            3
        });
        assert_eq!(val, 3);
    }

    #[tokio::test]
    async fn test_block_on_inside_tokio_runtime() {
        // When already inside a Tokio runtime, block_on should still work
        // for ready futures (falls back to futures::executor).
        let val = block_on(std::future::ready(99));
        assert_eq!(val, 99);
    }

    #[test]
    fn test_block_on_concurrent_from_threads() {
        let handles: Vec<_> = (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    block_on(async {
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                        i * 10
                    })
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results, vec![0, 10, 20, 30]);
    }
}
