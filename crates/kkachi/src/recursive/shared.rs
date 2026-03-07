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
/// When the `native` feature is enabled, creates a Tokio multi-thread
/// runtime (1 worker) with the time driver so that `tokio::time::sleep`
/// (used by retry and rate-limit wrappers) works correctly.
///
/// When already inside a Tokio runtime (e.g. nested `block_on` calls from
/// Python's `ApiLlm.__call__` inside `reason().go()`), uses
/// `block_in_place` + `Handle::block_on` to run the future on the
/// existing runtime without creating a new one.
#[cfg(feature = "native")]
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            // Inside an existing Tokio runtime — use block_in_place to
            // allow blocking on the current thread while the runtime
            // continues on its worker thread.
            tokio::task::block_in_place(|| handle.block_on(f))
        }
        Err(_) => {
            // No runtime (Python/CLI path) — create one with time driver.
            // Use multi_thread(1) so that nested block_on calls can use
            // block_in_place (which requires a multi-thread runtime).
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_time()
                .build()
                .expect("failed to create tokio runtime for block_on");
            rt.block_on(f)
        }
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_block_on_inside_tokio_runtime() {
        // When already inside a Tokio runtime, block_on uses
        // block_in_place + handle.block_on (requires multi-thread runtime).
        let val = block_on(std::future::ready(99));
        assert_eq!(val, 99);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_nested_block_on_with_sleep() {
        // Nested block_on with tokio::time::sleep — the exact scenario
        // when ApiLlm.__call__ (with retry) is invoked inside reason().go().
        let val = block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            "nested_slept"
        });
        assert_eq!(val, "nested_slept");
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
