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
/// Uses `futures::executor::block_on` which polls the future to completion
/// on the current thread. This is safe with `ApiLlm` (which uses
/// `reqwest::blocking` and returns `Ready` futures) and avoids conflicts
/// with `reqwest::blocking`'s internal tokio runtime.
pub(crate) fn block_on<F: std::future::Future>(f: F) -> F::Output {
    futures::executor::block_on(f)
}
