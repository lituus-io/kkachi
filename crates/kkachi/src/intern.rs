// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! String interning for efficient field name handling
//!
//! Provides [`Sym`] - a 4-byte interned string symbol that enables:
//! - O(1) equality comparison (just compare u32)
//! - Minimal memory usage (4 bytes vs 24 for String)
//! - Fast hashing (hash the u32 directly)
//!
//! Common DSPy field names are pre-interned at compile time for
//! instant lookup without any runtime overhead.

use dashmap::DashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

/// Interned string symbol - only 4 bytes.
///
/// A `Sym` is an index into a global string table. Two `Sym` values
/// with the same index represent the same string, enabling O(1)
/// equality comparison.
///
/// # Memory Layout
///
/// - `Sym` is 4 bytes (u32)
/// - `String` is 24 bytes (ptr + len + cap)
/// - `&str` is 16 bytes (ptr + len)
///
/// For field names that appear frequently, this 6x reduction in
/// storage adds up significantly.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[repr(transparent)]
pub struct Sym(u32);

impl Sym {
    /// Create a symbol from a raw index.
    ///
    /// # Safety
    ///
    /// The index must have been returned by a previous call to `sym()`.
    #[inline(always)]
    pub const unsafe fn from_raw(index: u32) -> Self {
        Sym(index)
    }

    /// Get the raw index.
    #[inline(always)]
    pub const fn as_raw(&self) -> u32 {
        self.0
    }

    /// Resolve the symbol to its string value.
    #[inline]
    pub fn as_str(&self) -> &'static str {
        resolve(*self)
    }
}

impl Hash for Sym {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Debug for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sym({:?})", self.as_str())
    }
}

impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Sym {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_str().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Sym {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(sym(&s))
    }
}

impl AsRef<str> for Sym {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

// Pre-interned common DSPy field names for instant lookup
mod static_strings {
    //! Pre-interned static symbols for common DSPy field names.
    //!
    //! These symbols have fixed indices (0-31) and require no runtime interning.

    use super::Sym;

    /// Pre-interned symbol for "question" field.
    pub const QUESTION: Sym = Sym(0);
    /// Pre-interned symbol for "answer" field.
    pub const ANSWER: Sym = Sym(1);
    /// Pre-interned symbol for "context" field.
    pub const CONTEXT: Sym = Sym(2);
    /// Pre-interned symbol for "thought" field (chain-of-thought).
    pub const THOUGHT: Sym = Sym(3);
    /// Pre-interned symbol for "action" field (ReAct).
    pub const ACTION: Sym = Sym(4);
    /// Pre-interned symbol for "observation" field (ReAct).
    pub const OBSERVATION: Sym = Sym(5);
    /// Pre-interned symbol for "reasoning" field.
    pub const REASONING: Sym = Sym(6);
    /// Pre-interned symbol for "code" field (program of thought).
    pub const CODE: Sym = Sym(7);
    /// Pre-interned symbol for "result" field.
    pub const RESULT: Sym = Sym(8);
    /// Pre-interned symbol for "input" field.
    pub const INPUT: Sym = Sym(9);
    /// Pre-interned symbol for "output" field.
    pub const OUTPUT: Sym = Sym(10);
    /// Pre-interned symbol for "query" field.
    pub const QUERY: Sym = Sym(11);
    /// Pre-interned symbol for "passage" field (retrieval).
    pub const PASSAGE: Sym = Sym(12);
    /// Pre-interned symbol for "document" field.
    pub const DOCUMENT: Sym = Sym(13);
    /// Pre-interned symbol for "summary" field.
    pub const SUMMARY: Sym = Sym(14);
    /// Pre-interned symbol for "rationale" field.
    pub const RATIONALE: Sym = Sym(15);
    /// Pre-interned symbol for "claim" field.
    pub const CLAIM: Sym = Sym(16);
    /// Pre-interned symbol for "evidence" field.
    pub const EVIDENCE: Sym = Sym(17);
    /// Pre-interned symbol for "label" field (classification).
    pub const LABEL: Sym = Sym(18);
    /// Pre-interned symbol for "score" field.
    pub const SCORE: Sym = Sym(19);
    /// Pre-interned symbol for "response" field.
    pub const RESPONSE: Sym = Sym(20);
    /// Pre-interned symbol for "instruction" field.
    pub const INSTRUCTION: Sym = Sym(21);
    /// Pre-interned symbol for "tool" field (tool use).
    pub const TOOL: Sym = Sym(22);
    /// Pre-interned symbol for "tool_input" field.
    pub const TOOL_INPUT: Sym = Sym(23);
    /// Pre-interned symbol for "tool_output" field.
    pub const TOOL_OUTPUT: Sym = Sym(24);
    /// Pre-interned symbol for "step" field.
    pub const STEP: Sym = Sym(25);
    /// Pre-interned symbol for "final_answer" field.
    pub const FINAL_ANSWER: Sym = Sym(26);
    /// Pre-interned symbol for "explanation" field.
    pub const EXPLANATION: Sym = Sym(27);
    /// Pre-interned symbol for "hypothesis" field.
    pub const HYPOTHESIS: Sym = Sym(28);
    /// Pre-interned symbol for "conclusion" field.
    pub const CONCLUSION: Sym = Sym(29);
    /// Pre-interned symbol for "feedback" field.
    pub const FEEDBACK: Sym = Sym(30);
    /// Pre-interned symbol for "error" field.
    pub const ERROR: Sym = Sym(31);

    /// Number of pre-interned static symbols.
    pub const STATIC_COUNT: u32 = 32;

    // Static string table
    pub static STRINGS: [&str; STATIC_COUNT as usize] = [
        "question",     // 0
        "answer",       // 1
        "context",      // 2
        "thought",      // 3
        "action",       // 4
        "observation",  // 5
        "reasoning",    // 6
        "code",         // 7
        "result",       // 8
        "input",        // 9
        "output",       // 10
        "query",        // 11
        "passage",      // 12
        "document",     // 13
        "summary",      // 14
        "rationale",    // 15
        "claim",        // 16
        "evidence",     // 17
        "label",        // 18
        "score",        // 19
        "response",     // 20
        "instruction",  // 21
        "tool",         // 22
        "tool_input",   // 23
        "tool_output",  // 24
        "step",         // 25
        "final_answer", // 26
        "explanation",  // 27
        "hypothesis",   // 28
        "conclusion",   // 29
        "feedback",     // 30
        "error",        // 31
    ];
}

// Re-export static symbols for convenient access
pub use static_strings::{
    ACTION, ANSWER, CLAIM, CODE, CONCLUSION, CONTEXT, DOCUMENT, ERROR, EVIDENCE, EXPLANATION,
    FEEDBACK, FINAL_ANSWER, HYPOTHESIS, INPUT, INSTRUCTION, LABEL, OBSERVATION, OUTPUT, PASSAGE,
    QUERY, QUESTION, RATIONALE, REASONING, RESPONSE, RESULT, SCORE, STEP, SUMMARY, THOUGHT, TOOL,
    TOOL_INPUT, TOOL_OUTPUT,
};

/// Global interner for runtime strings
struct Interner {
    /// Map from string to symbol index
    string_to_sym: DashMap<Box<str>, Sym>,
    /// Dynamic strings (indexed starting from STATIC_COUNT)
    dynamic_strings: RwLock<Vec<Box<str>>>,
    /// Next symbol index
    next_index: AtomicU32,
}

impl Interner {
    fn new() -> Self {
        Self {
            string_to_sym: DashMap::new(),
            dynamic_strings: RwLock::new(Vec::new()),
            next_index: AtomicU32::new(static_strings::STATIC_COUNT),
        }
    }

    fn intern(&self, s: &str) -> Sym {
        // Fast path: check static strings first
        if let Some(sym) = static_lookup(s) {
            return sym;
        }

        // Check if already interned
        if let Some(entry) = self.string_to_sym.get(s) {
            return *entry;
        }

        // Slow path: intern new string
        let boxed: Box<str> = s.into();

        // Double-check after acquiring (another thread might have interned it)
        *self.string_to_sym.entry(boxed.clone()).or_insert_with(|| {
            let index = self.next_index.fetch_add(1, Ordering::Relaxed);
            let sym = Sym(index);

            // Store the string for later resolution
            // Use unwrap_or_else to recover from poisoned lock (can happen in tests)
            let mut strings = self
                .dynamic_strings
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Index into dynamic_strings is (sym.0 - STATIC_COUNT)
            let dynamic_index = (index - static_strings::STATIC_COUNT) as usize;
            // Ensure vector is large enough (handles concurrent insertions)
            if strings.len() <= dynamic_index {
                strings.resize(dynamic_index + 1, "".into());
            }
            strings[dynamic_index] = boxed;

            sym
        })
    }

    fn resolve(&self, sym: Sym) -> &'static str {
        let index = sym.0;

        // Static string
        if index < static_strings::STATIC_COUNT {
            return static_strings::STRINGS[index as usize];
        }

        // Dynamic string
        let dynamic_index = (index - static_strings::STATIC_COUNT) as usize;
        // Use unwrap_or_else to recover from poisoned lock (can happen in tests)
        let strings = self
            .dynamic_strings
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // SAFETY: The string lives in the interner for the lifetime of the program.
        // We leak the reference to get 'static lifetime, which is safe because
        // the interner is never dropped (it's a static).
        let s: &str = &strings[dynamic_index];
        unsafe { std::mem::transmute::<&str, &'static str>(s) }
    }
}

/// Lookup a string in the static table.
#[inline]
fn static_lookup(s: &str) -> Option<Sym> {
    // Use a simple match for common strings (compiler can optimize this)
    match s {
        "question" => Some(QUESTION),
        "answer" => Some(ANSWER),
        "context" => Some(CONTEXT),
        "thought" => Some(THOUGHT),
        "action" => Some(ACTION),
        "observation" => Some(OBSERVATION),
        "reasoning" => Some(REASONING),
        "code" => Some(CODE),
        "result" => Some(RESULT),
        "input" => Some(INPUT),
        "output" => Some(OUTPUT),
        "query" => Some(QUERY),
        "passage" => Some(PASSAGE),
        "document" => Some(DOCUMENT),
        "summary" => Some(SUMMARY),
        "rationale" => Some(RATIONALE),
        "claim" => Some(CLAIM),
        "evidence" => Some(EVIDENCE),
        "label" => Some(LABEL),
        "score" => Some(SCORE),
        "response" => Some(RESPONSE),
        "instruction" => Some(INSTRUCTION),
        "tool" => Some(TOOL),
        "tool_input" => Some(TOOL_INPUT),
        "tool_output" => Some(TOOL_OUTPUT),
        "step" => Some(STEP),
        "final_answer" => Some(FINAL_ANSWER),
        "explanation" => Some(EXPLANATION),
        "hypothesis" => Some(HYPOTHESIS),
        "conclusion" => Some(CONCLUSION),
        "feedback" => Some(FEEDBACK),
        "error" => Some(ERROR),
        _ => None,
    }
}

// Global interner instance
static INTERNER: std::sync::OnceLock<Interner> = std::sync::OnceLock::new();

fn get_interner() -> &'static Interner {
    INTERNER.get_or_init(Interner::new)
}

/// Intern a string and return its symbol.
///
/// This is the primary API for string interning. Common DSPy field names
/// (question, answer, context, etc.) are pre-interned and return instantly.
/// Other strings are interned on first use.
///
/// # Example
///
/// ```
/// use kkachi::intern::{sym, QUESTION};
///
/// // Pre-interned strings return instantly
/// let q = sym("question");
/// assert_eq!(q, QUESTION);
///
/// // Custom strings are interned on first use
/// let custom = sym("my_field");
/// assert_eq!(custom.as_str(), "my_field");
///
/// // Same string always returns same symbol
/// assert_eq!(sym("my_field"), custom);
/// ```
#[inline]
pub fn sym(s: &str) -> Sym {
    // Fast path for static strings
    if let Some(sym) = static_lookup(s) {
        return sym;
    }
    get_interner().intern(s)
}

/// Resolve a symbol to its string value.
///
/// This is O(1) for pre-interned strings and O(1) for dynamic strings
/// (just an array lookup).
#[inline]
pub fn resolve(sym: Sym) -> &'static str {
    let index = sym.0;
    if index < static_strings::STATIC_COUNT {
        static_strings::STRINGS[index as usize]
    } else {
        get_interner().resolve(sym)
    }
}

/// Get the number of interned strings.
#[inline]
pub fn interned_count() -> u32 {
    get_interner().next_index.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_symbols() {
        assert_eq!(sym("question"), QUESTION);
        assert_eq!(sym("answer"), ANSWER);
        assert_eq!(sym("context"), CONTEXT);
        assert_eq!(sym("reasoning"), REASONING);
    }

    #[test]
    fn test_static_resolve() {
        assert_eq!(QUESTION.as_str(), "question");
        assert_eq!(ANSWER.as_str(), "answer");
        assert_eq!(REASONING.as_str(), "reasoning");
    }

    #[test]
    fn test_dynamic_interning() {
        let s1 = sym("custom_field_1");
        let s2 = sym("custom_field_2");

        assert_ne!(s1, s2);
        assert_eq!(s1.as_str(), "custom_field_1");
        assert_eq!(s2.as_str(), "custom_field_2");
    }

    #[test]
    fn test_same_string_same_symbol() {
        let s1 = sym("repeated_field");
        let s2 = sym("repeated_field");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_sym_size() {
        // Sym should be exactly 4 bytes
        assert_eq!(std::mem::size_of::<Sym>(), 4);
    }

    #[test]
    fn test_sym_hash() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(sym("key"), "value");
        assert_eq!(map.get(&sym("key")), Some(&"value"));
    }

    #[test]
    fn test_sym_ord() {
        // Static symbols should maintain order
        assert!(QUESTION < ANSWER);
        assert!(ANSWER < CONTEXT);
    }

    #[test]
    fn test_sym_display() {
        assert_eq!(format!("{}", QUESTION), "question");
        assert_eq!(format!("{}", sym("custom")), "custom");
    }

    #[test]
    fn test_sym_debug() {
        assert_eq!(format!("{:?}", QUESTION), "Sym(\"question\")");
    }

    #[test]
    fn test_sym_serde() {
        let original = sym("test_field");
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, "\"test_field\"");

        let deserialized: Sym = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_interned_count() {
        let initial = interned_count();
        sym("count_test_1");
        sym("count_test_2");
        assert!(interned_count() >= initial + 2);
    }

    #[test]
    fn test_all_static_strings() {
        // Verify all static strings resolve correctly
        for (i, &s) in static_strings::STRINGS.iter().enumerate() {
            let sym = Sym(i as u32);
            assert_eq!(sym.as_str(), s);
            assert_eq!(super::sym(s), sym);
        }
    }
}
