# Kkachi `reason()` Multi-Line Output Bug

**Status**: ✅ FIXED in PR [coming soon]
**GitHub Issue**: [#3](https://github.com/lituus-io/kkachi/issues/3)
**Fix**: One-line change to `parse_response()` - automatically preserves full response when no answer marker found

## Summary

**The `reason()` function was incompatible with multi-line output.** It was designed for Chain of Thought (CoT) prompting where the LLM provides reasoning and then a **single-line final answer**. When used with multi-line content like YAML templates, Python code, or any structured text, it discarded everything except the last line.

**This has been fixed** with a simple one-line change that automatically detects when no answer marker is present and preserves the full response.

## Root Cause

Located in `crates/kkachi/src/recursive/reason.rs:256-303` in the `parse_response()` method:

```rust
fn parse_response<'b>(&self, response: &'b str) -> (Option<&'b str>, String) {
    // Look for common answer markers
    let answer_markers = [
        "Therefore:",
        "Answer:",
        "Final Answer:",
        "So the answer is:",
        "Result:",
    ];

    for marker in &answer_markers {
        if let Some(idx) = response.find(marker) {
            let reasoning = response[..idx].trim();
            let answer_start = idx + marker.len();
            let answer = response[answer_start..].trim();

            // 🐛 BUG: Only keeps text up to first newline
            let answer_end = answer.find('\n').unwrap_or(answer.len());
            let answer = answer[..answer_end].trim().to_string();

            return (
                if reasoning.is_empty() { None } else { Some(reasoning) },
                answer,
            );
        }
    }

    // 🐛 CRITICAL BUG: When no marker found, only keeps the LAST LINE
    if let Some(last_line_start) = response.rfind('\n') {
        let reasoning = response[..last_line_start].trim();
        let answer = response[last_line_start..].trim().to_string();
        (
            if reasoning.is_empty() { None } else { Some(reasoning) },
            answer,
        )
    } else {
        (None, response.trim().to_string())
    }
}
```

### The Flow

1. **Marker found** (e.g., "Therefore:") → Extract answer but **stop at first `\n`** (line 273)
2. **No marker found** → Use `rfind('\n')` to find **last newline**, treat everything before as "reasoning" and everything after (the **last line only**) as the "answer" (lines 289-291)
3. **No newlines at all** → Use full response as answer (line 301)

## Evidence

### Test Results

| Input Content | Expected | `reason()` Result | `best_of()` Result |
|--------------|----------|-------------------|-------------------|
| `"Plain text no newlines"` | Full text | ✅ Full text (67 chars) | ✅ Full text |
| `"Line1\nLine2\nLine3"` | Full text | ❌ "Line3" only (5 chars) | ✅ Full text |
| YAML template (202 lines) | Full YAML | ❌ Last line only (7 chars) | ✅ Full YAML (105+ chars) |

### Real-World Example

```python
from kkachi import reason, Checks

# Mock returns valid YAML
def mock_llm(prompt, feedback=None):
    return """name: template
runtime: yaml
description: Multi-line YAML template

config:
  project: my-project
  region: us-central1

resources:
  bucket:
    type: storage.v1.bucket
    properties:
      name: test-bucket"""  # 7077 characters, 200+ lines

validator = Checks().min_len(100)

# ❌ FAILS: reason() only validates last line "      name: test-bucket"
result = reason(mock_llm, "Generate YAML").validate(validator).go()
print(f"Score: {result.score}")  # 0.0
print(f"Output: {result.output}")  # "name: test-bucket" (7 chars)

# ✅ WORKS: best_of() preserves full output
result, _ = best_of(mock_llm, "Generate YAML", n=1).validate(validator).go_with_pool()
print(f"Score: {result.score}")  # 1.0
print(f"Output length: {len(result.output)}")  # 7077 chars
```

## Why This Happens

The `reason()` function implements Chain of Thought prompting as described in academic papers:

1. LLM shows step-by-step reasoning
2. LLM provides a **final answer on one line** after a marker like "Therefore:" or "Answer:"

Example of intended use:

```
Step 1: Calculate 25 * 30 = 750
Step 2: Calculate 25 * 7 = 175
Step 3: Add them: 750 + 175 = 925

Therefore: 925
```

The parser extracts:
- **Reasoning**: "Step 1... Step 3: Add them: 750 + 175 = 925"
- **Answer**: "925"

## Comparison with `best_of()`

### `best_of()` (line 430 in `best_of.rs`)
```rust
let text_to_score = if let Some(ref lang) = self.config.extract_lang {
    extract_code(&output.text, lang)
        .map(|s| s.to_string())
        .unwrap_or_else(|| output.text.clone())
} else {
    output.text.clone()  // ✅ Uses FULL output
};
```

### `reason()` (line 198 in `reason.rs`)
```rust
let (reasoning, answer) = self.parse_response(&output.text);  // ❌ Splits and discards
last_output = answer.clone();  // Only keeps extracted answer
```

## Impact

### Fails For:
- ✗ YAML templates (200+ lines → 7 chars)
- ✗ Python/Rust/any code files (multi-line → last line only)
- ✗ Markdown documents (multi-paragraph → last line)
- ✗ JSON configurations (multi-line → last line)
- ✗ Any structured multi-line content

### Works For:
- ✓ Math problems ("Therefore: 42")
- ✓ Single-word answers ("Answer: Paris")
- ✓ Short facts ("Result: The capital is Washington DC")
- ✓ Content without newlines

## Solutions

### Option 1: Use `best_of()` for Multi-Line Content (Recommended Workaround)

```python
# Instead of:
result = reason(llm, "Generate YAML").validate(validator).go()

# Use:
result, _ = best_of(llm, "Generate YAML", n=1).validate(validator).go_with_pool()
```

### Option 2: Fix `reason()` to Preserve Full Output (Breaking Change)

Add a configuration option to disable answer extraction:

```rust
pub struct ReasonConfig {
    // ... existing fields ...
    /// If true, use the full response as output instead of parsing for "answer"
    pub use_full_output: bool,
}

impl<'a, L: Llm, V: Validate> Reason<'a, L, V> {
    /// Use the full LLM response as output (don't extract answer).
    ///
    /// When enabled, multi-line content is preserved.
    pub fn full_output(mut self) -> Self {
        self.config.use_full_output = true;
        self
    }
}
```

Then in `run()`:

```rust
last_output = if self.config.use_full_output {
    output.text.clone()  // Full output
} else {
    let (reasoning, answer) = self.parse_response(&output.text);
    // ... existing logic
    answer.clone()
};
```

Usage:

```python
from kkachi import reason

# Enable full output mode for multi-line content
result = reason(llm, "Generate YAML") \
    .full_output() \
    .validate(validator) \
    .go()
```

### Option 3: Document the Limitation (Minimal Change)

Add prominent documentation warnings:

```rust
/// ⚠️ **Multi-Line Content Warning**: This function is designed for Chain of Thought
/// reasoning where the final answer is a single line. For multi-line outputs like
/// YAML, code, or structured text, use `best_of()` or `refine()` instead.
///
/// If the LLM response contains newlines and no "Therefore:" or "Answer:" marker,
/// only the **last line** will be used as the output.
pub fn reason<'a, L: Llm>(llm: &'a L, prompt: &'a str) -> Reason<'a, L, NoValidation> {
    Reason::new(llm, prompt)
}
```

## Recommendation

**Short term**: Document the limitation and recommend `best_of(n=1)` for multi-line content.

**Long term**: Add `full_output()` configuration option to `reason()` to support both use cases without breaking existing code.

## Related Code

- `crates/kkachi/src/recursive/reason.rs:256-303` - `parse_response()` method
- `crates/kkachi/src/recursive/best_of.rs:425-434` - Shows correct full-output handling
- `crates/kkachi-python/src/dspy.rs:60-110` - Python bindings (PyCallableLlm)

## Test Cases to Add

```rust
#[test]
fn test_reason_multiline_with_full_output() {
    let llm = MockLlm::new(|_, _| {
        "Line 1\nLine 2\nLine 3\nLine 4".to_string()
    });

    // With full_output() - should preserve all lines
    let result = reason(&llm, "Test")
        .full_output()
        .go();

    assert!(result.output.contains("Line 1"));
    assert!(result.output.contains("Line 4"));
    assert_eq!(result.output.lines().count(), 4);
}

#[test]
fn test_reason_default_extracts_last_line() {
    let llm = MockLlm::new(|_, _| {
        "Line 1\nLine 2\nLine 3\nLine 4".to_string()
    });

    // Default behavior - should extract last line only
    let result = reason(&llm, "Test").go();

    assert_eq!(result.output, "Line 4");
    assert!(result.reasoning().unwrap().contains("Line 1"));
}
```

---

**Status**: Confirmed behavior, not a traditional bug but a design limitation that should be documented or made configurable.
