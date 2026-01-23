// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Markdown rewriting utilities.
//!
//! This module provides tools for extracting and manipulating code blocks
//! and sections within Markdown documents.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::rewrite::{rewrite, extract_code};
//!
//! let markdown = "# My Document\n\n```yaml\nkey: value\n```\n\n## Section A\nSome content\n";
//!
//! // Extract code
//! let yaml = extract_code(markdown, "yaml");
//! assert_eq!(yaml, Some("key: value"));
//!
//! // Rewrite code blocks
//! let updated = rewrite(markdown)
//!     .replace_code("yaml", "key: new_value")
//!     .build();
//! ```

use smallvec::SmallVec;

/// Create a new markdown rewriter.
pub fn rewrite(markdown: &str) -> Rewrite<'_> {
    Rewrite::new(markdown)
}

/// Extract the first code block of a given language.
///
/// Returns `None` if no matching code block is found.
pub fn extract_code<'a>(markdown: &'a str, lang: &str) -> Option<&'a str> {
    let fence_start = format!("```{}", lang);
    let start_idx = markdown.find(&fence_start)?;
    let content_start = start_idx + fence_start.len();

    // Skip to the next newline
    let content_start = markdown[content_start..].find('\n')? + content_start + 1;

    // Find the closing fence
    let content_end = markdown[content_start..].find("```")?;

    let content = &markdown[content_start..content_start + content_end];
    Some(content.trim())
}

/// Extract all code blocks of a given language.
pub fn extract_all_code<'a>(markdown: &'a str, lang: &str) -> Vec<&'a str> {
    let fence_start = format!("```{}", lang);
    let mut results = Vec::new();
    let mut search_start = 0;

    while let Some(start_idx) = markdown[search_start..].find(&fence_start) {
        let abs_start = search_start + start_idx;
        let content_start = abs_start + fence_start.len();

        // Skip to the next newline
        if let Some(nl_offset) = markdown[content_start..].find('\n') {
            let content_start = content_start + nl_offset + 1;

            // Find the closing fence
            if let Some(content_end) = markdown[content_start..].find("```") {
                let content = &markdown[content_start..content_start + content_end];
                results.push(content.trim());
                search_start = content_start + content_end + 3;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    results
}

/// Extract a section by its heading.
///
/// Returns the content between the heading and the next heading of the same
/// or higher level.
pub fn extract_section<'a>(markdown: &'a str, title: &str) -> Option<&'a str> {
    // Find the heading
    let heading_patterns = [
        format!("# {}", title),
        format!("## {}", title),
        format!("### {}", title),
        format!("#### {}", title),
    ];

    for pattern in &heading_patterns {
        if let Some(start_idx) = markdown.find(pattern.as_str()) {
            let level = pattern.chars().take_while(|c| *c == '#').count();

            // Find content start (after the heading line)
            let content_start = markdown[start_idx..]
                .find('\n')
                .map(|i| start_idx + i + 1)?;

            // Find the next heading of same or higher level
            let remaining = &markdown[content_start..];
            let content_end = find_next_heading(remaining, level).unwrap_or(remaining.len());

            return Some(remaining[..content_end].trim());
        }
    }

    None
}

/// Find the offset of the next heading at or above the given level.
fn find_next_heading(text: &str, level: usize) -> Option<usize> {
    let mut byte_offset = 0;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            let heading_level = trimmed.chars().take_while(|c| *c == '#').count();
            if heading_level <= level {
                return Some(byte_offset);
            }
        }
        // Add line length plus newline
        byte_offset += line.len() + 1;
    }
    None
}

/// A markdown rewriting operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operation<'a> {
    /// Replace a code block with new content.
    ReplaceCode { lang: &'a str, content: &'a str },
    /// Add or replace a section.
    Section { title: &'a str, content: &'a str },
    /// Add a section after another section.
    SectionAfter {
        after: &'a str,
        title: &'a str,
        content: &'a str,
    },
    /// Add a section before another section.
    SectionBefore {
        before: &'a str,
        title: &'a str,
        content: &'a str,
    },
    /// Remove a section.
    RemoveSection { title: &'a str },
    /// Append content to the end.
    Append { content: &'a str },
}

/// A builder for markdown rewriting operations.
#[derive(Debug)]
pub struct Rewrite<'a> {
    source: &'a str,
    operations: SmallVec<[Operation<'a>; 8]>,
}

impl<'a> Rewrite<'a> {
    /// Create a new rewriter for the given markdown.
    pub fn new(markdown: &'a str) -> Self {
        Self {
            source: markdown,
            operations: SmallVec::new(),
        }
    }

    /// Replace a code block of the given language.
    ///
    /// If multiple blocks exist, only the first is replaced.
    pub fn replace_code(mut self, lang: &'a str, content: &'a str) -> Self {
        self.operations
            .push(Operation::ReplaceCode { lang, content });
        self
    }

    /// Add or replace a section with the given title.
    ///
    /// If the section exists, its content is replaced.
    /// If it doesn't exist, it's appended to the document.
    pub fn section(mut self, title: &'a str, content: &'a str) -> Self {
        self.operations.push(Operation::Section { title, content });
        self
    }

    /// Add a section after another section.
    pub fn section_after(mut self, after: &'a str, title: &'a str, content: &'a str) -> Self {
        self.operations.push(Operation::SectionAfter {
            after,
            title,
            content,
        });
        self
    }

    /// Add a section before another section.
    pub fn section_before(mut self, before: &'a str, title: &'a str, content: &'a str) -> Self {
        self.operations.push(Operation::SectionBefore {
            before,
            title,
            content,
        });
        self
    }

    /// Remove a section by title.
    pub fn remove_section(mut self, title: &'a str) -> Self {
        self.operations.push(Operation::RemoveSection { title });
        self
    }

    /// Append content to the end of the document.
    pub fn append(mut self, content: &'a str) -> Self {
        self.operations.push(Operation::Append { content });
        self
    }

    /// Build the final markdown document.
    pub fn build(self) -> String {
        let mut result = self.source.to_string();

        for op in self.operations {
            result = match op {
                Operation::ReplaceCode { lang, content } => {
                    replace_code_block(&result, lang, content)
                }
                Operation::Section { title, content } => {
                    replace_or_append_section(&result, title, content)
                }
                Operation::SectionAfter {
                    after,
                    title,
                    content,
                } => insert_section_after(&result, after, title, content),
                Operation::SectionBefore {
                    before,
                    title,
                    content,
                } => insert_section_before(&result, before, title, content),
                Operation::RemoveSection { title } => remove_section(&result, title),
                Operation::Append { content } => {
                    let mut r = result;
                    if !r.ends_with('\n') {
                        r.push('\n');
                    }
                    r.push_str(content);
                    r
                }
            };
        }

        result
    }
}

/// Replace the first code block of a given language.
fn replace_code_block(markdown: &str, lang: &str, new_content: &str) -> String {
    let fence_start = format!("```{}", lang);

    if let Some(start_idx) = markdown.find(&fence_start) {
        let content_start = start_idx + fence_start.len();

        // Find the newline after the fence
        if let Some(nl_offset) = markdown[content_start..].find('\n') {
            let code_start = content_start + nl_offset + 1;

            // Find the closing fence
            if let Some(end_offset) = markdown[code_start..].find("```") {
                let code_end = code_start + end_offset;

                let mut result = String::with_capacity(markdown.len() + new_content.len());
                result.push_str(&markdown[..code_start]);
                result.push_str(new_content);
                result.push('\n');
                result.push_str(&markdown[code_end..]);
                return result;
            }
        }
    }

    // No matching block found, return original
    markdown.to_string()
}

/// Replace or append a section.
fn replace_or_append_section(markdown: &str, title: &str, content: &str) -> String {
    // Try to find and replace existing section
    // Search in reverse order (most specific first) to avoid "# X" matching inside "## X"
    let heading_patterns = [
        format!("#### {}", title),
        format!("### {}", title),
        format!("## {}", title),
        format!("# {}", title),
    ];

    for pattern in &heading_patterns {
        // Find the pattern, ensuring it's at the start of a line
        let mut search_start = 0;
        while let Some(pos) = markdown[search_start..].find(pattern.as_str()) {
            let abs_pos = search_start + pos;
            // Check if it's at the start of the string or after a newline
            if abs_pos == 0 || markdown.as_bytes().get(abs_pos.wrapping_sub(1)) == Some(&b'\n') {
                // Found a valid heading at start of line
                let level = pattern.chars().take_while(|c| *c == '#').count();

                // Find content start (after the heading line)
                if let Some(nl_offset) = markdown[abs_pos..].find('\n') {
                    let content_start = abs_pos + nl_offset + 1;

                    // Find the next heading of same or higher level
                    let remaining = &markdown[content_start..];
                    let content_end =
                        find_next_heading(remaining, level).unwrap_or(remaining.len());

                    let mut result = String::with_capacity(markdown.len() + content.len());
                    result.push_str(&markdown[..content_start]);
                    result.push_str(content);
                    if !content.ends_with('\n') {
                        result.push('\n');
                    }
                    result.push('\n');
                    result.push_str(&markdown[content_start + content_end..]);
                    return result;
                }
            }
            // Continue searching after this position
            search_start = abs_pos + 1;
        }
    }

    // Section not found, append it
    let mut result = markdown.to_string();
    if !result.ends_with('\n') {
        result.push('\n');
    }
    result.push('\n');
    result.push_str("## ");
    result.push_str(title);
    result.push('\n');
    result.push('\n');
    result.push_str(content);
    result.push('\n');
    result
}

/// Find a heading pattern at the start of a line.
/// Returns (position, level) if found.
fn find_heading_at_line_start(markdown: &str, title: &str) -> Option<(usize, usize)> {
    // Search in reverse order (most specific first) to avoid "# X" matching inside "## X"
    let heading_patterns = [
        format!("#### {}", title),
        format!("### {}", title),
        format!("## {}", title),
        format!("# {}", title),
    ];

    for pattern in &heading_patterns {
        let mut search_start = 0;
        while let Some(pos) = markdown[search_start..].find(pattern.as_str()) {
            let abs_pos = search_start + pos;
            // Check if it's at the start of the string or after a newline
            if abs_pos == 0 || markdown.as_bytes().get(abs_pos.wrapping_sub(1)) == Some(&b'\n') {
                let level = pattern.chars().take_while(|c| *c == '#').count();
                return Some((abs_pos, level));
            }
            search_start = abs_pos + 1;
        }
    }
    None
}

/// Insert a section after another section.
fn insert_section_after(markdown: &str, after: &str, title: &str, content: &str) -> String {
    if let Some((start_idx, level)) = find_heading_at_line_start(markdown, after) {
        // Find content start (after the heading line)
        if let Some(nl_offset) = markdown[start_idx..].find('\n') {
            let content_start = start_idx + nl_offset + 1;

            // Find the end of this section
            let remaining = &markdown[content_start..];
            let section_end = find_next_heading(remaining, level).unwrap_or(remaining.len());
            let insert_point = content_start + section_end;

            let hashes = "#".repeat(level);
            let new_section = format!("\n{} {}\n\n{}\n", hashes, title, content);

            let mut result = String::with_capacity(markdown.len() + new_section.len());
            result.push_str(&markdown[..insert_point]);
            result.push_str(&new_section);
            result.push_str(&markdown[insert_point..]);
            return result;
        }
    }

    // "after" section not found, append to end
    replace_or_append_section(markdown, title, content)
}

/// Insert a section before another section.
fn insert_section_before(markdown: &str, before: &str, title: &str, content: &str) -> String {
    if let Some((start_idx, level)) = find_heading_at_line_start(markdown, before) {
        let hashes = "#".repeat(level);
        let new_section = format!("{} {}\n\n{}\n\n", hashes, title, content);

        let mut result = String::with_capacity(markdown.len() + new_section.len());
        result.push_str(&markdown[..start_idx]);
        result.push_str(&new_section);
        result.push_str(&markdown[start_idx..]);
        return result;
    }

    // "before" section not found, append to end
    replace_or_append_section(markdown, title, content)
}

/// Remove a section from the document.
fn remove_section(markdown: &str, title: &str) -> String {
    if let Some((start_idx, level)) = find_heading_at_line_start(markdown, title) {
        // Find content start (after the heading line)
        if let Some(nl_offset) = markdown[start_idx..].find('\n') {
            let content_start = start_idx + nl_offset + 1;

            // Find the next heading of same or higher level
            let remaining = &markdown[content_start..];
            let section_end = find_next_heading(remaining, level).unwrap_or(remaining.len());

            let mut result = String::with_capacity(markdown.len());
            result.push_str(&markdown[..start_idx]);
            // Skip any leading newlines at the join point
            let after = &markdown[content_start + section_end..];
            result.push_str(after.trim_start_matches('\n'));
            return result;
        }
    }

    // Section not found, return original
    markdown.to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MD: &str = r#"# Title

Some intro text.

## Section A

Content of section A.

```yaml
key: old_value
nested:
  item: 1
```

## Section B

Content of section B.

```rust
fn main() {
    println!("Hello");
}
```

## Section C

Final section.
"#;

    #[test]
    fn test_extract_code_yaml() {
        let code = extract_code(SAMPLE_MD, "yaml");
        assert!(code.is_some());
        assert!(code.unwrap().contains("key: old_value"));
    }

    #[test]
    fn test_extract_code_rust() {
        let code = extract_code(SAMPLE_MD, "rust");
        assert!(code.is_some());
        assert!(code.unwrap().contains("fn main()"));
    }

    #[test]
    fn test_extract_code_not_found() {
        let code = extract_code(SAMPLE_MD, "python");
        assert!(code.is_none());
    }

    #[test]
    fn test_extract_all_code() {
        let md = r#"
```rust
fn one() {}
```

```rust
fn two() {}
```

```python
def three():
    pass
```
"#;
        let rust_blocks = extract_all_code(md, "rust");
        assert_eq!(rust_blocks.len(), 2);
        assert!(rust_blocks[0].contains("fn one"));
        assert!(rust_blocks[1].contains("fn two"));

        let python_blocks = extract_all_code(md, "python");
        assert_eq!(python_blocks.len(), 1);
    }

    #[test]
    fn test_extract_section() {
        let content = extract_section(SAMPLE_MD, "Section A");
        assert!(content.is_some());
        let content = content.unwrap();
        assert!(content.contains("Content of section A"));
        assert!(content.contains("key: old_value"));
    }

    #[test]
    fn test_replace_code() {
        let result = rewrite(SAMPLE_MD)
            .replace_code("yaml", "key: new_value")
            .build();

        assert!(result.contains("key: new_value"));
        assert!(!result.contains("key: old_value"));
        // Rust block should be unchanged
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn test_replace_section() {
        let result = rewrite(SAMPLE_MD)
            .section("Section B", "New content for B.")
            .build();

        assert!(result.contains("New content for B."));
        assert!(!result.contains("Content of section B."));
        // Other sections should be unchanged
        assert!(result.contains("Content of section A."));
        assert!(result.contains("Final section."));
    }

    #[test]
    fn test_add_new_section() {
        let result = rewrite(SAMPLE_MD)
            .section("Section D", "Brand new section.")
            .build();

        assert!(result.contains("## Section D"));
        assert!(result.contains("Brand new section."));
    }

    #[test]
    fn test_section_after() {
        let result = rewrite(SAMPLE_MD)
            .section_after("Section A", "Section A.5", "Inserted content.")
            .build();

        // Should appear after Section A but before Section B
        let a_pos = result.find("Section A").unwrap();
        let a5_pos = result.find("Section A.5").unwrap();
        let b_pos = result.find("Section B").unwrap();

        assert!(a_pos < a5_pos);
        assert!(a5_pos < b_pos);
    }

    #[test]
    fn test_section_before() {
        let result = rewrite(SAMPLE_MD)
            .section_before("Section B", "Section A.5", "Inserted content.")
            .build();

        let a_pos = result.find("Section A\n").unwrap();
        let a5_pos = result.find("Section A.5").unwrap();
        let b_pos = result.find("Section B").unwrap();

        assert!(a_pos < a5_pos);
        assert!(a5_pos < b_pos);
    }

    #[test]
    fn test_remove_section() {
        let result = rewrite(SAMPLE_MD).remove_section("Section B").build();

        assert!(!result.contains("Section B"));
        assert!(!result.contains("Content of section B."));
        // Other sections should remain
        assert!(result.contains("Section A"));
        assert!(result.contains("Section C"));
    }

    #[test]
    fn test_append() {
        let result = rewrite(SAMPLE_MD).append("Appended text.").build();

        assert!(result.ends_with("Appended text."));
    }

    #[test]
    fn test_chained_operations() {
        let result = rewrite(SAMPLE_MD)
            .replace_code("yaml", "key: updated")
            .section("Section B", "Updated B content.")
            .section("New Section", "New content.")
            .build();

        assert!(result.contains("key: updated"));
        assert!(result.contains("Updated B content."));
        assert!(result.contains("New Section"));
        assert!(result.contains("New content."));
    }

    #[test]
    fn test_empty_document() {
        let result = rewrite("")
            .section("First Section", "Some content.")
            .build();

        assert!(result.contains("## First Section"));
        assert!(result.contains("Some content."));
    }
}
