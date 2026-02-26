// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Chat Adapter
//!
//! Formats prompts for chat-style LM interfaces with clear field separators.

use crate::adapter::{find_field_value, format_field_name, Adapter, DemoData};
use crate::error::Result;
use crate::intern::Sym;
use crate::predict::FieldRange;
use crate::signature::Signature;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;

/// Chat adapter configuration.
#[derive(Clone, Copy)]
pub struct ChatConfig {
    /// Include field descriptions in prompt
    pub include_descriptions: bool,
    /// Use markdown formatting
    pub use_markdown: bool,
    /// Separator between demos
    pub demo_separator: &'static str,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            include_descriptions: true,
            use_markdown: false,
            demo_separator: "\n---\n",
        }
    }
}

impl ChatConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            include_descriptions: true,
            use_markdown: false,
            demo_separator: "\n---\n",
        }
    }

    /// Set description inclusion.
    pub const fn with_descriptions(mut self, include: bool) -> Self {
        self.include_descriptions = include;
        self
    }

    /// Set markdown formatting.
    pub const fn with_markdown(mut self, use_md: bool) -> Self {
        self.use_markdown = use_md;
        self
    }
}

/// Chat-style adapter for LM prompts.
///
/// Formats prompts as:
/// ```text
/// [Instructions]
///
/// [Demo 1]
/// Question: ...
/// Answer: ...
///
/// ---
///
/// [Demo 2]
/// ...
///
/// ---
///
/// Question: [input]
/// Answer:
/// ```
#[derive(Clone, Copy)]
pub struct ChatAdapter {
    config: ChatConfig,
}

impl ChatAdapter {
    /// Create a new chat adapter.
    pub const fn new(config: ChatConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub const fn default() -> Self {
        Self::new(ChatConfig::new())
    }

    /// Get the configuration.
    pub const fn config(&self) -> &ChatConfig {
        &self.config
    }

    /// Format a single demo.
    fn format_demo(&self, buffer: &mut Vec<u8>, demo: &DemoData<'_>, signature: &Signature<'_>) {
        use crate::intern::sym;

        // Format input fields
        for field in &signature.input_fields {
            let field_sym = sym(&field.name);
            if let Some(value) = demo.get_input(field_sym) {
                format_field_name(buffer, &field.name);
                buffer.extend_from_slice(b": ");
                buffer.extend_from_slice(value.as_str().as_bytes());
                buffer.push(b'\n');
            }
        }

        // Format output fields
        for field in &signature.output_fields {
            let field_sym = sym(&field.name);
            if let Some(value) = demo.get_output(field_sym) {
                format_field_name(buffer, &field.name);
                buffer.extend_from_slice(b": ");
                buffer.extend_from_slice(value.as_str().as_bytes());
                buffer.push(b'\n');
            }
        }
    }

    /// Format the current input (without demo answers).
    fn format_input(&self, buffer: &mut Vec<u8>, inputs: &Inputs<'_>, signature: &Signature<'_>) {
        for field in &signature.input_fields {
            let field_name = &field.name;
            if let Some(value) = inputs.get(field_name.as_ref()) {
                format_field_name(buffer, field_name);
                buffer.extend_from_slice(b": ");
                buffer.extend_from_slice(value.as_bytes());
                buffer.push(b'\n');
            }
        }

        // Add prompts for output fields
        for field in &signature.output_fields {
            format_field_name(buffer, &field.name);
            buffer.extend_from_slice(b": ");
        }
    }
}

impl Adapter for ChatAdapter {
    fn format<'a>(
        &self,
        buffer: &'a mut Vec<u8>,
        signature: &Signature<'_>,
        inputs: &Inputs<'_>,
        demos: &[DemoData<'_>],
    ) -> StrView<'a> {
        buffer.clear();

        // Add instructions
        if !signature.instructions.is_empty() {
            buffer.extend_from_slice(signature.instructions.as_bytes());
            buffer.extend_from_slice(b"\n\n");
        }

        // Add field descriptions if configured
        if self.config.include_descriptions && !signature.input_fields.is_empty() {
            buffer.extend_from_slice(b"Given the following fields:\n");
            for field in &signature.input_fields {
                buffer.extend_from_slice(b"- ");
                format_field_name(buffer, &field.name);
                if !field.desc.is_empty() {
                    buffer.extend_from_slice(b": ");
                    buffer.extend_from_slice(field.desc.as_bytes());
                }
                buffer.push(b'\n');
            }
            buffer.push(b'\n');

            buffer.extend_from_slice(b"Produce the following fields:\n");
            for field in &signature.output_fields {
                buffer.extend_from_slice(b"- ");
                format_field_name(buffer, &field.name);
                if !field.desc.is_empty() {
                    buffer.extend_from_slice(b": ");
                    buffer.extend_from_slice(field.desc.as_bytes());
                }
                buffer.push(b'\n');
            }
            buffer.push(b'\n');
        }

        // Add demos
        for (i, demo) in demos.iter().enumerate() {
            if i > 0 {
                buffer.extend_from_slice(self.config.demo_separator.as_bytes());
            }
            self.format_demo(buffer, demo, signature);
        }

        // Add separator before input if we had demos
        if !demos.is_empty() {
            buffer.extend_from_slice(self.config.demo_separator.as_bytes());
        }

        // Add current input
        self.format_input(buffer, inputs, signature);

        // SAFETY: We only write valid UTF-8
        unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) }
    }

    fn parse<'a>(
        &self,
        response: StrView<'a>,
        signature: &Signature<'_>,
    ) -> Result<SmallVec<[(Sym, FieldRange); 4]>> {
        use crate::intern::sym;

        let text = response.as_str();
        let mut fields = SmallVec::new();

        for field in &signature.output_fields {
            let field_name = &field.name;
            if let Some(range) = find_field_value(text, field_name) {
                fields.push((
                    sym(field_name),
                    FieldRange::new(range.start as u32, range.end as u32),
                ));
            }
        }

        Ok(fields)
    }

    fn name(&self) -> &'static str {
        "Chat"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::Signature;

    #[test]
    fn test_chat_adapter_creation() {
        let adapter = ChatAdapter::default();
        assert_eq!(adapter.name(), "Chat");
        assert!(adapter.config().include_descriptions);
    }

    #[test]
    fn test_chat_config() {
        let config = ChatConfig::new()
            .with_descriptions(false)
            .with_markdown(true);
        assert!(!config.include_descriptions);
        assert!(config.use_markdown);
    }

    #[test]
    fn test_format_basic() {
        let adapter = ChatAdapter::new(ChatConfig::new().with_descriptions(false));
        let sig = Signature::parse("question -> answer").unwrap();

        let mut buffer = Vec::new();
        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 2+2?");

        let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);

        assert!(prompt.as_str().contains("Question: What is 2+2?"));
        assert!(prompt.as_str().contains("Answer:"));
    }

    #[test]
    fn test_format_with_instructions() {
        use std::borrow::Cow;

        let adapter = ChatAdapter::new(ChatConfig::new().with_descriptions(false));
        let mut sig = Signature::parse("question -> answer").unwrap();
        sig.instructions = Cow::Borrowed("Answer the question.");

        let mut buffer = Vec::new();
        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 2+2?");

        let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);

        assert!(prompt.as_str().starts_with("Answer the question."));
    }

    #[test]
    fn test_parse_response() {
        use crate::intern::sym;

        let adapter = ChatAdapter::default();
        let sig = Signature::parse("question -> answer").unwrap();

        let response = StrView::new("Answer: 4");
        let fields = adapter.parse(response, &sig).unwrap();

        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, sym("answer"));

        let range = fields[0].1.as_range();
        assert_eq!(&response.as_str()[range], "4");
    }

    #[test]
    fn test_format_with_demo() {
        use crate::intern::sym;

        let adapter = ChatAdapter::new(ChatConfig::new().with_descriptions(false));
        let sig = Signature::parse("question -> answer").unwrap();

        let q_sym = sym("question");
        let a_sym = sym("answer");

        let demo_inputs = [(q_sym, StrView::new("What is 1+1?"))];
        let demo_outputs = [(a_sym, StrView::new("2"))];
        let demo = DemoData::new(&demo_inputs, &demo_outputs);

        let mut buffer = Vec::new();
        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 2+2?");

        let prompt = adapter.format(&mut buffer, &sig, &inputs, &[demo]);

        // Should contain demo
        assert!(prompt.as_str().contains("What is 1+1?"));
        assert!(prompt.as_str().contains("2"));
        // Should contain separator
        assert!(prompt.as_str().contains("---"));
        // Should contain input
        assert!(prompt.as_str().contains("What is 2+2?"));
    }
}
