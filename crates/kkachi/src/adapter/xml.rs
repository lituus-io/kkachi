// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! XML Adapter
//!
//! Formats prompts for XML output and parses XML responses.
//! Uses SAX-style streaming parsing for efficiency.

use crate::adapter::{Adapter, DemoData};
use crate::error::Result;
use crate::intern::{sym, Sym};
use crate::predict::FieldRange;
use crate::signature::Signature;
use crate::str_view::StrView;
use crate::types::Inputs;
use smallvec::SmallVec;

/// XML adapter configuration.
#[derive(Clone, Copy)]
pub struct XMLConfig {
    /// Pretty print XML (with indentation)
    pub pretty: bool,
    /// Root element name
    pub root_element: &'static str,
    /// Include XML declaration
    pub include_declaration: bool,
}

impl Default for XMLConfig {
    fn default() -> Self {
        Self {
            pretty: true,
            root_element: "response",
            include_declaration: false,
        }
    }
}

impl XMLConfig {
    /// Create new config.
    pub const fn new() -> Self {
        Self {
            pretty: true,
            root_element: "response",
            include_declaration: false,
        }
    }

    /// Set pretty printing.
    pub const fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Set root element name.
    pub const fn with_root(mut self, root: &'static str) -> Self {
        self.root_element = root;
        self
    }

    /// Set declaration inclusion.
    pub const fn with_declaration(mut self, include: bool) -> Self {
        self.include_declaration = include;
        self
    }
}

/// XML adapter for LM prompts.
///
/// Formats prompts to request XML output:
/// ```text
/// [Instructions]
///
/// Respond with XML in this format:
/// <response>
///   <answer>...</answer>
/// </response>
///
/// Input:
/// <input>
///   <question>...</question>
/// </input>
///
/// Output:
/// ```
#[derive(Clone, Copy)]
pub struct XMLAdapter {
    config: XMLConfig,
}

impl XMLAdapter {
    /// Create a new XML adapter.
    pub const fn new(config: XMLConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub const fn default() -> Self {
        Self::new(XMLConfig::new())
    }

    /// Get the configuration.
    pub const fn config(&self) -> &XMLConfig {
        &self.config
    }

    /// Write XML schema for output fields.
    fn write_schema(&self, buffer: &mut Vec<u8>, signature: &Signature<'_>) {
        buffer.push(b'<');
        buffer.extend_from_slice(self.config.root_element.as_bytes());
        buffer.extend_from_slice(b">\n");

        for field in &signature.output_fields {
            let name = &field.name;
            if self.config.pretty {
                buffer.extend_from_slice(b"  ");
            }
            buffer.push(b'<');
            buffer.extend_from_slice(name.as_bytes());
            buffer.push(b'>');
            buffer.extend_from_slice(b"...");
            buffer.extend_from_slice(b"</");
            buffer.extend_from_slice(name.as_bytes());
            buffer.extend_from_slice(b">\n");
        }

        buffer.extend_from_slice(b"</");
        buffer.extend_from_slice(self.config.root_element.as_bytes());
        buffer.push(b'>');
    }

    /// Write XML for inputs.
    fn write_input_xml(
        &self,
        buffer: &mut Vec<u8>,
        inputs: &Inputs<'_>,
        signature: &Signature<'_>,
    ) {
        buffer.extend_from_slice(b"<input>\n");

        for field in &signature.input_fields {
            let name = &field.name;
            if let Some(value) = inputs.get(name.as_ref()) {
                if self.config.pretty {
                    buffer.extend_from_slice(b"  ");
                }
                buffer.push(b'<');
                buffer.extend_from_slice(name.as_bytes());
                buffer.push(b'>');
                write_escaped_xml(buffer, value);
                buffer.extend_from_slice(b"</");
                buffer.extend_from_slice(name.as_bytes());
                buffer.extend_from_slice(b">\n");
            }
        }

        buffer.extend_from_slice(b"</input>");
    }

    /// Write demo as XML.
    fn write_demo_xml(&self, buffer: &mut Vec<u8>, demo: &DemoData<'_>, signature: &Signature<'_>) {
        use crate::intern::sym;

        // Input
        buffer.extend_from_slice(b"Input:\n<input>\n");
        for field in &signature.input_fields {
            let field_sym = sym(&field.name);
            if let Some(value) = demo.get_input(field_sym) {
                let name = &field.name;
                if self.config.pretty {
                    buffer.extend_from_slice(b"  ");
                }
                buffer.push(b'<');
                buffer.extend_from_slice(name.as_bytes());
                buffer.push(b'>');
                write_escaped_xml(buffer, value.as_str());
                buffer.extend_from_slice(b"</");
                buffer.extend_from_slice(name.as_bytes());
                buffer.extend_from_slice(b">\n");
            }
        }
        buffer.extend_from_slice(b"</input>\n\n");

        // Output
        buffer.extend_from_slice(b"Output:\n<");
        buffer.extend_from_slice(self.config.root_element.as_bytes());
        buffer.extend_from_slice(b">\n");
        for field in &signature.output_fields {
            let field_sym = sym(&field.name);
            if let Some(value) = demo.get_output(field_sym) {
                let name = &field.name;
                if self.config.pretty {
                    buffer.extend_from_slice(b"  ");
                }
                buffer.push(b'<');
                buffer.extend_from_slice(name.as_bytes());
                buffer.push(b'>');
                write_escaped_xml(buffer, value.as_str());
                buffer.extend_from_slice(b"</");
                buffer.extend_from_slice(name.as_bytes());
                buffer.extend_from_slice(b">\n");
            }
        }
        buffer.extend_from_slice(b"</");
        buffer.extend_from_slice(self.config.root_element.as_bytes());
        buffer.extend_from_slice(b">\n");
    }
}

/// Write XML-escaped text.
fn write_escaped_xml(buffer: &mut Vec<u8>, s: &str) {
    for c in s.chars() {
        match c {
            '<' => buffer.extend_from_slice(b"&lt;"),
            '>' => buffer.extend_from_slice(b"&gt;"),
            '&' => buffer.extend_from_slice(b"&amp;"),
            '"' => buffer.extend_from_slice(b"&quot;"),
            '\'' => buffer.extend_from_slice(b"&apos;"),
            c => {
                let mut buf = [0u8; 4];
                buffer.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
}

/// Unescape XML entities.
#[allow(dead_code)] // Reserved for future use in response parsing
fn unescape_xml(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Find an XML element value in text.
/// Returns the range of the content between tags.
fn find_xml_element(text: &str, element_name: &str) -> Option<std::ops::Range<usize>> {
    let open_tag = format!("<{}>", element_name);
    let close_tag = format!("</{}>", element_name);

    let start = text.find(&open_tag)?;
    let content_start = start + open_tag.len();

    let rest = &text[content_start..];
    let end = rest.find(&close_tag)?;

    Some(content_start..content_start + end)
}

/// Find XML element with attributes.
fn find_xml_element_with_attrs(text: &str, element_name: &str) -> Option<std::ops::Range<usize>> {
    // First try simple tags
    if let Some(range) = find_xml_element(text, element_name) {
        return Some(range);
    }

    // Try with attributes: <element attr="...">content</element>
    let pattern = format!("<{}", element_name);
    let start = text.find(&pattern)?;

    // Find the end of the opening tag
    let rest = &text[start..];
    let tag_end = rest.find('>')?;
    let content_start = start + tag_end + 1;

    // Find the closing tag
    let close_tag = format!("</{}>", element_name);
    let rest = &text[content_start..];
    let end = rest.find(&close_tag)?;

    Some(content_start..content_start + end)
}

impl Adapter for XMLAdapter {
    fn format<'a>(
        &self,
        buffer: &'a mut Vec<u8>,
        signature: &Signature<'_>,
        inputs: &Inputs<'_>,
        demos: &[DemoData<'_>],
    ) -> StrView<'a> {
        buffer.clear();

        // XML declaration
        if self.config.include_declaration {
            buffer.extend_from_slice(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n");
        }

        // Instructions
        if !signature.instructions.is_empty() {
            buffer.extend_from_slice(signature.instructions.as_bytes());
            buffer.extend_from_slice(b"\n\n");
        }

        // Schema
        buffer.extend_from_slice(b"Respond with XML in this format:\n");
        self.write_schema(buffer, signature);
        buffer.extend_from_slice(b"\n\n");

        // Demos
        for (i, demo) in demos.iter().enumerate() {
            if i > 0 {
                buffer.extend_from_slice(b"\n---\n\n");
            }
            self.write_demo_xml(buffer, demo, signature);
        }

        // Separator
        if !demos.is_empty() {
            buffer.extend_from_slice(b"\n---\n\n");
        }

        // Input
        buffer.extend_from_slice(b"Input:\n");
        self.write_input_xml(buffer, inputs, signature);
        buffer.extend_from_slice(b"\n\nOutput:\n");

        // SAFETY: We only write valid UTF-8
        unsafe { StrView::from_raw_parts(buffer.as_ptr(), buffer.len()) }
    }

    fn parse<'a>(
        &self,
        response: StrView<'a>,
        signature: &Signature<'_>,
    ) -> Result<SmallVec<[(Sym, FieldRange); 4]>> {
        let text = response.as_str();
        let mut fields = SmallVec::new();

        for field in &signature.output_fields {
            let field_name = &field.name;
            if let Some(range) = find_xml_element_with_attrs(text, field_name) {
                fields.push((
                    sym(field_name),
                    FieldRange::new(range.start as u32, range.end as u32),
                ));
            }
        }

        Ok(fields)
    }

    fn name(&self) -> &'static str {
        "XML"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::sym;

    #[test]
    fn test_xml_adapter_creation() {
        let adapter = XMLAdapter::default();
        assert_eq!(adapter.name(), "XML");
        assert!(adapter.config().pretty);
    }

    #[test]
    fn test_xml_config() {
        let config = XMLConfig::new()
            .with_pretty(false)
            .with_root("output")
            .with_declaration(true);
        assert!(!config.pretty);
        assert_eq!(config.root_element, "output");
        assert!(config.include_declaration);
    }

    #[test]
    fn test_escape_xml() {
        let mut buffer = Vec::new();
        write_escaped_xml(&mut buffer, "<script>alert('xss')</script>");
        assert_eq!(
            String::from_utf8(buffer).unwrap(),
            "&lt;script&gt;alert(&apos;xss&apos;)&lt;/script&gt;"
        );
    }

    #[test]
    fn test_unescape_xml() {
        assert_eq!(
            unescape_xml("&lt;script&gt;alert(&apos;xss&apos;)&lt;/script&gt;"),
            "<script>alert('xss')</script>"
        );
    }

    #[test]
    fn test_find_xml_element() {
        let xml = "<response><answer>42</answer></response>";
        let range = find_xml_element(xml, "answer");
        assert!(range.is_some());
        assert_eq!(&xml[range.unwrap()], "42");
    }

    #[test]
    fn test_find_xml_element_with_attrs() {
        let xml = r#"<response><answer type="number">42</answer></response>"#;
        let range = find_xml_element_with_attrs(xml, "answer");
        assert!(range.is_some());
        assert_eq!(&xml[range.unwrap()], "42");
    }

    #[test]
    fn test_format_basic() {
        let adapter = XMLAdapter::default();
        let sig = Signature::parse("question -> answer").unwrap();

        let mut buffer = Vec::new();
        let mut inputs = Inputs::new();
        inputs.insert("question", "What is 2+2?");

        let prompt = adapter.format(&mut buffer, &sig, &inputs, &[]);

        assert!(prompt
            .as_str()
            .contains("<question>What is 2+2?</question>"));
        assert!(prompt.as_str().contains("<answer>...</answer>"));
        assert!(prompt.as_str().contains("Output:"));
    }

    #[test]
    fn test_parse_response() {
        let adapter = XMLAdapter::default();
        let sig = Signature::parse("question -> answer").unwrap();

        let response = StrView::new("<response><answer>4</answer></response>");
        let fields = adapter.parse(response, &sig).unwrap();

        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, sym("answer"));

        let range = fields[0].1.as_range();
        assert_eq!(&response.as_str()[range], "4");
    }

    #[test]
    fn test_format_with_demo() {
        use crate::intern::sym;

        let adapter = XMLAdapter::default();
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
        assert!(prompt
            .as_str()
            .contains("<question>What is 1+1?</question>"));
        assert!(prompt.as_str().contains("<answer>2</answer>"));
        // Should contain input
        assert!(prompt
            .as_str()
            .contains("<question>What is 2+2?</question>"));
    }
}
