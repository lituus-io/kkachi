// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Template system for structured prompt optimization.
//!
//! Templates use Markdown with YAML frontmatter to define:
//! - Output format specifications
//! - System/user prompt templates
//! - Few-shot examples
//!
//! # Template Format
//!
//! ```text
//! ---
//! name: code_generation
//! version: "1.0"
//! signature: "requirement -> code"
//!
//! format:
//!   type: json
//!   schema:
//!     type: object
//!     required: [code]
//!
//! options:
//!   strict: true
//!   include_in_prompt: true
//! ---
//!
//! # System Prompt
//!
//! You are an expert programmer.
//!
//! ---examples---
//!
//! ## Example 1
//!
//! **Input:** Write hello world
//!
//! **Output:**
//! {"code": "println!(\"Hello, world!\")"}
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kkachi::recursive::Template;
//!
//! let template = Template::from_str(r#"
//! ---
//! name: qa
//! format:
//!   type: json
//! ---
//! Answer questions concisely.
//! "#)?;
//!
//! let prompt = template.assemble_prompt("What is 2+2?", 0, None);
//! ```

use std::borrow::Cow;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Output format types supported by templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FormatType {
    /// JSON format with optional schema validation.
    #[default]
    Json,
    /// YAML format with optional schema validation.
    Yaml,
    /// Markdown format.
    Markdown,
    /// XML format.
    Xml,
    /// Plain text (no format enforcement).
    Plain,
}

/// Tone for prompt instructions - affects language strictness.
///
/// This controls how requirements and constraints are phrased in prompts,
/// affecting the balance between recall (inclusive) and precision (restrictive).
///
/// # Example
///
/// ```rust
/// use kkachi::recursive::template::PromptTone;
///
/// let inclusive = PromptTone::Inclusive.modifiers();
/// assert!(inclusive.requirement_prefix.contains("Consider"));
///
/// let restrictive = PromptTone::Restrictive.modifiers();
/// assert!(restrictive.requirement_prefix.contains("MUST"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptTone {
    /// Inclusive language - encourages exploration, tolerates variation.
    /// Use phrases like "consider", "you may", "options include".
    /// Best for high-recall scenarios.
    Inclusive,
    /// Balanced tone - neutral instructions.
    /// Use phrases like "include", "please", "should".
    #[default]
    Balanced,
    /// Restrictive language - strict requirements, precise outputs.
    /// Use phrases like "must", "exactly", "only", "required".
    /// Best for high-precision scenarios.
    Restrictive,
}

impl PromptTone {
    /// Get instruction modifiers for this tone.
    ///
    /// Returns language patterns appropriate for this tone level.
    #[inline]
    pub fn modifiers(&self) -> ToneModifiers {
        match self {
            Self::Inclusive => ToneModifiers {
                requirement_prefix: "Consider including",
                possibility_prefix: "You may also",
                constraint_prefix: "Ideally",
                format_intro: "A suggested format is",
                output_verb: "could be",
                uncertainty_guidance: "When uncertain, include rather than exclude.",
            },
            Self::Balanced => ToneModifiers {
                requirement_prefix: "Include",
                possibility_prefix: "You can",
                constraint_prefix: "Please",
                format_intro: "Use the following format",
                output_verb: "should be",
                uncertainty_guidance: "Use your best judgment when uncertain.",
            },
            Self::Restrictive => ToneModifiers {
                requirement_prefix: "You MUST include",
                possibility_prefix: "Only include",
                constraint_prefix: "Required:",
                format_intro: "Output EXACTLY in this format",
                output_verb: "must be",
                uncertainty_guidance: "When uncertain, exclude rather than include.",
            },
        }
    }

    /// Map to RecallPrecisionMode equivalent threshold.
    ///
    /// - Inclusive → 0.6 (high recall)
    /// - Balanced → 0.8
    /// - Restrictive → 0.9 (high precision)
    #[inline]
    pub fn default_threshold(&self) -> f64 {
        match self {
            Self::Inclusive => 0.6,
            Self::Balanced => 0.8,
            Self::Restrictive => 0.9,
        }
    }

    /// Check if this tone favors recall over precision.
    #[inline]
    pub fn favors_recall(&self) -> bool {
        matches!(self, Self::Inclusive)
    }

    /// Check if this tone favors precision over recall.
    #[inline]
    pub fn favors_precision(&self) -> bool {
        matches!(self, Self::Restrictive)
    }
}

/// Modifiers for different prompt tones.
///
/// Contains language patterns appropriate for the tone level.
#[derive(Debug, Clone, Copy)]
pub struct ToneModifiers {
    /// Prefix for required elements (e.g., "Include" vs "You MUST include")
    pub requirement_prefix: &'static str,
    /// Prefix for optional elements (e.g., "You can" vs "Only include")
    pub possibility_prefix: &'static str,
    /// Prefix for constraints (e.g., "Please" vs "Required:")
    pub constraint_prefix: &'static str,
    /// Introduction for format instructions
    pub format_intro: &'static str,
    /// Verb for output descriptions (e.g., "should be" vs "must be")
    pub output_verb: &'static str,
    /// Guidance for handling uncertainty
    pub uncertainty_guidance: &'static str,
}

/// JSON Schema for output validation (simplified).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Schema type (object, array, string, etc.)
    #[serde(rename = "type", default)]
    pub schema_type: String,
    /// Required fields (for object type).
    #[serde(default)]
    pub required: Vec<String>,
    /// Property definitions (for object type).
    #[serde(default)]
    pub properties: serde_json::Map<String, serde_json::Value>,
}

/// Format specification for template outputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FormatSpec {
    /// The output format type.
    #[serde(rename = "type", default)]
    pub format_type: FormatType,
    /// Optional JSON schema for validation.
    #[serde(default)]
    pub schema: Option<JsonSchema>,
}

/// Template options controlling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateOptions {
    /// If true, fail when output doesn't match format.
    #[serde(default)]
    pub strict: bool,
    /// If true, include format instructions in the prompt.
    #[serde(default = "default_true")]
    pub include_in_prompt: bool,
    /// Prompt tone (inclusive, balanced, restrictive).
    /// Controls language strictness for recall/precision tuning.
    #[serde(default)]
    pub tone: PromptTone,
}

fn default_true() -> bool {
    true
}

impl Default for TemplateOptions {
    fn default() -> Self {
        Self {
            strict: false,
            include_in_prompt: true,
            tone: PromptTone::default(),
        }
    }
}

/// A single few-shot example in the template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateExample<'a> {
    /// The input/question for this example.
    #[serde(borrow)]
    pub input: Cow<'a, str>,
    /// The expected output for this example.
    #[serde(borrow)]
    pub output: Cow<'a, str>,
}

impl<'a> TemplateExample<'a> {
    /// Create a new example.
    pub fn new(input: impl Into<Cow<'a, str>>, output: impl Into<Cow<'a, str>>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
        }
    }

    /// Convert to owned version.
    pub fn into_owned(self) -> TemplateExample<'static> {
        TemplateExample {
            input: Cow::Owned(self.input.into_owned()),
            output: Cow::Owned(self.output.into_owned()),
        }
    }
}

/// YAML frontmatter structure for deserialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TemplateFrontmatter {
    #[serde(default)]
    name: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    signature: String,
    #[serde(default)]
    format: FormatSpec,
    #[serde(default)]
    options: TemplateOptions,
}

/// A template for structured prompt optimization.
///
/// Templates define the format, system prompt, and few-shot examples
/// for recursive refinement pipelines.
#[derive(Debug, Clone)]
pub struct Template<'a> {
    /// Template name (for identification).
    pub name: Cow<'a, str>,
    /// Template version.
    pub version: Cow<'a, str>,
    /// Signature string (e.g., "question -> answer").
    pub signature: Cow<'a, str>,
    /// Output format specification.
    pub format: FormatSpec,
    /// System prompt (the main instruction).
    pub system_prompt: Cow<'a, str>,
    /// Format instructions (extracted from markdown).
    pub format_instructions: Cow<'a, str>,
    /// Few-shot examples.
    pub examples: Vec<TemplateExample<'a>>,
    /// Template options.
    pub options: TemplateOptions,
}

impl Default for Template<'_> {
    fn default() -> Self {
        Self {
            name: Cow::Borrowed("default"),
            version: Cow::Borrowed("1.0"),
            signature: Cow::Borrowed("input -> output"),
            format: FormatSpec::default(),
            system_prompt: Cow::Borrowed(""),
            format_instructions: Cow::Borrowed(""),
            examples: Vec::new(),
            options: TemplateOptions::default(),
        }
    }
}

impl<'a> Template<'a> {
    /// Create a new empty template with the given name.
    pub fn new(name: impl Into<Cow<'a, str>>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Create a simple template with just a system prompt.
    ///
    /// This is useful for testing and quick prototyping.
    pub fn simple(prompt: impl Into<Cow<'a, str>>) -> Self {
        Self {
            name: Cow::Borrowed("simple"),
            system_prompt: prompt.into(),
            ..Default::default()
        }
    }

    /// Render the template with the given input.
    ///
    /// This is a simplified version of `assemble_prompt` that just
    /// combines the system prompt with the input.
    pub fn render(&self, input: &str) -> String {
        let mut output = String::with_capacity(self.system_prompt.len() + input.len() + 100);
        if !self.system_prompt.is_empty() {
            output.push_str(&self.system_prompt);
            output.push_str("\n\n");
        }
        output.push_str("Input: ");
        output.push_str(input);
        output
    }

    /// Parse a template from a string.
    ///
    /// The string should contain YAML frontmatter delimited by `---`,
    /// followed by the markdown body. Examples can be separated with
    /// `---examples---`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template = Template::from_str(r#"
    /// ---
    /// name: qa
    /// format:
    ///   type: json
    /// ---
    /// Answer questions.
    /// "#)?;
    /// ```
    pub fn from_str(content: &'a str) -> Result<Self> {
        let content = content.trim();

        // Check for frontmatter
        if !content.starts_with("---") {
            return Err(Error::parse(
                "Template must start with YAML frontmatter (---)",
            ));
        }

        // Find the end of frontmatter
        let after_first = &content[3..];
        let frontmatter_end = after_first
            .find("\n---")
            .ok_or_else(|| Error::parse("Missing closing --- for frontmatter"))?;

        let frontmatter_str = &after_first[..frontmatter_end].trim();
        let body_start = frontmatter_end + 4; // Skip \n---
        let body = if body_start < after_first.len() {
            after_first[body_start..].trim()
        } else {
            ""
        };

        // Parse frontmatter
        let frontmatter: TemplateFrontmatter = serde_yaml::from_str(frontmatter_str)
            .map_err(|e| Error::parse(format!("Invalid YAML frontmatter: {}", e)))?;

        // Split body into system prompt and examples
        let (system_prompt, examples_section) = if let Some(idx) = body.find("---examples---") {
            let prompt = body[..idx].trim();
            let examples_str = body[idx + 14..].trim(); // Skip ---examples---
            (prompt, Some(examples_str))
        } else {
            (body, None)
        };

        // Extract format instructions from system prompt
        let format_instructions = extract_format_instructions(system_prompt);

        // Parse examples
        let examples = if let Some(examples_str) = examples_section {
            parse_examples(examples_str)?
        } else {
            Vec::new()
        };

        Ok(Self {
            name: Cow::Borrowed(if frontmatter.name.is_empty() {
                "unnamed"
            } else {
                // We need to leak this or use owned
                return Ok(Self {
                    name: Cow::Owned(frontmatter.name),
                    version: Cow::Owned(frontmatter.version),
                    signature: Cow::Owned(frontmatter.signature),
                    format: frontmatter.format,
                    system_prompt: Cow::Borrowed(system_prompt),
                    format_instructions: Cow::Owned(format_instructions),
                    examples,
                    options: frontmatter.options,
                });
            }),
            version: Cow::Owned(frontmatter.version),
            signature: Cow::Owned(frontmatter.signature),
            format: frontmatter.format,
            system_prompt: Cow::Borrowed(system_prompt),
            format_instructions: Cow::Owned(format_instructions),
            examples,
            options: frontmatter.options,
        })
    }

    /// Load a template from a file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template = Template::from_file("templates/code_gen.md")?;
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Template<'static>> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::io(format!("Failed to read template file: {}", e)))?;

        // Parse and immediately convert to owned to avoid lifetime issues
        Self::parse_owned(&content)
    }

    /// Parse a template from a string and return an owned version.
    ///
    /// This is useful when the source string is temporary.
    fn parse_owned(content: &str) -> Result<Template<'static>> {
        let content = content.trim();

        // Check for frontmatter
        if !content.starts_with("---") {
            return Err(Error::parse(
                "Template must start with YAML frontmatter (---)",
            ));
        }

        // Find the end of frontmatter
        let after_first = &content[3..];
        let frontmatter_end = after_first
            .find("\n---")
            .ok_or_else(|| Error::parse("Missing closing --- for frontmatter"))?;

        let frontmatter_str = &after_first[..frontmatter_end].trim();
        let body_start = frontmatter_end + 4; // Skip \n---
        let body = if body_start < after_first.len() {
            after_first[body_start..].trim()
        } else {
            ""
        };

        // Parse frontmatter
        let frontmatter: TemplateFrontmatter = serde_yaml::from_str(frontmatter_str)
            .map_err(|e| Error::parse(format!("Invalid YAML frontmatter: {}", e)))?;

        // Split body into system prompt and examples
        let (system_prompt, examples_section) = if let Some(idx) = body.find("---examples---") {
            let prompt = body[..idx].trim();
            let examples_str = body[idx + 14..].trim(); // Skip ---examples---
            (prompt, Some(examples_str))
        } else {
            (body, None)
        };

        // Extract format instructions from system prompt
        let format_instructions = extract_format_instructions(system_prompt);

        // Parse examples
        let examples = if let Some(examples_str) = examples_section {
            parse_examples(examples_str)?
        } else {
            Vec::new()
        };

        Ok(Template {
            name: Cow::Owned(if frontmatter.name.is_empty() {
                "unnamed".to_string()
            } else {
                frontmatter.name
            }),
            version: Cow::Owned(frontmatter.version),
            signature: Cow::Owned(frontmatter.signature),
            format: frontmatter.format,
            system_prompt: Cow::Owned(system_prompt.to_string()),
            format_instructions: Cow::Owned(format_instructions),
            examples,
            options: frontmatter.options,
        })
    }

    /// Convert to an owned version with 'static lifetime.
    pub fn into_owned(self) -> Template<'static> {
        Template {
            name: Cow::Owned(self.name.into_owned()),
            version: Cow::Owned(self.version.into_owned()),
            signature: Cow::Owned(self.signature.into_owned()),
            format: self.format,
            system_prompt: Cow::Owned(self.system_prompt.into_owned()),
            format_instructions: Cow::Owned(self.format_instructions.into_owned()),
            examples: self.examples.into_iter().map(|e| e.into_owned()).collect(),
            options: self.options,
        }
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<Cow<'a, str>>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the format type.
    pub fn with_format(mut self, format_type: FormatType) -> Self {
        self.format.format_type = format_type;
        self
    }

    /// Add an example.
    pub fn with_example(mut self, example: TemplateExample<'a>) -> Self {
        self.examples.push(example);
        self
    }

    /// Set strict mode.
    pub fn strict(mut self, strict: bool) -> Self {
        self.options.strict = strict;
        self
    }

    /// Set the prompt tone.
    ///
    /// Controls language strictness for recall/precision tuning:
    /// - `Inclusive`: permissive language for high-recall scenarios
    /// - `Balanced`: neutral language (default)
    /// - `Restrictive`: strict language for high-precision scenarios
    ///
    /// # Example
    ///
    /// ```rust
    /// use kkachi::recursive::template::{Template, PromptTone};
    ///
    /// let template = Template::new("qa")
    ///     .with_tone(PromptTone::Restrictive);
    /// ```
    pub fn with_tone(mut self, tone: PromptTone) -> Self {
        self.options.tone = tone;
        self
    }

    /// Assemble the full prompt for an iteration.
    ///
    /// This combines the system prompt, format instructions, examples,
    /// and the current question into a single prompt string.
    ///
    /// The prompt tone affects how instructions are phrased:
    /// - `Inclusive`: permissive language ("consider", "may")
    /// - `Balanced`: neutral language ("include", "should")
    /// - `Restrictive`: strict language ("MUST", "required")
    pub fn assemble_prompt(
        &self,
        question: &str,
        iteration: u32,
        feedback: Option<&str>,
    ) -> String {
        let mut prompt = String::with_capacity(4096);
        let tone = self.options.tone.modifiers();

        // System prompt
        if !self.system_prompt.is_empty() {
            prompt.push_str(&self.system_prompt);
            prompt.push_str("\n\n");
        }

        // Format instructions
        if self.options.include_in_prompt && !self.format_instructions.is_empty() {
            prompt.push_str("## Output Format\n\n");
            prompt.push_str(&self.format_instructions);
            prompt.push_str("\n\n");
        } else if self.options.include_in_prompt {
            // Generate format instructions from format spec with tone-aware language
            match self.format.format_type {
                FormatType::Json => {
                    prompt.push_str("## Output Format\n\n");
                    prompt.push_str(tone.format_intro);
                    prompt.push_str(": valid JSON.\n");
                    if let Some(ref schema) = self.format.schema {
                        if !schema.required.is_empty() {
                            prompt.push_str(tone.requirement_prefix);
                            prompt.push_str(" these fields: ");
                            prompt.push_str(&schema.required.join(", "));
                            prompt.push('\n');
                        }
                    }
                    prompt.push_str(tone.uncertainty_guidance);
                    prompt.push_str("\n\n");
                }
                FormatType::Yaml => {
                    prompt.push_str("## Output Format\n\n");
                    prompt.push_str(tone.format_intro);
                    prompt.push_str(": valid YAML.\n");
                    if let Some(ref schema) = self.format.schema {
                        if !schema.required.is_empty() {
                            prompt.push_str(tone.requirement_prefix);
                            prompt.push_str(" these fields: ");
                            prompt.push_str(&schema.required.join(", "));
                            prompt.push('\n');
                        }
                    }
                    prompt.push_str(tone.uncertainty_guidance);
                    prompt.push_str("\n\n");
                }
                FormatType::Xml => {
                    prompt.push_str("## Output Format\n\n");
                    prompt.push_str(tone.format_intro);
                    prompt.push_str(": valid XML.\n");
                    prompt.push_str(tone.uncertainty_guidance);
                    prompt.push_str("\n\n");
                }
                FormatType::Markdown => {
                    prompt.push_str("## Output Format\n\n");
                    prompt.push_str(tone.format_intro);
                    prompt.push_str(": Markdown.\n");
                    prompt.push_str(tone.uncertainty_guidance);
                    prompt.push_str("\n\n");
                }
                FormatType::Plain => {}
            }
        }

        // Examples
        if !self.examples.is_empty() {
            prompt.push_str("## Examples\n\n");
            for (i, example) in self.examples.iter().enumerate() {
                prompt.push_str(&format!("### Example {}\n\n", i + 1));
                prompt.push_str("**Input:** ");
                prompt.push_str(&example.input);
                prompt.push_str("\n\n**Output:**\n");
                prompt.push_str(&example.output);
                prompt.push_str("\n\n");
            }
        }

        // Current question
        prompt.push_str("## Your Task\n\n");
        prompt.push_str("**Input:** ");
        prompt.push_str(question);
        prompt.push('\n');

        // Feedback from previous iteration
        if iteration > 0 {
            if let Some(fb) = feedback {
                prompt.push_str("\n**Feedback from previous attempt:**\n");
                prompt.push_str(fb);
                prompt.push('\n');
            }
        }

        prompt
    }

    /// Validate output against the format specification.
    ///
    /// Returns `Ok(())` if the output matches the expected format,
    /// or an error describing what's wrong.
    pub fn validate_output(&self, output: &str) -> Result<()> {
        match self.format.format_type {
            FormatType::Json => {
                // Extract JSON from output (may be wrapped in markdown code blocks)
                let json_str = extract_json_from_output(output);

                // Try to parse as JSON
                let value: serde_json::Value = serde_json::from_str(json_str)
                    .map_err(|e| Error::validation(format!("Invalid JSON: {}", e)))?;

                // Check schema if provided
                if let Some(ref schema) = self.format.schema {
                    if schema.schema_type == "object" {
                        if let serde_json::Value::Object(obj) = &value {
                            // Check required fields
                            for field in &schema.required {
                                if !obj.contains_key(field) {
                                    return Err(Error::validation(format!(
                                        "Missing required field: {}",
                                        field
                                    )));
                                }
                            }
                        } else {
                            return Err(Error::validation("Expected JSON object"));
                        }
                    }
                }
                Ok(())
            }
            FormatType::Yaml => {
                // Extract YAML from output (may be wrapped in markdown code blocks)
                let yaml_str = extract_yaml_from_output(output);

                // Try to parse as YAML
                let value: serde_yaml::Value = serde_yaml::from_str(yaml_str)
                    .map_err(|e| Error::validation(format!("Invalid YAML: {}", e)))?;

                // Check schema if provided (YAML maps to similar structure as JSON)
                if let Some(ref schema) = self.format.schema {
                    if schema.schema_type == "object" {
                        if let serde_yaml::Value::Mapping(map) = &value {
                            // Check required fields
                            for field in &schema.required {
                                let key = serde_yaml::Value::String(field.clone());
                                if !map.contains_key(&key) {
                                    return Err(Error::validation(format!(
                                        "Missing required field: {}",
                                        field
                                    )));
                                }
                            }
                        } else {
                            return Err(Error::validation("Expected YAML mapping"));
                        }
                    }
                }
                Ok(())
            }
            FormatType::Xml => {
                // Basic XML validation - check for balanced tags
                if !output.trim().starts_with('<') || !output.trim().ends_with('>') {
                    return Err(Error::validation(
                        "Invalid XML: must start with < and end with >",
                    ));
                }
                Ok(())
            }
            FormatType::Markdown | FormatType::Plain => {
                // No validation for markdown/plain
                Ok(())
            }
        }
    }

    /// Parse and extract structured data from the output.
    ///
    /// For JSON format, this deserializes into the requested type.
    pub fn parse_output<T: serde::de::DeserializeOwned>(&self, output: &str) -> Result<T> {
        match self.format.format_type {
            FormatType::Json => {
                // Try to extract JSON from the output (may be wrapped in markdown code blocks)
                let json_str = extract_json_from_output(output);
                serde_json::from_str(json_str)
                    .map_err(|e| Error::parse(format!("Failed to parse JSON output: {}", e)))
            }
            FormatType::Yaml => {
                // Try to extract YAML from the output (may be wrapped in markdown code blocks)
                let yaml_str = extract_yaml_from_output(output);
                serde_yaml::from_str(yaml_str)
                    .map_err(|e| Error::parse(format!("Failed to parse YAML output: {}", e)))
            }
            _ => Err(Error::parse(
                "parse_output only supports JSON and YAML formats",
            )),
        }
    }

    /// Get format instructions string.
    pub fn get_format_instructions(&self) -> &str {
        &self.format_instructions
    }
}

/// Extract format instructions from a markdown prompt.
///
/// Looks for sections like "## Output Format" or "### Format".
fn extract_format_instructions(markdown: &str) -> String {
    let lower = markdown.to_lowercase();

    // Look for format-related headings
    for marker in &[
        "## output format",
        "### output format",
        "## format",
        "### format",
    ] {
        if let Some(start) = lower.find(marker) {
            // Find the content after the heading
            let content_start = start + marker.len();
            let rest = &markdown[content_start..];

            // Find the next heading or end
            let end = rest
                .find("\n## ")
                .or_else(|| rest.find("\n### "))
                .unwrap_or(rest.len());

            return rest[..end].trim().to_string();
        }
    }

    String::new()
}

/// Parse examples from the examples section of a template.
fn parse_examples(content: &str) -> Result<Vec<TemplateExample<'static>>> {
    let mut examples = Vec::new();

    // Split by example headers (## Example, ### Example, etc.)
    let parts: Vec<&str> = content
        .split(|c| c == '#')
        .filter(|s| !s.trim().is_empty())
        .collect();

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Skip the "Example N" header line
        let content_start = part.find('\n').unwrap_or(0);
        let example_content = &part[content_start..].trim();

        // Look for Input: and Output: markers
        let input = extract_field(example_content, "input")?;
        let output = extract_field(example_content, "output")?;

        if !input.is_empty() && !output.is_empty() {
            examples.push(TemplateExample {
                input: Cow::Owned(input),
                output: Cow::Owned(output),
            });
        }
    }

    Ok(examples)
}

/// Extract a field value from example content.
fn extract_field(content: &str, field_name: &str) -> Result<String> {
    let lower = content.to_lowercase();
    let markers = [
        format!("**{}:**", field_name),
        format!("**{}**:", field_name),
        format!("{}:", field_name),
    ];

    for marker in &markers {
        let marker_lower = marker.to_lowercase();
        if let Some(start) = lower.find(&marker_lower) {
            let value_start = start + marker.len();
            let rest = &content[value_start..];

            // Find the end (next field or end of content)
            let end = rest
                .to_lowercase()
                .find("**output")
                .or_else(|| rest.to_lowercase().find("**input"))
                .or_else(|| rest.find("\n## "))
                .or_else(|| rest.find("\n### "))
                .unwrap_or(rest.len());

            return Ok(rest[..end].trim().to_string());
        }
    }

    Ok(String::new())
}

/// Extract JSON from output that may be wrapped in markdown code blocks.
fn extract_json_from_output(output: &str) -> &str {
    let trimmed = output.trim();

    // Check for ```json ... ``` blocks
    if let Some(start) = trimmed.find("```json") {
        let json_start = start + 7;
        if let Some(end) = trimmed[json_start..].find("```") {
            return trimmed[json_start..json_start + end].trim();
        }
    }

    // Check for ``` ... ``` blocks (generic code block)
    if let Some(start) = trimmed.find("```") {
        let json_start = start + 3;
        // Skip language identifier if present
        let content_start = trimmed[json_start..]
            .find('\n')
            .map(|i| json_start + i + 1)
            .unwrap_or(json_start);
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    // Return as-is
    trimmed
}

/// Extract YAML from output that may be wrapped in markdown code blocks.
///
/// Supports both ```yaml and ```yml language identifiers.
fn extract_yaml_from_output(output: &str) -> &str {
    let trimmed = output.trim();

    // Check for ```yaml ... ``` blocks
    if let Some(start) = trimmed.find("```yaml") {
        let yaml_start = start + 7;
        if let Some(end) = trimmed[yaml_start..].find("```") {
            return trimmed[yaml_start..yaml_start + end].trim();
        }
    }

    // Check for ```yml ... ``` blocks (common alternative)
    if let Some(start) = trimmed.find("```yml") {
        let yaml_start = start + 6;
        if let Some(end) = trimmed[yaml_start..].find("```") {
            return trimmed[yaml_start..yaml_start + end].trim();
        }
    }

    // Check for ``` ... ``` blocks (generic code block)
    if let Some(start) = trimmed.find("```") {
        let yaml_start = start + 3;
        // Skip language identifier if present
        let content_start = trimmed[yaml_start..]
            .find('\n')
            .map(|i| yaml_start + i + 1)
            .unwrap_or(yaml_start);
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    // Return as-is
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_template() {
        let content = r#"---
name: test_template
version: "1.0"
signature: "question -> answer"
format:
  type: json
options:
  strict: true
---
You are a helpful assistant.

Answer questions concisely.
"#;

        let template = Template::from_str(content).unwrap();
        assert_eq!(template.name, "test_template");
        assert_eq!(template.version, "1.0");
        assert_eq!(template.signature, "question -> answer");
        assert_eq!(template.format.format_type, FormatType::Json);
        assert!(template.options.strict);
        assert!(template.system_prompt.contains("helpful assistant"));
    }

    #[test]
    fn test_parse_template_with_examples() {
        let content = r#"---
name: qa
format:
  type: json
---
Answer questions.

---examples---

## Example 1

**Input:** What is 2+2?

**Output:**
```json
{"answer": "4"}
```

## Example 2

**Input:** What color is the sky?

**Output:**
```json
{"answer": "blue"}
```
"#;

        let template = Template::from_str(content).unwrap();
        assert_eq!(template.examples.len(), 2);
        assert!(template.examples[0].input.contains("2+2"));
        assert!(template.examples[0].output.contains("4"));
    }

    #[test]
    fn test_parse_template_with_schema() {
        let content = r#"---
name: code_gen
format:
  type: json
  schema:
    type: object
    required:
      - code
      - explanation
    properties:
      code:
        type: string
      explanation:
        type: string
---
Generate code.
"#;

        let template = Template::from_str(content).unwrap();
        let schema = template.format.schema.as_ref().unwrap();
        assert_eq!(schema.required, vec!["code", "explanation"]);
    }

    #[test]
    fn test_validate_json_output() {
        let template = Template::new("test").with_format(FormatType::Json);

        // Valid JSON
        assert!(template.validate_output(r#"{"key": "value"}"#).is_ok());

        // Invalid JSON
        assert!(template.validate_output("not json").is_err());
    }

    #[test]
    fn test_validate_json_with_schema() {
        let content = r#"---
format:
  type: json
  schema:
    type: object
    required:
      - answer
---
Test
"#;

        let template = Template::from_str(content).unwrap();

        // Has required field
        assert!(template.validate_output(r#"{"answer": "yes"}"#).is_ok());

        // Missing required field
        assert!(template.validate_output(r#"{"other": "no"}"#).is_err());
    }

    #[test]
    fn test_assemble_prompt() {
        let content = r#"---
name: qa
format:
  type: json
---
You are a helpful assistant.

## Output Format

Return JSON with an "answer" field.

---examples---

## Example 1

**Input:** What is 1+1?

**Output:**
```json
{"answer": "2"}
```
"#;

        let template = Template::from_str(content).unwrap();
        let prompt = template.assemble_prompt("What is 2+2?", 0, None);

        assert!(prompt.contains("helpful assistant"));
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.contains("Example 1"));
    }

    #[test]
    fn test_assemble_prompt_with_feedback() {
        let template = Template::new("test").with_system_prompt("Answer questions.");

        let prompt = template.assemble_prompt("What is 2+2?", 1, Some("Previous answer was wrong"));

        assert!(prompt.contains("Feedback from previous attempt"));
        assert!(prompt.contains("Previous answer was wrong"));
    }

    #[test]
    fn test_extract_json_from_code_block() {
        let output = r#"Here is the answer:

```json
{"answer": "42"}
```
"#;

        let json = extract_json_from_output(output);
        assert_eq!(json, r#"{"answer": "42"}"#);
    }

    #[test]
    fn test_extract_yaml_from_code_block() {
        let output = r#"Here is the configuration:

```yaml
name: test
version: 1.0
enabled: true
```
"#;

        let yaml = extract_yaml_from_output(output);
        assert_eq!(yaml, "name: test\nversion: 1.0\nenabled: true");
    }

    #[test]
    fn test_extract_yaml_from_yml_code_block() {
        let output = r#"Config file:

```yml
database:
  host: localhost
  port: 5432
```
"#;

        let yaml = extract_yaml_from_output(output);
        assert_eq!(yaml, "database:\n  host: localhost\n  port: 5432");
    }

    #[test]
    fn test_extract_yaml_plain() {
        let output = "name: test\nvalue: 42";
        let yaml = extract_yaml_from_output(output);
        assert_eq!(yaml, "name: test\nvalue: 42");
    }

    #[test]
    fn test_validate_yaml_output() {
        let template = Template::new("test").with_format(FormatType::Yaml);

        // Valid YAML
        assert!(template.validate_output("name: test\nvalue: 42").is_ok());

        // Valid YAML in code block
        let with_block = "```yaml\nname: test\n```";
        assert!(template.validate_output(with_block).is_ok());

        // Invalid YAML
        assert!(template.validate_output("name: [unclosed").is_err());
    }

    #[test]
    fn test_validate_yaml_with_schema() {
        let content = r#"---
name: yaml_test
format:
  type: yaml
  schema:
    type: object
    required:
      - name
      - value
---
Generate YAML output.
"#;

        let template = Template::from_str(content).unwrap();

        // Valid - has all required fields
        let valid = "name: test\nvalue: 42";
        assert!(template.validate_output(valid).is_ok());

        // Invalid - missing required field
        let missing = "name: test";
        assert!(template.validate_output(missing).is_err());
    }

    #[test]
    fn test_parse_yaml_output() {
        #[derive(Debug, serde::Deserialize, PartialEq)]
        struct Config {
            name: String,
            port: u16,
        }

        let template = Template::new("test").with_format(FormatType::Yaml);

        // Plain YAML
        let output = "name: myapp\nport: 8080";
        let config: Config = template.parse_output(output).unwrap();
        assert_eq!(config.name, "myapp");
        assert_eq!(config.port, 8080);

        // YAML in code block
        let with_block = "```yaml\nname: other\nport: 3000\n```";
        let config2: Config = template.parse_output(with_block).unwrap();
        assert_eq!(config2.name, "other");
        assert_eq!(config2.port, 3000);
    }

    #[test]
    fn test_format_type_default() {
        let format = FormatSpec::default();
        assert_eq!(format.format_type, FormatType::Json);
    }

    #[test]
    fn test_template_into_owned() {
        let content = r#"---
name: test
---
Prompt
"#;

        let template = Template::from_str(content).unwrap();
        let owned: Template<'static> = template.into_owned();
        assert_eq!(owned.name, "test");
    }

    #[test]
    fn test_extract_format_instructions() {
        let markdown = r#"
# Main Prompt

Do something.

## Output Format

Return JSON with the following fields:
- answer: string
- confidence: number

## Other Section

More content.
"#;

        let instructions = extract_format_instructions(markdown);
        assert!(instructions.contains("Return JSON"));
        assert!(instructions.contains("answer: string"));
    }

    #[test]
    fn test_parse_output_json() {
        #[derive(Debug, serde::Deserialize, PartialEq)]
        struct Answer {
            answer: String,
        }

        let template = Template::new("test").with_format(FormatType::Json);

        let result: Answer = template.parse_output(r#"{"answer": "42"}"#).unwrap();

        assert_eq!(result.answer, "42");
    }

    #[test]
    fn test_template_example_into_owned() {
        let example = TemplateExample::new("input", "output");
        let owned = example.into_owned();
        assert_eq!(owned.input, "input");
        assert_eq!(owned.output, "output");
    }

    #[test]
    fn test_prompt_tone_modifiers() {
        let inclusive = PromptTone::Inclusive.modifiers();
        assert!(inclusive.requirement_prefix.contains("Consider"));
        assert!(inclusive
            .uncertainty_guidance
            .contains("include rather than exclude"));

        let balanced = PromptTone::Balanced.modifiers();
        assert!(balanced.requirement_prefix.contains("Include"));
        assert!(!balanced.requirement_prefix.contains("MUST"));

        let restrictive = PromptTone::Restrictive.modifiers();
        assert!(restrictive.requirement_prefix.contains("MUST"));
        assert!(restrictive
            .uncertainty_guidance
            .contains("exclude rather than include"));
    }

    #[test]
    fn test_prompt_tone_default_thresholds() {
        assert!((PromptTone::Inclusive.default_threshold() - 0.6).abs() < 0.001);
        assert!((PromptTone::Balanced.default_threshold() - 0.8).abs() < 0.001);
        assert!((PromptTone::Restrictive.default_threshold() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_prompt_tone_favors() {
        assert!(PromptTone::Inclusive.favors_recall());
        assert!(!PromptTone::Inclusive.favors_precision());

        assert!(!PromptTone::Balanced.favors_recall());
        assert!(!PromptTone::Balanced.favors_precision());

        assert!(!PromptTone::Restrictive.favors_recall());
        assert!(PromptTone::Restrictive.favors_precision());
    }

    #[test]
    fn test_template_with_tone() {
        let template = Template::new("test")
            .with_format(FormatType::Json)
            .with_tone(PromptTone::Restrictive);

        assert_eq!(template.options.tone, PromptTone::Restrictive);

        let prompt = template.assemble_prompt("What is 2+2?", 0, None);
        assert!(prompt.contains("EXACTLY"));
        assert!(prompt.contains("exclude rather than include"));
    }

    #[test]
    fn test_template_inclusive_tone_prompt() {
        let template = Template::new("test")
            .with_format(FormatType::Json)
            .with_tone(PromptTone::Inclusive);

        let prompt = template.assemble_prompt("What is 2+2?", 0, None);
        assert!(prompt.contains("suggested format"));
        assert!(prompt.contains("include rather than exclude"));
    }

    #[test]
    fn test_prompt_tone_default() {
        assert_eq!(PromptTone::default(), PromptTone::Balanced);
    }

    #[test]
    fn test_template_options_default_tone() {
        let options = TemplateOptions::default();
        assert_eq!(options.tone, PromptTone::Balanced);
    }
}
