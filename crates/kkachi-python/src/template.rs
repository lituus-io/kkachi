// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for the template system.
//!
//! Provides structured prompt engineering with format specs, tone control,
//! few-shot examples, and output validation.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use kkachi::recursive::{FormatType, PromptTone, Template, TemplateExample};

// ============================================================================
// PyFormatType — Output format enum
// ============================================================================

/// Output format type for templates.
#[pyclass(name = "FormatType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyFormatType {
    /// JSON format with optional schema validation.
    Json = 0,
    /// YAML format with optional schema validation.
    Yaml = 1,
    /// Markdown format.
    Markdown = 2,
    /// XML format.
    Xml = 3,
    /// Plain text (no format enforcement).
    Plain = 4,
}

impl From<PyFormatType> for FormatType {
    fn from(py_fmt: PyFormatType) -> Self {
        match py_fmt {
            PyFormatType::Json => FormatType::Json,
            PyFormatType::Yaml => FormatType::Yaml,
            PyFormatType::Markdown => FormatType::Markdown,
            PyFormatType::Xml => FormatType::Xml,
            PyFormatType::Plain => FormatType::Plain,
        }
    }
}

impl From<FormatType> for PyFormatType {
    fn from(fmt: FormatType) -> Self {
        match fmt {
            FormatType::Json => PyFormatType::Json,
            FormatType::Yaml => PyFormatType::Yaml,
            FormatType::Markdown => PyFormatType::Markdown,
            FormatType::Xml => PyFormatType::Xml,
            FormatType::Plain => PyFormatType::Plain,
        }
    }
}

// ============================================================================
// PyPromptTone — Tone control enum
// ============================================================================

/// Prompt tone controlling language strictness.
///
/// Example:
/// ```python
/// tone = PromptTone.RESTRICTIVE
/// print(tone.default_threshold())   # 0.9
/// print(tone.favors_precision())    # True
/// ```
#[pyclass(name = "PromptTone", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyPromptTone {
    /// Inclusive — encourages exploration, tolerates variation.
    Inclusive = 0,
    /// Balanced — neutral instructions (default).
    Balanced = 1,
    /// Restrictive — strict requirements, precise outputs.
    Restrictive = 2,
}

#[pymethods]
impl PyPromptTone {
    /// Get the default validation threshold for this tone.
    fn default_threshold(&self) -> f64 {
        self.to_rust().default_threshold()
    }

    /// Check if this tone favors recall over precision.
    fn favors_recall(&self) -> bool {
        self.to_rust().favors_recall()
    }

    /// Check if this tone favors precision over recall.
    fn favors_precision(&self) -> bool {
        self.to_rust().favors_precision()
    }
}

impl PyPromptTone {
    fn to_rust(self) -> PromptTone {
        match self {
            Self::Inclusive => PromptTone::Inclusive,
            Self::Balanced => PromptTone::Balanced,
            Self::Restrictive => PromptTone::Restrictive,
        }
    }
}

impl From<PyPromptTone> for PromptTone {
    fn from(tone: PyPromptTone) -> Self {
        tone.to_rust()
    }
}

impl From<PromptTone> for PyPromptTone {
    fn from(tone: PromptTone) -> Self {
        match tone {
            PromptTone::Inclusive => PyPromptTone::Inclusive,
            PromptTone::Balanced => PyPromptTone::Balanced,
            PromptTone::Restrictive => PyPromptTone::Restrictive,
        }
    }
}

// ============================================================================
// PyTemplate — Python-facing template class
// ============================================================================

/// A template for structured prompt optimization.
///
/// Example:
/// ```python
/// template = Template("code_gen") \
///     .system_prompt("You are an expert Rust programmer.") \
///     .format(FormatType.JSON) \
///     .tone(PromptTone.RESTRICTIVE) \
///     .strict(True) \
///     .example("Write hello world", '{"code": "println!(\"Hello\")"}')
///
/// prompt = template.render("Write a URL parser")
/// template.validate_output('{"code": "fn parse() {}"}')
/// ```
#[pyclass(name = "Template")]
#[derive(Clone)]
pub struct PyTemplate {
    inner: Template<'static>,
}

#[pymethods]
impl PyTemplate {
    /// Create a new template with the given name.
    #[new]
    fn new(name: String) -> Self {
        Self {
            inner: Template::new(name).into_owned(),
        }
    }

    /// Create a simple template with just a system prompt.
    #[staticmethod]
    fn simple(prompt: String) -> Self {
        Self {
            inner: Template::simple(prompt).into_owned(),
        }
    }

    /// Parse a template from a YAML frontmatter + markdown string.
    #[staticmethod]
    fn from_str(content: String) -> PyResult<Self> {
        // parse_owned handles lifetime correctly
        let template = Template::from_str(&content)
            .map_err(|e| PyRuntimeError::new_err(format!("Template parse error: {}", e)))?;
        Ok(Self {
            inner: template.into_owned(),
        })
    }

    /// Load a template from a file.
    #[staticmethod]
    fn from_file(path: String) -> PyResult<Self> {
        let template = Template::from_file(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Template file error: {}", e)))?;
        Ok(Self { inner: template })
    }

    /// Set the system prompt.
    fn system_prompt(&self, prompt: String) -> Self {
        Self {
            inner: self.inner.clone().with_system_prompt(prompt),
        }
    }

    /// Set the output format type.
    fn format(&self, format_type: PyFormatType) -> Self {
        Self {
            inner: self.inner.clone().with_format(format_type.into()),
        }
    }

    /// Set the prompt tone.
    fn tone(&self, tone: PyPromptTone) -> Self {
        Self {
            inner: self.inner.clone().with_tone(tone.into()),
        }
    }

    /// Set strict mode (fail on format mismatch).
    fn strict(&self, strict: bool) -> Self {
        Self {
            inner: self.inner.clone().strict(strict),
        }
    }

    /// Add a few-shot example.
    fn example(&self, input: String, output: String) -> Self {
        let example = TemplateExample::new(input, output).into_owned();
        Self {
            inner: self.inner.clone().with_example(example),
        }
    }

    /// Render the template with the given input.
    fn render(&self, input: String) -> String {
        self.inner.render(&input)
    }

    /// Assemble the full prompt for an iteration.
    #[pyo3(signature = (question, iteration=0, feedback=None))]
    fn assemble_prompt(
        &self,
        question: String,
        iteration: u32,
        feedback: Option<String>,
    ) -> String {
        self.inner
            .assemble_prompt(&question, iteration, feedback.as_deref())
    }

    /// Validate output against the format specification.
    ///
    /// Raises RuntimeError if the output doesn't match the expected format.
    fn validate_output(&self, output: String) -> PyResult<()> {
        self.inner
            .validate_output(&output)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }

    /// Parse JSON output into a Python dict.
    fn parse_json(&self, output: String, py: Python<'_>) -> PyResult<PyObject> {
        let value: serde_json::Value = serde_json::from_str(extract_json_content(&output))
            .map_err(|e| PyRuntimeError::new_err(format!("JSON parse error: {}", e)))?;

        json_to_pyobject(py, &value)
    }

    /// Get format instructions string.
    fn get_format_instructions(&self) -> String {
        self.inner.get_format_instructions().to_string()
    }

    /// Get the template name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.to_string()
    }

    /// Get the template signature.
    #[getter]
    fn signature(&self) -> String {
        self.inner.signature.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Template(name='{}', format={:?}, tone={:?})",
            self.inner.name, self.inner.format.format_type, self.inner.options.tone,
        )
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Extract JSON content from a string (handles markdown code blocks).
fn extract_json_content(output: &str) -> &str {
    let trimmed = output.trim();

    if let Some(start) = trimmed.find("```json") {
        let json_start = start + 7;
        if let Some(end) = trimmed[json_start..].find("```") {
            return trimmed[json_start..json_start + end].trim();
        }
    }

    if let Some(start) = trimmed.find("```") {
        let json_start = start + 3;
        let content_start = trimmed[json_start..]
            .find('\n')
            .map(|i| json_start + i + 1)
            .unwrap_or(json_start);
        if let Some(end) = trimmed[content_start..].find("```") {
            return trimmed[content_start..content_start + end].trim();
        }
    }

    trimmed
}

/// Convert a serde_json::Value to a PyObject.
fn json_to_pyobject(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    use pyo3::IntoPy;

    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_py(py)),
        serde_json::Value::Array(arr) => {
            let list = pyo3::types::PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_pyobject(py, item)?)?;
            }
            Ok(list.into_py(py))
        }
        serde_json::Value::Object(map) => {
            let dict = pyo3::types::PyDict::new_bound(py);
            for (key, val) in map {
                dict.set_item(key, json_to_pyobject(py, val)?)?;
            }
            Ok(dict.into_py(py))
        }
    }
}
