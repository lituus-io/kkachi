// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Python bindings for Jinja2 template support.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use kkachi::declarative::{JinjaFormatter, JinjaTemplate};
use minijinja;

// ============================================================================
// Value Conversion: minijinja::Value ↔ PyObject
// ============================================================================

/// Convert Python object to minijinja::Value
fn pyobject_to_jinja_value(obj: &Bound<'_, PyAny>) -> PyResult<minijinja::Value> {
    if obj.is_none() {
        Ok(minijinja::Value::from(()))
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(minijinja::Value::from(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(minijinja::Value::from(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(minijinja::Value::from(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(minijinja::Value::from(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let mut vec = Vec::new();
        for item in list.iter() {
            vec.push(pyobject_to_jinja_value(&item)?);
        }
        Ok(minijinja::Value::from(vec))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = std::collections::HashMap::new();
        for (key, value) in dict.iter() {
            let key_str = key.extract::<String>()?;
            map.insert(key_str, pyobject_to_jinja_value(&value)?);
        }
        Ok(minijinja::Value::from_serialize(&map))
    } else {
        Err(PyRuntimeError::new_err(format!(
            "Unsupported type for Jinja context: {}",
            obj.get_type().name()?
        )))
    }
}

/// Convert dict to HashMap<String, minijinja::Value>
fn pydict_to_context(
    dict: &Bound<'_, PyDict>,
) -> PyResult<std::collections::HashMap<String, minijinja::Value>> {
    let mut map = std::collections::HashMap::new();
    for (key, value) in dict.iter() {
        let key_str = key.extract::<String>()?;
        map.insert(key_str, pyobject_to_jinja_value(&value)?);
    }
    Ok(map)
}

// ============================================================================
// PyJinjaTemplate — Standalone template rendering
// ============================================================================

/// A Jinja2-compatible template for dynamic prompt generation.
///
/// Example:
/// ```python
/// from kkachi import JinjaTemplate
///
/// # Load from file
/// template = JinjaTemplate.from_file("./templates/prompt.j2")
///
/// # Or create from string
/// template = JinjaTemplate.from_str("code_gen", '''
/// ## Task
/// {{ task }}
///
/// {% if examples %}
/// ## Examples
/// {% for ex in examples %}
/// - {{ ex }}
/// {% endfor %}
/// {% endif %}
/// ''')
///
/// # Render with context
/// output = template.render({
///     "task": "Write a parser",
///     "examples": ["Example 1", "Example 2"]
/// })
/// ```
#[pyclass(name = "JinjaTemplate")]
pub struct PyJinjaTemplate {
    inner: JinjaTemplate,
}

#[pymethods]
impl PyJinjaTemplate {
    /// Load a template from a file.
    ///
    /// Args:
    ///     path (str): Path to template file (e.g., "./templates/prompt.j2")
    ///
    /// Returns:
    ///     JinjaTemplate: Loaded template
    ///
    /// Raises:
    ///     RuntimeError: If file cannot be read or template is invalid
    #[staticmethod]
    fn from_file(path: String) -> PyResult<Self> {
        let template = JinjaTemplate::from_file(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load template: {}", e)))?;
        Ok(Self { inner: template })
    }

    /// Create a template from a string.
    ///
    /// Args:
    ///     name (str): Template name for identification
    ///     content (str): Jinja2 template content
    ///
    /// Returns:
    ///     JinjaTemplate: Created template
    ///
    /// Raises:
    ///     RuntimeError: If template syntax is invalid
    #[staticmethod]
    fn from_str(name: String, content: String) -> PyResult<Self> {
        let template = JinjaTemplate::from_str(&name, &content)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse template: {}", e)))?;
        Ok(Self { inner: template })
    }

    /// Get the template name.
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Render the template with a context dictionary.
    ///
    /// Supports nested structures (dicts, lists, primitives).
    ///
    /// Args:
    ///     context (dict): Variables for template rendering
    ///
    /// Returns:
    ///     str: Rendered output
    ///
    /// Example:
    ///     ```python
    ///     output = template.render({
    ///         "name": "Alice",
    ///         "items": ["item1", "item2"],
    ///         "config": {"debug": True}
    ///     })
    ///     ```
    fn render(&self, context: &Bound<'_, PyDict>) -> PyResult<String> {
        let vars = pydict_to_context(context)?;
        self.inner
            .render(&vars)
            .map_err(|e| PyRuntimeError::new_err(format!("Template render error: {}", e)))
    }

    /// Render with simple string-to-string mappings (convenience method).
    ///
    /// Args:
    ///     **kwargs: Keyword arguments as template variables
    ///
    /// Returns:
    ///     str: Rendered output
    ///
    /// Example:
    ///     ```python
    ///     output = template.render_strings(task="Write code", language="Rust")
    ///     ```
    #[pyo3(signature = (**kwargs))]
    fn render_strings(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        let context = kwargs.unwrap();
        let pairs: Vec<(String, String)> = context
            .iter()
            .map(|(k, v)| {
                let key = k.extract::<String>()?;
                let val = v.extract::<String>()?;
                Ok((key, val))
            })
            .collect::<PyResult<Vec<_>>>()?;

        let pairs_ref: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        self.inner
            .render_strings(&pairs_ref)
            .map_err(|e| PyRuntimeError::new_err(format!("Template render error: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!("JinjaTemplate(name='{}')", self.inner.name())
    }
}

// ============================================================================
// PyJinjaFormatter — Formatter for refinement loops
// ============================================================================

/// A prompt formatter using Jinja2 templates for refinement loops.
///
/// The formatter receives three variables at each iteration:
/// - `task`: The original prompt/task
/// - `feedback`: Feedback from previous iteration (empty string if none)
/// - `iteration`: Current iteration number (0-indexed)
///
/// Example:
/// ```python
/// from kkachi import JinjaTemplate, JinjaFormatter, refine, Checks
///
/// template = JinjaTemplate.from_str("refine", '''
/// ## Task
/// {{ task }}
///
/// {% if feedback %}
/// ## Feedback from Previous Attempt
/// {{ feedback }}
/// {% endif %}
///
/// ## Iteration
/// This is attempt #{{ iteration + 1 }}
/// ''')
///
/// formatter = JinjaFormatter(template)
///
/// result = refine(llm, "Write a parser") \
///     .with_formatter(formatter) \
///     .validate(Checks().require("fn ")) \
///     .go()
/// ```
#[pyclass(name = "JinjaFormatter")]
#[derive(Clone)]
pub struct PyJinjaFormatter {
    inner: JinjaFormatter,
}

#[pymethods]
impl PyJinjaFormatter {
    /// Create a formatter from a template.
    ///
    /// Args:
    ///     template (JinjaTemplate): The template to use for formatting
    ///
    /// Returns:
    ///     JinjaFormatter: Formatter instance
    #[new]
    fn new(template: &PyJinjaTemplate) -> Self {
        Self {
            inner: JinjaFormatter::new(template.inner.clone()),
        }
    }

    fn __repr__(&self) -> String {
        "JinjaFormatter(...)".to_string()
    }
}

// Internal trait implementation for use in builders
impl PyJinjaFormatter {
    pub(crate) fn into_inner(self) -> JinjaFormatter {
        self.inner
    }
}
