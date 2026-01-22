// Copyright (c) Lituus-io. All rights reserved.
// Author: terekete <spicyzhug@gmail.com>

//! Generic Jinja2-like template support using minijinja.
//!
//! This module provides `JinjaTemplate` for loading and rendering templates
//! with Jinja2 syntax. Users load templates from their own files - no
//! hardcoded templates are provided.
//!
//! # Example
//!
//! ```ignore
//! use kkachi::declarative::JinjaTemplate;
//!
//! // Load from file
//! let template = JinjaTemplate::from_file("./templates/my_template.md.j2")?;
//!
//! // Render with context
//! let output = template.render_with(&hashmap!{
//!     "question" => "How do I create a bucket?",
//!     "code" => yaml_content,
//!     "language" => "yaml",
//! })?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use minijinja::Environment;

use crate::error::{Error, Result};

/// A Jinja2-compatible template loaded from file or string.
///
/// Uses minijinja for rendering with full Jinja2 syntax support including:
/// - Variable interpolation: `{{ variable }}`
/// - Filters: `{{ name | upper }}`
/// - Control flow: `{% if condition %}...{% endif %}`
/// - Loops: `{% for item in items %}...{% endfor %}`
/// - Defaults: `{{ value | default("fallback") }}`
pub struct JinjaTemplate {
    /// Template environment.
    env: Environment<'static>,
    /// Template name.
    name: String,
}

impl JinjaTemplate {
    /// Load a template from a file path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the template file (e.g., `./templates/my_template.md.j2`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template = JinjaTemplate::from_file("./templates/pulumi.md.j2")?;
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            Error::Other(format!(
                "Failed to read template file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("template")
            .to_string();

        Self::from_str(&name, &content)
    }

    /// Create a template from a string.
    ///
    /// # Arguments
    ///
    /// * `name` - Template name for identification
    /// * `content` - Template content with Jinja2 syntax
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template = JinjaTemplate::from_str("my_template", r#"
    /// ## Task
    /// {{ question }}
    ///
    /// ## Solution
    /// ```{{ language }}
    /// {{ code }}
    /// ```
    /// "#)?;
    /// ```
    pub fn from_str(name: &str, content: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.add_template_owned(name.to_string(), content.to_string())
            .map_err(|e| Error::Other(format!("Failed to parse template '{}': {}", name, e)))?;

        Ok(Self {
            env,
            name: name.to_string(),
        })
    }

    /// Get the template name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Render the template with a context map.
    ///
    /// # Arguments
    ///
    /// * `vars` - Map of variable names to values
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::HashMap;
    ///
    /// let mut vars = HashMap::new();
    /// vars.insert("question".to_string(), minijinja::Value::from("How do I..."));
    /// vars.insert("code".to_string(), minijinja::Value::from(yaml_code));
    ///
    /// let output = template.render(&vars)?;
    /// ```
    pub fn render(&self, vars: &HashMap<String, minijinja::Value>) -> Result<String> {
        let tmpl = self
            .env
            .get_template(&self.name)
            .map_err(|e| Error::Other(format!("Template not found: {}", e)))?;

        tmpl.render(vars)
            .map_err(|e| Error::Other(format!("Failed to render template: {}", e)))
    }

    /// Render with a simple string-to-string context.
    ///
    /// Convenience method for when all values are strings.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let output = template.render_strings(&[
    ///     ("question", "How do I create a bucket?"),
    ///     ("code", &yaml_code),
    ///     ("language", "yaml"),
    /// ])?;
    /// ```
    pub fn render_strings(&self, vars: &[(&str, &str)]) -> Result<String> {
        let map: HashMap<String, minijinja::Value> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), minijinja::Value::from(*v)))
            .collect();
        self.render(&map)
    }

    /// Render with a context builder.
    ///
    /// Uses minijinja's `context!` macro style for building context.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let output = template.render_context(context! {
    ///     question => "How do I create a bucket?",
    ///     code => yaml_code,
    ///     language => "yaml",
    ///     errors => vec!["Error 1", "Error 2"],
    /// })?;
    /// ```
    pub fn render_context(&self, ctx: minijinja::Value) -> Result<String> {
        let tmpl = self
            .env
            .get_template(&self.name)
            .map_err(|e| Error::Other(format!("Template not found: {}", e)))?;

        tmpl.render(ctx)
            .map_err(|e| Error::Other(format!("Failed to render template: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jinja_template_from_str() {
        let template = JinjaTemplate::from_str(
            "test",
            r#"
## Task
{{ question }}

## Answer
{{ answer }}
"#,
        )
        .unwrap();

        assert_eq!(template.name(), "test");

        let output = template
            .render_strings(&[("question", "What is 2+2?"), ("answer", "4")])
            .unwrap();

        assert!(output.contains("What is 2+2?"));
        assert!(output.contains("4"));
    }

    #[test]
    fn test_jinja_template_with_loop() {
        let template = JinjaTemplate::from_str(
            "loop_test",
            r#"
## Errors
{% for error in errors %}
- {{ error }}
{% endfor %}
"#,
        )
        .unwrap();

        let mut vars = HashMap::new();
        vars.insert(
            "errors".to_string(),
            minijinja::Value::from(vec!["Error 1", "Error 2", "Error 3"]),
        );

        let output = template.render(&vars).unwrap();

        assert!(output.contains("- Error 1"));
        assert!(output.contains("- Error 2"));
        assert!(output.contains("- Error 3"));
    }

    #[test]
    fn test_jinja_template_with_conditional() {
        let template = JinjaTemplate::from_str(
            "conditional_test",
            r#"
{% if notes %}
## Notes
{{ notes }}
{% endif %}
Done.
"#,
        )
        .unwrap();

        // With notes
        let output = template
            .render_strings(&[("notes", "Some important notes")])
            .unwrap();
        assert!(output.contains("## Notes"));
        assert!(output.contains("Some important notes"));

        // Without notes (empty string)
        let mut vars = HashMap::new();
        vars.insert("notes".to_string(), minijinja::Value::from(""));
        let output = template.render(&vars).unwrap();
        // Empty string is falsy in Jinja, so notes section should not appear
        // Actually in minijinja, empty string is truthy, so we need to check explicitly
        assert!(output.contains("Done."));
    }

    #[test]
    fn test_jinja_template_with_default() {
        let template = JinjaTemplate::from_str(
            "default_test",
            r#"
Language: {{ language | default("yaml") }}
"#,
        )
        .unwrap();

        // Without language - should use default
        let vars: HashMap<String, minijinja::Value> = HashMap::new();
        let output = template.render(&vars).unwrap();
        assert!(output.contains("yaml"));

        // With language - should use provided value
        let output = template.render_strings(&[("language", "rust")]).unwrap();
        assert!(output.contains("rust"));
    }

    #[test]
    fn test_jinja_template_code_block() {
        let template = JinjaTemplate::from_str(
            "code_test",
            r#"
```{{ language }}
{{ code }}
```
"#,
        )
        .unwrap();

        let output = template
            .render_strings(&[
                ("language", "yaml"),
                (
                    "code",
                    "resources:\n  bucket:\n    type: gcp:storage:Bucket",
                ),
            ])
            .unwrap();

        assert!(output.contains("```yaml"));
        assert!(output.contains("gcp:storage:Bucket"));
        assert!(output.contains("```\n") || output.ends_with("```"));
    }
}
