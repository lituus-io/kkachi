// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for the template system.

use kkachi::recursive::{FormatType, Template};
use std::path::PathBuf;

fn test_templates_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples")
        .join("templates")
}

#[test]
fn test_load_rust_code_generator_template() {
    let path = test_templates_dir().join("rust_code_generator.md");

    // Skip if template doesn't exist (e.g., in CI without examples)
    if !path.exists() {
        println!("Skipping test: template file not found at {:?}", path);
        return;
    }

    let template = Template::from_file(&path).expect("Failed to load template");

    assert_eq!(template.name, "rust_code_generator");
    assert_eq!(template.version, "1.0");
    assert_eq!(template.signature, "requirement -> code");
    assert!(matches!(template.format.format_type, FormatType::Json));
    assert!(!template.examples.is_empty());
    assert!(template.system_prompt.contains("Rust"));
}

#[test]
fn test_load_api_documentation_template() {
    let path = test_templates_dir().join("api_documentation.md");

    if !path.exists() {
        println!("Skipping test: template file not found at {:?}", path);
        return;
    }

    let template = Template::from_file(&path).expect("Failed to load template");

    assert_eq!(template.name, "api_documentation");
    assert_eq!(template.version, "1.0");
    assert_eq!(template.signature, "code -> documentation");
    assert!(matches!(template.format.format_type, FormatType::Json));
}

#[test]
fn test_template_assemble_prompt_from_file() {
    let path = test_templates_dir().join("rust_code_generator.md");

    if !path.exists() {
        return;
    }

    let template = Template::from_file(&path).unwrap();
    let prompt = template.assemble_prompt("Write a config parser", 0, None);

    // Should include system prompt
    assert!(prompt.contains("Rust"));
    // Should include the question
    assert!(prompt.contains("Write a config parser"));
    // Should include examples
    assert!(prompt.contains("Example"));
}

#[test]
fn test_template_validate_output_from_file() {
    let path = test_templates_dir().join("rust_code_generator.md");

    if !path.exists() {
        return;
    }

    let template = Template::from_file(&path).unwrap();

    // Valid JSON output
    let valid_output = r#"{"code": "fn main() {}", "explanation": "Hello world", "tests": ""}"#;
    assert!(template.validate_output(valid_output).is_ok());

    // Invalid output (not JSON)
    let invalid_output = "This is not JSON";
    assert!(template.validate_output(invalid_output).is_err());
}

#[test]
fn test_template_with_json_in_code_block() {
    let path = test_templates_dir().join("rust_code_generator.md");

    if !path.exists() {
        return;
    }

    let template = Template::from_file(&path).unwrap();

    // JSON wrapped in code block (common LLM output)
    let output_with_block = r#"
Here is the solution:

```json
{"code": "fn hello() { println!(\"Hello\"); }", "explanation": "Simple function", "tests": ""}
```
"#;

    assert!(template.validate_output(output_with_block).is_ok());
}

#[test]
fn test_load_yaml_config_generator_template() {
    let path = test_templates_dir().join("config_generator.md");

    if !path.exists() {
        println!("Skipping test: YAML template file not found at {:?}", path);
        return;
    }

    let template = Template::from_file(&path).expect("Failed to load YAML template");

    assert_eq!(template.name, "config_generator");
    assert_eq!(template.version, "1.0");
    assert_eq!(template.signature, "requirements -> config");
    assert!(matches!(template.format.format_type, FormatType::Yaml));
    assert!(!template.examples.is_empty());
    assert!(template.system_prompt.contains("configuration"));
}

#[test]
fn test_yaml_template_validate_output_from_file() {
    let path = test_templates_dir().join("config_generator.md");

    if !path.exists() {
        return;
    }

    let template = Template::from_file(&path).unwrap();

    // Valid YAML output
    let valid_output = "name: test_config\nsettings:\n  key: value";
    assert!(template.validate_output(valid_output).is_ok());

    // Valid YAML in code block
    let yaml_in_block = "```yaml\nname: test_config\nsettings:\n  key: value\n```";
    assert!(template.validate_output(yaml_in_block).is_ok());

    // Invalid output (missing required field)
    let missing_field = "name: test_config";
    assert!(template.validate_output(missing_field).is_err());
}

#[test]
fn test_yaml_template_assemble_prompt() {
    let path = test_templates_dir().join("config_generator.md");

    if !path.exists() {
        return;
    }

    let template = Template::from_file(&path).unwrap();
    let prompt = template.assemble_prompt("Create a web server config", 0, None);

    // Should include format instructions for YAML
    assert!(prompt.contains("YAML") || prompt.contains("yaml"));
    // Should include the question
    assert!(prompt.contains("web server config"));
    // Should include examples
    assert!(prompt.contains("Example"));
}

#[test]
fn test_yaml_extraction_with_yml_extension() {
    let path = test_templates_dir().join("config_generator.md");

    if !path.exists() {
        return;
    }

    let template = Template::from_file(&path).unwrap();

    // YAML wrapped in ```yml code block (alternative extension)
    let output_with_yml = r#"
Here is your configuration:

```yml
name: server_config
settings:
  host: 0.0.0.0
  port: 8080
```
"#;

    assert!(template.validate_output(output_with_yml).is_ok());
}
