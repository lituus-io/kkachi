// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! YAML Configuration Pipeline Example
//!
//! This example demonstrates:
//! 1. Loading YAML-format templates
//! 2. YAML extraction from markdown code blocks
//! 3. Schema validation for YAML output
//! 4. Parsing YAML into typed Rust structs
//!
//! Run with: cargo run --example yaml_config_pipeline

use std::collections::HashMap;

use kkachi::error::Result;
use kkachi::recursive::{
    Template,
    Critic, CriticResult,
    StandaloneRunner, RecursiveConfig, RecursiveState,
};
use kkachi::str_view::StrView;

// ============================================================================
// Typed Configuration Structures
// ============================================================================

#[derive(Debug, Clone, serde::Deserialize)]
struct ServerConfig {
    name: String,
    settings: ServerSettings,
    #[serde(default)]
    enabled: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ServerSettings {
    host: String,
    port: u16,
    #[serde(default)]
    workers: Option<u32>,
    #[serde(default)]
    tls: Option<TlsConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TlsConfig {
    cert_path: String,
    key_path: String,
}

// ============================================================================
// YAML Template (inline)
// ============================================================================

const YAML_CONFIG_TEMPLATE: &str = r#"---
name: server_config_generator
version: "1.0"
signature: "requirements -> config"
format:
  type: yaml
  schema:
    type: object
    required:
      - name
      - settings
options:
  strict: true
  include_in_prompt: true
---

You are a DevOps expert. Generate server configuration in YAML format.

## Output Format

Return valid YAML with:
- `name`: Configuration identifier
- `settings`: Server settings (host, port, workers, tls)
- `enabled`: Whether the server is enabled (default: true)

## Best Practices

1. Use sensible defaults for optional fields
2. Include TLS configuration for production
3. Set appropriate worker counts based on workload

---examples---

## Example 1

**Input:** Create a development server config

**Output:**
```yaml
name: dev_server
settings:
  host: localhost
  port: 3000
  workers: 2
enabled: true
```

## Example 2

**Input:** Create a production web server with TLS

**Output:**
```yaml
name: prod_web_server
settings:
  host: 0.0.0.0
  port: 443
  workers: 8
  tls:
    cert_path: /etc/ssl/certs/server.crt
    key_path: /etc/ssl/private/server.key
enabled: true
```
"#;

// ============================================================================
// YAML Critic
// ============================================================================

struct YamlConfigCritic<'t> {
    template: &'t Template<'t>,
}

impl<'t> YamlConfigCritic<'t> {
    fn new(template: &'t Template<'t>) -> Self {
        Self { template }
    }
}

impl<'t> Critic for YamlConfigCritic<'t> {
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        let text = output.as_str();
        let mut score: f64 = 0.0;
        let mut issues = Vec::new();
        let mut breakdown = Vec::new();

        // 1. Format validation (0.3)
        let format_score = match self.template.validate_output(text) {
            Ok(()) => 1.0,
            Err(e) => {
                issues.push(format!("Format error: {}", e));
                0.0
            }
        };
        breakdown.push(("format".to_string(), format_score));
        score += format_score * 0.3;

        // 2. Parse into typed struct (0.3)
        let parse_score = match self.template.parse_output::<ServerConfig>(text) {
            Ok(config) => {
                // Validate the parsed config
                if config.settings.port == 0 {
                    issues.push("Invalid port: 0".to_string());
                    0.5
                } else if config.settings.host.is_empty() {
                    issues.push("Empty host".to_string());
                    0.5
                } else {
                    1.0
                }
            }
            Err(e) => {
                issues.push(format!("Parse error: {}", e));
                0.0
            }
        };
        breakdown.push(("parsing".to_string(), parse_score));
        score += parse_score * 0.3;

        // 3. Best practices (0.4)
        let mut practices_score: f64 = 1.0;

        // Check for production-ready settings
        if text.contains("port: 443") || text.contains("port: 8443") {
            if !text.contains("tls:") {
                practices_score -= 0.3;
                issues.push("HTTPS port without TLS config".to_string());
            }
        }

        // Check for reasonable worker count
        if text.contains("workers:") {
            if text.contains("workers: 0") {
                practices_score -= 0.2;
                issues.push("Zero workers configured".to_string());
            }
        }

        // Bonus for complete configs
        if text.contains("tls:") && text.contains("workers:") {
            practices_score = (practices_score + 0.1).min(1.0);
        }

        breakdown.push(("best_practices".to_string(), practices_score));
        score += practices_score * 0.4;

        let mut result = CriticResult::new(score).with_breakdown(breakdown);
        if !issues.is_empty() {
            result = result.with_feedback(issues.join("\n"));
        }
        result
    }
}

// ============================================================================
// Mock LLM
// ============================================================================

struct MockConfigLLM {
    responses: HashMap<u32, String>,
}

impl MockConfigLLM {
    fn new(requirement: &str) -> Self {
        let mut responses = HashMap::new();

        if requirement.contains("production") || requirement.contains("TLS") {
            // Production config progression
            responses.insert(0, r#"Here is the config:

```yaml
name: server
settings:
  host: 0.0.0.0
  port: 443
```
"#.to_string());

            responses.insert(1, r#"Updated config with TLS:

```yml
name: prod_server
settings:
  host: 0.0.0.0
  port: 443
  workers: 4
  tls:
    cert_path: /etc/ssl/certs/server.crt
    key_path: /etc/ssl/private/server.key
enabled: true
```
"#.to_string());
        } else {
            // Development config
            responses.insert(0, r#"```yaml
name: dev_server
settings:
  host: localhost
  port: 3000
  workers: 2
enabled: true
```
"#.to_string());
        }

        Self { responses }
    }

    fn generate(&self, iteration: u32) -> String {
        self.responses
            .get(&iteration)
            .or_else(|| self.responses.values().last())
            .cloned()
            .unwrap_or_else(|| "name: error\nsettings:\n  host: localhost\n  port: 8080".to_string())
    }
}

// ============================================================================
// Main Pipeline
// ============================================================================

fn run_pipeline() -> Result<()> {
    println!("================================================================");
    println!("   YAML Configuration Pipeline");
    println!("================================================================\n");

    // Load template
    println!("[Step 1] Loading YAML template...");
    let template = Template::from_str(YAML_CONFIG_TEMPLATE)?;
    println!("  Name: {}", template.name);
    println!("  Format: {:?}", template.format.format_type);
    println!("  Examples: {}\n", template.examples.len());

    // Define requirements
    let requirements = vec![
        "Create a development server config for local testing",
        "Create a production web server with TLS on port 443",
    ];

    for (i, requirement) in requirements.iter().enumerate() {
        println!("================================================================");
        println!("  Scenario {}: {}", i + 1, requirement);
        println!("================================================================\n");

        // Set up critic
        let critic = YamlConfigCritic::new(&template);
        let config = RecursiveConfig {
            max_iterations: 3,
            score_threshold: 0.8,
            ..Default::default()
        };

        let runner = StandaloneRunner::with_config(&critic, "yaml_config", config);
        let mock_llm = MockConfigLLM::new(requirement);

        // Run refinement
        let result = runner.refine(requirement, |iteration, feedback| {
            println!("  --- Iteration {} ---", iteration);
            if let Some(fb) = feedback {
                println!("  Feedback: {}", fb);
            }

            let response = mock_llm.generate(iteration);
            println!("  Generated {} bytes", response.len());
            Ok(response)
        })?;

        println!("\n  Results:");
        println!("    Score: {:.2}", result.score);
        println!("    Iterations: {}", result.iterations);

        // Parse final config
        match template.parse_output::<ServerConfig>(&result.answer) {
            Ok(config) => {
                println!("\n  Parsed Configuration:");
                println!("    Name: {}", config.name);
                println!("    Host: {}", config.settings.host);
                println!("    Port: {}", config.settings.port);
                if let Some(workers) = config.settings.workers {
                    println!("    Workers: {}", workers);
                }
                if let Some(tls) = &config.settings.tls {
                    println!("    TLS Cert: {}", tls.cert_path);
                }
                println!("    Enabled: {}", config.enabled);
            }
            Err(e) => {
                println!("\n  Failed to parse config: {}", e);
            }
        }
        println!();
    }

    println!("================================================================");
    println!("  Pipeline Complete!");
    println!("================================================================\n");

    Ok(())
}

fn main() {
    if let Err(e) = run_pipeline() {
        eprintln!("Pipeline error: {}", e);
        std::process::exit(1);
    }
}
