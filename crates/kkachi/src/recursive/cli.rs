// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Generic CLI validation primitives.
//!
//! This module provides abstract primitives for CLI-based validation.
//! Users compose their own validators using these building blocks.
//!
//! # Example
//!
//! ```rust,ignore
//! use kkachi::declarative::{Cli, CliPipeline};
//!
//! // Single command validation
//! let validator = Cli::new("rustfmt")
//!     .args(["--check"])
//!     .file_ext("rs");
//!
//! // Multi-stage validation pipeline
//! let pipeline = CliPipeline::new()
//!     .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
//!     .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
//!     .file_ext("rs");
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use smallvec::SmallVec;

use crate::error::{Error, Result};
use crate::str_view::StrView;

use super::critic::{Critic, CriticResult};
use super::state::RecursiveState;

// =============================================================================
// Command Execution
// =============================================================================

/// Result of executing a CLI command.
#[derive(Debug, Clone)]
pub struct CommandResult {
    /// Whether the command succeeded (exit code 0).
    pub success: bool,
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Exit code.
    pub exit_code: i32,
    /// Execution duration.
    pub duration: Duration,
}

/// CLI command executor with timeout support.
pub struct CliExecutor {
    /// Command timeout.
    timeout: Duration,
    /// Working directory.
    working_dir: Option<PathBuf>,
    /// Environment variables to set.
    env_vars: Vec<(String, String)>,
}

impl Default for CliExecutor {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            working_dir: None,
            env_vars: Vec::new(),
        }
    }
}

impl CliExecutor {
    /// Create a new CLI executor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set working directory.
    pub fn with_working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set environment variables.
    pub fn with_envs(mut self, vars: Vec<(String, String)>) -> Self {
        self.env_vars = vars;
        self
    }

    /// Execute a command.
    pub fn execute(&self, cmd: &str, args: &[&str]) -> Result<CommandResult> {
        let start = Instant::now();

        let mut command = Command::new(cmd);
        command
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(dir) = &self.working_dir {
            command.current_dir(dir);
        }

        // Set environment variables
        for (key, value) in &self.env_vars {
            command.env(key, value);
        }

        let output = command
            .output()
            .map_err(|e| Error::Other(format!("Failed to execute '{}': {}", cmd, e)))?;

        Ok(CommandResult {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
            duration: start.elapsed(),
        })
    }

    /// Execute a command with input piped to stdin.
    pub fn execute_with_stdin(
        &self,
        cmd: &str,
        args: &[&str],
        input: &str,
    ) -> Result<CommandResult> {
        let start = Instant::now();

        let mut command = Command::new(cmd);
        command
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(dir) = &self.working_dir {
            command.current_dir(dir);
        }

        // Set environment variables
        for (key, value) in &self.env_vars {
            command.env(key, value);
        }

        let mut child = command
            .spawn()
            .map_err(|e| Error::Other(format!("Failed to spawn '{}': {}", cmd, e)))?;

        // Write input to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(input.as_bytes())
                .map_err(|e| Error::Other(format!("Failed to write to stdin: {}", e)))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| Error::Other(format!("Failed to wait for '{}': {}", cmd, e)))?;

        Ok(CommandResult {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
            duration: start.elapsed(),
        })
    }
}

// =============================================================================
// Validation Result
// =============================================================================

/// Result of CLI validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Overall score (0.0 - 1.0).
    pub score: f64,
    /// Whether all required stages passed.
    pub passed: bool,
    /// Error messages from failed stages.
    pub errors: Vec<String>,
    /// Per-stage results: (stage_name, score).
    pub stage_results: Vec<(String, f64)>,
}

impl ValidationResult {
    /// Create a passing result.
    pub fn pass() -> Self {
        Self {
            score: 1.0,
            passed: true,
            errors: Vec::new(),
            stage_results: Vec::new(),
        }
    }

    /// Create a failing result with error message.
    pub fn fail(error: impl Into<String>) -> Self {
        Self {
            score: 0.0,
            passed: false,
            errors: vec![error.into()],
            stage_results: Vec::new(),
        }
    }
}

// =============================================================================
// Validator Trait
// =============================================================================

/// Trait for CLI-based validators.
///
/// Implement this trait to create custom validators beyond `Cli` and `CliPipeline`.
pub trait Validator: Send + Sync {
    /// Validate content and return a result.
    fn validate(&self, content: &str) -> Result<ValidationResult>;
}

// =============================================================================
// Cli - Single Command Validator
// =============================================================================

/// A single CLI command for validation.
///
/// # Example
///
/// ```rust,ignore
/// let validator = Cli::new("rustfmt")
///     .args(["--check"])
///     .file_ext("rs");
/// ```
#[derive(Clone)]
pub struct Cli {
    /// Command to run.
    command: String,
    /// Command arguments.
    args: Vec<String>,
    /// Weight for scoring (0.0 - 1.0).
    weight: f32,
    /// Whether this command must pass.
    required: bool,
    /// Use stdin instead of temp file.
    stdin_mode: bool,
    /// File extension for temp file.
    file_extension: Option<String>,
    /// Custom error parser (Arc for Clone support).
    error_parser: Option<std::sync::Arc<dyn Fn(&CommandResult) -> Vec<String> + Send + Sync>>,
    /// Environment variables to set for command execution.
    env_vars: Vec<(String, String)>,
}

impl Cli {
    /// Create a new CLI command validator.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            weight: 1.0,
            required: false,
            stdin_mode: false,
            file_extension: None,
            error_parser: None,
            env_vars: Vec::new(),
        }
    }

    /// Add arguments.
    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args = args.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Add a single argument.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Set weight for scoring (default: 1.0).
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Mark as required (must pass).
    pub fn required(self) -> Self {
        Self {
            required: true,
            ..self
        }
    }

    /// Use stdin instead of temp file.
    pub fn stdin(self) -> Self {
        Self {
            stdin_mode: true,
            ..self
        }
    }

    /// Set file extension for temp file (e.g., "rs", "py", "ts").
    pub fn file_ext(mut self, ext: impl Into<String>) -> Self {
        self.file_extension = Some(ext.into());
        self
    }

    /// Set custom error parser.
    pub fn with_error_parser<F>(mut self, parser: F) -> Self
    where
        F: Fn(&CommandResult) -> Vec<String> + Send + Sync + 'static,
    {
        self.error_parser = Some(std::sync::Arc::new(parser));
        self
    }

    /// Set an environment variable for command execution.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let validator = Cli::new("pulumi")
    ///     .args(["preview"])
    ///     .env("GOOGLE_PROJECT", "my-project")
    ///     .env("PULUMI_CONFIG_PASSPHRASE", "");
    /// ```
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.push((key.into(), value.into()));
        self
    }

    /// Set multiple environment variables.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let validator = Cli::new("pulumi")
    ///     .args(["preview"])
    ///     .envs([
    ///         ("GOOGLE_PROJECT", "my-project"),
    ///         ("PULUMI_CONFIG_PASSPHRASE", ""),
    ///     ]);
    /// ```
    pub fn envs<I, K, V>(mut self, vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.env_vars
            .extend(vars.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Inherit an environment variable from the current process.
    ///
    /// If the environment variable is not set in the current process,
    /// this method does nothing (no error).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let validator = Cli::new("pulumi")
    ///     .args(["preview"])
    ///     .env_inherit("GOOGLE_APPLICATION_CREDENTIALS")
    ///     .env_inherit("GOOGLE_PROJECT");
    /// ```
    pub fn env_inherit(mut self, key: &str) -> Self {
        if let Ok(value) = std::env::var(key) {
            self.env_vars.push((key.to_string(), value));
        }
        self
    }

    /// Get the command name.
    pub fn command(&self) -> &str {
        &self.command
    }

    /// Get the arguments.
    pub fn arguments(&self) -> &[String] {
        &self.args
    }

    /// Get the weight.
    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    /// Check if required.
    pub fn is_required(&self) -> bool {
        self.required
    }

    /// Execute the command with content.
    fn execute(&self, content: &str, temp_dir: &std::path::Path) -> Result<CommandResult> {
        let executor = CliExecutor::new().with_envs(self.env_vars.clone());

        if self.stdin_mode {
            // Pipe content to stdin
            let args: Vec<&str> = self.args.iter().map(|s| s.as_str()).collect();
            executor.execute_with_stdin(&self.command, &args, content)
        } else {
            // Write to temp file
            let ext = self.file_extension.as_deref().unwrap_or("txt");
            let file_path = temp_dir.join(format!("kkachi_validate.{}", ext));

            std::fs::write(&file_path, content)
                .map_err(|e| Error::Other(format!("Failed to write temp file: {}", e)))?;

            let mut args: Vec<&str> = self.args.iter().map(|s| s.as_str()).collect();
            args.push(file_path.to_str().unwrap_or(""));

            let result = executor.execute(&self.command, &args);

            // Cleanup
            let _ = std::fs::remove_file(&file_path);

            result
        }
    }

    /// Parse errors from command result.
    fn parse_errors(&self, result: &CommandResult) -> Vec<String> {
        if let Some(ref parser) = self.error_parser {
            parser(result)
        } else {
            // Default: use stderr lines as errors
            if result.success {
                Vec::new()
            } else {
                result
                    .stderr
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| l.to_string())
                    .collect()
            }
        }
    }
}

impl Validator for Cli {
    fn validate(&self, content: &str) -> Result<ValidationResult> {
        let temp_dir = std::env::temp_dir();
        let result = self.execute(content, &temp_dir)?;
        let errors = self.parse_errors(&result);

        let score = if result.success && errors.is_empty() {
            1.0
        } else if errors.is_empty() {
            0.5 // Partial success
        } else {
            0.0
        };

        Ok(ValidationResult {
            score,
            passed: result.success,
            errors,
            stage_results: vec![(self.command.clone(), score)],
        })
    }
}

// =============================================================================
// CliPipeline - Multi-Stage Validator
// =============================================================================

/// Multi-stage CLI validation pipeline.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = CliPipeline::new()
///     .stage("format", Cli::new("rustfmt").args(["--check"]).weight(0.1))
///     .stage("compile", Cli::new("rustc").args(["--emit=metadata"]).required())
///     .stage("lint", Cli::new("cargo").args(["clippy"]).weight(0.3))
///     .file_ext("rs");
/// ```
#[derive(Clone)]
pub struct CliPipeline {
    /// Validation stages.
    stages: Vec<(String, Cli)>,
    /// File extension for temp file.
    file_extension: Option<String>,
    /// Temporary directory.
    temp_dir: PathBuf,
}

impl Default for CliPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CliPipeline {
    /// Create a new CLI pipeline.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            file_extension: None,
            temp_dir: std::env::temp_dir(),
        }
    }

    /// Add a validation stage.
    pub fn stage(mut self, name: impl Into<String>, cli: Cli) -> Self {
        self.stages.push((name.into(), cli));
        self
    }

    /// Set file extension for temp file.
    pub fn file_ext(mut self, ext: impl Into<String>) -> Self {
        self.file_extension = Some(ext.into());
        self
    }

    /// Set temporary directory.
    pub fn temp_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.temp_dir = dir.into();
        self
    }

    /// Get the number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Get stage names.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|(name, _)| name.as_str()).collect()
    }
}

impl Validator for CliPipeline {
    fn validate(&self, content: &str) -> Result<ValidationResult> {
        // Write content to temp file
        let ext = self.file_extension.as_deref().unwrap_or("txt");
        let file_path = self.temp_dir.join(format!("kkachi_validate.{}", ext));

        std::fs::write(&file_path, content)
            .map_err(|e| Error::Other(format!("Failed to write temp file: {}", e)))?;

        let executor = CliExecutor::new();
        let mut total_score = 0.0f64;
        let mut total_weight = 0.0f32;
        let mut all_errors: Vec<String> = Vec::new();
        let mut stage_results: SmallVec<[(String, f64); 4]> = SmallVec::new();
        let mut all_passed = true;

        for (name, cli) in &self.stages {
            // Build args, appending file path
            let mut args: Vec<&str> = cli.args.iter().map(|s| s.as_str()).collect();
            if !cli.stdin_mode {
                args.push(file_path.to_str().unwrap_or(""));
            }

            // Execute
            let result = if cli.stdin_mode {
                executor.execute_with_stdin(&cli.command, &args, content)?
            } else {
                executor.execute(&cli.command, &args)?
            };

            // Parse errors
            let errors = cli.parse_errors(&result);
            let stage_score = if result.success && errors.is_empty() {
                1.0
            } else if errors.is_empty() {
                0.5 // Partial success
            } else {
                0.0
            };

            stage_results.push((name.clone(), stage_score));
            total_score += stage_score * cli.weight as f64;
            total_weight += cli.weight;

            if !errors.is_empty() {
                for error in errors {
                    all_errors.push(format!("[{}] {}", name, error));
                }
            }

            // Stop on required failure
            if cli.required && stage_score < 1.0 {
                all_passed = false;
                break;
            }

            if stage_score < 1.0 {
                all_passed = false;
            }
        }

        // Cleanup temp file
        let _ = std::fs::remove_file(&file_path);

        let final_score = if total_weight == 0.0 {
            0.0
        } else {
            total_score / total_weight as f64
        };

        Ok(ValidationResult {
            score: final_score,
            passed: all_passed,
            errors: all_errors,
            stage_results: stage_results.into_iter().collect(),
        })
    }
}

// =============================================================================
// Critic Implementation for Validators
// =============================================================================

/// Wrapper that adapts any Validator to the Critic trait.
pub struct ValidatorCritic<V: Validator> {
    validator: V,
}

impl<V: Validator> ValidatorCritic<V> {
    /// Create a new validator critic.
    pub fn new(validator: V) -> Self {
        Self { validator }
    }
}

impl<V: Validator> Critic for ValidatorCritic<V> {
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        match self.validator.validate(output.as_str()) {
            Ok(result) => {
                let mut critic_result = CriticResult::new(result.score);

                // Add breakdown
                let breakdown: Vec<(String, f64)> = result.stage_results;
                critic_result = critic_result.with_breakdown(breakdown);

                // Add feedback if there were errors
                if !result.errors.is_empty() {
                    let feedback = result.errors.join("\n");
                    critic_result = critic_result.with_feedback(feedback);
                }

                critic_result
            }
            Err(e) => {
                // Execution error - return score 0 with error message
                CriticResult::new(0.0).with_feedback(format!("Validation failed: {}", e))
            }
        }
    }
}

// =============================================================================
// CliBinaryCritic - Simple Pass/Fail
// =============================================================================

/// Simple pass/fail critic based on a CLI check.
pub struct CliBinaryCritic {
    cli: Cli,
    failure_feedback: &'static str,
}

impl CliBinaryCritic {
    /// Create a new binary CLI critic.
    pub fn new(cli: Cli, failure_feedback: &'static str) -> Self {
        Self {
            cli,
            failure_feedback,
        }
    }
}

impl Critic for CliBinaryCritic {
    fn evaluate<'a>(&self, output: StrView<'a>, _state: &RecursiveState<'a>) -> CriticResult<'a> {
        match self.cli.validate(output.as_str()) {
            Ok(result) if result.passed => CriticResult::new(1.0),
            Ok(result) => {
                let feedback = if result.errors.is_empty() {
                    self.failure_feedback.to_string()
                } else {
                    format!("{}\n{}", self.failure_feedback, result.errors.join("\n"))
                };
                CriticResult::new(0.0).with_feedback(feedback)
            }
            Err(e) => CriticResult::new(0.0).with_feedback(format!("Execution failed: {}", e)),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_executor_echo() {
        let executor = CliExecutor::new();
        let result = executor.execute("echo", &["hello"]).unwrap();

        assert!(result.success);
        assert_eq!(result.stdout.trim(), "hello");
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_cli_executor_failure() {
        let executor = CliExecutor::new();
        let result = executor.execute("false", &[]).unwrap(); // `false` exits with 1

        assert!(!result.success);
        assert_eq!(result.exit_code, 1);
    }

    #[test]
    fn test_cli_executor_stdin() {
        let executor = CliExecutor::new();
        let result = executor
            .execute_with_stdin("cat", &[], "test input")
            .unwrap();

        assert!(result.success);
        assert_eq!(result.stdout, "test input");
    }

    #[test]
    fn test_cli_builder() {
        let cli = Cli::new("echo")
            .args(["hello", "world"])
            .weight(0.5)
            .required()
            .file_ext("txt");

        assert_eq!(cli.command(), "echo");
        assert_eq!(cli.arguments(), &["hello", "world"]);
        assert_eq!(cli.get_weight(), 0.5);
        assert!(cli.is_required());
    }

    #[test]
    fn test_cli_pipeline_builder() {
        let pipeline = CliPipeline::new()
            .stage("stage1", Cli::new("echo").arg("hello"))
            .stage("stage2", Cli::new("echo").arg("world"))
            .file_ext("txt");

        assert_eq!(pipeline.len(), 2);
        assert_eq!(pipeline.stage_names(), vec!["stage1", "stage2"]);
    }

    #[test]
    fn test_cli_validate_echo() {
        let cli = Cli::new("cat").stdin();
        let result = cli.validate("hello world").unwrap();

        assert!(result.passed);
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn test_cli_pipeline_validate() {
        let pipeline = CliPipeline::new()
            .stage("stage1", Cli::new("cat").stdin())
            .stage("stage2", Cli::new("cat").stdin());

        let result = pipeline.validate("hello").unwrap();

        assert!(result.passed);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.stage_results.len(), 2);
    }

    #[test]
    fn test_validation_result_pass() {
        let result = ValidationResult::pass();
        assert!(result.passed);
        assert_eq!(result.score, 1.0);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validation_result_fail() {
        let result = ValidationResult::fail("Test error");
        assert!(!result.passed);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.errors, vec!["Test error"]);
    }

    // =========================================================================
    // Environment Variable Tests
    // =========================================================================

    #[test]
    fn test_cli_env_single() {
        let cli = Cli::new("printenv")
            .args(["TEST_VAR"])
            .env("TEST_VAR", "test_value");

        assert_eq!(cli.env_vars.len(), 1);
        assert_eq!(
            cli.env_vars[0],
            ("TEST_VAR".to_string(), "test_value".to_string())
        );
    }

    #[test]
    fn test_cli_env_multiple() {
        let cli = Cli::new("printenv")
            .env("VAR1", "value1")
            .env("VAR2", "value2")
            .env("VAR3", "value3");

        assert_eq!(cli.env_vars.len(), 3);
        assert_eq!(cli.env_vars[0].0, "VAR1");
        assert_eq!(cli.env_vars[1].0, "VAR2");
        assert_eq!(cli.env_vars[2].0, "VAR3");
    }

    #[test]
    fn test_cli_envs_batch() {
        let cli = Cli::new("printenv").envs([("VAR1", "value1"), ("VAR2", "value2")]);

        assert_eq!(cli.env_vars.len(), 2);
        assert_eq!(cli.env_vars[0], ("VAR1".to_string(), "value1".to_string()));
        assert_eq!(cli.env_vars[1], ("VAR2".to_string(), "value2".to_string()));
    }

    #[test]
    fn test_cli_env_inherit_existing() {
        // Set a test environment variable
        std::env::set_var("KKACHI_TEST_INHERIT", "inherited_value");

        let cli = Cli::new("printenv").env_inherit("KKACHI_TEST_INHERIT");

        assert_eq!(cli.env_vars.len(), 1);
        assert_eq!(cli.env_vars[0].0, "KKACHI_TEST_INHERIT");
        assert_eq!(cli.env_vars[0].1, "inherited_value");

        // Clean up
        std::env::remove_var("KKACHI_TEST_INHERIT");
    }

    #[test]
    fn test_cli_env_inherit_missing() {
        // Ensure the variable doesn't exist
        std::env::remove_var("KKACHI_NONEXISTENT_VAR");

        let cli = Cli::new("printenv").env_inherit("KKACHI_NONEXISTENT_VAR");

        // Should not add anything if var doesn't exist
        assert_eq!(cli.env_vars.len(), 0);
    }

    #[test]
    fn test_cli_executor_with_envs() {
        let executor = CliExecutor::new().with_envs(vec![(
            "MY_TEST_VAR".to_string(),
            "my_test_value".to_string(),
        )]);

        let result = executor.execute("printenv", &["MY_TEST_VAR"]).unwrap();

        assert!(result.success);
        assert_eq!(result.stdout.trim(), "my_test_value");
    }

    #[test]
    fn test_cli_validate_with_env() {
        // Test that env vars are passed to the command
        let cli = Cli::new("sh")
            .args(["-c", "echo $MY_VAR"])
            .env("MY_VAR", "hello_from_env");

        let result = cli.validate("").unwrap();

        assert!(result.passed);
        // The output should contain the env var value
        assert!(
            result.errors.is_empty() || result.errors.iter().any(|e| e.contains("hello_from_env"))
        );
    }

    #[test]
    fn test_cli_combined_env_methods() {
        // Test combining all env methods
        std::env::set_var("KKACHI_COMBINED_TEST", "inherited");

        let cli = Cli::new("printenv")
            .env("EXPLICIT", "explicit_value")
            .envs([("BATCH1", "batch1_value"), ("BATCH2", "batch2_value")])
            .env_inherit("KKACHI_COMBINED_TEST");

        assert_eq!(cli.env_vars.len(), 4);
        assert_eq!(cli.env_vars[0].0, "EXPLICIT");
        assert_eq!(cli.env_vars[1].0, "BATCH1");
        assert_eq!(cli.env_vars[2].0, "BATCH2");
        assert_eq!(cli.env_vars[3].0, "KKACHI_COMBINED_TEST");

        std::env::remove_var("KKACHI_COMBINED_TEST");
    }
}
