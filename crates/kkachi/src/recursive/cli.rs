// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! CLI-based validation for external tool integration.
//!
//! This module provides the [`Cli`] builder for creating validators that
//! execute external commands to validate generated content.
//!
//! # Examples
//!
//! ```rust,ignore
//! use kkachi::recursive::cli;
//!
//! // Single command
//! let v = cli("rustfmt").arg("--check").ext("rs");
//!
//! // Multi-stage pipeline
//! let v = cli("rustfmt").arg("--check")
//!     .then("rustc").arg("--emit=metadata").required()
//!     .then("clippy")
//!     .ext("rs");
//!
//! // With environment variables
//! let v = cli("pulumi").arg("preview")
//!     .env_from("GOOGLE_APPLICATION_CREDENTIALS")
//!     .capture();
//! ```

use crate::error::{Error, Result};
use crate::recursive::validate::{Score, Validate};
use smallvec::SmallVec;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Create a new CLI validator builder.
///
/// This is the entry point for building CLI-based validators.
#[inline]
pub fn cli(command: &str) -> Cli {
    Cli::new(command)
}

/// Captured output from a CLI command.
#[derive(Debug, Clone)]
pub struct CliCapture {
    /// Stage name.
    pub stage: String,
    /// Command that was run.
    pub command: String,
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Whether the command succeeded.
    pub success: bool,
    /// Exit code.
    pub exit_code: Option<i32>,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
}

impl CliCapture {
    /// Get combined output (stdout + stderr).
    pub fn combined(&self) -> String {
        if self.stderr.is_empty() {
            self.stdout.clone()
        } else if self.stdout.is_empty() {
            self.stderr.clone()
        } else {
            format!("{}\n{}", self.stdout, self.stderr)
        }
    }

    /// Get error lines from stderr.
    pub fn errors(&self) -> Vec<&str> {
        self.stderr
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect()
    }
}

/// A single CLI stage in the validation pipeline.
#[derive(Clone)]
struct Stage {
    command: String,
    args: SmallVec<[String; 8]>,
    weight: f64,
    required: bool,
}

impl Stage {
    fn new(command: &str) -> Self {
        Self {
            command: command.to_string(),
            args: SmallVec::new(),
            weight: 1.0,
            required: false,
        }
    }
}

/// CLI validator builder.
///
/// Builds a validator that runs one or more CLI commands to validate content.
/// Commands can be chained with `.then()` to create multi-stage pipelines.
pub struct Cli {
    stages: SmallVec<[Stage; 4]>,
    extension: String,
    inherit_env: bool,
    env_vars: SmallVec<[(String, String); 8]>,
    env_passthrough: SmallVec<[String; 8]>,
    workdir: Option<PathBuf>,
    timeout: Duration,
    capture: bool,
    use_stdin: bool,
    /// Captured outputs (populated during validation).
    captures: std::sync::Mutex<SmallVec<[CliCapture; 4]>>,
}

impl Clone for Cli {
    fn clone(&self) -> Self {
        Self {
            stages: self.stages.clone(),
            extension: self.extension.clone(),
            inherit_env: self.inherit_env,
            env_vars: self.env_vars.clone(),
            env_passthrough: self.env_passthrough.clone(),
            workdir: self.workdir.clone(),
            timeout: self.timeout,
            capture: self.capture,
            use_stdin: self.use_stdin,
            // Create a new empty mutex for the clone
            captures: std::sync::Mutex::new(SmallVec::new()),
        }
    }
}

impl Cli {
    /// Create a new CLI validator with the given command.
    pub fn new(command: &str) -> Self {
        let mut stages = SmallVec::new();
        stages.push(Stage::new(command));

        Self {
            stages,
            extension: "txt".to_string(),
            inherit_env: true,
            env_vars: SmallVec::new(),
            env_passthrough: SmallVec::new(),
            workdir: None,
            timeout: Duration::from_secs(120),
            capture: false,
            use_stdin: false,
            captures: std::sync::Mutex::new(SmallVec::new()),
        }
    }

    /// Add an argument to the current stage.
    pub fn arg(mut self, arg: &str) -> Self {
        if let Some(stage) = self.stages.last_mut() {
            stage.args.push(arg.to_string());
        }
        self
    }

    /// Add multiple arguments to the current stage.
    pub fn args(mut self, args: &[&str]) -> Self {
        if let Some(stage) = self.stages.last_mut() {
            for arg in args {
                stage.args.push((*arg).to_string());
            }
        }
        self
    }

    /// Set the weight for the current stage (for scoring).
    pub fn weight(mut self, w: f64) -> Self {
        if let Some(stage) = self.stages.last_mut() {
            stage.weight = w;
        }
        self
    }

    /// Mark the current stage as required (must pass).
    pub fn required(mut self) -> Self {
        if let Some(stage) = self.stages.last_mut() {
            stage.required = true;
        }
        self
    }

    /// Add another command stage to the pipeline.
    pub fn then(mut self, command: &str) -> Self {
        self.stages.push(Stage::new(command));
        self
    }

    /// Set the file extension for the temp file.
    pub fn ext(mut self, extension: &str) -> Self {
        self.extension = extension.to_string();
        self
    }

    /// Set an environment variable.
    pub fn env(mut self, key: &str, value: &str) -> Self {
        self.env_vars.push((key.to_string(), value.to_string()));
        self
    }

    /// Inherit an environment variable from the current process.
    ///
    /// If the variable is not set, this is a no-op.
    pub fn env_from(mut self, key: &str) -> Self {
        self.env_passthrough.push(key.to_string());
        self
    }

    /// Set whether to inherit the parent's environment.
    ///
    /// Default is true.
    pub fn inherit_env(mut self, inherit: bool) -> Self {
        self.inherit_env = inherit;
        self
    }

    /// Set the working directory.
    pub fn workdir(mut self, path: &str) -> Self {
        self.workdir = Some(PathBuf::from(path));
        self
    }

    /// Set the timeout in seconds.
    pub fn timeout(mut self, secs: u64) -> Self {
        self.timeout = Duration::from_secs(secs);
        self
    }

    /// Enable output capture.
    pub fn capture(mut self) -> Self {
        self.capture = true;
        self
    }

    /// Use stdin instead of a temp file.
    pub fn stdin(mut self) -> Self {
        self.use_stdin = true;
        self
    }

    /// Get captured outputs after validation.
    pub fn get_captures(&self) -> SmallVec<[CliCapture; 4]> {
        self.captures.lock().unwrap().clone()
    }

    /// Execute a single stage.
    fn execute_stage(
        &self,
        stage: &Stage,
        content: &str,
        temp_file: &std::path::Path,
    ) -> Result<CliCapture> {
        let start = Instant::now();

        let mut cmd = Command::new(&stage.command);

        // Add arguments
        for arg in &stage.args {
            cmd.arg(arg);
        }

        // Add temp file path if not using stdin
        if !self.use_stdin {
            cmd.arg(temp_file);
        }

        // Environment
        if !self.inherit_env {
            cmd.env_clear();
        }

        for (key, value) in &self.env_vars {
            cmd.env(key, value);
        }

        for key in &self.env_passthrough {
            if let Ok(value) = std::env::var(key) {
                cmd.env(key, value);
            }
        }

        // Working directory
        if let Some(ref dir) = self.workdir {
            cmd.current_dir(dir);
        }

        // Set up I/O
        if self.use_stdin {
            cmd.stdin(Stdio::piped());
        }
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Execute
        let output = if self.use_stdin {
            let mut child = cmd.spawn().map_err(|e| {
                Error::Other(format!("Failed to spawn '{}': {}", stage.command, e))
            })?;

            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(content.as_bytes()).map_err(|e| {
                    Error::Other(format!("Failed to write to stdin: {}", e))
                })?;
            }

            child.wait_with_output().map_err(|e| {
                Error::Other(format!("Failed to wait for '{}': {}", stage.command, e))
            })?
        } else {
            cmd.output().map_err(|e| {
                Error::Other(format!("Failed to execute '{}': {}", stage.command, e))
            })?
        };

        let duration = start.elapsed();

        Ok(CliCapture {
            stage: stage.command.clone(),
            command: format!(
                "{} {}",
                stage.command,
                stage.args.join(" ")
            ),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            success: output.status.success(),
            exit_code: output.status.code(),
            duration_ms: duration.as_millis() as u64,
        })
    }
}

impl Validate for Cli {
    fn validate(&self, text: &str) -> Score<'static> {
        // Create temp file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("kkachi_validate.{}", self.extension));

        // Write content to temp file if not using stdin
        if !self.use_stdin {
            if let Err(e) = std::fs::write(&temp_file, text) {
                return Score::with_feedback(0.0, format!("Failed to write temp file: {}", e));
            }
        }

        let mut total_weight = 0.0f64;
        let mut weighted_score = 0.0f64;
        let mut errors: Vec<String> = Vec::new();
        let mut all_captures: SmallVec<[CliCapture; 4]> = SmallVec::new();
        let mut required_failed = false;

        // Execute each stage
        for stage in &self.stages {
            let capture = match self.execute_stage(stage, text, &temp_file) {
                Ok(c) => c,
                Err(e) => {
                    // Cleanup
                    let _ = std::fs::remove_file(&temp_file);
                    return Score::with_feedback(0.0, e.to_string());
                }
            };

            let stage_score = if capture.success { 1.0 } else { 0.0 };
            weighted_score += stage_score * stage.weight;
            total_weight += stage.weight;

            if !capture.success {
                if stage.required {
                    required_failed = true;
                }
                // Add error lines
                for line in capture.stderr.lines().filter(|l| !l.trim().is_empty()) {
                    errors.push(line.to_string());
                }
            }

            if self.capture {
                all_captures.push(capture);
            }
        }

        // Cleanup temp file
        let _ = std::fs::remove_file(&temp_file);

        // Store captures
        if self.capture {
            *self.captures.lock().unwrap() = all_captures;
        }

        // Calculate final score
        let final_score = if required_failed {
            0.0
        } else if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            1.0
        };

        if errors.is_empty() {
            Score::new(final_score)
        } else {
            Score::with_feedback(final_score, errors.join("\n"))
        }
    }

    fn name(&self) -> &'static str {
        "cli"
    }
}

/// Older-style CLI executor for direct command execution.
///
/// This is a lower-level API for executing commands directly.
pub struct CliExecutor {
    timeout: Duration,
    working_dir: Option<PathBuf>,
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
    pub fn execute(&self, cmd: &str, args: &[&str]) -> Result<CliCapture> {
        let start = Instant::now();

        let mut command = Command::new(cmd);
        command
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(ref dir) = self.working_dir {
            command.current_dir(dir);
        }

        for (key, value) in &self.env_vars {
            command.env(key, value);
        }

        let output = command
            .output()
            .map_err(|e| Error::Other(format!("Failed to execute '{}': {}", cmd, e)))?;

        Ok(CliCapture {
            stage: cmd.to_string(),
            command: format!("{} {}", cmd, args.join(" ")),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            success: output.status.success(),
            exit_code: output.status.code(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Execute a command with input piped to stdin.
    pub fn execute_with_stdin(&self, cmd: &str, args: &[&str], input: &str) -> Result<CliCapture> {
        let start = Instant::now();

        let mut command = Command::new(cmd);
        command
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(ref dir) = self.working_dir {
            command.current_dir(dir);
        }

        for (key, value) in &self.env_vars {
            command.env(key, value);
        }

        let mut child = command
            .spawn()
            .map_err(|e| Error::Other(format!("Failed to spawn '{}': {}", cmd, e)))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(input.as_bytes())
                .map_err(|e| Error::Other(format!("Failed to write to stdin: {}", e)))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| Error::Other(format!("Failed to wait for '{}': {}", cmd, e)))?;

        Ok(CliCapture {
            stage: cmd.to_string(),
            command: format!("{} {}", cmd, args.join(" ")),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            success: output.status.success(),
            exit_code: output.status.code(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_builder() {
        let c = cli("echo")
            .arg("hello")
            .arg("world")
            .ext("txt");

        assert_eq!(c.stages.len(), 1);
        assert_eq!(c.stages[0].command, "echo");
        assert_eq!(c.stages[0].args.len(), 2);
        assert_eq!(c.extension, "txt");
    }

    #[test]
    fn test_cli_multi_stage() {
        let c = cli("echo")
            .arg("1")
            .then("echo")
            .arg("2")
            .required()
            .then("echo")
            .arg("3");

        assert_eq!(c.stages.len(), 3);
        assert!(c.stages[1].required);
    }

    #[test]
    fn test_cli_env() {
        let c = cli("echo")
            .env("FOO", "bar")
            .env_from("PATH");

        assert_eq!(c.env_vars.len(), 1);
        assert_eq!(c.env_passthrough.len(), 1);
    }

    #[test]
    fn test_cli_validate_echo() {
        let v = cli("echo").arg("ok").stdin();
        let score = v.validate("test input");
        assert!(score.is_perfect());
    }

    #[test]
    fn test_cli_validate_false() {
        let v = cli("false");
        let score = v.validate("anything");
        assert!(!score.is_perfect());
    }

    #[test]
    fn test_cli_capture() {
        let v = cli("echo").arg("captured").stdin().capture();
        let _ = v.validate("input");

        let captures = v.get_captures();
        assert_eq!(captures.len(), 1);
        assert!(captures[0].stdout.contains("captured"));
    }

    #[test]
    fn test_cli_executor() {
        let exec = CliExecutor::new();
        let result = exec.execute("echo", &["test"]).unwrap();
        assert!(result.success);
        assert!(result.stdout.contains("test"));
    }

    #[test]
    fn test_cli_executor_stdin() {
        let exec = CliExecutor::new();
        let result = exec.execute_with_stdin("cat", &[], "hello world").unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "hello world");
    }

    #[test]
    fn test_cli_capture_errors() {
        let capture = CliCapture {
            stage: "test".to_string(),
            command: "test".to_string(),
            stdout: String::new(),
            stderr: "error: something failed\nwarning: be careful\n".to_string(),
            success: false,
            exit_code: Some(1),
            duration_ms: 100,
        };

        let errors = capture.errors();
        assert_eq!(errors.len(), 2);
        assert!(errors[0].contains("error"));
    }

    #[test]
    fn test_cli_weighted_stages() {
        let c = cli("true").weight(0.3)
            .then("true").weight(0.7);

        let score = c.validate("test");
        assert!(score.is_perfect());
    }
}
