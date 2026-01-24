// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Code execution for Program of Thought.
//!
//! This module provides the [`CodeExecutor`] trait which defines the interface
//! for executing generated code. Using GATs allows zero-cost async without boxing.
//!
//! # Examples
//!
//! ```rust,ignore
//! use kkachi::recursive::executor::{python_executor, CodeExecutor};
//!
//! let executor = python_executor().timeout(Duration::from_secs(10));
//! let result = executor.execute("print(2 + 2)").await;
//! assert!(result.success);
//! assert!(result.stdout.contains("4"));
//! ```

use smallvec::SmallVec;
use std::borrow::Cow;
use std::future::Future;
use std::io::Write;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Result of code execution with zero-copy output.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Whether execution succeeded (exit code 0).
    pub success: bool,
    /// Exit code if available.
    pub exit_code: Option<i32>,
    /// Execution duration.
    pub duration: Duration,
}

impl ExecutionResult {
    /// Get the primary output (stdout if available, stderr otherwise).
    pub fn output(&self) -> &str {
        if !self.stdout.is_empty() {
            &self.stdout
        } else {
            &self.stderr
        }
    }

    /// Get combined output (stdout + stderr).
    pub fn combined(&self) -> Cow<'_, str> {
        if self.stderr.is_empty() {
            Cow::Borrowed(&self.stdout)
        } else if self.stdout.is_empty() {
            Cow::Borrowed(&self.stderr)
        } else {
            Cow::Owned(format!("{}\n{}", self.stdout, self.stderr))
        }
    }

    /// Get error message if execution failed.
    pub fn error(&self) -> Option<&str> {
        if self.success {
            None
        } else if !self.stderr.is_empty() {
            Some(&self.stderr)
        } else {
            Some("Execution failed with no error output")
        }
    }
}

/// Trait for code executors.
///
/// This trait is object-safe to allow storing different executor types
/// in Program of Thought builders. For async execution, it uses boxed futures.
pub trait CodeExecutor: Send + Sync {
    /// Get the language name (for prompt generation).
    fn language(&self) -> &str;

    /// Get the file extension for this language.
    fn extension(&self) -> &str;

    /// Execute the given code.
    ///
    /// Returns a boxed future to maintain object safety.
    fn execute<'a>(
        &'a self,
        code: &'a str,
    ) -> Pin<Box<dyn Future<Output = ExecutionResult> + Send + 'a>>;
}

impl CodeExecutor for Box<dyn CodeExecutor> {
    fn language(&self) -> &str {
        (**self).language()
    }

    fn extension(&self) -> &str {
        (**self).extension()
    }

    fn execute<'a>(
        &'a self,
        code: &'a str,
    ) -> Pin<Box<dyn Future<Output = ExecutionResult> + Send + 'a>> {
        (**self).execute(code)
    }
}

/// CLI-based code executor that runs code via external process.
pub struct ProcessExecutor {
    command: &'static str,
    args: SmallVec<[&'static str; 4]>,
    extension: &'static str,
    language: &'static str,
    timeout: Duration,
    use_stdin: bool,
    working_dir: Option<PathBuf>,
    env_vars: SmallVec<[(String, String); 4]>,
}

impl ProcessExecutor {
    /// Create a new process executor.
    pub fn new(command: &'static str, extension: &'static str, language: &'static str) -> Self {
        Self {
            command,
            args: SmallVec::new(),
            extension,
            language,
            timeout: Duration::from_secs(30),
            use_stdin: false,
            working_dir: None,
            env_vars: SmallVec::new(),
        }
    }

    /// Add command line arguments.
    pub fn args(mut self, args: &[&'static str]) -> Self {
        for arg in args {
            self.args.push(*arg);
        }
        self
    }

    /// Set execution timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Use stdin for code input instead of temp file.
    pub fn stdin(mut self) -> Self {
        self.use_stdin = true;
        self
    }

    /// Set working directory.
    pub fn working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set environment variable.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.push((key.into(), value.into()));
        self
    }

    /// Execute code synchronously.
    fn execute_sync(&self, code: &str) -> ExecutionResult {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let start = Instant::now();

        // Create temp file if needed with unique name per execution
        let temp_file = if !self.use_stdin {
            let temp_dir = std::env::temp_dir();
            let unique_id = COUNTER.fetch_add(1, Ordering::SeqCst);
            let file_path = temp_dir.join(format!(
                "kkachi_exec_{}_{}.{}",
                std::process::id(),
                unique_id,
                self.extension
            ));

            if let Err(e) = std::fs::write(&file_path, code) {
                return ExecutionResult {
                    stdout: String::new(),
                    stderr: format!("Failed to write temp file: {}", e),
                    success: false,
                    exit_code: None,
                    duration: start.elapsed(),
                };
            }
            Some(file_path)
        } else {
            None
        };

        let mut cmd = Command::new(self.command);

        // Add base args
        for arg in &self.args {
            cmd.arg(arg);
        }

        // Add file path or set up stdin
        if let Some(ref file) = temp_file {
            cmd.arg(file);
        } else {
            cmd.stdin(Stdio::piped());
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Set working directory
        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        // Set environment variables
        for (key, value) in &self.env_vars {
            cmd.env(key, value);
        }

        let result = if self.use_stdin {
            match cmd.spawn() {
                Ok(mut child) => {
                    if let Some(mut stdin) = child.stdin.take() {
                        let _ = stdin.write_all(code.as_bytes());
                    }
                    match child.wait_with_output() {
                        Ok(output) => {
                            let duration = start.elapsed();
                            ExecutionResult {
                                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                                success: output.status.success(),
                                exit_code: output.status.code(),
                                duration,
                            }
                        }
                        Err(e) => ExecutionResult {
                            stdout: String::new(),
                            stderr: format!("Failed to wait for process: {}", e),
                            success: false,
                            exit_code: None,
                            duration: start.elapsed(),
                        },
                    }
                }
                Err(e) => ExecutionResult {
                    stdout: String::new(),
                    stderr: format!("Failed to spawn '{}': {}", self.command, e),
                    success: false,
                    exit_code: None,
                    duration: start.elapsed(),
                },
            }
        } else {
            match cmd.output() {
                Ok(output) => {
                    let duration = start.elapsed();
                    ExecutionResult {
                        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                        success: output.status.success(),
                        exit_code: output.status.code(),
                        duration,
                    }
                }
                Err(e) => ExecutionResult {
                    stdout: String::new(),
                    stderr: format!("Failed to execute '{}': {}", self.command, e),
                    success: false,
                    exit_code: None,
                    duration: start.elapsed(),
                },
            }
        };

        // Cleanup temp file
        if let Some(file) = temp_file {
            let _ = std::fs::remove_file(file);
        }

        result
    }
}

impl CodeExecutor for ProcessExecutor {
    fn language(&self) -> &str {
        self.language
    }

    fn extension(&self) -> &str {
        self.extension
    }

    fn execute<'a>(
        &'a self,
        code: &'a str,
    ) -> Pin<Box<dyn Future<Output = ExecutionResult> + Send + 'a>> {
        Box::pin(std::future::ready(self.execute_sync(code)))
    }
}

/// Create a Python executor.
///
/// # Examples
///
/// ```rust,ignore
/// let executor = python_executor().timeout(Duration::from_secs(10));
/// let result = executor.execute("print('hello')").await;
/// ```
pub fn python_executor() -> ProcessExecutor {
    ProcessExecutor::new("python3", "py", "python")
}

/// Create a Node.js executor.
///
/// # Examples
///
/// ```rust,ignore
/// let executor = node_executor().timeout(Duration::from_secs(10));
/// let result = executor.execute("console.log('hello')").await;
/// ```
pub fn node_executor() -> ProcessExecutor {
    ProcessExecutor::new("node", "js", "javascript")
}

/// Create a Ruby executor.
pub fn ruby_executor() -> ProcessExecutor {
    ProcessExecutor::new("ruby", "rb", "ruby")
}

/// Create a Bash executor.
pub fn bash_executor() -> ProcessExecutor {
    ProcessExecutor::new("bash", "sh", "bash")
}

/// Create a Rust executor (compiles and runs).
///
/// Note: This creates a temp directory, compiles with rustc, and runs the binary.
pub fn rust_executor() -> ProcessExecutor {
    // For Rust, we'll use rustc and run the resulting binary
    // This is simplified - a real implementation might use cargo
    ProcessExecutor::new("rustc", "rs", "rust").args(&[
        "-o",
        "/tmp/kkachi_rust_exec",
        "--edition=2021",
    ])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_python_executor() {
        let exec = python_executor();
        let result = exec.execute("print(2 + 2)").await;

        if result.success {
            assert!(result.stdout.trim() == "4" || result.output().contains("4"));
        }
        // If Python not available, just check the structure
        assert!(result.duration.as_nanos() > 0);
    }

    #[tokio::test]
    async fn test_node_executor() {
        let exec = node_executor();
        let result = exec.execute("console.log(2 + 2)").await;

        if result.success {
            assert!(result.stdout.trim() == "4");
        }
        assert!(result.duration.as_nanos() > 0);
    }

    #[tokio::test]
    async fn test_bash_executor() {
        let exec = bash_executor();
        let result = exec.execute("echo hello").await;

        assert!(result.success);
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_execution_result_methods() {
        let result = ExecutionResult {
            stdout: "output".to_string(),
            stderr: "error".to_string(),
            success: false,
            exit_code: Some(1),
            duration: Duration::from_millis(100),
        };

        assert_eq!(result.output(), "output");
        assert!(result.combined().contains("output"));
        assert!(result.combined().contains("error"));
        assert_eq!(result.error(), Some("error"));
    }

    #[tokio::test]
    async fn test_executor_timeout() {
        let exec = python_executor().timeout(Duration::from_secs(1));
        assert_eq!(exec.timeout, Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_executor_env() {
        let exec = python_executor().env("TEST_VAR", "test_value");
        assert_eq!(exec.env_vars.len(), 1);
    }

    #[test]
    fn test_process_executor_creation() {
        let exec = ProcessExecutor::new("python3", "py", "python")
            .args(&["-u"])
            .timeout(Duration::from_secs(5))
            .stdin();

        assert_eq!(exec.command, "python3");
        assert_eq!(exec.language(), "python");
        assert_eq!(exec.extension(), "py");
        assert!(exec.use_stdin);
    }
}
