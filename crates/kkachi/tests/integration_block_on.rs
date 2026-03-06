// Integration tests for block_on with retry and rate-limit wrappers.

use kkachi::recursive::{
    reason, FailingLlm, LlmExt, MockLlm, RateLimitConfig, RateLimitExt, RetryConfig,
};
use std::time::Duration;

#[test]
fn test_reason_go_with_retry_and_retryable_error() {
    let llm = FailingLlm::new("HTTP 429 rate limit exceeded").with_retry_config(RetryConfig {
        max_retries: 2,
        initial_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(10),
        backoff_factor: 2.0,
    });

    let result = reason(&llm, "test prompt").go();
    // Should complete without panic. Error stored in result.error.
    assert!(result.error.is_some());
}

#[test]
fn test_reason_go_with_rate_limit() {
    let llm = MockLlm::new(|_, _| "answer".to_string())
        .with_rate_limit_config(RateLimitConfig::new(10.0).with_burst(1));

    let result = reason(&llm, "test prompt").go();
    assert_eq!(result.output, "answer");
}

#[test]
fn test_reason_go_with_retry_exhausted() {
    let llm = FailingLlm::new("503 Service Unavailable").with_retry_config(RetryConfig {
        max_retries: 1,
        initial_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        backoff_factor: 1.0,
    });

    // Should not panic even though retries are exhausted.
    let _result = reason(&llm, "test").go();
}

#[test]
fn test_full_chain_via_block_on() {
    let llm = MockLlm::new(|_, _| "ok".to_string())
        .with_rate_limit_config(RateLimitConfig::new(100.0).with_burst(2))
        .with_retry(2);

    let result = reason(&llm, "test").go();
    assert_eq!(result.output, "ok");
}
