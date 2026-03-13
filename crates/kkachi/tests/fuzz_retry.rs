// Property-based tests for retry with block_on.

use kkachi::recursive::{block_on, FailingLlm, Llm, LlmExt, RetryConfig};
use proptest::prelude::*;
use std::time::Duration;

proptest! {
    #[test]
    fn retry_arbitrary_config_no_panic(
        max_retries in 0u32..5,
        initial_ms in 1u64..100,
        max_ms in 1u64..100,
        backoff in 1.0f64..4.0,
    ) {
        let config = RetryConfig {
            max_retries,
            initial_delay: Duration::from_millis(initial_ms),
            max_delay: Duration::from_millis(max_ms),
            backoff_factor: backoff,
        };
        let llm = FailingLlm::new("429 rate limit").with_retry_config(config);

        // Must never panic, always returns Err.
        let result = block_on(llm.generate("test", "", None));
        prop_assert!(result.is_err());
    }
}
