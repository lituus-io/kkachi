// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Cache key generation

use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// A cache key for LM requests
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Model name
    pub model: String,

    /// Request hash
    pub request_hash: u64,

    /// Temperature (multiplied by 1000 for integer comparison)
    pub temperature_x1000: u32,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(model: String, request_hash: u64, temperature: f32) -> Self {
        Self {
            model,
            request_hash,
            temperature_x1000: (temperature * 1000.0) as u32,
        }
    }

    /// Create from request data
    pub fn from_request(model: &str, messages: &str, temperature: f32) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        messages.hash(&mut hasher);
        let request_hash = hasher.finish();

        Self::new(model.to_string(), request_hash, temperature)
    }

    /// Convert to string for file-based storage
    pub fn to_string(&self) -> String {
        format!(
            "{}_{:x}_{}",
            self.model, self.request_hash, self.temperature_x1000
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key() {
        let key1 = CacheKey::from_request("gpt-4", "test message", 0.0);
        let key2 = CacheKey::from_request("gpt-4", "test message", 0.0);
        let key3 = CacheKey::from_request("gpt-4", "different message", 0.0);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
