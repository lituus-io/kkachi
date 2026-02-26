// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! State serialization for saving and loading module configurations.
//!
//! Provides a generic state system that can serialize any module's
//! configuration to JSON or binary format. This enables checkpointing
//! optimized prompts, saving trained module parameters, and restoring
//! state across sessions.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::state::{StateMap, StateValue, Saveable};
//!
//! let mut state = StateMap::new("my_module");
//! state.set("prompt", StateValue::Str("optimized prompt".into()));
//! state.set("temperature", StateValue::Float(0.7));
//! state.set("max_tokens", StateValue::Int(1024));
//! state.set("verbose", StateValue::Bool(false));
//!
//! // Retrieve typed values
//! assert_eq!(state.get_str("prompt"), Some("optimized prompt"));
//! assert_eq!(state.get_float("temperature"), Some(0.7));
//! assert_eq!(state.get_int("max_tokens"), Some(1024));
//! assert_eq!(state.get_bool("verbose"), Some(false));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A serializable state map for a module.
///
/// Contains the module identifier, a version number for forward
/// compatibility, and a map of string keys to [`StateValue`]s.
/// Using `BTreeMap` ensures deterministic serialization order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateMap {
    /// Module name/type identifier.
    pub module: String,
    /// Version for forward compatibility.
    pub version: u32,
    /// The state data.
    pub data: BTreeMap<String, StateValue>,
}

/// A value in the state map.
///
/// Supports common primitive types plus nested maps and lists,
/// enabling arbitrarily complex state structures. Uses externally-tagged
/// serialization for compatibility with both JSON and binary formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StateValue {
    /// Boolean value.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating point.
    Float(f64),
    /// UTF-8 string.
    Str(String),
    /// Raw byte data (serialized as array of numbers in JSON, compact in bincode).
    Bytes(Vec<u8>),
    /// Nested map of state values.
    Map(BTreeMap<String, StateValue>),
    /// List of state values.
    List(Vec<StateValue>),
}

impl StateValue {
    /// Get as a string reference if this is a `Str` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            StateValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Get as f64 if this is a `Float` variant.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            StateValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Get as i64 if this is an `Int` variant.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            StateValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as bool if this is a `Bool` variant.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            StateValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as byte slice if this is a `Bytes` variant.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            StateValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// Get as a map reference if this is a `Map` variant.
    pub fn as_map(&self) -> Option<&BTreeMap<String, StateValue>> {
        match self {
            StateValue::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Get as a list reference if this is a `List` variant.
    pub fn as_list(&self) -> Option<&Vec<StateValue>> {
        match self {
            StateValue::List(l) => Some(l),
            _ => None,
        }
    }

    /// Check if this value is a string.
    pub fn is_str(&self) -> bool {
        matches!(self, StateValue::Str(_))
    }

    /// Check if this value is a float.
    pub fn is_float(&self) -> bool {
        matches!(self, StateValue::Float(_))
    }

    /// Check if this value is an integer.
    pub fn is_int(&self) -> bool {
        matches!(self, StateValue::Int(_))
    }

    /// Check if this value is a boolean.
    pub fn is_bool(&self) -> bool {
        matches!(self, StateValue::Bool(_))
    }
}

impl StateMap {
    /// Create a new empty state map for the given module.
    pub fn new(module: impl Into<String>) -> Self {
        Self {
            module: module.into(),
            version: 1,
            data: BTreeMap::new(),
        }
    }

    /// Set the version number (builder pattern).
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Set a key-value pair in the state data.
    pub fn set(&mut self, key: impl Into<String>, value: StateValue) {
        self.data.insert(key.into(), value);
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&StateValue> {
        self.data.get(key)
    }

    /// Get a string value by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.data.get(key).and_then(|v| v.as_str())
    }

    /// Get a float value by key.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.data.get(key).and_then(|v| v.as_float())
    }

    /// Get an integer value by key.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.data.get(key).and_then(|v| v.as_int())
    }

    /// Get a boolean value by key.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.data.get(key).and_then(|v| v.as_bool())
    }

    /// Get a byte slice value by key.
    pub fn get_bytes(&self, key: &str) -> Option<&[u8]> {
        self.data.get(key).and_then(|v| v.as_bytes())
    }

    /// Get a nested map value by key.
    pub fn get_map(&self, key: &str) -> Option<&BTreeMap<String, StateValue>> {
        self.data.get(key).and_then(|v| v.as_map())
    }

    /// Remove a key from the state data, returning the value if it existed.
    pub fn remove(&mut self, key: &str) -> Option<StateValue> {
        self.data.remove(key)
    }

    /// Check if the state map contains a key.
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get the number of entries in the state data.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the state data is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &StateValue)> {
        self.data.iter()
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> crate::error::Result<String> {
        serde_json::to_string_pretty(self).map_err(crate::error::Error::Json)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> crate::error::Result<Self> {
        serde_json::from_str(json).map_err(crate::error::Error::Json)
    }

    /// Serialize to bincode bytes.
    pub fn to_binary(&self) -> crate::error::Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| crate::error::Error::Serialization(e.to_string()))
    }

    /// Deserialize from bincode bytes.
    pub fn from_binary(bytes: &[u8]) -> crate::error::Result<Self> {
        bincode::deserialize(bytes).map_err(|e| crate::error::Error::Serialization(e.to_string()))
    }
}

/// Trait for types that can save and load their state.
///
/// Implement this trait on modules that have learnable or configurable
/// state that should be persisted across sessions.
///
/// # Examples
///
/// ```
/// use kkachi::recursive::state::{StateMap, StateValue, Saveable};
///
/// struct MyModule {
///     prompt: String,
///     temperature: f64,
/// }
///
/// impl Saveable for MyModule {
///     fn dump_state(&self) -> StateMap {
///         let mut state = StateMap::new("MyModule");
///         state.set("prompt", StateValue::Str(self.prompt.clone()));
///         state.set("temperature", StateValue::Float(self.temperature));
///         state
///     }
///
///     fn load_state(&mut self, state: &StateMap) -> kkachi::error::Result<()> {
///         if let Some(p) = state.get_str("prompt") {
///             self.prompt = p.to_string();
///         }
///         if let Some(t) = state.get_float("temperature") {
///             self.temperature = t;
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait Saveable {
    /// Dump the current state to a [`StateMap`].
    fn dump_state(&self) -> StateMap;

    /// Load state from a [`StateMap`], updating internal fields.
    fn load_state(&mut self, state: &StateMap) -> crate::error::Result<()>;
}

/// Extension trait providing file I/O for [`Saveable`] types.
///
/// Automatically implemented for all types that implement [`Saveable`].
/// Provides convenience methods for saving/loading to JSON and binary files.
pub trait SaveableExt: Saveable {
    /// Save state to a JSON file.
    fn save_json(&self, path: &str) -> crate::error::Result<()>;

    /// Load state from a JSON file.
    fn load_json(&mut self, path: &str) -> crate::error::Result<()>;

    /// Save state to a binary (bincode) file.
    fn save_binary(&self, path: &str) -> crate::error::Result<()>;

    /// Load state from a binary (bincode) file.
    fn load_binary(&mut self, path: &str) -> crate::error::Result<()>;
}

impl<T: Saveable> SaveableExt for T {
    fn save_json(&self, path: &str) -> crate::error::Result<()> {
        let state = self.dump_state();
        let json = state.to_json()?;
        std::fs::write(path, json).map_err(|e| {
            crate::error::Error::Serialization(format!("Failed to write JSON to {}: {}", path, e))
        })
    }

    fn load_json(&mut self, path: &str) -> crate::error::Result<()> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            crate::error::Error::Serialization(format!(
                "Failed to read JSON from {}: {}",
                path, e
            ))
        })?;
        let state = StateMap::from_json(&json)?;
        self.load_state(&state)
    }

    fn save_binary(&self, path: &str) -> crate::error::Result<()> {
        let state = self.dump_state();
        let bytes = state.to_binary()?;
        std::fs::write(path, bytes).map_err(|e| {
            crate::error::Error::Serialization(format!(
                "Failed to write binary to {}: {}",
                path, e
            ))
        })
    }

    fn load_binary(&mut self, path: &str) -> crate::error::Result<()> {
        let bytes = std::fs::read(path).map_err(|e| {
            crate::error::Error::Serialization(format!(
                "Failed to read binary from {}: {}",
                path, e
            ))
        })?;
        let state = StateMap::from_binary(&bytes)?;
        self.load_state(&state)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // StateMap basic tests
    // ========================================================================

    #[test]
    fn test_state_map_new() {
        let state = StateMap::new("test_module");
        assert_eq!(state.module, "test_module");
        assert_eq!(state.version, 1);
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_state_map_with_version() {
        let state = StateMap::new("mod").with_version(3);
        assert_eq!(state.version, 3);
    }

    #[test]
    fn test_state_map_set_and_get() {
        let mut state = StateMap::new("mod");
        state.set("key", StateValue::Str("value".into()));

        assert!(state.contains_key("key"));
        assert!(!state.contains_key("missing"));
        assert_eq!(state.len(), 1);
        assert!(!state.is_empty());

        let val = state.get("key").unwrap();
        assert_eq!(val.as_str(), Some("value"));
    }

    #[test]
    fn test_state_map_typed_getters() {
        let mut state = StateMap::new("mod");
        state.set("s", StateValue::Str("hello".into()));
        state.set("f", StateValue::Float(3.14));
        state.set("i", StateValue::Int(42));
        state.set("b", StateValue::Bool(true));
        state.set("bytes", StateValue::Bytes(vec![1, 2, 3]));

        assert_eq!(state.get_str("s"), Some("hello"));
        assert_eq!(state.get_float("f"), Some(3.14));
        assert_eq!(state.get_int("i"), Some(42));
        assert_eq!(state.get_bool("b"), Some(true));
        assert_eq!(state.get_bytes("bytes"), Some([1u8, 2, 3].as_slice()));

        // Wrong type returns None
        assert_eq!(state.get_str("f"), None);
        assert_eq!(state.get_float("s"), None);
        assert_eq!(state.get_int("b"), None);
        assert_eq!(state.get_bool("i"), None);

        // Missing key returns None
        assert_eq!(state.get_str("missing"), None);
    }

    #[test]
    fn test_state_map_nested_map() {
        let mut inner = BTreeMap::new();
        inner.insert("nested_key".into(), StateValue::Int(99));

        let mut state = StateMap::new("mod");
        state.set("config", StateValue::Map(inner));

        let map = state.get_map("config").unwrap();
        assert_eq!(map.get("nested_key").unwrap().as_int(), Some(99));
    }

    #[test]
    fn test_state_map_list() {
        let mut state = StateMap::new("mod");
        state.set(
            "items",
            StateValue::List(vec![
                StateValue::Str("a".into()),
                StateValue::Str("b".into()),
                StateValue::Int(3),
            ]),
        );

        let val = state.get("items").unwrap();
        let list = val.as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_str(), Some("a"));
        assert_eq!(list[1].as_str(), Some("b"));
        assert_eq!(list[2].as_int(), Some(3));
    }

    #[test]
    fn test_state_map_remove() {
        let mut state = StateMap::new("mod");
        state.set("key", StateValue::Int(1));
        assert!(state.contains_key("key"));

        let removed = state.remove("key");
        assert_eq!(removed.unwrap().as_int(), Some(1));
        assert!(!state.contains_key("key"));

        assert!(state.remove("nonexistent").is_none());
    }

    #[test]
    fn test_state_map_iter() {
        let mut state = StateMap::new("mod");
        state.set("a", StateValue::Int(1));
        state.set("b", StateValue::Int(2));
        state.set("c", StateValue::Int(3));

        let keys: Vec<_> = state.iter().map(|(k, _)| k.as_str()).collect();
        // BTreeMap guarantees sorted order
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_state_map_overwrite() {
        let mut state = StateMap::new("mod");
        state.set("key", StateValue::Int(1));
        state.set("key", StateValue::Int(2));

        assert_eq!(state.get_int("key"), Some(2));
        assert_eq!(state.len(), 1);
    }

    // ========================================================================
    // StateValue tests
    // ========================================================================

    #[test]
    fn test_state_value_type_checks() {
        assert!(StateValue::Str("x".into()).is_str());
        assert!(StateValue::Float(1.0).is_float());
        assert!(StateValue::Int(1).is_int());
        assert!(StateValue::Bool(true).is_bool());

        assert!(!StateValue::Str("x".into()).is_int());
        assert!(!StateValue::Float(1.0).is_bool());
    }

    #[test]
    fn test_state_value_as_bytes_none() {
        assert!(StateValue::Str("x".into()).as_bytes().is_none());
    }

    #[test]
    fn test_state_value_as_map_none() {
        assert!(StateValue::Int(1).as_map().is_none());
    }

    #[test]
    fn test_state_value_as_list_none() {
        assert!(StateValue::Float(1.0).as_list().is_none());
    }

    // ========================================================================
    // JSON serialization roundtrip tests
    // ========================================================================

    #[test]
    fn test_json_roundtrip_simple() {
        let mut state = StateMap::new("test_mod").with_version(2);
        state.set("prompt", StateValue::Str("Write Rust code".into()));
        state.set("temperature", StateValue::Float(0.7));
        state.set("max_iter", StateValue::Int(5));
        state.set("verbose", StateValue::Bool(true));

        let json = state.to_json().unwrap();
        let restored = StateMap::from_json(&json).unwrap();

        assert_eq!(restored.module, "test_mod");
        assert_eq!(restored.version, 2);
        assert_eq!(restored.get_str("prompt"), Some("Write Rust code"));
        assert_eq!(restored.get_float("temperature"), Some(0.7));
        assert_eq!(restored.get_int("max_iter"), Some(5));
        assert_eq!(restored.get_bool("verbose"), Some(true));
    }

    #[test]
    fn test_json_roundtrip_nested() {
        let mut inner = BTreeMap::new();
        inner.insert("lr".into(), StateValue::Float(0.001));
        inner.insert("epochs".into(), StateValue::Int(10));

        let mut state = StateMap::new("optimizer");
        state.set("config", StateValue::Map(inner));
        state.set(
            "history",
            StateValue::List(vec![
                StateValue::Float(0.5),
                StateValue::Float(0.7),
                StateValue::Float(0.9),
            ]),
        );

        let json = state.to_json().unwrap();
        let restored = StateMap::from_json(&json).unwrap();

        let config = restored.get_map("config").unwrap();
        assert_eq!(config.get("lr").unwrap().as_float(), Some(0.001));
        assert_eq!(config.get("epochs").unwrap().as_int(), Some(10));

        let history = restored.get("history").unwrap().as_list().unwrap();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_json_roundtrip_empty() {
        let state = StateMap::new("empty");
        let json = state.to_json().unwrap();
        let restored = StateMap::from_json(&json).unwrap();

        assert_eq!(restored.module, "empty");
        assert_eq!(restored.version, 1);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_json_roundtrip_preserves_equality() {
        let mut state = StateMap::new("eq_test");
        state.set("a", StateValue::Str("hello".into()));
        state.set("b", StateValue::Int(42));

        let json = state.to_json().unwrap();
        let restored = StateMap::from_json(&json).unwrap();

        assert_eq!(state, restored);
    }

    #[test]
    fn test_json_invalid_input() {
        let result = StateMap::from_json("not valid json {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_json_pretty_format() {
        let mut state = StateMap::new("pretty");
        state.set("key", StateValue::Str("value".into()));

        let json = state.to_json().unwrap();
        // Pretty-printed JSON should contain newlines and indentation
        assert!(json.contains('\n'));
        assert!(json.contains("  "));
    }

    // ========================================================================
    // Binary serialization roundtrip tests
    // ========================================================================

    #[test]
    fn test_binary_roundtrip_simple() {
        let mut state = StateMap::new("bin_test").with_version(3);
        state.set("prompt", StateValue::Str("binary roundtrip".into()));
        state.set("score", StateValue::Float(0.95));
        state.set("count", StateValue::Int(100));
        state.set("active", StateValue::Bool(false));

        let bytes = state.to_binary().unwrap();
        let restored = StateMap::from_binary(&bytes).unwrap();

        assert_eq!(restored.module, "bin_test");
        assert_eq!(restored.version, 3);
        assert_eq!(restored.get_str("prompt"), Some("binary roundtrip"));
        assert_eq!(restored.get_float("score"), Some(0.95));
        assert_eq!(restored.get_int("count"), Some(100));
        assert_eq!(restored.get_bool("active"), Some(false));
    }

    #[test]
    fn test_binary_roundtrip_nested() {
        let mut inner = BTreeMap::new();
        inner.insert("key".into(), StateValue::Str("nested_value".into()));

        let mut state = StateMap::new("nested_bin");
        state.set("map", StateValue::Map(inner));
        state.set(
            "list",
            StateValue::List(vec![StateValue::Int(1), StateValue::Int(2)]),
        );
        state.set("raw", StateValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]));

        let bytes = state.to_binary().unwrap();
        let restored = StateMap::from_binary(&bytes).unwrap();

        assert_eq!(
            restored.get_map("map").unwrap().get("key").unwrap().as_str(),
            Some("nested_value")
        );
        assert_eq!(restored.get("list").unwrap().as_list().unwrap().len(), 2);
        assert_eq!(
            restored.get_bytes("raw"),
            Some([0xDE, 0xAD, 0xBE, 0xEF].as_slice())
        );
    }

    #[test]
    fn test_binary_roundtrip_empty() {
        let state = StateMap::new("empty_bin");
        let bytes = state.to_binary().unwrap();
        let restored = StateMap::from_binary(&bytes).unwrap();

        assert_eq!(state, restored);
    }

    #[test]
    fn test_binary_roundtrip_preserves_equality() {
        let mut state = StateMap::new("eq_bin");
        state.set("x", StateValue::Float(1.23456789));
        state.set("y", StateValue::Int(-999));

        let bytes = state.to_binary().unwrap();
        let restored = StateMap::from_binary(&bytes).unwrap();

        assert_eq!(state, restored);
    }

    #[test]
    fn test_binary_invalid_input() {
        let result = StateMap::from_binary(&[0xFF, 0x00, 0x01]);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_is_compact() {
        let mut state = StateMap::new("compact");
        state.set("key", StateValue::Str("value".into()));

        let json_bytes = state.to_json().unwrap().len();
        let bin_bytes = state.to_binary().unwrap().len();

        // Binary format should be more compact than pretty JSON
        assert!(bin_bytes < json_bytes);
    }

    // ========================================================================
    // Saveable trait tests
    // ========================================================================

    struct TestModule {
        prompt: String,
        temperature: f64,
        iterations: i64,
        enabled: bool,
    }

    impl TestModule {
        fn new() -> Self {
            Self {
                prompt: "default prompt".into(),
                temperature: 0.5,
                iterations: 10,
                enabled: true,
            }
        }
    }

    impl Saveable for TestModule {
        fn dump_state(&self) -> StateMap {
            let mut state = StateMap::new("TestModule").with_version(1);
            state.set("prompt", StateValue::Str(self.prompt.clone()));
            state.set("temperature", StateValue::Float(self.temperature));
            state.set("iterations", StateValue::Int(self.iterations));
            state.set("enabled", StateValue::Bool(self.enabled));
            state
        }

        fn load_state(&mut self, state: &StateMap) -> crate::error::Result<()> {
            if state.module != "TestModule" {
                return Err(crate::error::Error::Serialization(format!(
                    "Expected TestModule state, got {}",
                    state.module
                )));
            }
            if let Some(p) = state.get_str("prompt") {
                self.prompt = p.to_string();
            }
            if let Some(t) = state.get_float("temperature") {
                self.temperature = t;
            }
            if let Some(i) = state.get_int("iterations") {
                self.iterations = i;
            }
            if let Some(b) = state.get_bool("enabled") {
                self.enabled = b;
            }
            Ok(())
        }
    }

    #[test]
    fn test_saveable_dump_and_load() {
        let mut module = TestModule::new();
        module.prompt = "optimized prompt".into();
        module.temperature = 0.8;
        module.iterations = 20;
        module.enabled = false;

        let state = module.dump_state();
        assert_eq!(state.module, "TestModule");
        assert_eq!(state.get_str("prompt"), Some("optimized prompt"));

        let mut restored = TestModule::new();
        restored.load_state(&state).unwrap();
        assert_eq!(restored.prompt, "optimized prompt");
        assert_eq!(restored.temperature, 0.8);
        assert_eq!(restored.iterations, 20);
        assert!(!restored.enabled);
    }

    #[test]
    fn test_saveable_roundtrip_via_json() {
        let mut module = TestModule::new();
        module.prompt = "json roundtrip".into();
        module.temperature = 0.3;

        let state = module.dump_state();
        let json = state.to_json().unwrap();
        let restored_state = StateMap::from_json(&json).unwrap();

        let mut restored = TestModule::new();
        restored.load_state(&restored_state).unwrap();
        assert_eq!(restored.prompt, "json roundtrip");
        assert_eq!(restored.temperature, 0.3);
    }

    #[test]
    fn test_saveable_roundtrip_via_binary() {
        let mut module = TestModule::new();
        module.prompt = "binary roundtrip".into();
        module.iterations = 99;

        let state = module.dump_state();
        let bytes = state.to_binary().unwrap();
        let restored_state = StateMap::from_binary(&bytes).unwrap();

        let mut restored = TestModule::new();
        restored.load_state(&restored_state).unwrap();
        assert_eq!(restored.prompt, "binary roundtrip");
        assert_eq!(restored.iterations, 99);
    }

    #[test]
    fn test_saveable_load_wrong_module() {
        let state = StateMap::new("WrongModule");
        let mut module = TestModule::new();
        let result = module.load_state(&state);
        assert!(result.is_err());
    }

    #[test]
    fn test_saveable_partial_load() {
        // Loading a state with missing keys should keep defaults
        let mut state = StateMap::new("TestModule");
        state.set("prompt", StateValue::Str("partial".into()));
        // temperature, iterations, enabled are missing

        let mut module = TestModule::new();
        module.load_state(&state).unwrap();
        assert_eq!(module.prompt, "partial");
        assert_eq!(module.temperature, 0.5); // default preserved
        assert_eq!(module.iterations, 10); // default preserved
        assert!(module.enabled); // default preserved
    }

    // ========================================================================
    // SaveableExt file I/O tests
    // ========================================================================

    #[test]
    fn test_saveable_ext_json_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        let path_str = path.to_str().unwrap();

        let mut module = TestModule::new();
        module.prompt = "file test".into();
        module.temperature = 0.99;

        module.save_json(path_str).unwrap();

        let mut restored = TestModule::new();
        restored.load_json(path_str).unwrap();
        assert_eq!(restored.prompt, "file test");
        assert_eq!(restored.temperature, 0.99);
    }

    #[test]
    fn test_saveable_ext_binary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.bin");
        let path_str = path.to_str().unwrap();

        let mut module = TestModule::new();
        module.prompt = "binary file test".into();
        module.iterations = 77;

        module.save_binary(path_str).unwrap();

        let mut restored = TestModule::new();
        restored.load_binary(path_str).unwrap();
        assert_eq!(restored.prompt, "binary file test");
        assert_eq!(restored.iterations, 77);
    }

    #[test]
    fn test_saveable_ext_load_missing_file() {
        let mut module = TestModule::new();
        let result = module.load_json("/nonexistent/path/state.json");
        assert!(result.is_err());

        let result = module.load_binary("/nonexistent/path/state.bin");
        assert!(result.is_err());
    }

    // ========================================================================
    // Clone / Debug / PartialEq tests
    // ========================================================================

    #[test]
    fn test_state_map_clone() {
        let mut state = StateMap::new("clone_test");
        state.set("key", StateValue::Str("value".into()));

        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_state_map_debug() {
        let state = StateMap::new("debug_test");
        let debug = format!("{:?}", state);
        assert!(debug.contains("debug_test"));
    }

    #[test]
    fn test_state_value_clone() {
        let val = StateValue::List(vec![StateValue::Int(1), StateValue::Int(2)]);
        let cloned = val.clone();
        assert_eq!(val, cloned);
    }

    #[test]
    fn test_state_value_debug() {
        let val = StateValue::Float(3.14);
        let debug = format!("{:?}", val);
        assert!(debug.contains("3.14"));
    }
}
