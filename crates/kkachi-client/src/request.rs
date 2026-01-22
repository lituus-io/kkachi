// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! LM request types

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message<'a> {
    /// Role (system, user, assistant)
    #[serde(borrow)]
    pub role: Cow<'a, str>,

    /// Content
    #[serde(borrow)]
    pub content: Cow<'a, str>,
}

impl<'a> Message<'a> {
    /// Create a system message
    pub fn system(content: impl Into<Cow<'a, str>>) -> Self {
        Self {
            role: Cow::Borrowed("system"),
            content: content.into(),
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<Cow<'a, str>>) -> Self {
        Self {
            role: Cow::Borrowed("user"),
            content: content.into(),
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<Cow<'a, str>>) -> Self {
        Self {
            role: Cow::Borrowed("assistant"),
            content: content.into(),
        }
    }
}

/// Request to language model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMRequest<'a> {
    /// Messages in the conversation
    #[serde(borrow)]
    pub messages: Vec<Message<'a>>,

    /// Optional system prompt
    #[serde(borrow)]
    pub system: Option<Cow<'a, str>>,

    /// Override temperature
    pub temperature: Option<f32>,

    /// Override max tokens
    pub max_tokens: Option<u32>,
}

impl<'a> LMRequest<'a> {
    /// Create a new request
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            system: None,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Add a message
    pub fn add_message(mut self, message: Message<'a>) -> Self {
        self.messages.push(message);
        self
    }

    /// Set system prompt
    pub fn with_system(mut self, system: impl Into<Cow<'a, str>>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl<'a> Default for LMRequest<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_system() {
        let msg = Message::system("You are helpful");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are helpful");
    }

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_message_assistant() {
        let msg = Message::assistant("Hi there");
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "Hi there");
    }

    #[test]
    fn test_lm_request_builder() {
        let req = LMRequest::new()
            .add_message(Message::user("test"))
            .with_system("sys")
            .with_temperature(0.7)
            .with_max_tokens(100);

        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.system, Some(Cow::Borrowed("sys")));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
    }
}
