// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Multi-turn conversation support.
//!
//! Provides a [`Conversation`] builder that wraps any [`Llm`] and maintains
//! message history, formatting it into the generate call. This enables
//! multi-turn chat interactions without modifying the underlying Llm trait.
//!
//! # Example
//!
//! ```no_run
//! use kkachi::recursive::{MockLlm, Llm};
//! use kkachi::recursive::conversation::{Conversation, Role};
//!
//! # async fn example() {
//! let llm = MockLlm::new(|prompt, _| {
//!     if prompt.contains("ownership") {
//!         "Ownership is Rust's memory management system.".to_string()
//!     } else {
//!         "Here is an example of ownership.".to_string()
//!     }
//! });
//!
//! let mut chat = Conversation::new(&llm)
//!     .system("You are a helpful Rust tutor.");
//!
//! let response = chat.send("What is ownership?").await.unwrap();
//! assert!(response.contains("Ownership"));
//! # }
//! ```

use crate::error::Result;
use crate::recursive::llm::{Llm, LmOutput};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::fmt;

/// The role of a message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// System instructions that guide model behavior.
    System,
    /// User input messages.
    User,
    /// Assistant (model) responses.
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "System"),
            Self::User => write!(f, "User"),
            Self::Assistant => write!(f, "Assistant"),
        }
    }
}

/// A message in the conversation history.
#[derive(Debug, Clone)]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,
    /// The message content.
    pub content: String,
}

impl Message {
    /// Create a new message.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.role, self.content)
    }
}

/// A multi-turn conversation builder wrapping any [`Llm`].
///
/// Maintains message history and formats it into the `generate()` call,
/// enabling chat-like interactions over the single-turn Llm trait.
///
/// # Type Parameters
///
/// * `L` - The underlying Llm implementation (no boxing, fully generic).
pub struct Conversation<'a, L: Llm> {
    llm: &'a L,
    system: Option<Cow<'a, str>>,
    history: SmallVec<[Message; 8]>,
}

impl<'a, L: Llm> Conversation<'a, L> {
    /// Create a new conversation with the given LLM.
    pub fn new(llm: &'a L) -> Self {
        Self {
            llm,
            system: None,
            history: SmallVec::new(),
        }
    }

    /// Set the system message that guides model behavior.
    pub fn system(mut self, msg: &'a str) -> Self {
        self.system = Some(Cow::Borrowed(msg));
        self
    }

    /// Set the system message from an owned string.
    pub fn system_owned(mut self, msg: String) -> Self {
        self.system = Some(Cow::Owned(msg));
        self
    }

    /// Add a user message and get the assistant's response.
    ///
    /// This appends the user message to history, generates a response,
    /// and appends the assistant response to history.
    ///
    /// Returns a reference to the assistant's response text.
    pub async fn send(&mut self, msg: &str) -> Result<&str> {
        self.history.push(Message::new(Role::User, msg));
        let prompt = self.format_messages();
        let output: LmOutput = self.llm.generate(&prompt, "", None).await?;
        self.history
            .push(Message::new(Role::Assistant, output.text));
        Ok(&self.history.last().unwrap().content)
    }

    /// Add a user message with additional context and get the assistant's response.
    ///
    /// The context is passed to the LLM's context parameter (e.g., RAG results).
    pub async fn send_with_context(&mut self, msg: &str, context: &str) -> Result<&str> {
        self.history.push(Message::new(Role::User, msg));
        let prompt = self.format_messages();
        let output: LmOutput = self.llm.generate(&prompt, context, None).await?;
        self.history
            .push(Message::new(Role::Assistant, output.text));
        Ok(&self.history.last().unwrap().content)
    }

    /// Get the full conversation history (excludes system message).
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Get the system message, if set.
    pub fn system_message(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// Get the number of messages in the history.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if the conversation has no messages.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get the last assistant response, if any.
    pub fn last_response(&self) -> Option<&str> {
        self.history
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .map(|m| m.content.as_str())
    }

    /// Clear the conversation history (keeps the system message).
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Add a message to the history without generating a response.
    ///
    /// Useful for injecting context or pre-filling history.
    pub fn push(&mut self, role: Role, content: impl Into<String>) {
        self.history.push(Message::new(role, content));
    }

    /// Format all messages (system + history) into a single prompt string.
    fn format_messages(&self) -> String {
        let mut prompt = String::new();

        if let Some(ref system) = self.system {
            prompt.push_str("System: ");
            prompt.push_str(system);
            prompt.push_str("\n\n");
        }

        for msg in &self.history {
            prompt.push_str(&msg.role.to_string());
            prompt.push_str(": ");
            prompt.push_str(&msg.content);
            prompt.push('\n');
        }

        prompt
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::llm::MockLlm;

    #[tokio::test]
    async fn test_basic_conversation() {
        let llm = MockLlm::new(|_prompt, _| "Hello! How can I help?".to_string());

        let mut chat = Conversation::new(&llm);
        let response = chat.send("Hi there").await.unwrap();
        assert_eq!(response, "Hello! How can I help?");
        assert_eq!(chat.len(), 2); // user + assistant
    }

    #[tokio::test]
    async fn test_system_message() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("System: You are a pirate") {
                "Arrr! How can I help ye?".to_string()
            } else {
                "Hello!".to_string()
            }
        });

        let mut chat = Conversation::new(&llm).system("You are a pirate.");
        let response = chat.send("Hello").await.unwrap();
        assert_eq!(response, "Arrr! How can I help ye?");
        assert_eq!(chat.system_message(), Some("You are a pirate."));
    }

    #[tokio::test]
    async fn test_multi_turn() {
        let llm = MockLlm::new(|prompt, _| {
            // Check more specific condition first (second message includes history)
            if prompt.contains("And 3+3") {
                "6".to_string()
            } else if prompt.contains("What is 2+2") {
                "4".to_string()
            } else {
                "I don't understand".to_string()
            }
        });

        let mut chat = Conversation::new(&llm);

        let r1 = chat.send("What is 2+2?").await.unwrap();
        assert_eq!(r1, "4");

        let r2 = chat.send("And 3+3?").await.unwrap();
        assert_eq!(r2, "6");

        assert_eq!(chat.len(), 4); // 2 user + 2 assistant
        assert_eq!(chat.last_response(), Some("6"));
    }

    #[tokio::test]
    async fn test_history_included_in_prompt() {
        let llm = MockLlm::new(|prompt, _| {
            // Verify previous messages are included
            if prompt.contains("User: first message") && prompt.contains("User: second") {
                "I see both messages".to_string()
            } else {
                "Missing history".to_string()
            }
        });

        let mut chat = Conversation::new(&llm);
        chat.send("first message").await.unwrap();
        let r2 = chat.send("second").await.unwrap();
        assert_eq!(r2, "I see both messages");
    }

    #[tokio::test]
    async fn test_clear_history() {
        let llm = MockLlm::new(|_prompt, _| "response".to_string());

        let mut chat = Conversation::new(&llm).system("Be helpful.");
        chat.send("hello").await.unwrap();
        assert_eq!(chat.len(), 2);

        chat.clear();
        assert_eq!(chat.len(), 0);
        assert!(chat.is_empty());
        // System message preserved
        assert_eq!(chat.system_message(), Some("Be helpful."));
    }

    #[tokio::test]
    async fn test_push_message() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("User: pre-filled question")
                && prompt.contains("Assistant: pre-filled answer")
            {
                "I see the pre-filled history".to_string()
            } else {
                "No history found".to_string()
            }
        });

        let mut chat = Conversation::new(&llm);
        chat.push(Role::User, "pre-filled question");
        chat.push(Role::Assistant, "pre-filled answer");

        let response = chat.send("continue").await.unwrap();
        assert_eq!(response, "I see the pre-filled history");
    }

    #[tokio::test]
    async fn test_send_with_context() {
        let llm = MockLlm::new(|_prompt, _| "response with context".to_string());

        let mut chat = Conversation::new(&llm);
        let response = chat
            .send_with_context("question", "some RAG context")
            .await
            .unwrap();
        assert_eq!(response, "response with context");
    }

    #[test]
    fn test_message_display() {
        let msg = Message::new(Role::User, "hello");
        assert_eq!(format!("{}", msg), "User: hello");
    }

    #[test]
    fn test_role_display() {
        assert_eq!(format!("{}", Role::System), "System");
        assert_eq!(format!("{}", Role::User), "User");
        assert_eq!(format!("{}", Role::Assistant), "Assistant");
    }

    #[tokio::test]
    async fn test_last_response_none_when_empty() {
        let llm = MockLlm::new(|_prompt, _| "response".to_string());
        let chat = Conversation::new(&llm);
        assert_eq!(chat.last_response(), None);
    }

    #[tokio::test]
    async fn test_system_owned() {
        let llm = MockLlm::new(|prompt, _| {
            if prompt.contains("System: dynamic system") {
                "got it".to_string()
            } else {
                "nope".to_string()
            }
        });

        let system_msg = format!("dynamic {}", "system");
        let mut chat = Conversation::new(&llm).system_owned(system_msg);
        let response = chat.send("test").await.unwrap();
        assert_eq!(response, "got it");
    }
}
