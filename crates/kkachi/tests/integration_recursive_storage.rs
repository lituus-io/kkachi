// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for recursive context storage.

#[cfg(feature = "storage")]
mod storage_tests {
    use kkachi::recursive::{ContextId, ContextStore, ContextUpdate, ContextView, UpsertResult};

    #[test]
    fn test_full_storage_workflow() {
        // Create in-memory store
        let store = ContextStore::in_memory().expect("Failed to create in-memory store");

        let question = "How do I create an S3 bucket with Terraform?";
        let domain = "terraform";

        // Initially empty
        assert!(store.lookup(question, domain).is_none());

        // First insert
        let result = store
            .upsert(&ContextUpdate {
                question,
                domain,
                answer: "Use the aws_s3_bucket resource...",
                summary: "Create S3 bucket with aws_s3_bucket",
                score: 0.7,
                iterations: 2,
                error_corrections: &[],
            })
            .expect("Failed to upsert");

        assert_eq!(result, UpsertResult::Updated);

        // Verify lookup
        let view = store.lookup(question, domain).expect("Should find entry");
        assert_eq!(view.score, 0.7);
        assert_eq!(view.iterations, 2);

        // Lower score should skip
        let result = store
            .upsert(&ContextUpdate {
                question,
                domain,
                answer: "Worse answer",
                summary: "",
                score: 0.5,
                iterations: 3,
                error_corrections: &[],
            })
            .expect("Failed to upsert");

        assert_eq!(result, UpsertResult::Skipped);

        // Original answer preserved
        let view = store.lookup(question, domain).expect("Should find entry");
        assert!(view.answer.contains("aws_s3_bucket"));

        // Higher score should update
        let result = store
            .upsert(&ContextUpdate {
                question,
                domain,
                answer: "resource \"aws_s3_bucket\" \"example\" { bucket = \"my-bucket\" }",
                summary: "Complete S3 bucket with versioning",
                score: 0.95,
                iterations: 5,
                error_corrections: &[(
                    "Missing bucket name".to_string(),
                    "Added bucket parameter".to_string(),
                )],
            })
            .expect("Failed to upsert");

        assert_eq!(result, UpsertResult::Updated);

        // Verify update
        let view = store.lookup(question, domain).expect("Should find entry");
        assert_eq!(view.score, 0.95);
        assert_eq!(view.iterations, 5);
        assert!(view.answer.contains("my-bucket"));
    }

    #[test]
    fn test_context_id_consistency() {
        // Same question normalized differently should produce same ID
        let id1 = ContextId::from_question("What is Rust?", "programming");
        let id2 = ContextId::from_question("  WHAT   IS   RUST?  ", "programming");
        let id3 = ContextId::from_question("what is rust?", "programming");

        assert_eq!(id1, id2);
        assert_eq!(id2, id3);

        // Different domain = different ID
        let id4 = ContextId::from_question("What is Rust?", "games");
        assert_ne!(id1, id4);
    }

    #[test]
    fn test_list_by_domain() {
        let store = ContextStore::in_memory().expect("Failed to create store");

        // Insert multiple entries in same domain
        for i in 0..10 {
            store
                .upsert(&ContextUpdate {
                    question: &format!("Question {}", i),
                    domain: "test_domain",
                    answer: &format!("Answer {}", i),
                    summary: "",
                    score: (i as f32) / 10.0,
                    iterations: 1,
                    error_corrections: &[],
                })
                .expect("Failed to upsert");
        }

        // Also insert in different domain
        store
            .upsert(&ContextUpdate {
                question: "Other question",
                domain: "other_domain",
                answer: "Other answer",
                summary: "",
                score: 0.99,
                iterations: 1,
                error_corrections: &[],
            })
            .expect("Failed to upsert");

        // List by domain
        let results = store
            .list_by_domain("test_domain", 5)
            .expect("Failed to list");

        assert_eq!(results.len(), 5);

        // Should be sorted by score descending
        for i in 0..4 {
            assert!(results[i].1.score >= results[i + 1].1.score);
        }

        // Count
        let count = store
            .count_by_domain("test_domain")
            .expect("Failed to count");
        assert_eq!(count, 10);

        let other_count = store
            .count_by_domain("other_domain")
            .expect("Failed to count");
        assert_eq!(other_count, 1);
    }

    #[test]
    fn test_delete_context() {
        let store = ContextStore::in_memory().expect("Failed to create store");

        let question = "Test question";
        let domain = "test";

        // Insert
        store
            .upsert(&ContextUpdate {
                question,
                domain,
                answer: "Test answer",
                summary: "",
                score: 0.8,
                iterations: 1,
                error_corrections: &[],
            })
            .expect("Failed to upsert");

        // Verify exists
        assert!(store.lookup(question, domain).is_some());

        // Delete
        let id = ContextId::from_question(question, domain);
        let deleted = store.delete(&id).expect("Failed to delete");
        assert!(deleted);

        // Verify gone
        assert!(store.lookup(question, domain).is_none());

        // Delete again should return false
        let deleted_again = store.delete(&id).expect("Failed to delete");
        assert!(!deleted_again);
    }

    #[test]
    fn test_improvement_log() {
        let store = ContextStore::in_memory().expect("Failed to create store");

        let id = ContextId::from_question("test", "test");

        // Log improvements
        store
            .log_improvement(&id, 1, "Missing semicolon", "Added semicolon", 0.3, 0.5)
            .expect("Failed to log");

        store
            .log_improvement(&id, 2, "Type error", "Fixed type annotation", 0.5, 0.8)
            .expect("Failed to log");

        store
            .log_improvement(&id, 3, "Test failure", "Fixed test assertion", 0.8, 1.0)
            .expect("Failed to log");

        // Improvements are logged (no query API in current impl, but inserts succeed)
    }

    #[test]
    fn test_lookup_by_id() {
        let store = ContextStore::in_memory().expect("Failed to create store");

        let question = "Test lookup by ID";
        let domain = "test";

        store
            .upsert(&ContextUpdate {
                question,
                domain,
                answer: "Direct ID lookup answer",
                summary: "Summary",
                score: 0.9,
                iterations: 2,
                error_corrections: &[],
            })
            .expect("Failed to upsert");

        let id = ContextId::from_question(question, domain);

        // Lookup by ID directly
        let view = store.lookup_by_id(&id).expect("Should find by ID");
        assert_eq!(view.answer, "Direct ID lookup answer");
        assert_eq!(view.summary, "Summary");
        assert_eq!(view.score, 0.9);
    }
}

#[cfg(not(feature = "storage"))]
fn main() {
    // Storage feature not enabled, skip tests
}
