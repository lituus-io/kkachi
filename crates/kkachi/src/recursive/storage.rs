// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! DuckDB-backed context storage for recursive refinement.
//!
//! Provides persistent storage of refined outputs with:
//! - Content-based addressing via blake3 hash
//! - Upsert-on-improvement semantics
//! - Semantic cache lookup support

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use std::path::Path;

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
use crate::error::{Error, Result};

/// Context ID - blake3 hash of normalized question + domain.
///
/// Consistent across recursive improvements for the same question.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct ContextId([u8; 16]);

impl ContextId {
    /// Generate consistent ID from question and domain.
    #[inline]
    pub fn from_question(question: &str, domain: &str) -> Self {
        let normalized = Self::normalize(question);
        let mut hasher = blake3::Hasher::new();
        hasher.update(normalized.as_bytes());
        hasher.update(b"::");
        hasher.update(domain.as_bytes());
        let hash = hasher.finalize();
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        Self(id)
    }

    /// Normalize question: lowercase, trim, collapse whitespace.
    #[inline]
    fn normalize(s: &str) -> String {
        s.to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Convert to hex string for SQL.
    #[inline]
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Get raw bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Create from raw bytes.
    #[inline]
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
}

impl std::fmt::Debug for ContextId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContextId({})", self.to_hex())
    }
}

impl std::fmt::Display for ContextId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Result of upserting a context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsertResult {
    /// Context was inserted or updated.
    Updated,
    /// Context was skipped (score not improved).
    Skipped,
}

/// Context view returned from storage lookup.
#[derive(Debug, Clone)]
pub struct ContextView {
    /// The stored answer.
    pub answer: String,
    /// Concise summary.
    pub summary: String,
    /// Quality score (0.0 - 1.0).
    pub score: f32,
    /// Number of iterations taken.
    pub iterations: u16,
}

/// Context update for upserting.
pub struct ContextUpdate<'a> {
    /// The question being answered.
    pub question: &'a str,
    /// Domain namespace.
    pub domain: &'a str,
    /// Full answer.
    pub answer: &'a str,
    /// Concise summary.
    pub summary: &'a str,
    /// Quality score.
    pub score: f32,
    /// Iteration count.
    pub iterations: u16,
    /// Error corrections made during refinement.
    pub error_corrections: &'a [(String, String)],
}

/// DuckDB-backed context store with zero-copy reads.
#[cfg(any(feature = "storage", feature = "storage-bundled"))]
pub struct ContextStore {
    /// DuckDB connection (embedded, no network).
    conn: super::db::Connection,
}

#[cfg(any(feature = "storage", feature = "storage-bundled"))]
impl ContextStore {
    /// Open or create context store at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = super::db::Connection::open(path.as_ref())
            .map_err(|e| Error::Other(format!("Failed to open DuckDB: {}", e)))?;

        // Disable extension autoloading to avoid dependency on external extensions
        conn.execute_batch(
            "SET autoload_known_extensions = false; SET autoinstall_known_extensions = false;",
        )
        .map_err(|e| Error::Other(format!("Failed to configure DuckDB: {}", e)))?;

        // Create tables if not exists
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS context (
                id BLOB PRIMARY KEY,
                question_hash BIGINT,
                question_text TEXT,
                domain VARCHAR,
                answer TEXT,
                summary VARCHAR(500),
                score FLOAT,
                iteration_count USMALLINT,
                error_corrections TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_question_hash ON context(question_hash);
            CREATE INDEX IF NOT EXISTS idx_domain_score ON context(domain, score DESC);

            CREATE TABLE IF NOT EXISTS improvement_log (
                id INTEGER PRIMARY KEY,
                context_id BLOB,
                iteration USMALLINT,
                error_message TEXT,
                correction TEXT,
                score_before FLOAT,
                score_after FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )
        .map_err(|e| Error::Other(format!("Failed to create tables: {}", e)))?;

        Ok(Self { conn })
    }

    /// Create an in-memory context store (useful for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = super::db::Connection::open_in_memory()
            .map_err(|e| Error::Other(format!("Failed to open in-memory DuckDB: {}", e)))?;

        // Disable extension autoloading to avoid dependency on external extensions
        conn.execute_batch(
            "SET autoload_known_extensions = false; SET autoinstall_known_extensions = false;",
        )
        .map_err(|e| Error::Other(format!("Failed to configure DuckDB: {}", e)))?;

        // Create tables
        conn.execute_batch(
            r#"
            CREATE TABLE context (
                id BLOB PRIMARY KEY,
                question_hash BIGINT,
                question_text TEXT,
                domain VARCHAR,
                answer TEXT,
                summary VARCHAR(500),
                score FLOAT,
                iteration_count USMALLINT,
                error_corrections TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE improvement_log (
                id INTEGER PRIMARY KEY,
                context_id BLOB,
                iteration USMALLINT,
                error_message TEXT,
                correction TEXT,
                score_before FLOAT,
                score_after FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )
        .map_err(|e| Error::Other(format!("Failed to create tables: {}", e)))?;

        Ok(Self { conn })
    }

    /// Lookup context by question and domain.
    pub fn lookup(&self, question: &str, domain: &str) -> Option<ContextView> {
        let id = ContextId::from_question(question, domain);
        let hash = Self::question_hash(question);

        self.conn
            .query_row(
                "SELECT answer, summary, score, iteration_count
                 FROM context WHERE question_hash = ? AND id = ?",
                super::db::params![hash as i64, id.as_bytes()],
                |row| {
                    Ok(ContextView {
                        answer: row.get(0)?,
                        summary: row.get(1)?,
                        score: row.get(2)?,
                        iterations: row.get(3)?,
                    })
                },
            )
            .ok()
    }

    /// Lookup by context ID directly.
    pub fn lookup_by_id(&self, id: &ContextId) -> Option<ContextView> {
        self.conn
            .query_row(
                "SELECT answer, summary, score, iteration_count
                 FROM context WHERE id = ?",
                [id.as_bytes()],
                |row| {
                    Ok(ContextView {
                        answer: row.get(0)?,
                        summary: row.get(1)?,
                        score: row.get(2)?,
                        iterations: row.get(3)?,
                    })
                },
            )
            .ok()
    }

    /// Upsert context - only if score improved or new.
    pub fn upsert(&self, ctx: &ContextUpdate<'_>) -> Result<UpsertResult> {
        let id = ContextId::from_question(ctx.question, ctx.domain);
        let hash = Self::question_hash(ctx.question);

        // Check existing score
        let existing_score: Option<f32> = self
            .conn
            .query_row(
                "SELECT score FROM context WHERE id = ?",
                [id.as_bytes()],
                |row| row.get(0),
            )
            .ok();

        // Only update if score improved or new entry
        if let Some(old_score) = existing_score {
            if ctx.score <= old_score {
                return Ok(UpsertResult::Skipped);
            }
        }

        // Serialize error corrections
        let error_corrections_json =
            serde_json::to_string(&ctx.error_corrections).unwrap_or_else(|_| "[]".to_string());

        // Upsert with ON CONFLICT
        self.conn
            .execute(
                r#"INSERT INTO context
                   (id, question_hash, question_text, domain, answer, summary, score,
                    iteration_count, error_corrections, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, now(), now())
                   ON CONFLICT (id) DO UPDATE SET
                       answer = excluded.answer,
                       summary = excluded.summary,
                       score = excluded.score,
                       iteration_count = excluded.iteration_count,
                       error_corrections = excluded.error_corrections,
                       updated_at = now()"#,
                super::db::params![
                    id.as_bytes(),
                    hash as i64,
                    ctx.question,
                    ctx.domain,
                    ctx.answer,
                    ctx.summary,
                    ctx.score,
                    ctx.iterations as i32,
                    error_corrections_json,
                ],
            )
            .map_err(|e| Error::Other(format!("Failed to upsert context: {}", e)))?;

        Ok(UpsertResult::Updated)
    }

    /// Delete a context by ID.
    pub fn delete(&self, id: &ContextId) -> Result<bool> {
        let rows = self
            .conn
            .execute("DELETE FROM context WHERE id = ?", [id.as_bytes()])
            .map_err(|e| Error::Other(format!("Failed to delete context: {}", e)))?;

        Ok(rows > 0)
    }

    /// Get all contexts for a domain, ordered by score descending.
    pub fn list_by_domain(
        &self,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<(ContextId, ContextView)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, answer, summary, score, iteration_count
                 FROM context WHERE domain = ?
                 ORDER BY score DESC LIMIT ?",
            )
            .map_err(|e| Error::Other(format!("Failed to prepare statement: {}", e)))?;

        let rows = stmt
            .query_map(super::db::params![domain, limit as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let mut id_arr = [0u8; 16];
                id_arr.copy_from_slice(&id_bytes);
                Ok((
                    ContextId::from_bytes(id_arr),
                    ContextView {
                        answer: row.get(1)?,
                        summary: row.get(2)?,
                        score: row.get(3)?,
                        iterations: row.get(4)?,
                    },
                ))
            })
            .map_err(|e| Error::Other(format!("Failed to query: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Error::Other(format!("Failed to read row: {}", e)))?);
        }
        Ok(results)
    }

    /// Get count of contexts in a domain.
    pub fn count_by_domain(&self, domain: &str) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM context WHERE domain = ?",
                [domain],
                |row| row.get(0),
            )
            .map_err(|e| Error::Other(format!("Failed to count: {}", e)))?;

        Ok(count as usize)
    }

    /// Log an improvement made during refinement.
    pub fn log_improvement(
        &self,
        context_id: &ContextId,
        iteration: u16,
        error_message: &str,
        correction: &str,
        score_before: f32,
        score_after: f32,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO improvement_log
                 (context_id, iteration, error_message, correction, score_before, score_after)
                 VALUES (?, ?, ?, ?, ?, ?)",
                super::db::params![
                    context_id.as_bytes(),
                    iteration as i32,
                    error_message,
                    correction,
                    score_before,
                    score_after,
                ],
            )
            .map_err(|e| Error::Other(format!("Failed to log improvement: {}", e)))?;

        Ok(())
    }

    /// Compute hash of normalized question for fast lookup.
    #[inline]
    fn question_hash(question: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        ContextId::normalize(question).hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_id_generation() {
        let id1 = ContextId::from_question("What is Rust?", "programming");
        let id2 = ContextId::from_question("what is rust?", "programming"); // Different case
        let id3 = ContextId::from_question("What is Rust?", "general"); // Different domain

        // Same question (normalized) + same domain = same ID
        assert_eq!(id1, id2);

        // Different domain = different ID
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_context_id_normalization() {
        let id1 = ContextId::from_question("  What   is   Rust?  ", "test");
        let id2 = ContextId::from_question("what is rust?", "test");

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_context_id_hex() {
        let id = ContextId::from_question("test", "domain");
        let hex = id.to_hex();

        assert_eq!(hex.len(), 32); // 16 bytes = 32 hex chars
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[cfg(any(feature = "storage", feature = "storage-bundled"))]
    #[test]
    fn test_context_store_in_memory() {
        let store = ContextStore::in_memory().unwrap();

        // Initially empty
        assert!(store.lookup("test question", "test").is_none());

        // Insert
        let result = store
            .upsert(&ContextUpdate {
                question: "test question",
                domain: "test",
                answer: "test answer",
                summary: "summary",
                score: 0.8,
                iterations: 3,
                error_corrections: &[],
            })
            .unwrap();

        assert_eq!(result, UpsertResult::Updated);

        // Lookup
        let view = store.lookup("test question", "test").unwrap();
        assert_eq!(view.answer, "test answer");
        assert_eq!(view.score, 0.8);
        assert_eq!(view.iterations, 3);
    }

    #[cfg(any(feature = "storage", feature = "storage-bundled"))]
    #[test]
    fn test_context_store_upsert_only_on_improvement() {
        let store = ContextStore::in_memory().unwrap();

        // Insert with score 0.8
        store
            .upsert(&ContextUpdate {
                question: "test",
                domain: "test",
                answer: "answer v1",
                summary: "",
                score: 0.8,
                iterations: 1,
                error_corrections: &[],
            })
            .unwrap();

        // Try to update with lower score - should skip
        let result = store
            .upsert(&ContextUpdate {
                question: "test",
                domain: "test",
                answer: "answer v2",
                summary: "",
                score: 0.7,
                iterations: 2,
                error_corrections: &[],
            })
            .unwrap();

        assert_eq!(result, UpsertResult::Skipped);

        // Verify original answer is still there
        let view = store.lookup("test", "test").unwrap();
        assert_eq!(view.answer, "answer v1");

        // Update with higher score - should update
        let result = store
            .upsert(&ContextUpdate {
                question: "test",
                domain: "test",
                answer: "answer v3",
                summary: "",
                score: 0.9,
                iterations: 3,
                error_corrections: &[],
            })
            .unwrap();

        assert_eq!(result, UpsertResult::Updated);

        // Verify new answer
        let view = store.lookup("test", "test").unwrap();
        assert_eq!(view.answer, "answer v3");
        assert_eq!(view.score, 0.9);
    }

    #[cfg(any(feature = "storage", feature = "storage-bundled"))]
    #[test]
    fn test_context_store_list_by_domain() {
        let store = ContextStore::in_memory().unwrap();

        // Insert multiple contexts
        for i in 0..5 {
            store
                .upsert(&ContextUpdate {
                    question: &format!("question {}", i),
                    domain: "test_domain",
                    answer: &format!("answer {}", i),
                    summary: "",
                    score: i as f32 / 10.0,
                    iterations: 1,
                    error_corrections: &[],
                })
                .unwrap();
        }

        // List should return in score order (descending)
        let results = store.list_by_domain("test_domain", 10).unwrap();
        assert_eq!(results.len(), 5);

        // Verify descending score order
        for i in 0..4 {
            assert!(results[i].1.score >= results[i + 1].1.score);
        }
    }
}
