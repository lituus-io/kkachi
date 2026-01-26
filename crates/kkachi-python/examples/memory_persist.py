# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Example demonstrating persistent Memory storage with DuckDB and CRUD operations."""

from kkachi import Memory

# Create memory with persistent storage
mem = Memory().persist("./knowledge_base.db")

print("=== CREATE (Insert) ===")
# Add documents - they persist across restarts
doc1_id = mem.add("Q: How to parse JSON in Rust? A: Use serde_json::from_str()")
doc2_id = mem.add_with_id("rust-file-io", "Q: How to read files in Rust? A: Use std::fs::read_to_string()")
doc3_id = mem.add_tagged("http", "Q: How to make HTTP requests? A: Use reqwest crate")
print(f"Documents stored: {len(mem)}")

print("\n=== READ (Retrieve) ===")
# Get specific document
content = mem.get(doc1_id)
print(f"Document {doc1_id[:12]}...: {content[:50]}...")

# Search persisted documents
results = mem.search("JSON parsing", k=2)
for r in results:
    print(f"Score: {r.score:.2f} - {r.content[:50]}...")

print("\n=== UPDATE (Modify) ===")
# Update existing document
success = mem.update(doc1_id, "Q: How to parse JSON? A: Use serde_json crate with from_str()")
print(f"Updated document {doc1_id[:12]}...: {success}")
print(f"New content: {mem.get(doc1_id)[:60]}...")

print("\n=== DELETE (Remove) ===")
# Remove document
removed = mem.remove(doc3_id)
print(f"Removed document: {removed}")
print(f"Documents remaining: {len(mem)}")

print("\n=== PERSISTENCE TEST ===")
# Close and reopen - data persists
del mem

# Open existing database
mem2 = Memory().persist("./knowledge_base.db")
print(f"Reopened database with {len(mem2)} documents")

# Verify updated content persists
content = mem2.get(doc1_id)
print(f"Persisted update: {content[:60]}...")

# Search still works
results = mem2.search("Rust", k=2)
print(f"Found {len(results)} results for 'Rust'")
