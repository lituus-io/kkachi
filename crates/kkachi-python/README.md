# kkachi-python

Python bindings for the [kkachi](https://github.com/lituus-io/kkachi) LLM optimization library.

## Installation

```bash
pip install kkachi
```

## Usage

```python
from kkachi import Signature, Example

sig = Signature("question -> answer")
ex = Example({"question": "What is 2+2?", "answer": "4"})
```

## Memory / RAG with Persistent Storage

The `Memory` class provides a vector store for RAG (Retrieval-Augmented Generation) with full CRUD operations and optional persistent storage using DuckDB.

### Basic Usage (In-Memory)

```python
from kkachi import Memory

# Create in-memory store
mem = Memory()

# CREATE: Add documents
doc_id = mem.add("Document content here")
mem.add_with_id("custom-id", "Another document")
mem.add_tagged("category", "Tagged document")

# READ: Retrieve documents
content = mem.get(doc_id)
results = mem.search("query text", k=3)

# UPDATE: Modify existing documents
mem.update(doc_id, "Updated content")

# DELETE: Remove documents
mem.remove(doc_id)
```

### Persistent Storage

Enable DuckDB-backed persistent storage to preserve data across program restarts:

```python
from kkachi import Memory

# Create or open a persistent database
mem = Memory().persist("./my_rag_db.db")

# Add documents (persists to disk)
mem.add("Important knowledge that survives restarts")
mem.add("More documents here")

# Data persists across program runs
results = mem.search("knowledge", k=3)

# Close and reopen
del mem
mem = Memory().persist("./my_rag_db.db")  # Data is still there!
```

The database file will be created if it doesn't exist. Subsequent calls to `.persist()` with the same path will reopen the existing database.

**Requirements**:
- The `storage` feature is enabled by default in pip package
- DuckDB native library (usually handled automatically)

### Complete Example

See `examples/memory_persist.py` for a complete demonstration of CRUD operations with persistent storage.
