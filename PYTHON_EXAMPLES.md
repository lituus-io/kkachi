# Kkachi Python Examples

Comprehensive examples for using kkachi in Python.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Client Setup](#api-client-setup)
- [Iterative Refinement](#iterative-refinement)
- [Best-of-N Selection](#best-of-n-selection)
- [Memory & RAG](#memory--rag)
- [CLI Validators](#cli-validators)
- [Jinja2 Templates](#jinja2-templates)
- [Ensemble & Voting](#ensemble--voting)
- [Complete Workflow](#complete-workflow)

## Installation

```bash
pip install kkachi
```

## Quick Start

```python
from kkachi import refine, Checks, ApiLlm

# Create LLM client (auto-detects from environment)
llm = ApiLlm.from_env()

# Generate with iterative refinement
result = refine(llm, "Write a Python function to parse URLs") \
    .validate(
        Checks()
            .require("def ")
            .require("return")
            .forbid("eval(")
            .min_len(50)
    ) \
    .max_iter(5) \
    .target(0.9) \
    .go()

print(f"Score: {result.score * 100:.0f}%")
print(f"Iterations: {result.iterations}")
print(f"Output:\n{result.output}")
```

## API Client Setup

### Auto-Detection from Environment

```python
from kkachi import ApiLlm

# Automatically detects ANTHROPIC_API_KEY, OPENAI_API_KEY, or Claude CLI
llm = ApiLlm.from_env()
print(f"Using: {llm.model_name()}")
```

### Anthropic Claude

```python
llm = ApiLlm.anthropic(
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514"
)
```

### OpenAI

```python
llm = ApiLlm.openai(
    api_key="sk-...",
    model="gpt-4o"
)
```

### Claude Code CLI (No API Key)

```python
# Uses locally installed 'claude' CLI
llm = ApiLlm.claude_code()
```

### Custom Endpoints

```python
# Together.ai (OpenAI-compatible)
llm = ApiLlm.openai_with_url(
    api_key="your-together-key",
    model="meta-llama/Llama-3-70b-chat-hf",
    base_url="https://api.together.xyz"
)

# Groq (OpenAI-compatible)
llm = ApiLlm.openai_with_url(
    api_key="your-groq-key",
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai"
)
```

## Iterative Refinement

### Basic Refinement

```python
from kkachi import refine, Checks, ApiLlm

llm = ApiLlm.from_env()

result = refine(llm, "Write a function to validate email addresses") \
    .validate(
        Checks()
            .require("def validate_email")
            .require("@")
            .require("return")
            .forbid("eval")
            .min_len(100)
    ) \
    .max_iter(10) \
    .target(0.95) \
    .go()

print(f"Final score: {result.score:.2%}")
print(f"Took {result.iterations} iterations")
```

### Adaptive Target

Automatically increase target if easily achieved:

```python
result = refine(llm, "Write a sorting algorithm") \
    .validate(Checks().require("def sort").min_len(50)) \
    .max_iter(5) \
    .adaptive(0.8)  # Start at 80%, increase if achieved quickly
    .go()
```

### With Budget Limit

```python
result = refine(llm, "Explain recursion") \
    .validate(Checks().min_len(200)) \
    .with_budget(5000)  # Max 5000 tokens
    .go()
```

## Best-of-N Selection

### Basic Best-of-N

```python
from kkachi import best_of, Checks, ApiLlm

llm = ApiLlm.from_env()

result, pool = best_of(llm, "Write a haiku about Python") \
    .n(5) \
    .validate(Checks().min_len(10).max_len(100)) \
    .go_with_pool()

print(f"Best: {result.output}")
print(f"Score: {result.score:.2f}")
print(f"Pool stats: {pool.count()} candidates, mean={pool.mean():.2f}")
```

### Custom Metric Function

```python
def haiku_metric(output: str) -> float:
    """Score haikus based on line count."""
    lines = output.strip().split('\n')
    if len(lines) == 3:
        return 0.9
    return 0.3

result = best_of(llm, "Write a haiku about Rust") \
    .n(7) \
    .metric(haiku_metric) \
    .go()
```

### With Validation

```python
result = best_of(llm, "Write a limerick") \
    .n(10) \
    .metric(lambda text: 0.8 if len(text.split('\n')) == 5 else 0.2) \
    .validate(Checks().min_len(50).forbid("inappropriate")) \
    .go()
```

## Memory & RAG

### Basic Memory Usage

```python
from kkachi import Memory

# Create in-memory store
mem = Memory()

# Add documents
mem.add("Rust uses ownership for memory safety")
mem.add("Python uses reference counting and garbage collection")
mem.add("JavaScript is single-threaded with an event loop")

# Search with semantic similarity
results = mem.search("memory management", k=2)
for recall in results:
    print(f"Score: {recall.score:.2f} - {recall.content}")
```

### Persistent Memory

```python
# Persist to disk
mem = Memory().persist("./knowledge.db")

# Add documents (saved to disk)
mem.add("FastAPI is a modern Python web framework")
mem.add("Django is a batteries-included web framework")

# Search (from disk)
results = mem.search("web frameworks", k=2)
```

### Tagged Documents

```python
mem = Memory()

# Add with tags
mem.add("Rust is memory safe", tag="languages")
mem.add("Python is dynamically typed", tag="languages")
mem.add("FastAPI uses Pydantic", tag="frameworks")

# Search within tag
results = mem.search("typed", k=2, tag="languages")
```

### Diversity Search

```python
# Get diverse results (avoid near-duplicates)
results = mem.search_diverse("programming", k=5, min_similarity=0.7)
```

### List All Documents

```python
all_docs = mem.list()
for doc in all_docs:
    print(f"{doc.id}: {doc.content[:50]}...")
```

### Update and Delete

```python
# Update by ID
mem.update(doc_id, new_content="Updated content")

# Delete by ID
mem.delete(doc_id)

# Clear all
mem.clear()
```

## CLI Validators

### Basic CLI Validation

```python
from kkachi import refine, CliValidator, ApiLlm

llm = ApiLlm.from_env()

# Validate with rustfmt
validator = CliValidator("rustfmt").args(["--check"]).stdin()

result = refine(llm, "Write a Rust function") \
    .validate(validator) \
    .go()
```

### Combining Validators

```python
from kkachi import CliValidator, Checks

# CLI validation + pattern checks
validator = CliValidator("rustfmt") \
    .args(["--check"]) \
    .stdin() \
    .and_(Checks().forbid(".unwrap()").require("Result"))

result = refine(llm, "Write safe Rust code") \
    .validate(validator) \
    .go()
```

### Multiple CLI Tools

```python
# Format check AND lint check
validator = CliValidator("rustfmt") \
    .args(["--check"]) \
    .stdin() \
    .and_(CliValidator("cargo").args(["clippy", "--"]).stdin())
```

### With Working Directory

```python
validator = CliValidator("npm") \
    .args(["test"]) \
    .cwd("./my-project")
```

## Jinja2 Templates

### Basic Template

```python
from kkachi import JinjaTemplate, JinjaFormatter

template = JinjaTemplate.from_str("task", """
Task: {{ task }}

Requirements:
{% for req in requirements %}
- {{ req }}
{% endfor %}

Output format: {{ format }}
""")

formatter = JinjaFormatter() \
    .template(template) \
    .context({
        "task": "Parse URLs",
        "requirements": ["Handle edge cases", "Return Result type"],
        "format": "Rust code"
    })

# Use with refinement
result = refine(llm, formatter).go()
```

### Template from File

```python
template = JinjaTemplate.from_file("task", "./templates/code_gen.j2")

formatter = JinjaFormatter() \
    .template(template) \
    .context({"task": "URL parser", "language": "Rust"})
```

### Dynamic Context with Memory

```python
from kkachi import Memory, JinjaTemplate, JinjaFormatter

# Build knowledge base
mem = Memory()
mem.add("Use Result for errors")
mem.add("Avoid .unwrap() in production")

# Search for relevant patterns
patterns = mem.search("best practices", k=3)
pattern_list = [p.content for p in patterns]

# Create template with patterns
template = JinjaTemplate.from_str("task", """
Task: {{ task }}

Best Practices:
{% for pattern in patterns %}
- {{ pattern }}
{% endfor %}
""")

formatter = JinjaFormatter() \
    .template(template) \
    .context({"task": "Write safe code", "patterns": pattern_list})
```

## Ensemble & Voting

### Basic Ensemble

```python
from kkachi import ensemble, Checks, ApiLlm

llm = ApiLlm.from_env()

result = ensemble(llm, "Explain Rust ownership") \
    .strategies([
        "Explain with analogies",
        "Explain with code examples",
        "Explain step-by-step"
    ]) \
    .vote_threshold(2) \
    .validate(Checks().min_len(100)) \
    .go()
```

### Weighted Strategies

```python
result = ensemble(llm, "Design a REST API") \
    .strategies([
        ("Focus on security", 2.0),
        ("Focus on performance", 1.0),
        ("Focus on simplicity", 1.5)
    ]) \
    .go()
```

## Complete Workflow

### Memory + Templates + CLI Validation

```python
from kkachi import (
    Memory,
    JinjaTemplate,
    JinjaFormatter,
    CliValidator,
    Checks,
    refine,
    ApiLlm
)

# 1. Setup knowledge base
mem = Memory().persist("./code_patterns.db")
mem.add("Use Result<T, E> for error handling")
mem.add("Avoid .unwrap() in production code")
mem.add("Prefer &str over String in signatures")

# 2. Search for relevant patterns
patterns = mem.search("rust best practices", k=3)
pattern_texts = [p.content for p in patterns]

# 3. Create dynamic template
template = JinjaTemplate.from_str("code_gen", """
Task: {{ task }}

Best Practices:
{% for pattern in patterns %}
- {{ pattern }}
{% endfor %}

Requirements:
- Must compile with rustfmt
- Must pass clippy
- No .unwrap() calls
""")

formatter = JinjaFormatter() \
    .template(template) \
    .context({"task": "Parse URLs", "patterns": pattern_texts})

# 4. Create validator (CLI + pattern checks)
validator = CliValidator("rustfmt") \
    .args(["--check"]) \
    .stdin() \
    .and_(Checks().forbid(".unwrap()").require("Result"))

# 5. Generate with refinement
llm = ApiLlm.from_env()

result = refine(llm, formatter) \
    .validate(validator) \
    .max_iter(10) \
    .target(0.95) \
    .go()

print(f"Success! Score: {result.score:.2%}")
print(f"Iterations: {result.iterations}")
print(f"Output:\n{result.output}")
```

### Multi-Stage Pipeline

```python
from kkachi import best_of, refine, Checks, ApiLlm

llm = ApiLlm.from_env()

# Stage 1: Generate multiple candidates
print("Stage 1: Generating candidates...")
result, pool = best_of(llm, "Write a URL parser in Python") \
    .n(5) \
    .metric(lambda text: 0.8 if "def parse_url" in text else 0.3) \
    .go_with_pool()

print(f"Best candidate (score={result.score:.2f})")

# Stage 2: Refine the best candidate
print("\nStage 2: Refining best candidate...")
refined = refine(llm, f"Improve this code:\n{result.output}") \
    .validate(
        Checks()
            .require("def parse_url")
            .require("return")
            .forbid("eval")
            .min_len(100)
    ) \
    .max_iter(5) \
    .target(0.9) \
    .go()

print(f"Final score: {refined.score:.2%}")
print(f"Final output:\n{refined.output}")
```

## Error Handling

### Handling API Errors

```python
from kkachi import ApiLlm, refine, Checks

try:
    llm = ApiLlm.from_env()
    result = refine(llm, "Write a function").go()
except RuntimeError as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Validation Failures

```python
result = refine(llm, "Write code") \
    .validate(Checks().require("impossible_string")) \
    .max_iter(3) \
    .go()

if result.score < 0.5:
    print("Warning: Low score, validation may have failed")
    print(f"Best attempt:\n{result.output}")
```

## Advanced Features

### Custom Scoring

```python
def custom_scorer(output: str) -> float:
    """Complex scoring logic."""
    score = 0.0

    # Check structure
    if "def " in output:
        score += 0.3

    # Check length
    if 50 <= len(output) <= 500:
        score += 0.3

    # Check complexity
    if output.count("if ") >= 2:
        score += 0.2

    # Check documentation
    if '"""' in output:
        score += 0.2

    return min(score, 1.0)

result = best_of(llm, "Write a validator") \
    .n(5) \
    .metric(custom_scorer) \
    .go()
```

### Streaming Results

```python
# For long-running operations
result = refine(llm, "Write complex code") \
    .max_iter(20) \
    .validate(Checks().min_len(500)) \
    .go()

# Check progress
print(f"Completed in {result.iterations} iterations")
```

## Tips & Best Practices

### 1. Start Simple

Begin with basic validators and add complexity as needed:

```python
# Start simple
result = refine(llm, prompt).validate(Checks().min_len(50)).go()

# Add more validation as you learn what works
result = refine(llm, prompt) \
    .validate(
        Checks()
            .min_len(50)
            .require("def ")
            .forbid("eval")
    ) \
    .go()
```

### 2. Use Appropriate Iteration Limits

```python
# Simple tasks: 3-5 iterations
result = refine(llm, "Write a hello world").max_iter(3).go()

# Complex tasks: 10-15 iterations
result = refine(llm, "Write a web server").max_iter(15).go()
```

### 3. Combine Memory with Templates

```python
# Build reusable knowledge base
mem = Memory().persist("./patterns.db")

# Use in templates for consistent generation
patterns = mem.search("best practices", k=5)
# ... use in JinjaTemplate
```

### 4. Layer Validators

```python
# Layer validators from fast to slow
validator = Checks() \
    .min_len(50) \  # Fast: check length first
    .require("def ") \  # Fast: check patterns
    .and_(CliValidator("rustfmt").stdin())  # Slow: external validation last
```

## Next Steps

- **Main README**: See [README.md](README.md) for overview
- **Optimizer Guide**: Check [OPTIMIZER_GUIDE.md](OPTIMIZER_GUIDE.md)
- **Performance Baseline**: Review [PERFORMANCE_BASELINE.md](PERFORMANCE_BASELINE.md)
- **Rust Examples**: See [examples/](examples/) for Rust examples
- **Source Code**: Explore [crates/kkachi-python/](crates/kkachi-python/)

## Getting Help

- **Issues**: https://github.com/lituus-io/kkachi/issues
- **Discussions**: https://github.com/lituus-io/kkachi/discussions
