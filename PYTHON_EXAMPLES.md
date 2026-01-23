# Kkachi Python Examples

Python bindings for the Kkachi recursive language prompting library.

## Installation

```bash
pip install kkachi
```

## Quick Start

### Iterative Refinement

```python
from kkachi import Kkachi, Checks

def my_llm(prompt, context, feedback):
    return "fn parse(s: &str) -> Result<i32, _> { s.parse() }"

result = Kkachi.refine("Write a string parser") \
    .validate(Checks().require("fn ").require("Result").forbid(".unwrap()")) \
    .max_iter(5) \
    .target(0.9) \
    .run(my_llm)

print(f"Output: {result.output}")
print(f"Score: {result.score}")
```

### Pattern-Based Validation

```python
from kkachi import Checks

validator = Checks() \
    .require("fn ") \
    .require("Result") \
    .forbid(".unwrap()") \
    .forbid("panic!") \
    .regex(r"fn \w+\(") \
    .min_len(50)

score = validator.validate("/// Parses.\nfn parse(s: &str) -> Result<i32, _> { s.parse() }")
print(f"Score: {score.value}")
print(f"Feedback: {score.feedback}")
```

### CLI Validation

```python
from kkachi import Cli, CliPipeline

# Single command
simple = Cli("rustfmt").args(["--check"])

# Multi-stage pipeline
validator = CliPipeline() \
    .stage("format", Cli("rustfmt").args(["--check"]).weight(0.1)) \
    .stage("compile", Cli("rustc").args(["--emit=metadata", "-o", "/dev/null"]).weight(0.6).required()) \
    .stage("lint", Cli("cargo").args(["clippy", "--", "-D", "warnings"]).weight(0.3)) \
    .file_ext("rs")
```

### Semantic Validation (LLM-as-Judge)

```python
from kkachi import Semantic

def judge(prompt, feedback=None):
    # Your judge LLM - evaluates quality and returns JSON scores
    return '{"overall": 0.85, "confidence": 0.9}'

validator = Semantic(judge) \
    .criterion("Code is idiomatic Rust") \
    .criterion("Error handling is complete") \
    .criterion("Has documentation") \
    .threshold(0.8)

score = validator.validate("fn parse(s: &str) -> i32 { s.parse().unwrap() }")
print(f"Score: {score.value}")
print(f"Feedback: {score.feedback}")
print(f"Confidence: {score.confidence}")
```

### Validator Composition

```python
from kkachi import Checks, Semantic, Validator

# Pattern checks
checks = Checks().require("fn ").forbid(".unwrap()").min_len(50)

# Semantic validation
semantic = Semantic(judge) \
    .criterion("Code quality") \
    .threshold(0.8)

# Compose with .and_() / .or_()
strict = checks.and_(semantic)     # Both must pass
relaxed = checks.or_(semantic)     # At least one passes

# Multi-validator composition
all_pass = Validator.all([checks, semantic])
any_pass = Validator.any([checks, semantic])

# Chain compositions
triple = checks.and_(semantic).and_(Checks().regex(r"\d+"))

# Test composed validator directly
score = strict.validate("fn parse(s: &str) -> Result<i32, _> { s.parse() }")
print(f"Score: {score.value}, Passes: {score.passes(0.8)}")
```

### Chain of Thought (Reasoning)

```python
from kkachi import reason, Checks

def llm(prompt, feedback=None):
    return "Step 1: 25*30=750\nStep 2: 25*7=175\nStep 3: 750+175=925\n\nTherefore: 925"

result = reason(llm, "What is 25 * 37?") \
    .regex(r"\d+") \
    .max_iter(5) \
    .go()

print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.output}")
print(f"Iterations: {result.iterations}")

# With composed validator
result = reason(llm, "What is 25 * 37?") \
    .validate(Checks().regex(r"\d+").min_len(10)) \
    .max_iter(5) \
    .go()
```

### Best of N (Candidate Selection)

```python
from kkachi import best_of, Checks

def llm(prompt, feedback=None):
    return "A haiku about code\nCompiler speaks in errors\nBugs fade with the dawn"

result, pool = best_of(llm, "Write a haiku about programming", 5) \
    .score_with(lambda x: 1.0 if len(x.splitlines()) == 3 else 0.0) \
    .validate(Checks().min_len(10)) \
    .go_with_pool()

print(f"Best: {result.output}")
print(f"Score: {result.score:.2f}")

# Precision/recall tuning
stats = pool.stats()
print(f"Mean: {stats.mean:.2f}, StdDev: {stats.std_dev:.2f}")
high_quality = pool.filter_by_threshold(0.9)
top_3 = pool.top_k(3)
```

### Multi-Chain Ensemble

```python
from kkachi import ensemble

def llm(prompt, feedback=None):
    return "Paris"

result, consensus = ensemble(llm, "What is the capital of France?", 7) \
    .aggregate("majority_vote") \
    .go_with_consensus()

print(f"Answer: {result.output}")
print(f"Agreement: {consensus.agreement_ratio():.0%}")
print(f"Unanimous: {consensus.has_unanimous_agreement()}")

# Inspect votes
for answer, count in consensus.vote_counts().items():
    print(f"  '{answer}': {count} votes")

# Check dissenting opinions
for chain in consensus.dissenting_chains():
    print(f"  Dissent: '{chain.answer}'")
```

### ReAct Agent (Tool Calling)

```python
from kkachi import agent, ToolDef

# Define tools
search = ToolDef("search", "Search for information",
    lambda q: f"Tokyo population: 14 million")

calc = ToolDef("calculator", "Evaluate math",
    lambda expr: str(eval(expr)))

def llm(prompt, feedback=None):
    if "Observation:" in prompt:
        return "Thought: I have the answer\nFinal Answer: 7 million"
    return "Thought: I need to search\nAction: search\nAction Input: Tokyo population"

result = agent(llm, "What is Tokyo's population divided by 2?") \
    .tool(search) \
    .tool(calc) \
    .max_steps(10) \
    .go()

print(f"Answer: {result.output}")
print(f"Steps: {result.steps}")

for step in result.trajectory():
    print(f"  Thought: {step.thought}")
    print(f"  Action: {step.action}({step.action_input})")
    print(f"  Result: {step.observation}")
```

### Program of Thought (Code Execution)

```python
from kkachi import program, Executor, Checks

def llm(prompt, feedback=None):
    return "```python\nprint(42)\n```"

result = program(llm, "Print 42") \
    .executor(Executor.python()) \
    .validate(Checks().regex(r"\d+")) \
    .max_attempts(3) \
    .go()

print(f"Output: {result.output}")     # "42"
print(f"Code: {result.code}")
print(f"Success: {result.success}")

# Direct code execution
executor = Executor.bash().timeout(10)
exec_result = executor.execute("echo hello world")
print(f"stdout: {exec_result.stdout}")
print(f"exit_code: {exec_result.exit_code}")
```

### Templates (Structured Prompts)

```python
from kkachi import Template, FormatType, PromptTone

# Builder-style template
template = Template("code_gen") \
    .system_prompt("You are an expert Rust programmer.") \
    .format(FormatType.JSON) \
    .tone(PromptTone.RESTRICTIVE) \
    .strict(True) \
    .example("Write hello world", '{"code": "fn main() { println!(\"Hello\"); }"}') \
    .example("Add numbers", '{"code": "fn add(a: i32, b: i32) -> i32 { a + b }"}')

# Simple template
simple = Template.simple("Answer questions concisely.")

# Parse from YAML frontmatter + markdown
template = Template.from_str("""---
name: qa
format:
  type: json
  schema:
    type: object
    required: [answer]
options:
  tone: balanced
  strict: true
---
Answer the question precisely.

---examples---

## Example 1

**Input:** What is 2+2?

**Output:**
{"answer": "4"}
""")

# Render prompt
prompt = template.render("What is the capital of France?")

# Assemble with iteration context
prompt = template.assemble_prompt("Question", iteration=2, feedback="Be more specific")

# Validate output format
template.validate_output('{"answer": "Paris"}')  # OK

# Parse structured output
data = template.parse_json('{"answer": "Paris"}')  # -> dict
print(data["answer"])  # "Paris"

# Prompt tone utilities
tone = PromptTone.RESTRICTIVE
print(tone.default_threshold())   # 0.9
print(tone.favors_precision())    # True
print(tone.favors_recall())       # False
```

### Memory / RAG

```python
from kkachi import InMemoryVectorStore

store = InMemoryVectorStore()

# Add documents
store.add("Example content for retrieval")
store.add_tagged("rust:json", "use serde_json; fn parse() { ... }")
store.add_tagged("rust:yaml", "use serde_yaml; fn parse() { ... }")

# Search
results = store.search("how to parse JSON", 3)
for result in results:
    print(f"Score: {result.score:.2f}, Content: {result.content}")
```

### RAG + Refinement Pipeline

```python
from kkachi import Kkachi, Checks, InMemoryVectorStore

def my_llm(prompt, context, feedback):
    return "/// Reads config.\nfn read(p: &str) -> Result<Config, _> { ... }"

store = InMemoryVectorStore()
store.add_tagged("example:file", "fn read_file(p: &str) -> io::Result<String> { ... }")
store.add_tagged("example:json", "fn parse_json(s: &str) -> Result<Value, _> { ... }")

result = Kkachi.refine("Write a config reader") \
    .memory(store) \
    .k(3) \
    .validate(Checks().require("fn ").require("Result").forbid(".unwrap()")) \
    .max_iter(5) \
    .target(1.0) \
    .run(my_llm)
```

## API Reference

### Entry Point Functions

| Function | Description |
|----------|-------------|
| `reason(llm, prompt)` | Chain of Thought reasoning |
| `best_of(llm, prompt, n)` | Best of N candidate selection |
| `ensemble(llm, prompt, n)` | Multi-chain ensemble voting |
| `agent(llm, goal)` | ReAct agent with tools |
| `program(llm, problem)` | Code generation + execution |

### Validator Types

| Class | Description |
|-------|-------------|
| `Checks()` | Pattern-based validator |
| `Semantic(llm)` | LLM-as-judge validator |
| `Validator.all([...])` | All validators must pass |
| `Validator.any([...])` | At least one must pass |

### Composition

```python
v1.and_(v2)          # Both must pass
v1.or_(v2)           # At least one passes
Validator.all([...]) # All must pass
Validator.any([...]) # At least one passes
```

### ScoreResult

```python
score.value          # Score (0.0-1.0)
score.feedback       # Optional feedback string
score.confidence     # Optional confidence (0.0-1.0)
score.passes(0.8)    # Check against threshold
score.is_perfect()   # Check if 1.0
```

### Template

```python
Template(name)
    .system_prompt(text)
    .format(FormatType.JSON)
    .tone(PromptTone.BALANCED)
    .strict(True)
    .example(input, output)
    .render(input) -> str
    .assemble_prompt(q, iteration, feedback) -> str
    .validate_output(output)
    .parse_json(output) -> dict
```

### Executor

```python
Executor.python()    # Python executor
Executor.node()      # Node.js executor
Executor.bash()      # Bash executor
Executor.ruby()      # Ruby executor
    .timeout(secs)   # Set timeout
    .execute(code)   # Run code directly
```

### Result Types

| Type | Key Fields |
|------|------------|
| `ReasonResult` | `output`, `reasoning`, `score`, `iterations` |
| `BestOfResult` | `output`, `score`, `candidates_generated` |
| `EnsembleResult` | `output`, `chains_generated` |
| `AgentResult` | `output`, `steps`, `success`, `trajectory()` |
| `ProgramResult` | `output`, `code`, `attempts`, `success` |
| `ScoreResult` | `value`, `feedback`, `confidence` |
| `RefinementResult` | `output`, `score`, `iterations` |
