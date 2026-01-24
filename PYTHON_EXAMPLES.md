# Kkachi Python Examples

Python bindings for the Kkachi LLM prompt optimization library.

## Installation

```bash
pip install kkachi anthropic
```

## Setup

Kkachi's DSPy-style builders accept any callable `(prompt, feedback) -> str` as an LLM:

```python
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY

def llm(prompt: str, feedback: str = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    if feedback:
        messages.append({"role": "user", "content": f"Feedback: {feedback}"})
    response = client.messages.create(
        model="your-model-id",
        max_tokens=4096,
        messages=messages,
    )
    return response.content[0].text
```

## Quick Start

```python
from kkachi import Kkachi

result = Kkachi.refine("Write a URL parser in Rust") \
    .require("fn ") \
    .require("Result") \
    .forbid(".unwrap()") \
    .max_iter(5) \
    .target(0.9) \
    .run(lambda i, prompt, feedback: llm(prompt, feedback))

print(f"Score: {result.score:.0%} after {result.iterations} iterations")
print(result.output)
```

> **Note:** `Kkachi.refine().run()` takes a callable `(iteration, prompt, feedback) -> str`.
> The DSPy-style builders below use the simpler `(prompt, feedback) -> str` signature.

## Features

### Iterative Refinement

```python
from kkachi import Kkachi

result = Kkachi.refine("Write a binary search in Rust") \
    .require("fn ") \
    .require("-> Option<usize>") \
    .forbid(".unwrap()") \
    .min_len(100) \
    .max_iter(5) \
    .target(1.0) \
    .run(lambda i, prompt, feedback: llm(prompt, feedback))

print(f"Score: {result.score:.2f}, Iterations: {result.iterations}")
```

### Best of N

Generate N candidates and select the highest-scoring:

```python
from kkachi import best_of, Checks

result, pool = best_of(llm, "Write a haiku about Rust", 5) \
    .metric(lambda output: 1.0 if len(output.strip().splitlines()) == 3 else 0.0) \
    .validate(Checks().min_len(10)) \
    .go_with_pool()

print(f"Best: {result.output} (score: {result.score:.2f})")
stats = pool.stats()
print(f"Pool mean: {stats.mean:.2f}, std: {stats.std_dev:.2f}")
```

### Ensemble Voting

Multiple chains vote on the answer:

```python
from kkachi import ensemble

result, consensus = ensemble(llm, "What is the capital of Australia?", 7) \
    .aggregate("majority_vote") \
    .go_with_consensus()

print(f"Answer: {result.output} (agreement: {consensus.agreement_ratio():.0%})")
for answer, count in consensus.vote_counts().items():
    print(f"  '{answer}': {count} votes")
```

### Chain of Thought

Step-by-step reasoning:

```python
from kkachi import reason, Checks

result = reason(llm, "A farmer has 17 sheep. All but 9 die. How many are left?") \
    .validate(Checks().regex(r"\d+")) \
    .max_iter(3) \
    .go()

print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.output}")
```

### ReAct Agent

Tool-calling agent with reasoning loop:

```python
from kkachi import agent, ToolDef

search = ToolDef("search", "Search the web for information",
    lambda query: f"Tokyo population: 14 million")

calc = ToolDef("calculator", "Evaluate a math expression",
    lambda expr: str(eval(expr)))

result = agent(llm, "What is Tokyo's population divided by 2?") \
    .tool(search) \
    .tool(calc) \
    .max_steps(10) \
    .go()

print(f"Answer: {result.output}")
for step in result.trajectory():
    print(f"  {step.action}({step.action_input}) -> {step.observation}")
```

### Program of Thought

Generate and execute code:

```python
from kkachi import program, Executor, Checks

result = program(llm, "Calculate the first 10 Fibonacci numbers") \
    .executor(Executor.python()) \
    .validate(Checks().regex(r"\d+")) \
    .max_iter(3) \
    .go()

print(f"Output: {result.output}")
print(f"Code: {result.code}")
print(f"Success: {result.success}")
```

### Validators

Pattern-based, CLI, and semantic validation with batch variants:

```python
from kkachi import Checks, Semantic, Validator, CliValidator

# Single checks
v = Checks() \
    .require("fn ") \
    .forbid(".unwrap()") \
    .regex(r"Result<.*>") \
    .min_len(50) \
    .max_len(500)

# Batch checks (multiple patterns at once)
v = Checks() \
    .require_all(["fn ", "->", "Result"]) \
    .forbid_all([".unwrap()", "panic!", "todo!"]) \
    .regex_all([r"fn \w+", r"-> \w+"])

# CLI validation (external tools)
v = CliValidator("rustfmt") \
    .args(["--check"]) \
    .then("rustc") \
    .args(["--emit=metadata", "-o", "/dev/null"]) \
    .required() \
    .ext("rs")

# LLM-as-judge
v = Semantic(llm) \
    .criterion("Code is idiomatic Rust") \
    .criterion("Error handling is complete") \
    .threshold(0.8)

# Compose validators
strict = Checks().require("fn ").and_(Semantic(llm).criterion("Clean code").threshold(0.8))
combined = Validator.all([
    Checks().require("fn "),
    Checks().forbid("panic!"),
    Semantic(llm).criterion("Well documented").threshold(0.7),
])
```

### Memory / RAG

Store and retrieve examples for few-shot learning:

```python
from kkachi import Memory

mem = Memory()
mem.add("fn read_file(p: &str) -> io::Result<String> { fs::read_to_string(p) }")
mem.add("fn parse_json(s: &str) -> Result<Value, _> { serde_json::from_str(s) }")

results = mem.search("how to read files", 3)
for r in results:
    print(f"Score: {r.score:.2f}, Content: {r.content}")
```

### Templates

Structured prompts with format enforcement:

```python
from kkachi import Template, FormatType, PromptTone

template = Template("code_gen") \
    .system_prompt("You are an expert Rust programmer.") \
    .format(FormatType.JSON) \
    .tone(PromptTone.RESTRICTIVE) \
    .strict(True) \
    .example("Write hello world", '{"code": "fn main() { println!(\"Hello\"); }"}')

prompt = template.render("Write a URL parser")
template.validate_output('{"code": "fn parse() {}"}')
data = template.parse_json('{"code": "fn parse() {}"}')
```

## API Reference

### Entry Points

| Function | Description |
|----------|-------------|
| `Kkachi.refine(prompt).run(fn)` | Iterative refinement (`fn(iter, prompt, feedback) -> str`) |
| `reason(llm, prompt)` | Chain of Thought reasoning |
| `best_of(llm, prompt, n)` | Best of N candidate selection |
| `ensemble(llm, prompt, n)` | Multi-chain ensemble voting |
| `agent(llm, goal)` | ReAct agent with tools |
| `program(llm, problem)` | Code generation + execution |

### Validators

| Class | Variants |
|-------|----------|
| `Checks()` | `.require()`, `.require_all()`, `.forbid()`, `.forbid_all()`, `.regex()`, `.regex_all()`, `.min_len()`, `.max_len()` |
| `CliValidator(cmd)` | `.args()`, `.ext()`, `.then()`, `.required()`, `.weight()` |
| `Semantic(llm)` | `.criterion()`, `.threshold()` |
| `v1.and_(v2)` | Both must pass |
| `v1.or_(v2)` | At least one passes |
| `Validator.all([...])` | All must pass |
| `Validator.any([...])` | At least one passes |

### Result Types

| Type | Key Fields |
|------|------------|
| `RefineResult` | `output`, `score`, `iterations` |
| `ReasonResult` | `output`, `reasoning`, `score`, `iterations` |
| `BestOfResult` | `output`, `score`, `candidates_generated` |
| `EnsembleResult` | `output`, `chains_generated` |
| `AgentResult` | `output`, `steps`, `success`, `trajectory()` |
| `ProgramResult` | `output`, `code`, `attempts`, `success` |

### LLM Callable

DSPy-style builders use `(prompt: str, feedback: str = None) -> str`:

```python
# Anthropic
import anthropic
client = anthropic.Anthropic()
def llm(prompt, feedback=None):
    messages = [{"role": "user", "content": prompt}]
    if feedback:
        messages.append({"role": "user", "content": f"Previous attempt feedback: {feedback}"})
    return client.messages.create(
        model="your-model-id", max_tokens=4096, messages=messages
    ).content[0].text

# OpenAI
from openai import OpenAI
client = OpenAI()
def llm(prompt, feedback=None):
    messages = [{"role": "user", "content": prompt}]
    if feedback:
        messages.append({"role": "user", "content": feedback})
    return client.chat.completions.create(
        model="gpt-4o", messages=messages
    ).choices[0].message.content
```

## License

PolyForm Noncommercial 1.0.0
