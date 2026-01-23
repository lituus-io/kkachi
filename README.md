# Kkachi

Recursive Language Prompting library for building LLM pipelines with iterative refinement, validation, and DSPy-style reasoning patterns.

## Installation

```toml
[dependencies]
kkachi = "0.1"
```

For persistent storage:
```toml
kkachi = { version = "0.1", features = ["storage"] }
```

## Quick Start

### Iterative Refinement

```rust
use kkachi::recursive::{refine, checks, MockLlm};

let llm = MockLlm::new(|prompt, _| {
    "fn parse(s: &str) -> Result<i32, ParseIntError> { s.parse() }".to_string()
});

let code = refine(&llm, "Write a string parser")
    .validate(checks().require("fn ").require("Result").forbid(".unwrap()"))
    .max_iter(5)
    .target(1.0)
    .go();
```

### CLI Validation Pipeline

Validate generated code using external tools:

```rust
use kkachi::recursive::{refine, cli, IterativeMockLlm};

let validator = cli("rustfmt")
    .args(&["--check"])
    .weight(0.1)
    .then("rustc")
    .args(&["--emit=metadata", "-o", "/dev/null"])
    .weight(0.6)
    .required()
    .then("cargo")
    .args(&["clippy", "--", "-D", "warnings"])
    .weight(0.3)
    .ext("rs");

let llm = IterativeMockLlm::new(|iter, _, _| match iter {
    0 => "fn main() { println!(\"hello\") }".into(),
    _ => "fn main() {\n    println!(\"hello\");\n}".into(),
});

let result = refine(&llm, "Write a hello world program")
    .validate(validator)
    .max_iter(5)
    .go_full()?;

println!("Score: {:.0}%, Iterations: {}", result.score * 100.0, result.iterations);
```

### Pattern-Based Validation

```rust
use kkachi::recursive::{checks, Validate};

let validator = checks()
    .require("fn ")
    .require("Result")
    .require("///")
    .forbid(".unwrap()")
    .forbid("panic!")
    .regex(r"fn \w+\(")
    .min_len(50);

let score = validator.validate("/// Parses.\nfn parse(s: &str) -> Result<i32, _> { s.parse() }");
println!("Score: {}", score.value);
```

### Semantic Validation (LLM-as-Judge)

```rust
use kkachi::recursive::{semantic, MockLlm, Validate};

let judge = MockLlm::new(|_, _| {
    r#"{"overall": 0.9, "confidence": 0.85}"#.to_string()
});

let validator = semantic(&judge)
    .criterion("Code is idiomatic Rust")
    .criterion("Error handling is complete")
    .threshold(0.8)
    .build();

let score = validator.validate("fn parse(s: &str) -> i32 { s.parse().unwrap() }");
println!("Semantic score: {:.2}", score.value);
```

### Validator Composition

Combine validators with `and`/`or`/`all`/`any`:

```rust
use kkachi::recursive::{checks, cli, semantic, ValidateExt, all, any, MockLlm, Validate};

let pattern = checks().require("fn ").forbid(".unwrap()");
let cli_check = cli("rustfmt").args(&["--check"]).ext("rs");

// Both must pass
let strict = pattern.clone().and(cli_check.clone());

// Either can pass
let lenient = pattern.clone().or(cli_check);

// Multi-validator: all must pass
let combined = all(vec![
    checks().require("fn "),
    checks().require("Result"),
    checks().forbid("panic!"),
]);

// With semantic validation
let judge = MockLlm::new(|_, _| r#"{"overall": 0.9}"#.to_string());
let quality = pattern.and(semantic(&judge).criterion("Clean code").threshold(0.8).build());

let score = quality.validate("fn parse(s: &str) -> Result<i32, _> { s.parse() }");
```

### Chain of Thought (Reasoning)

```rust
use kkachi::recursive::{reason, checks, MockLlm};

let llm = MockLlm::new(|_, _| {
    "Step 1: 25 * 30 = 750\nStep 2: 25 * 7 = 175\nStep 3: 750 + 175 = 925\n\nTherefore: 925".to_string()
});

let result = reason(&llm, "What is 25 * 37?")
    .validate(checks().regex(r"\d+"))
    .max_iter(5)
    .target(1.0)
    .go();

println!("Reasoning: {}", result.reasoning());
println!("Answer: {}", result.output);
println!("Iterations: {}", result.iterations);
```

### Best of N (Candidate Selection)

```rust
use kkachi::recursive::{best_of, checks, MockLlm};

let llm = MockLlm::new(|_, _| "A haiku about code\nCompiler speaks in errors\nBugs fade with the dawn".to_string());

let (result, pool) = best_of(&llm, "Write a haiku about programming", 5)
    .score_with(|output| {
        let lines: Vec<_> = output.lines().collect();
        if lines.len() == 3 { 1.0 } else { 0.0 }
    })
    .validate(checks().min_len(10))
    .go_with_pool();

println!("Best: {}", result.output);
println!("Score: {:.2}", result.score);

// Precision/recall tuning
let stats = pool.stats();
println!("Mean: {:.2}, StdDev: {:.2}", stats.mean, stats.std_dev);
let high_quality = pool.filter_by_threshold(0.9);
```

### Multi-Chain Ensemble

```rust
use kkachi::recursive::{ensemble, Aggregate, MockLlm};

let llm = MockLlm::new(|_, _| "Paris".to_string());

let (result, consensus) = ensemble(&llm, "What is the capital of France?", 7)
    .aggregate(Aggregate::MajorityVote)
    .go_with_consensus();

println!("Answer: {}", result.output);
println!("Agreement: {:.0}%", consensus.agreement_ratio() * 100.0);

if consensus.agreement_ratio() < 0.5 {
    println!("Low confidence - {} dissenting chains", consensus.dissenting_chains().count());
}
```

### ReAct Agent (Tool Calling)

```rust
use kkachi::recursive::{agent, tool, MockLlm};

let search = tool("search")
    .description("Search for information")
    .execute(|query| Ok(format!("Tokyo population: 14 million")));

let calc = tool("calculator")
    .description("Evaluate math expressions")
    .execute(|expr| Ok("7000000".to_string()));

let llm = MockLlm::new(|prompt, _| {
    if prompt.contains("Observation:") {
        "Thought: I have the answer\nFinal Answer: 7 million".to_string()
    } else {
        "Thought: I need to search\nAction: search\nAction Input: Tokyo population".to_string()
    }
});

let result = agent(&llm, "What is Tokyo's population divided by 2?")
    .tool(&search)
    .tool(&calc)
    .max_steps(10)
    .go();

println!("Answer: {}", result.output);
for step in result.trajectory() {
    println!("  {} -> {}", step.action, step.observation);
}
```

### Program of Thought (Code Execution)

```rust
use kkachi::recursive::{program, bash_executor, checks, MockLlm};

let llm = MockLlm::new(|_, _| "```bash\necho 42\n```".to_string());

let result = program(&llm, "Print 42")
    .executor(bash_executor())
    .validate(checks().regex(r"\d+"))
    .max_attempts(3)
    .go();

println!("Output: {}", result.output);
println!("Code: {}", result.code);
println!("Success: {}", result.success);
```

### Templates (Structured Prompts)

```rust
use kkachi::recursive::{Template, FormatType, PromptTone, TemplateExample};

// Builder-style
let template = Template::new("code_gen")
    .with_system_prompt("You are an expert Rust programmer.")
    .with_format(FormatType::Json)
    .with_tone(PromptTone::Restrictive)
    .strict(true)
    .with_example(TemplateExample::new(
        "Write hello world",
        r#"{"code": "fn main() { println!(\"Hello\"); }"}"#
    ));

// Render prompt
let prompt = template.assemble_prompt("Write a URL parser", 0, None);

// Validate output format
template.validate_output(r#"{"code": "fn parse() {}"}"#)?;

// Parse structured output
use serde::Deserialize;
#[derive(Deserialize)]
struct CodeOutput { code: String }
let parsed: CodeOutput = template.parse_output(r#"{"code": "fn main() {}"}"#)?;
```

### Memory / RAG

```rust
use kkachi::recursive::{refine, checks, memory, IterativeMockLlm};

let mut mem = memory();
mem.add_tagged("rust:io", "fn read_file(p: &str) -> io::Result<String> { fs::read_to_string(p) }");
mem.add_tagged("rust:json", "fn parse(s: &str) -> Result<Value, _> { serde_json::from_str(s) }");

let llm = IterativeMockLlm::new(|_, _, _| {
    "/// Reads config.\nfn read(p: &str) -> Result<Config, _> { Ok(toml::from_str(&fs::read_to_string(p)?)?) }".into()
});

let result = refine(&llm, "Write a config reader")
    .memory(&mut mem)
    .k(3)
    .validate(checks().require("fn ").require("Result").forbid(".unwrap()"))
    .max_iter(5)
    .learn_above(0.8)
    .go_full()?;
```

### Markdown Rewriting

```rust
use kkachi::recursive::rewrite::{rewrite, extract_code, extract_all_code};

let markdown = "# Config\n\n```yaml\nname: myapp\n```\n\n```rust\nfn main() {}\n```";

let yaml = extract_code(markdown, "yaml");
let all_rust = extract_all_code(markdown, "rust");

let updated = rewrite(markdown)
    .replace_code("yaml", "name: updated\nversion: 2.0")
    .build();
```

## API Reference

### Entry Points

| Function | Description |
|----------|-------------|
| `refine(llm, prompt)` | Iterative refinement pipeline |
| `reason(llm, prompt)` | Chain of Thought reasoning |
| `best_of(llm, prompt, n)` | Best of N candidate selection |
| `ensemble(llm, prompt, n)` | Multi-chain ensemble voting |
| `agent(llm, goal)` | ReAct agent with tools |
| `program(llm, problem)` | Code generation + execution |
| `checks()` | Pattern-based validator |
| `cli(cmd)` | CLI-based validator |
| `semantic(llm)` | LLM-as-judge validator |
| `memory()` | RAG memory store |
| `tool(name)` | Tool definition for agents |
| `Template::new(name)` | Structured prompt template |

### Validator Composition

| Function | Description |
|----------|-------------|
| `v1.and(v2)` | Both must pass |
| `v1.or(v2)` | At least one passes |
| `all(vec![...])` | All must pass |
| `any(vec![...])` | At least one passes |

### Code Executors

| Function | Language |
|----------|----------|
| `python_executor()` | Python |
| `node_executor()` | JavaScript/Node.js |
| `bash_executor()` | Bash |
| `ruby_executor()` | Ruby |

## Features

- `storage` — DuckDB persistent storage for Memory
- `storage-bundled` — Bundled DuckDB (no system install needed)
- `jinja` — Jinja2 template support
- `embeddings-onnx` — ONNX-based embeddings
- `hnsw` — HNSW vector index

## License

PolyForm Noncommercial 1.0.0
