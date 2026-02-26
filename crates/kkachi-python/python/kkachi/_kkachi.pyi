# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Type stubs for the kkachi._kkachi module."""

from typing import Callable, Optional, Dict, List, Tuple, Any

__version__: str

# =============================================================================
# LLM Implementations
# =============================================================================

class ApiLlm:
    """Real LLM API client supporting Anthropic, OpenAI, and Claude Code CLI.

    Example:
        llm = ApiLlm.from_env()
        llm = ApiLlm.anthropic("sk-ant-...", "claude-sonnet-4-20250514")
        llm = ApiLlm.openai("sk-...", "gpt-4o")
        llm = ApiLlm.claude_code()

        # With optimization wrappers
        llm = llm.with_cache(100).with_rate_limit(10.0).with_retry(3)
    """

    @staticmethod
    def from_env() -> ApiLlm:
        """Auto-detect LLM provider from environment variables."""
        ...
    @staticmethod
    def anthropic(api_key: str, model: str) -> ApiLlm: ...
    @staticmethod
    def anthropic_with_url(api_key: str, model: str, base_url: str) -> ApiLlm: ...
    @staticmethod
    def openai(api_key: str, model: str) -> ApiLlm: ...
    @staticmethod
    def openai_with_url(api_key: str, model: str, base_url: str) -> ApiLlm: ...
    @staticmethod
    def claude_code() -> ApiLlm: ...
    def with_cache(self, capacity: int) -> ApiLlm: ...
    def with_rate_limit(self, requests_per_second: float) -> ApiLlm: ...
    def with_retry(self, max_retries: int) -> ApiLlm: ...
    def with_timeout(self, secs: int) -> ApiLlm:
        """Set HTTP request timeout in seconds (default: 300)."""
        ...
    def model_name(self) -> str: ...
    def max_context(self) -> int: ...
    def __call__(self, prompt: str, feedback: Optional[str] = None) -> str: ...

# =============================================================================
# Runtime Defaults
# =============================================================================

class DefaultAnnotation:
    """Annotation metadata for a single default entry."""
    key: str
    original_pattern: str
    replacement: str
    note: Optional[str]
    source: str

    def __repr__(self) -> str: ...

class Defaults:
    """Runtime defaults for regex substitution on LLM output.

    Stores regex->replacement pairs with metadata. Integrates at two points:
    1. Prompt injection via `.context()` — renders a summary for the LLM
    2. Output transform via `.apply(text)` — regex replacements before validation

    Example:
        defaults = (
            Defaults()
            .set("iam_user", r"user:\\S+@example\\.com", "user:mark@company.com",
                 note="Replace with actual IAM user")
            .from_env("project", r"my-gcp-project",
                      "GOOGLE_CLOUD_PROJECT", fallback="default-proj")
        )

        # Inject into prompt
        prompt = f"...{defaults.context()}..."

        # Use with reason()
        result = reason(llm, prompt).defaults(defaults).go()
    """

    def __init__(self) -> None: ...
    def set(self, key: str, pattern: str, replacement: str, note: Optional[str] = None) -> Defaults: ...
    def from_env(self, key: str, pattern: str, env_var: str, fallback: str, note: Optional[str] = None) -> Defaults: ...
    def apply(self, text: str) -> str: ...
    def context(self) -> str: ...
    def annotations(self) -> List[DefaultAnnotation]: ...
    def is_empty(self) -> bool: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

# =============================================================================
# Skills (Persistent Prompt Context)
# =============================================================================

class Skill:
    """A reusable set of instructions injected into LLM prompts.

    Example:
        skill = Skill() \\
            .instruct("naming", "Use snake_case for all names.") \\
            .instruct("policy", "Always set deletionProtection: false.")

        result = reason(llm, "Generate config").skill(skill).go()
    """

    def __init__(self) -> None: ...
    def instruct(self, label: str, instruction: str) -> Skill: ...
    def instruct_at(self, label: str, instruction: str, priority: int = 128) -> Skill: ...
    def render(self) -> str: ...
    def is_empty(self) -> bool: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

# =============================================================================
# Memory / RAG
# =============================================================================

class Recall:
    """Result from a memory search."""
    id: str
    content: str
    score: float
    tag: Optional[str]

    def __init__(self, id: str, content: str, score: float) -> None: ...

class PackageResult:
    """Result from packaging a Memory into a pip wheel."""
    wheel_path: str
    wheel_name: str
    size_bytes: int
    db_size_bytes: int
    file_count: int

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Memory:
    """In-memory vector store for RAG.

    Example:
        mem = Memory()
        doc_id = mem.add("Example content")
        results = mem.search("query", 3)
    """

    def __init__(self) -> None: ...
    def add(self, content: str) -> str: ...
    def add_with_id(self, id: str, content: str) -> None: ...
    def add_tagged(self, tag: str, content: str) -> str: ...
    def search(self, query: str, k: int) -> List[Recall]: ...
    def get(self, id: str) -> Optional[str]: ...
    def remove(self, id: str) -> bool: ...
    def update(self, id: str, content: str) -> bool: ...
    def is_empty(self) -> bool: ...
    def tags(self) -> List[str]: ...
    def persist(self, path: str) -> Memory: ...
    def package(
        self,
        name: str,
        version: str = "0.1.0",
        output_dir: str = ".",
        description: str = "Kkachi knowledge base",
        author: str = "",
    ) -> PackageResult:
        """Package the persistent memory into a pip-installable wheel.

        Raises:
            RuntimeError: If memory is not persistent or packaging fails.
        """
        ...
    def __len__(self) -> int: ...

# =============================================================================
# Rewrite Utilities
# =============================================================================

class Rewrite:
    """Markdown rewriter with fluent API.

    Example:
        result = Rewrite(markdown).replace_code("yaml", new_yaml).build()
    """

    def __init__(self, markdown: str) -> None: ...
    def replace_code(self, lang: str, content: str) -> Rewrite: ...
    def build(self) -> str: ...

def extract_code(markdown: str, lang: str) -> Optional[str]:
    """Extract the first code block of a given language from markdown."""
    ...

def extract_all_code(markdown: str, lang: str) -> List[str]:
    """Extract all code blocks of a given language from markdown."""
    ...

# =============================================================================
# Validators & Composition
# =============================================================================

class ScoreResult:
    """Result from calling .validate() directly."""
    value: float
    feedback: Optional[str]
    confidence: Optional[float]

    def passes(self, threshold: float) -> bool: ...
    def is_perfect(self) -> bool: ...

class CliValidator:
    """A CLI command validator with fluent API.

    Example:
        validator = CliValidator("rustfmt") \\
            .args(["--check"]) \\
            .weight(0.1) \\
            .then("rustc") \\
            .args(["--emit=metadata"]) \\
            .required() \\
            .ext("rs")
    """

    def __init__(self, command: str) -> None: ...
    def arg(self, arg: str) -> CliValidator: ...
    def args(self, args: List[str]) -> CliValidator: ...
    def weight(self, weight: float) -> CliValidator: ...
    def required(self) -> CliValidator: ...
    def then(self, command: str) -> CliValidator: ...
    def ext(self, ext: str) -> CliValidator: ...
    def env(self, key: str, value: str) -> CliValidator: ...
    def env_from(self, key: str) -> CliValidator: ...
    def workdir(self, path: str) -> CliValidator: ...
    def timeout(self, secs: int) -> CliValidator: ...
    def capture(self) -> CliValidator: ...
    def and_(self, other: Any) -> Validator: ...
    def or_(self, other: Any) -> Validator: ...
    def validate(self, text: str) -> ScoreResult: ...
    def validate_with_captures(self, text: str) -> Tuple[ScoreResult, List[CliCapture]]: ...
    def get_captures(self) -> List[CliCapture]: ...

class CliCapture:
    """Capture result from a CLI validation stage."""
    stage: str
    stdout: str
    stderr: str
    success: bool
    exit_code: Optional[int]
    duration_ms: int

class Checks:
    """Pattern-based validator with fluent API.

    Example:
        validator = Checks() \\
            .require("fn ") \\
            .forbid(".unwrap()") \\
            .regex(r"fn \\w+\\(") \\
            .min_len(50)
    """

    def __init__(self) -> None: ...
    def require(self, pattern: str) -> Checks: ...
    def require_all(self, patterns: list[str]) -> Checks: ...
    def forbid(self, pattern: str) -> Checks: ...
    def forbid_all(self, patterns: list[str]) -> Checks: ...
    def min_len(self, n: int) -> Checks: ...
    def max_len(self, n: int) -> Checks: ...
    def regex(self, pattern: str) -> Checks: ...
    def regex_all(self, patterns: list[str]) -> Checks: ...
    def and_(self, other: Any) -> Validator: ...
    def or_(self, other: Any) -> Validator: ...
    def validate(self, text: str) -> ScoreResult: ...

class Semantic:
    """Semantic validator using LLM-as-judge.

    Example:
        validator = Semantic(judge) \\
            .criterion("Code is idiomatic Rust") \\
            .criterion("Error handling is complete") \\
            .threshold(0.8)
    """

    def __init__(self, llm: Callable[[str, Optional[str]], str]) -> None: ...
    def criterion(self, criterion: str) -> Semantic: ...
    def threshold(self, threshold: float) -> Semantic: ...
    def system_prompt(self, prompt: str) -> Semantic: ...
    def and_(self, other: Any) -> Validator: ...
    def or_(self, other: Any) -> Validator: ...
    def validate(self, text: str) -> ScoreResult: ...

class Validator:
    """A composable validator combining Checks and Semantic validators.

    Example:
        strict = checks.and_(semantic)
        relaxed = checks.or_(semantic)
        combined = Validator.all([checks, semantic])
    """

    def and_(self, other: Any) -> Validator: ...
    def or_(self, other: Any) -> Validator: ...
    @staticmethod
    def all(validators: List[Any]) -> Validator: ...
    @staticmethod
    def any(validators: List[Any]) -> Validator: ...
    def validate(self, text: str) -> ScoreResult: ...

# =============================================================================
# Template System
# =============================================================================

class FormatType:
    """Output format type for templates."""
    JSON: int
    YAML: int
    MARKDOWN: int
    XML: int
    PLAIN: int

class PromptTone:
    """Prompt tone controlling language strictness.

    Example:
        tone = PromptTone.RESTRICTIVE
        print(tone.default_threshold())   # 0.9
        print(tone.favors_precision())    # True
    """
    INCLUSIVE: int
    BALANCED: int
    RESTRICTIVE: int

    def default_threshold(self) -> float: ...
    def favors_recall(self) -> bool: ...
    def favors_precision(self) -> bool: ...

class Template:
    """A template for structured prompt optimization.

    Example:
        template = Template("code_gen") \\
            .system_prompt("You are an expert Rust programmer.") \\
            .format(FormatType.JSON) \\
            .tone(PromptTone.RESTRICTIVE) \\
            .strict(True) \\
            .example("Write hello world", '{"code": "println!(\\"Hello\\")"}')
    """

    def __init__(self, name: str) -> None: ...
    @staticmethod
    def simple(prompt: str) -> Template: ...
    @staticmethod
    def from_str(content: str) -> Template: ...
    @staticmethod
    def from_file(path: str) -> Template: ...
    def system_prompt(self, prompt: str) -> Template: ...
    def format(self, format_type: FormatType) -> Template: ...
    def tone(self, tone: PromptTone) -> Template: ...
    def strict(self, strict: bool) -> Template: ...
    def example(self, input: str, output: str) -> Template: ...
    def render(self, input: str) -> str: ...
    def assemble_prompt(
        self,
        question: str,
        iteration: int = 0,
        feedback: Optional[str] = None,
    ) -> str: ...
    def validate_output(self, output: str) -> None: ...
    def parse_json(self, output: str) -> Any: ...
    def get_format_instructions(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def signature(self) -> str: ...

# =============================================================================
# Jinja Template System
# =============================================================================

class JinjaTemplate:
    """Jinja2-compatible template for dynamic prompt generation.

    Example:
        # Load from file
        template = JinjaTemplate.from_file("./templates/prompt.j2")

        # Or create from string
        template = JinjaTemplate.from_str("code_gen", '''
        ## Task
        {{ task }}

        {% if examples %}
        ## Examples
        {% for ex in examples %}
        - {{ ex }}
        {% endfor %}
        {% endif %}
        ''')

        # Render with context
        output = template.render({
            "task": "Write a parser",
            "examples": ["Example 1", "Example 2"]
        })
    """

    @staticmethod
    def from_file(path: str) -> JinjaTemplate:
        """Load a template from a file.

        Args:
            path: Path to template file (e.g., "./templates/prompt.j2")

        Returns:
            Loaded template

        Raises:
            RuntimeError: If file cannot be read or template is invalid
        """
        ...

    @staticmethod
    def from_str(name: str, content: str) -> JinjaTemplate:
        """Create a template from a string.

        Args:
            name: Template name for identification
            content: Jinja2 template content

        Returns:
            Created template

        Raises:
            RuntimeError: If template syntax is invalid
        """
        ...

    def name(self) -> str:
        """Get the template name."""
        ...

    def render(self, context: Dict[str, Any]) -> str:
        """Render the template with a context dictionary.

        Supports nested structures (dicts, lists, primitives).

        Args:
            context: Variables for template rendering

        Returns:
            Rendered output

        Example:
            output = template.render({
                "name": "Alice",
                "items": ["item1", "item2"],
                "config": {"debug": True}
            })
        """
        ...

    def render_strings(self, **kwargs: str) -> str:
        """Render with simple string-to-string mappings (convenience method).

        Args:
            **kwargs: Keyword arguments as template variables

        Returns:
            Rendered output

        Example:
            output = template.render_strings(task="Write code", language="Rust")
        """
        ...

class JinjaFormatter:
    """Prompt formatter using Jinja2 templates for refinement loops.

    The formatter receives three variables at each iteration:
    - task: The original prompt/task
    - feedback: Feedback from previous iteration (empty string if none)
    - iteration: Current iteration number (0-indexed)

    Example:
        template = JinjaTemplate.from_str("refine", '''
        ## Task
        {{ task }}

        {% if feedback %}
        ## Feedback from Previous Attempt
        {{ feedback }}
        {% endif %}

        ## Iteration
        This is attempt #{{ iteration + 1 }}
        ''')

        formatter = JinjaFormatter(template)

        # Use with DSPy-style builders
        result = reason(llm, "Write a parser") \\
            .with_formatter(formatter) \\
            .validate(Checks().require("fn ")) \\
            .go()
    """

    def __init__(self, template: JinjaTemplate) -> None:
        """Create a formatter from a template.

        Args:
            template: The template to use for formatting
        """
        ...

class RefineResult:
    """Result from the declarative API."""
    answer: str
    summary: str
    score: float
    iterations: int
    from_cache: bool
    domain: Optional[str]

class RefineBuilder:
    """Builder for declarative recursive refinement."""

    def __init__(self, signature: str) -> None: ...

    # Domain & Storage
    def domain(self, domain: str) -> RefineBuilder: ...
    def storage(self, path: str) -> RefineBuilder: ...

    # Convergence Criteria
    def max_iterations(self, n: int) -> RefineBuilder: ...
    def until_score(self, threshold: float) -> RefineBuilder: ...
    def until_plateau(self, min_improvement: float, window: int) -> RefineBuilder: ...

    # Validation
    def validate(self, validator: Any) -> RefineBuilder:
        """Use a validator for refinement.

        Example:
            validator = CliValidator("rustfmt") \\
                .args(["--check"]) \\
                .then("rustc") \\
                .args(["--emit=metadata"]) \\
                .required() \\
                .ext("rs")

            result = Kkachi.refine("question -> code") \\
                .validate(validator) \\
                .run("Write a URL parser", generate)
        """
        ...
    def critic_heuristic(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> RefineBuilder: ...

    # Similarity & Retrieval
    def semantic_cache(self, enabled: bool) -> RefineBuilder: ...
    def similarity_threshold(self, threshold: float) -> RefineBuilder: ...
    def auto_condense(self, enabled: bool) -> RefineBuilder: ...
    def cluster_threshold(self, threshold: float) -> RefineBuilder: ...
    def min_cluster_size(self, size: int) -> RefineBuilder: ...
    def similarity_weights(self, weights: Any) -> RefineBuilder: ...

    # Few-Shot
    def few_shot_k(self, k: int) -> RefineBuilder: ...
    def few_shot_as_demos(self, enabled: bool) -> RefineBuilder: ...
    def few_shot_refresh(self, enabled: bool) -> RefineBuilder: ...

    # DSPy Integration
    def with_reasoning(self) -> RefineBuilder: ...
    def with_best_of_n(self, n: int) -> RefineBuilder: ...

    # Execute
    def run(
        self,
        question: str,
        generate: Callable[[int, Optional[str]], str],
    ) -> RefineResult: ...

class Kkachi:
    """Main entry point for Kkachi."""

    @staticmethod
    def refine(signature: str) -> RefineBuilder:
        """Start building a recursive refinement pipeline.

        Args:
            signature: The signature string (e.g., "question -> code")

        Returns:
            A builder for configuring the refinement pipeline.
        """
        ...

# =============================================================================
# DSPy-style modules
# =============================================================================

# LLM callable type: (prompt: str, feedback: Optional[str]) -> str
LlmCallable = Callable[[str, Optional[str]], str]

# --- Result types ---

class ReasonResult:
    """Result from Chain of Thought reasoning."""
    output: str
    reasoning: Optional[str]
    score: float
    iterations: int
    tokens: int
    error: Optional[str]

    def success(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class BestOfResult:
    """Result from Best of N generation."""
    output: str
    score: float
    candidates_generated: int
    tokens: int
    error: Optional[str]

    def success(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ScoredCandidate:
    """A scored candidate from Best of N generation."""
    index: int
    output: str
    scorer_score: float
    validator_score: float
    combined_score: float
    feedback: Optional[str]

    def __repr__(self) -> str: ...

class PoolStats:
    """Statistics about a candidate pool."""
    count: int
    mean: float
    std_dev: float
    min: float
    max: float

    def __repr__(self) -> str: ...

class CandidatePool:
    """Pool of all generated candidates for recall/precision tuning."""
    candidates: List[ScoredCandidate]
    total_tokens: int

    def filter_by_threshold(self, threshold: float) -> List[ScoredCandidate]: ...
    def best(self) -> Optional[ScoredCandidate]: ...
    def top_k(self, k: int) -> List[ScoredCandidate]: ...
    def stats(self) -> PoolStats: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class EnsembleResult:
    """Result from ensemble (multi-chain) generation."""
    output: str
    chains_generated: int
    tokens: int
    error: Optional[str]

    def success(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ChainResult:
    """A single chain result in the consensus pool."""
    index: int
    answer: str
    normalized_answer: str
    agrees_with_majority: bool

    def __repr__(self) -> str: ...

class ConsensusPool:
    """Consensus pool from ensemble generation."""
    chains: List[ChainResult]
    selected: str

    def agreement_ratio(self) -> float: ...
    def has_unanimous_agreement(self) -> bool: ...
    def dissenting_chains(self) -> List[ChainResult]: ...
    def vote_counts(self) -> Dict[str, int]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class Step:
    """A single step in the agent trajectory."""
    thought: str
    action: str
    action_input: str
    observation: str

    def __repr__(self) -> str: ...

class AgentResult:
    """Result from agent execution."""
    output: str
    steps: int
    tokens: int
    success: bool
    error: Optional[str]

    def trajectory(self) -> List[Step]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ProgramResult:
    """Result from Program of Thought execution."""
    output: str
    code: str
    attempts: int
    tokens: int
    success: bool
    error: Optional[str]

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ExecutionResult:
    """Result from running code directly."""
    stdout: str
    stderr: str
    success: bool
    exit_code: Optional[int]
    duration_ms: int

    def output(self) -> str: ...
    def __repr__(self) -> str: ...

# --- Tool and Executor ---

class ToolDef:
    """A tool definition for use with agents.

    Example:
        tool = ToolDef("calculator", "Perform math", lambda x: str(eval(x)))
        result = agent(llm, "What is 2+2?").tool(tool).go()
    """

    def __init__(self, name: str, description: str, execute: Callable[[str], str]) -> None: ...
    def __repr__(self) -> str: ...

class Executor:
    """A code executor configuration.

    Example:
        executor = Executor.python()
        executor = Executor.bash().timeout(10)
        result = executor.execute("echo hello")
    """

    @staticmethod
    def python() -> Executor: ...
    @staticmethod
    def node() -> Executor: ...
    @staticmethod
    def bash() -> Executor: ...
    @staticmethod
    def ruby() -> Executor: ...
    def timeout(self, secs: int) -> Executor: ...
    def execute(self, code: str) -> ExecutionResult: ...
    def __repr__(self) -> str: ...

# --- Builders ---

class ReasonBuilder:
    """Builder for Chain of Thought reasoning.

    Example:
        result = reason(llm, "What is 25 * 37?") \\
            .max_iter(5) \\
            .require(r"\\d+") \\
            .go()
    """

    def __init__(self, llm: LlmCallable, prompt: str) -> None: ...
    def validate(self, validator: Any) -> ReasonBuilder: ...
    def defaults(self, defaults: Defaults) -> ReasonBuilder: ...
    def skill(self, skill: Skill) -> ReasonBuilder:
        """Attach a Skill (persistent prompt context) to this reasoning chain."""
        ...
    def max_iter(self, n: int) -> ReasonBuilder: ...
    def target(self, score: float) -> ReasonBuilder: ...
    def no_reasoning(self) -> ReasonBuilder: ...
    def require(self, pattern: str) -> ReasonBuilder: ...
    def forbid(self, pattern: str) -> ReasonBuilder: ...
    def regex(self, pattern: str) -> ReasonBuilder: ...
    def go(self) -> ReasonResult: ...
    def __repr__(self) -> str: ...

class BestOfBuilder:
    """Builder for Best of N generation.

    Example:
        result = best_of(llm, "Write a haiku", 5) \\
            .metric(lambda x: 1.0 if len(x.splitlines()) == 3 else 0.0) \\
            .go()
    """

    def __init__(self, llm: LlmCallable, prompt: str, n: int) -> None: ...
    def validate(self, validator: Any) -> BestOfBuilder: ...
    def defaults(self, defaults: Defaults) -> BestOfBuilder: ...
    def skill(self, skill: Skill) -> BestOfBuilder: ...
    def metric(self, scorer: Callable[[str], float]) -> BestOfBuilder: ...
    def scorer_weight(self, weight: float) -> BestOfBuilder: ...
    def with_reasoning(self) -> BestOfBuilder: ...
    def require(self, pattern: str) -> BestOfBuilder: ...
    def forbid(self, pattern: str) -> BestOfBuilder: ...
    def go(self) -> BestOfResult: ...
    def go_with_pool(self) -> Tuple[BestOfResult, CandidatePool]: ...
    def __repr__(self) -> str: ...

class EnsembleBuilder:
    """Builder for ensemble (multi-chain) generation.

    Example:
        result = ensemble(llm, "What is the capital of France?", 5) \\
            .aggregate("majority_vote") \\
            .go()
    """

    def __init__(self, llm: LlmCallable, prompt: str, n: int) -> None: ...
    def validate(self, validator: Any) -> EnsembleBuilder: ...
    def defaults(self, defaults: Defaults) -> EnsembleBuilder: ...
    def skill(self, skill: Skill) -> EnsembleBuilder: ...
    def aggregate(self, strategy: str) -> EnsembleBuilder: ...
    def with_reasoning(self) -> EnsembleBuilder: ...
    def no_normalize(self) -> EnsembleBuilder: ...
    def require(self, pattern: str) -> EnsembleBuilder: ...
    def forbid(self, pattern: str) -> EnsembleBuilder: ...
    def go(self) -> EnsembleResult: ...
    def go_with_consensus(self) -> Tuple[EnsembleResult, ConsensusPool]: ...
    def __repr__(self) -> str: ...

class AgentBuilder:
    """Builder for ReAct agent.

    Example:
        calc = ToolDef("calculator", "Do math", lambda x: str(eval(x)))
        result = agent(llm, "What is 2+2?") \\
            .tool(calc) \\
            .max_steps(10) \\
            .go()
    """

    def __init__(self, llm: LlmCallable, goal: str) -> None: ...
    def defaults(self, defaults: Defaults) -> AgentBuilder: ...
    def skill(self, skill: Skill) -> AgentBuilder: ...
    def tool(self, tool_def: ToolDef) -> AgentBuilder: ...
    def max_steps(self, n: int) -> AgentBuilder: ...
    def go(self) -> AgentResult: ...
    def __repr__(self) -> str: ...

class ProgramBuilder:
    """Builder for Program of Thought.

    Example:
        result = program(llm, "Calculate Fibonacci(50)") \\
            .executor(Executor.python()) \\
            .max_iter(3) \\
            .go()
    """

    def __init__(self, llm: LlmCallable, problem: str) -> None: ...
    def validate(self, validator: Any) -> ProgramBuilder: ...
    def defaults(self, defaults: Defaults) -> ProgramBuilder: ...
    def skill(self, skill: Skill) -> ProgramBuilder: ...
    def executor(self, executor: Executor) -> ProgramBuilder: ...
    def max_iter(self, n: int) -> ProgramBuilder: ...
    def no_code(self) -> ProgramBuilder: ...
    def language(self, lang: str) -> ProgramBuilder: ...
    def require(self, pattern: str) -> ProgramBuilder: ...
    def forbid(self, pattern: str) -> ProgramBuilder: ...
    def regex(self, pattern: str) -> ProgramBuilder: ...
    def go(self) -> ProgramResult: ...
    def __repr__(self) -> str: ...

# --- Entry point functions ---

def reason(llm: LlmCallable, prompt: str) -> ReasonBuilder:
    """Chain of Thought reasoning.

    Args:
        llm: A callable (prompt: str, feedback: Optional[str]) -> str
        prompt: The question/problem.

    Returns:
        A builder to configure and execute reasoning.
    """
    ...

def best_of(llm: LlmCallable, prompt: str, n: int) -> BestOfBuilder:
    """Best of N candidate generation.

    Args:
        llm: A callable (prompt: str, feedback: Optional[str]) -> str
        prompt: The prompt for generation.
        n: Number of candidates to generate.

    Returns:
        A builder to configure and execute generation.
    """
    ...

def ensemble(llm: LlmCallable, prompt: str, n: int) -> EnsembleBuilder:
    """Multi-chain ensemble with voting.

    Args:
        llm: A callable (prompt: str, feedback: Optional[str]) -> str
        prompt: The prompt/question.
        n: Number of chains to generate.

    Returns:
        A builder to configure and execute ensemble.
    """
    ...

def agent(llm: LlmCallable, goal: str) -> AgentBuilder:
    """ReAct agent with tool calling.

    Args:
        llm: A callable (prompt: str, feedback: Optional[str]) -> str
        goal: The goal for the agent.

    Returns:
        A builder to configure and execute the agent.
    """
    ...

def program(llm: LlmCallable, problem: str) -> ProgramBuilder:
    """Program of Thought - code generation and execution.

    Args:
        llm: A callable (prompt: str, feedback: Optional[str]) -> str
        problem: The problem to solve with code.

    Returns:
        A builder to configure and execute.
    """
    ...
