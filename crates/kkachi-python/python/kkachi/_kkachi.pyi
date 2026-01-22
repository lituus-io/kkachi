# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Type stubs for the kkachi._kkachi module."""

from typing import Callable, Optional, Dict, List, Tuple
from enum import IntEnum

__version__: str

class Cli:
    """A CLI command validator.

    Example:
        validator = Cli("rustfmt") \\
            .args(["--check"]) \\
            .weight(0.1) \\
            .required()
    """

    def __init__(self, command: str) -> None: ...
    def arg(self, arg: str) -> Cli: ...
    def args(self, args: List[str]) -> Cli: ...
    def weight(self, weight: float) -> Cli: ...
    def required(self) -> Cli: ...
    def stdin(self) -> Cli: ...
    def file_ext(self, ext: str) -> Cli: ...
    def env(self, key: str, value: str) -> Cli: ...
    def env_inherit(self, key: str) -> Cli: ...

class CliPipeline:
    """A pipeline of CLI validators with multiple stages.

    Example:
        pipeline = CliPipeline() \\
            .stage("format", Cli("rustfmt").args(["--check"]).weight(0.1)) \\
            .stage("compile", Cli("rustc").args(["--emit=metadata"]).required()) \\
            .stage("lint", Cli("cargo").args(["clippy"]).weight(0.3)) \\
            .file_ext("rs")
    """

    def __init__(self) -> None: ...
    def stage(self, name: str, cli: Cli) -> CliPipeline: ...
    def file_ext(self, ext: str) -> CliPipeline: ...

class ToolType(IntEnum):
    """Tool types for built-in CLI critics.

    DEPRECATED: Use Cli and CliPipeline to compose your own validators.
    """
    Rust = 0
    Python = 1
    Terraform = 2
    Pulumi = 3
    Kubernetes = 4
    JavaScript = 5
    Go = 6

class SimilarityWeights:
    """Weights for multi-signal similarity scoring."""
    embedding: float
    keyword: float
    metadata: float
    hierarchy: float

    def __init__(
        self,
        embedding: float = 0.40,
        keyword: float = 0.25,
        metadata: float = 0.20,
        hierarchy: float = 0.15,
    ) -> None: ...
    @staticmethod
    def default_weights() -> SimilarityWeights: ...
    @staticmethod
    def semantic_focus() -> SimilarityWeights: ...
    @staticmethod
    def keyword_focus() -> SimilarityWeights: ...
    def normalized(self) -> SimilarityWeights: ...

class FewShotConfig:
    """Configuration for few-shot learning."""
    k: int
    min_similarity: float
    include_in_prompt: bool
    as_demonstrations: bool
    refresh_per_iteration: bool

    def __init__(
        self,
        k: int = 3,
        min_similarity: float = 0.7,
        include_in_prompt: bool = True,
        as_demonstrations: bool = True,
        refresh_per_iteration: bool = False,
    ) -> None: ...
    @staticmethod
    def default_config() -> FewShotConfig: ...
    @staticmethod
    def with_k(k: int) -> FewShotConfig: ...

class RefinementResult:
    """Result from a refinement operation."""
    answer: str
    summary: str
    score: float
    iterations: int
    from_cache: bool

    def is_successful(self) -> bool: ...

class RefineResult:
    """Result from the declarative API."""
    answer: str
    summary: str
    score: float
    iterations: int
    from_cache: bool
    domain: Optional[str]

class VectorSearchResult:
    """Result from a vector store search."""
    id: str
    content: str
    score: float
    metadata: Optional[Dict[str, str]]

    def __init__(self, id: str, content: str, score: float) -> None: ...

class InMemoryVectorStore:
    """In-memory vector store for testing and small-scale use."""
    dimension: int

    def __init__(self, dimension: int = 64) -> None: ...
    def add(self, id: str, content: str) -> None: ...
    def add_batch(self, docs: List[Tuple[str, str]]) -> None: ...
    def clear(self) -> None: ...
    def search(self, query: str, k: int) -> List[VectorSearchResult]: ...
    def is_empty(self) -> bool: ...
    def __len__(self) -> int: ...

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
    def validate(self, pipeline: CliPipeline) -> RefineBuilder:
        """Use a custom CLI pipeline for validation.

        Example:
            validator = CliPipeline() \\
                .stage("format", Cli("rustfmt").args(["--check"]).weight(0.1)) \\
                .stage("compile", Cli("rustc").args(["--emit=metadata"]).required()) \\
                .file_ext("rs")

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
    def similarity_weights(self, weights: SimilarityWeights) -> RefineBuilder: ...

    # Few-Shot
    def few_shot_k(self, k: int) -> RefineBuilder: ...
    def few_shot_as_demos(self, enabled: bool) -> RefineBuilder: ...
    def few_shot_refresh(self, enabled: bool) -> RefineBuilder: ...

    # DSPy Integration
    def with_chain_of_thought(self) -> RefineBuilder: ...
    def with_best_of_n(self, n: int) -> RefineBuilder: ...

    # Execute
    def run(
        self,
        question: str,
        generate: Callable[[int, Optional[str]], str],
    ) -> RefinementResult: ...

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
