# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Kkachi - High-performance LLM prompt optimization library.

This package provides Python bindings for the Kkachi Rust library,
enabling high-performance recursive language prompting and optimization.

Example:
    >>> from kkachi import Kkachi, Cli, CliPipeline
    >>>
    >>> def generate(iteration, feedback):
    ...     if feedback:
    ...         return f"Improved output for iteration {iteration}"
    ...     return "Initial output"
    >>>
    >>> # Compose your own validator
    >>> validator = CliPipeline() \\
    ...     .stage("syntax", Cli("python").args(["-m", "py_compile"]).required()) \\
    ...     .stage("lint", Cli("ruff").args(["check"])) \\
    ...     .file_ext("py")
    >>>
    >>> result = Kkachi.refine("question -> code") \\
    ...     .domain("python") \\
    ...     .validate(validator) \\
    ...     .max_iterations(5) \\
    ...     .run("Write a URL parser", generate)
    >>>
    >>> print(f"Score: {result.score}")
"""

from kkachi._kkachi import (
    # Main entry point
    Kkachi,
    # Builder
    RefineBuilder,
    # Result types
    RefinementResult,
    RefineResult,
    # CLI Validators
    Cli,
    CliPipeline,
    # Tool types (deprecated)
    ToolType,
    # Similarity
    SimilarityWeights,
    # Few-shot config
    FewShotConfig,
    # VectorStore types
    VectorSearchResult,
    InMemoryVectorStore,
    # Version
    __version__,
)

__all__ = [
    # Main entry point
    "Kkachi",
    # Builder
    "RefineBuilder",
    # Result types
    "RefinementResult",
    "RefineResult",
    # CLI Validators
    "Cli",
    "CliPipeline",
    # Tool types (deprecated)
    "ToolType",
    # Similarity
    "SimilarityWeights",
    # Few-shot config
    "FewShotConfig",
    # VectorStore types
    "VectorSearchResult",
    "InMemoryVectorStore",
    # Version
    "__version__",
]
