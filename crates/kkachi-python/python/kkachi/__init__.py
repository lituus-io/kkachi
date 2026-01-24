# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Kkachi - High-performance LLM prompt optimization library.

This package provides Python bindings for the Kkachi Rust library,
enabling high-performance recursive language prompting and optimization.

Example:
    >>> from kkachi import Kkachi, Checks
    >>>
    >>> def generate(iteration, feedback):
    ...     if feedback:
    ...         return f"Improved output for iteration {iteration}"
    ...     return "Initial output"
    >>>
    >>> result = Kkachi.refine("question -> code") \\
    ...     .domain("python") \\
    ...     .validate(Checks().require("def ")) \\
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
    RefineResult,
    # CLI Validator
    CliValidator,
    # Validators & Composition
    Checks,
    Semantic,
    Validator,
    ScoreResult,
    # Template system
    Template,
    FormatType,
    PromptTone,
    # DSPy-style entry point functions
    reason,
    best_of,
    ensemble,
    agent,
    program,
    # DSPy-style builders
    ReasonBuilder,
    BestOfBuilder,
    EnsembleBuilder,
    AgentBuilder,
    ProgramBuilder,
    # DSPy-style result types
    ReasonResult,
    BestOfResult,
    EnsembleResult,
    AgentResult,
    ProgramResult,
    ExecutionResult,
    # DSPy-style pool/candidate types
    CandidatePool,
    ScoredCandidate,
    PoolStats,
    ConsensusPool,
    ChainResult,
    Step,
    # DSPy-style tool and executor
    ToolDef,
    Executor,
    # Version
    __version__,
)

__all__ = [
    # Main entry point
    "Kkachi",
    # Builder
    "RefineBuilder",
    # Result types
    "RefineResult",
    # CLI Validator
    "CliValidator",
    # Validators & Composition
    "Checks",
    "Semantic",
    "Validator",
    "ScoreResult",
    # Template system
    "Template",
    "FormatType",
    "PromptTone",
    # DSPy-style entry point functions
    "reason",
    "best_of",
    "ensemble",
    "agent",
    "program",
    # DSPy-style builders
    "ReasonBuilder",
    "BestOfBuilder",
    "EnsembleBuilder",
    "AgentBuilder",
    "ProgramBuilder",
    # DSPy-style result types
    "ReasonResult",
    "BestOfResult",
    "EnsembleResult",
    "AgentResult",
    "ProgramResult",
    "ExecutionResult",
    # DSPy-style pool/candidate types
    "CandidatePool",
    "ScoredCandidate",
    "PoolStats",
    "ConsensusPool",
    "ChainResult",
    "Step",
    # DSPy-style tool and executor
    "ToolDef",
    "Executor",
    # Version
    "__version__",
]
