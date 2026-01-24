# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Kkachi - High-performance LLM prompt optimization library.

This package provides Python bindings for the Kkachi Rust library,
enabling high-performance recursive language prompting and optimization.

Example:
    >>> from kkachi import Kkachi
    >>>
    >>> result = Kkachi.refine("Write a URL parser in Rust") \\
    ...     .require("fn ") \\
    ...     .forbid(".unwrap()") \\
    ...     .max_iter(5) \\
    ...     .run(lambda i, prompt, feedback: llm(prompt, feedback))
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
    # Memory / RAG
    Memory,
    Recall,
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
    # Memory / RAG
    "Memory",
    "Recall",
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
