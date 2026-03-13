#!/usr/bin/env python3
"""
Iterative refinement with validation using a real LLM.

Run with: python refine.py
Requires: KKACHI_API_KEY or OPENAI_API_KEY environment variable.
"""

import sys

from kkachi import ApiLlm, Checks, pipeline


def main():
    try:
        llm = ApiLlm.from_env()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Set KKACHI_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    print(f"Using model: {llm.model_name()}")
    print("Prompt: Write a Rust function that parses a URL into its components\n")

    checks = (
        Checks()
        .require("fn ")
        .require("->")
        .require("Result")
        .forbid(".unwrap()")
        .min_len(80)
    )

    result = (
        pipeline(llm, "Write a Rust function that parses a URL into its components (scheme, host, path)")
        .refine(checks, max_iter=5, target=0.9)
        .go()
    )

    print(f"Steps: {result.steps_count}")
    print(f"Output:\n{result.output}")


if __name__ == "__main__":
    main()
