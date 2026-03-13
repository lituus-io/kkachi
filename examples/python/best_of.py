#!/usr/bin/env python3
"""
Best-of-N candidate selection using a real LLM.

Run with: python best_of.py
Requires: KKACHI_API_KEY or OPENAI_API_KEY environment variable.
"""

import sys

from kkachi import ApiLlm, Checks, best_of


def main():
    try:
        llm = ApiLlm.from_env()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Set KKACHI_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    print(f"Using model: {llm.model_name()}")
    print("Prompt: Write a haiku about Rust programming")
    print("Generating 5 candidates...\n")

    def haiku_metric(output: str) -> float:
        lines = output.strip().splitlines()
        return 0.8 if len(lines) == 3 else 0.2

    checks = Checks().min_len(10).forbid("```")

    result, pool = (
        best_of(llm, "Write a haiku about Rust programming")
        .n(5)
        .metric(haiku_metric)
        .validate(checks)
        .go_with_pool()
    )

    print(f"Best (score={result.score:.2f}):\n{result.output}")
    stats = pool.stats()
    print(f"\nPool: count={stats.count}, mean={stats.mean:.2f}, std_dev={stats.std_dev:.2f}")


if __name__ == "__main__":
    main()
