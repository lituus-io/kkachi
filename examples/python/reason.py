#!/usr/bin/env python3
"""
Chain of Thought reasoning using a real LLM.

Run with: python reason.py
Requires: KKACHI_API_KEY or OPENAI_API_KEY environment variable.
"""

import sys

from kkachi import ApiLlm, Checks, reason


def main():
    try:
        llm = ApiLlm.from_env()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Set KKACHI_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    print(f"Using model: {llm.model_name()}")
    print("Prompt: A farmer has 17 sheep. All but 9 die. How many are left?\n")

    checks = Checks().regex(r"\d+")

    result = (
        reason(llm, "A farmer has 17 sheep. All but 9 die. How many are left?")
        .validate(checks)
        .max_iter(3)
        .go()
    )

    print(f"Reasoning:\n{result.reasoning()}")
    print(f"\nAnswer: {result.output}")
    print(f"Score: {result.score:.2f}")


if __name__ == "__main__":
    main()
