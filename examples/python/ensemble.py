#!/usr/bin/env python3
"""
Multi-chain ensemble voting using a real LLM.

Run with: python ensemble.py
Requires: KKACHI_API_KEY or OPENAI_API_KEY environment variable.
"""

import sys

from kkachi import ApiLlm, ensemble


def main():
    try:
        llm = ApiLlm.from_env()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Set KKACHI_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    print(f"Using model: {llm.model_name()}")
    print("Prompt: What is the capital of Australia?")
    print("Generating 5 chains with majority vote...\n")

    result, consensus = (
        ensemble(llm, "What is the capital of Australia?")
        .n(5)
        .aggregate("majority_vote")
        .go_with_consensus()
    )

    print(f"Answer: {result.output}")
    print(f"Agreement: {consensus.agreement_ratio() * 100.0:.0f}%")
    print(f"Chains generated: {result.chains_generated}")


if __name__ == "__main__":
    main()
