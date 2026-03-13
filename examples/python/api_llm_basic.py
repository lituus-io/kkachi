#!/usr/bin/env python3
"""
Basic ApiLlm usage example.

This demonstrates how to use the ApiLlm client to work with real LLM APIs
including OpenAI-compatible endpoints and custom providers.
"""

from kkachi import ApiLlm

def main():
    print("=" * 70)
    print("Kkachi ApiLlm - Basic Usage Examples")
    print("=" * 70)
    print()

    # Example 1: Auto-detect from environment variables
    print("Example 1: Auto-detect from Environment")
    print("-" * 70)
    print("""
# Set one of these environment variables:
# - KKACHI_API_KEY="your-api-key"
# - OPENAI_API_KEY="sk-..."

try:
    llm = ApiLlm.from_env()
    print(f"Detected LLM: {llm}")
    print(f"  Model: {llm.model_name()}")
    print(f"  Max context: {llm.max_context():,} tokens")
except RuntimeError as e:
    print(f"No API key found: {e}")
""")

    try:
        llm = ApiLlm.from_env()
        print(f"  Detected LLM: {llm}")
        print(f"  Model: {llm.model_name()}")
        print(f"  Max context: {llm.model_name()}")
    except RuntimeError as e:
        print(f"  No API key found: {e}")
    print()

    # Example 2: OpenAI
    print("Example 2: OpenAI")
    print("-" * 70)
    print("""
llm = ApiLlm.openai(
    api_key="sk-...",
    model="gpt-4o"
)
print(f"Model: {llm.model_name()}")
print(f"Max context: {llm.max_context():,} tokens")
""")
    llm_openai = ApiLlm.openai("demo-key", "gpt-4o")
    print(f"  Created: {llm_openai}")
    print()

    # Example 3: Custom endpoints (proxies, self-hosted, etc.)
    print("Example 3: Custom Endpoints")
    print("-" * 70)
    print("""
# OpenAI-compatible endpoint (e.g. Together.ai)
llm = ApiLlm.openai_with_url(
    api_key="your-key",
    model="meta-llama/Llama-3-70b-chat-hf",
    base_url="https://api.together.xyz"
)

# Another OpenAI-compatible endpoint (e.g. Groq)
llm = ApiLlm.openai_with_url(
    api_key="your-key",
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai"
)
""")

    llm_custom = ApiLlm.openai_with_url(
        "demo-key",
        "meta-llama/Llama-3-70b-chat-hf",
        "https://api.together.xyz"
    )
    print(f"  Created Together.ai client: {llm_custom}")
    print()

    # Example 4: Environment Variable Overrides
    print("Example 4: Environment Variable Overrides")
    print("-" * 70)
    print("""
# Set these to override defaults:
# export KKACHI_MODEL="gpt-4-turbo"
# export KKACHI_BASE_URL="https://custom.endpoint.com"
# export KKACHI_API_KEY="your-key"

llm = ApiLlm.from_env()  # Will use overrides
""")
    print()

    print("=" * 70)
    print("For actual usage with refinement, see:")
    print("  - examples/python/refine.py")
    print("  - examples/python/reason.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
