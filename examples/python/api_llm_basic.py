#!/usr/bin/env python3
"""
Basic ApiLlm usage example.

This demonstrates how to use the ApiLlm client to work with real LLM APIs
including Anthropic Claude, OpenAI, and custom endpoints.
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
# - ANTHROPIC_API_KEY="sk-ant-..."
# - OPENAI_API_KEY="sk-..."
# - Or have 'claude' CLI in your PATH

try:
    llm = ApiLlm.from_env()
    print(f"✓ Detected LLM: {llm}")
    print(f"  Model: {llm.model_name()}")
    print(f"  Max context: {llm.max_context():,} tokens")
except RuntimeError as e:
    print(f"✗ No API key found: {e}")
""")

    try:
        llm = ApiLlm.from_env()
        print(f"✓ Detected LLM: {llm}")
        print(f"  Model: {llm.model_name()}")
        print(f"  Max context: {llm.model_name()}")
    except RuntimeError as e:
        print(f"✗ No API key found: {e}")
    print()

    # Example 2: Explicit Anthropic Claude
    print("Example 2: Anthropic Claude")
    print("-" * 70)
    print("""
llm = ApiLlm.anthropic(
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514"
)
print(f"Model: {llm.model_name()}")
print(f"Max context: {llm.max_context():,} tokens")
""")
    llm_anthropic = ApiLlm.anthropic(
        "demo-key",
        "claude-sonnet-4-20250514"
    )
    print(f"✓ Created: {llm_anthropic}")
    print()

    # Example 3: OpenAI
    print("Example 3: OpenAI")
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
    print(f"✓ Created: {llm_openai}")
    print()

    # Example 4: Claude Code CLI (no API key needed)
    print("Example 4: Claude Code CLI")
    print("-" * 70)
    print("""
# Uses locally installed 'claude' CLI - no API key needed!
llm = ApiLlm.claude_code()
print(f"Model: {llm.model_name()}")
""")
    llm_cli = ApiLlm.claude_code()
    print(f"✓ Created: {llm_cli}")
    print()

    # Example 5: Custom endpoints (proxies, self-hosted, etc.)
    print("Example 5: Custom Endpoints")
    print("-" * 70)
    print("""
# Anthropic-compatible endpoint
llm = ApiLlm.anthropic_with_url(
    api_key="your-key",
    model="claude-sonnet-4-20250514",
    base_url="https://your-proxy.com"
)

# Together.ai (OpenAI-compatible)
llm = ApiLlm.openai_with_url(
    api_key="your-together-key",
    model="meta-llama/Llama-3-70b-chat-hf",
    base_url="https://api.together.xyz"
)

# Groq (OpenAI-compatible)
llm = ApiLlm.openai_with_url(
    api_key="your-groq-key",
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai"
)
""")

    llm_custom = ApiLlm.openai_with_url(
        "demo-key",
        "meta-llama/Llama-3-70b-chat-hf",
        "https://api.together.xyz"
    )
    print(f"✓ Created Together.ai client: {llm_custom}")
    print()

    # Example 6: Override with environment variables
    print("Example 6: Environment Variable Overrides")
    print("-" * 70)
    print("""
# Set these to override defaults:
# export KKACHI_MODEL="gpt-4-turbo"
# export KKACHI_BASE_URL="https://custom.endpoint.com"
# export ANTHROPIC_API_KEY="sk-ant-..."

llm = ApiLlm.from_env()  # Will use overrides
""")
    print()

    print("=" * 70)
    print("For actual usage with refinement, see:")
    print("  - examples/python/refine_with_api_llm.py")
    print("  - examples/python/custom_endpoint.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
