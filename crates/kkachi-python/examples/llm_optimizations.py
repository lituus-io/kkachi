"""Example demonstrating LLM optimization methods: cache, rate limiting, and retry."""

import os
from kkachi import ApiLlm

# Set up API key (use your actual key or environment variable)
# os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

def main():
    """Demonstrate LLM optimization features."""

    print("="*60)
    print("LLM Optimization Methods Demo")
    print("="*60)

    # Create base LLM
    print("\n1. Creating base LLM from environment...")
    try:
        llm = ApiLlm.from_env()
        print(f"   ✓ Created: {llm}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("\n   Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        return

    # Apply caching
    print("\n2. Adding LRU cache (capacity=100)...")
    llm = llm.with_cache(100)
    print(f"   ✓ Cached LLM: {llm}")
    print("   → Identical prompts will return cached results")

    # Apply rate limiting
    print("\n3. Adding rate limiting (10 requests/second)...")
    llm = llm.with_rate_limit(10.0)
    print(f"   ✓ Rate-limited LLM: {llm}")
    print("   → Requests will be automatically paced")

    # Apply retry logic
    print("\n4. Adding automatic retry (max 3 attempts)...")
    llm = llm.with_retry(3)
    print(f"   ✓ Retry-enabled LLM: {llm}")
    print("   → Transient errors will be retried with backoff")

    print("\n" + "="*60)
    print("Optimization Chain Summary")
    print("="*60)
    print(f"Model: {llm.model_name()}")
    print(f"Context: {llm.max_context():,} tokens")
    print("\nOptimizations applied (in order):")
    print("  1. Cache      → Avoids redundant API calls")
    print("  2. Rate Limit → Prevents 429 errors")
    print("  3. Retry      → Handles transient failures")

    # Example usage notes
    print("\n" + "="*60)
    print("Usage Examples")
    print("="*60)
    print("""
# Chaining optimizations (order matters for best results):
llm = (ApiLlm.from_env()
       .with_cache(100)          # Apply cache first
       .with_rate_limit(10.0)    # Then rate limiting
       .with_retry(3))           # Finally retry

# Cache alone (for development/testing):
llm = ApiLlm.from_env().with_cache(50)

# Rate limiting alone (for production API limits):
llm = ApiLlm.anthropic(api_key, model).with_rate_limit(5.0)

# Retry alone (for unreliable networks):
llm = ApiLlm.openai(api_key, model).with_retry(5)
    """)

    print("\n" + "="*60)
    print("Recommended Patterns")
    print("="*60)
    print("""
1. Development:   cache only (fast iteration)
2. Testing:       cache + retry (repeatable + resilient)
3. Production:    cache + rate_limit + retry (full optimization)
4. High-load:     rate_limit + retry (prevent overload)
    """)


if __name__ == "__main__":
    main()
