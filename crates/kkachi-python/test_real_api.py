#!/usr/bin/env python3
"""Test all DSPy modules with real Anthropic API (claude-sonnet-4-5)."""

import os
import sys

# Check for FUELIX_API_KEY or ANTHROPIC_API_KEY
api_key = os.environ.get("FUELIX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ERROR: No API key found!")
    print("   Please set one of:")
    print("   - export FUELIX_API_KEY='your-key-here'")
    print("   - export ANTHROPIC_API_KEY='your-key-here'")
    sys.exit(1)

# Set the API key for kkachi to use
os.environ["ANTHROPIC_API_KEY"] = api_key
print(f"✓ Using API key: {api_key[:8]}***")

from kkachi import ApiLlm, refine, reason, best_of, ensemble, agent, program, Executor, ToolDef

print("=" * 70)
print("Real API Testing with claude-sonnet-4-5")
print("=" * 70)
print(f"✓ API Key found: {api_key[:8]}...")
print()

# Initialize ApiLlm with claude-sonnet-4-5
# Using from_env() which will pick up ANTHROPIC_API_KEY
llm = ApiLlm.from_env()

def test_refine():
    """Test refine() with real API."""
    print("🧪 Testing refine() with claude-sonnet-4-5...")
    result = refine(llm, "Say hello in French").max_iter(1).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Tokens: {result.tokens}")
    return result

def test_reason():
    """Test reason() with real API."""
    print("🧪 Testing reason() with claude-sonnet-4-5...")
    result = reason(llm, "What is 15 * 23?").max_iter(1).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Reasoning: {result.reasoning[:100] if result.reasoning else 'N/A'}...")
    print(f"   Tokens: {result.tokens}")
    return result

def test_best_of():
    """Test best_of() with real API."""
    print("🧪 Testing best_of() with claude-sonnet-4-5...")
    result = best_of(llm, "Write a haiku about coding", 2).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Score: {result.score}")
    print(f"   Candidates: {result.candidates_generated}")
    print(f"   Tokens: {result.tokens}")
    return result

def test_ensemble():
    """Test ensemble() with real API."""
    print("🧪 Testing ensemble() with claude-sonnet-4-5...")
    result = ensemble(llm, "What is the capital of France?", 2).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Chains: {result.chains_generated}")
    print(f"   Tokens: {result.tokens}")
    return result

def test_agent():
    """Test agent() with real API."""
    print("🧪 Testing agent() with claude-sonnet-4-5...")

    # Simple tool for testing
    def get_time():
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tool = ToolDef("get_time", "Get current time", '{"type": "object", "properties": {}}')
    result = agent(llm, "What time is it?").tool(tool).max_steps(2).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Steps: {result.steps}")
    print(f"   Tokens: {result.tokens}")
    return result

def test_program():
    """Test program() with real API."""
    print("🧪 Testing program() with claude-sonnet-4-5...")
    result = program(llm, "Calculate 5!") \
        .executor(Executor.python()) \
        .max_iter(1) \
        .go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Tokens: {result.tokens}")
    return result

if __name__ == "__main__":
    print()

    tests = [
        ("refine", test_refine),
        ("reason", test_reason),
        ("best_of", test_best_of),
        ("ensemble", test_ensemble),
        ("agent", test_agent),
        ("program", test_program),
    ]

    results = {}
    total_tokens = 0

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = "PASS"
            total_tokens += result.tokens
            print()
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"   ❌ FAILED: {e}")
            print()
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, status in results.items():
        emoji = "✅" if status == "PASS" else "❌"
        print(f"  {emoji} {name}: {status}")

    print()
    print(f"Total API tokens used: {total_tokens}")
    print("=" * 70)

    # Exit with error if any failed
    if any(v != "PASS" for v in results.values()):
        sys.exit(1)
