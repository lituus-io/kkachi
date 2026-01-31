#!/usr/bin/env python3
"""Test FUELIX API using OpenAI-compatible format."""

import os
import sys

# FUELIX uses OpenAI-compatible API format
FUELIX_API_KEY = "ak-Nn0cLhhf0KkHLeWn0p4OSK8m2ZWC"
FUELIX_BASE_URL = "https://api.fuelix.ai"  # Don't include /v1, it's added automatically

# Configure as OpenAI with FUELIX endpoint
os.environ["OPENAI_API_KEY"] = FUELIX_API_KEY
os.environ["KKACHI_BASE_URL"] = FUELIX_BASE_URL
os.environ["KKACHI_MODEL"] = "claude-3-5-sonnet"  # FUELIX model name

from kkachi import ApiLlm, refine, reason, best_of, ensemble, agent, program, Executor, ToolDef

print("=" * 70)
print("FUELIX API Testing (OpenAI-compatible format)")
print("=" * 70)
print(f"✓ Endpoint: {FUELIX_BASE_URL}")
print(f"✓ Model: claude-3-5-sonnet")
print()

# Initialize ApiLlm - it will use OpenAI format with FUELIX endpoint
llm = ApiLlm.from_env()
print(f"✓ LLM initialized: {llm}")
print()

def test_refine():
    """Test refine() with FUELIX API."""
    print("🧪 Testing refine()...")
    result = refine(llm, "Say hello in French").max_iter(1).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Score: {result.score}")
    print(f"   Iterations: {result.iterations}")
    return result

def test_reason():
    """Test reason() with FUELIX API."""
    print("🧪 Testing reason()...")
    result = reason(llm, "What is 15 * 23?").max_iter(1).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Reasoning: {result.reasoning[:100] if result.reasoning else 'N/A'}...")
    print(f"   Tokens: {result.tokens}")
    return result

def test_best_of():
    """Test best_of() with FUELIX API."""
    print("🧪 Testing best_of()...")
    result = best_of(llm, "Write a haiku about coding", 2).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Score: {result.score}")
    print(f"   Candidates: {result.candidates_generated}")
    print(f"   Tokens: {result.tokens}")
    return result

def test_ensemble():
    """Test ensemble() with FUELIX API."""
    print("🧪 Testing ensemble()...")
    result = ensemble(llm, "What is the capital of France?", 2).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Chains: {result.chains_generated}")
    print(f"   Tokens: {result.tokens}")
    return result

def test_agent():
    """Test agent() with FUELIX API."""
    print("🧪 Testing agent()...")

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
    """Test program() with FUELIX API."""
    print("🧪 Testing program()...")
    result = program(llm, "Calculate 5!") \
        .executor(Executor.python()) \
        .max_iter(1) \
        .go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Tokens: {result.tokens}")
    return result

def test_refine_with_llm_wrapper():
    """Test refine() with LLM wrapped in callable."""
    print("🧪 Testing refine() with LLM wrapper...")
    # Test that refine() works with both ApiLlm and Python callables
    def llm_wrapper(prompt, feedback=None):
        return llm(prompt, feedback)

    result = refine(llm_wrapper, "Say hi").max_iter(1).go()
    print(f"   ✅ Output: {result.output[:100]}...")
    print(f"   Score: {result.score}")
    return result

if __name__ == "__main__":
    tests = [
        ("refine", test_refine),
        ("reason", test_reason),
        ("best_of", test_best_of),
        ("ensemble", test_ensemble),
        ("agent", test_agent),
        ("program", test_program),
        ("refine_with_llm_wrapper", test_refine_with_llm_wrapper),
    ]

    results = {}
    total_tokens = 0

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = "PASS"
            total_tokens += getattr(result, 'tokens', 0)
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
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    print(f"\nFinal: {passed}/{total} tests passed")

    if passed < total:
        sys.exit(1)
