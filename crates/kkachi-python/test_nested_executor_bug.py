#!/usr/bin/env python3
"""Test that ApiLlm works with all DSPy modules (nested executor bug fix)."""

import os
import sys

# Check for API key
if "ANTHROPIC_API_KEY" not in os.environ:
    print("⚠️  ANTHROPIC_API_KEY not set, using mock LLM")
    def mock_llm(prompt, feedback=None):
        return f"Mock response to: {prompt[:50]}"
    llm = mock_llm
else:
    from kkachi import ApiLlm
    llm = ApiLlm.claude_code()

from kkachi import refine, reason, best_of, ensemble, agent, program, Executor, ToolDef

def test_refine():
    """Test refine() with ApiLlm (reference implementation)."""
    print("\n🧪 Testing refine()...")
    try:
        result = refine(llm, "Say hello").max_iter(1).go()
        assert result.output is not None
        print(f"   ✅ refine() works - output: {result.output[:50]}...")
        return True
    except Exception as e:
        print(f"   ❌ refine() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reason():
    """Test reason() with ApiLlm (FIXED)."""
    print("\n🧪 Testing reason()...")
    try:
        result = reason(llm, "What is 2+2?").max_iter(1).go()
        assert result.output is not None
        print(f"   ✅ reason() works - output: {result.output[:50]}...")
        return True
    except Exception as e:
        print(f"   ❌ reason() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_best_of():
    """Test best_of() with ApiLlm (FIXED)."""
    print("\n🧪 Testing best_of()...")
    try:
        result = best_of(llm, "Say hello", 2).go()
        assert result.output is not None
        print(f"   ✅ best_of() works - output: {result.output[:50]}...")
        return True
    except Exception as e:
        print(f"   ❌ best_of() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble():
    """Test ensemble() with ApiLlm (FIXED)."""
    print("\n🧪 Testing ensemble()...")
    try:
        result = ensemble(llm, "Capital of France?", 2).go()
        assert result.output is not None
        print(f"   ✅ ensemble() works - output: {result.output[:50]}...")
        return True
    except Exception as e:
        print(f"   ❌ ensemble() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent():
    """Test agent() with ApiLlm (FIXED)."""
    print("\n🧪 Testing agent()...")
    try:
        # Simple tool that returns the current time
        def get_time(_):
            import datetime
            return datetime.datetime.now().strftime("%H:%M:%S")

        tool = ToolDef("get_time", "Get current time", get_time)
        result = agent(llm, "What time is it?").tool(tool).max_steps(2).go()
        assert result.output is not None
        print(f"   ✅ agent() works - output: {result.output[:50]}...")
        return True
    except Exception as e:
        print(f"   ❌ agent() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_program():
    """Test program() with ApiLlm (FIXED)."""
    print("\n🧪 Testing program()...")
    try:
        result = program(llm, "Calculate 2+2") \
            .executor(Executor.python()) \
            .max_iter(1) \
            .go()
        assert result.output is not None
        print(f"   ✅ program() works - output: {result.output[:50]}...")
        return True
    except Exception as e:
        print(f"   ❌ program() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_refine_with_mock_callable():
    """Test refine() with Python callable (not ApiLlm)."""
    print("\n🧪 Testing refine() with mock callable...")
    try:
        # Test that refine() works with a simple Python callable
        def mock_callable(prompt, feedback=None):
            return "Hello from mock callable!"

        result = refine(mock_callable, "Say hi").max_iter(1).go()
        assert result.output is not None
        assert "Hello" in result.output
        print(f"   ✅ refine() with callable works - output: {result.output}")
        return True
    except Exception as e:
        print(f"   ❌ refine() with callable failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Nested Executor Bug Fix Verification")
    print("=" * 70)

    if "ANTHROPIC_API_KEY" in os.environ:
        print("\n🔑 Using ApiLlm (real API calls)")
    else:
        print("\n🔧 Using mock LLM (set ANTHROPIC_API_KEY to test with real API)")

    tests = [
        ("refine()", test_refine),
        ("reason()", test_reason),
        ("best_of()", test_best_of),
        ("ensemble()", test_ensemble),
        ("agent()", test_agent),
        ("program()", test_program),
        ("refine() with callable", test_refine_with_mock_callable),
    ]

    results = []
    for name, test_func in tests:
        try:
            results.append((name, test_func()))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)
