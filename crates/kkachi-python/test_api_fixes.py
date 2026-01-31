#!/usr/bin/env python3
"""Test script to verify all Python API bug fixes."""

import sys
import os

# Make sure we're using the local build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.venv/lib/python3.11/site-packages'))

def test_imports():
    """Test that all new APIs can be imported."""
    print("🧪 Testing imports...")
    try:
        from kkachi import (
            refine, RefineBuilderV2, ApiLlm,
            reason, best_of, ensemble,
            Memory, Checks
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_refine_function():
    """Test that refine() module function exists and works."""
    print("\n🧪 Testing refine() function...")
    try:
        from kkachi import refine

        # Test with a mock callable
        def mock_llm(prompt, feedback):
            return f"Response to: {prompt}"

        builder = refine(mock_llm, "Test prompt")
        print(f"  Created builder: {builder}")

        # Test method chaining
        builder = builder.max_iter(3).target(0.9)
        print(f"  Chained methods: {builder}")

        print("✅ refine() function works")
        return True
    except Exception as e:
        print(f"❌ refine() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_llm_callable():
    """Test that ApiLlm is callable."""
    print("\n🧪 Testing ApiLlm callable interface...")
    try:
        from kkachi import ApiLlm

        # Create ApiLlm instance (will fail if no API key, but that's ok)
        try:
            llm = ApiLlm.claude_code()
            print(f"  Created ApiLlm: {llm}")

            # Check if __call__ method exists
            if not hasattr(llm, '__call__'):
                print("❌ ApiLlm missing __call__ method")
                return False

            # Check if generate method exists
            if not hasattr(llm, 'generate'):
                print("❌ ApiLlm missing generate method")
                return False

            print("✅ ApiLlm has callable interface")
            return True
        except Exception as e:
            # If we can't create the LLM (no API key), just check the class has the methods
            if "__call__" in dir(ApiLlm) or "generate" in dir(ApiLlm):
                print("✅ ApiLlm has callable interface (class check)")
                return True
            else:
                print(f"❌ ApiLlm not callable: {e}")
                return False
    except Exception as e:
        print(f"❌ ApiLlm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dspy_functions():
    """Test that DSPy functions work."""
    print("\n🧪 Testing DSPy functions...")
    try:
        from kkachi import reason, best_of, ensemble

        def mock_llm(prompt, feedback):
            return f"Response to: {prompt}"

        # Test reason
        reason_builder = reason(mock_llm, "Test prompt")
        print(f"  reason() works: {reason_builder}")

        # Test best_of
        best_builder = best_of(mock_llm, "Test prompt", 3)
        print(f"  best_of() works: {best_builder}")

        # Test ensemble
        ensemble_builder = ensemble(mock_llm, "Test prompt", 3)
        print(f"  ensemble() works: {ensemble_builder}")

        print("✅ All DSPy functions work")
        return True
    except Exception as e:
        print(f"❌ DSPy functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_methods():
    """Test new Memory methods."""
    print("\n🧪 Testing Memory methods...")
    try:
        from kkachi import Memory

        mem = Memory()

        # Add some test documents
        id1 = mem.add("First document about Python")
        id2 = mem.add("Second document about Rust")
        id3 = mem.add("Third document about Python")
        print(f"  Added 3 documents")

        # Test list()
        if not hasattr(mem, 'list'):
            print("❌ Memory missing list() method")
            return False
        all_docs = mem.list()
        if len(all_docs) != 3:
            print(f"❌ list() returned {len(all_docs)} docs, expected 3")
            return False
        print(f"  list() works: {len(all_docs)} documents")

        # Test search_diverse()
        if not hasattr(mem, 'search_diverse'):
            print("❌ Memory missing search_diverse() method")
            return False
        diverse = mem.search_diverse("Python", k=2, min_similarity=0.5)
        print(f"  search_diverse() works: {len(diverse)} results")

        # Test delete()
        if not hasattr(mem, 'delete'):
            print("❌ Memory missing delete() method")
            return False
        deleted = mem.delete(id1)
        if not deleted:
            print("❌ delete() failed to delete document")
            return False
        print(f"  delete() works: deleted {id1}")

        # Test clear()
        if not hasattr(mem, 'clear'):
            print("❌ Memory missing clear() method")
            return False
        mem.clear()
        if len(mem) != 0:
            print(f"❌ clear() failed, {len(mem)} documents remain")
            return False
        print(f"  clear() works: all documents cleared")

        print("✅ All Memory methods work")
        return True
    except Exception as e:
        print(f"❌ Memory methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_refine_builder_methods():
    """Test RefineBuilder adaptive and with_budget methods."""
    print("\n🧪 Testing RefineBuilder methods...")
    try:
        from kkachi import Kkachi

        # Test adaptive()
        builder = Kkachi.refine("Test task")
        if not hasattr(builder, 'adaptive'):
            print("❌ RefineBuilder missing adaptive() method")
            return False
        builder = builder.adaptive()
        print(f"  adaptive() works: {builder}")

        # Test with_budget()
        builder = Kkachi.refine("Test task")
        if not hasattr(builder, 'with_budget'):
            print("❌ RefineBuilder missing with_budget() method")
            return False
        builder = builder.with_budget(5000)
        print(f"  with_budget() works: {builder}")

        print("✅ RefineBuilder methods work")
        return True
    except Exception as e:
        print(f"❌ RefineBuilder methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_documented_api():
    """Test the exact API shown in PYTHON_EXAMPLES.md."""
    print("\n🧪 Testing documented API from PYTHON_EXAMPLES.md...")
    try:
        from kkachi import refine, Checks

        def mock_llm(prompt, feedback):
            return """def parse_url(url):
    return url.split('/')"""

        # This is the exact API from PYTHON_EXAMPLES.md
        result = refine(mock_llm, "Write a Python function to parse URLs") \
            .validate(
                Checks()
                    .require("def ")
                    .require("return")
                    .forbid("eval(")
                    .min_len(50)
            ) \
            .max_iter(5) \
            .target(0.9) \
            .go()

        print(f"  Result: {result}")
        print(f"  Score: {result.score:.2f}")
        print(f"  Iterations: {result.iterations}")

        print("✅ Documented API works!")
        return True
    except Exception as e:
        print(f"❌ Documented API failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Python API Bug Fixes")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("refine() function", test_refine_function),
        ("ApiLlm callable", test_api_llm_callable),
        ("DSPy functions", test_dspy_functions),
        ("Memory methods", test_memory_methods),
        ("RefineBuilder methods", test_refine_builder_methods),
        ("Documented API", test_documented_api),
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

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
