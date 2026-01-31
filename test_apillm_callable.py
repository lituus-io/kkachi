#!/usr/bin/env python3
"""Test if ApiLlm is actually callable."""

try:
    from kkachi import ApiLlm, reason

    print("Test: Can we actually call ApiLlm?")
    print("=" * 60)

    # Create an ApiLlm instance (this will fail without API keys, but that's fine)
    # We just want to see if it's callable
    try:
        llm = ApiLlm.claude_code()
        print(f"✓ Created ApiLlm: {llm}")

        # Try to call it like a function
        print("\nTrying to call llm('test prompt', None)...")
        try:
            result = llm("test prompt", None)
            print(f"✓ ApiLlm is callable! Result: {result}")
        except TypeError as e:
            print(f"✗ ApiLlm.__call__() failed: {e}")
        except Exception as e:
            print(f"? ApiLlm.__call__() raised: {type(e).__name__}: {e}")

        # Try to use with reason()
        print("\nTrying reason(llm, 'test').go()...")
        try:
            result = reason(llm, "What is 2+2?").go()
            print(f"✓ reason() worked! Result: {result}")
        except Exception as e:
            print(f"✗ reason() failed: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"✗ Failed to create ApiLlm: {e}")
        print("\nThis is expected if claude binary is not installed")
        print("The key question is whether the API is exposed, not whether it works")

except Exception as e:
    print(f"✗ Import error: {e}")
