#!/usr/bin/env python3
"""Test FUELIX API connection."""

import os
import sys

# Set the FUELIX API key
os.environ["ANTHROPIC_API_KEY"] = "ak-Nn0cLhhf0KkHLeWn0p4OSK8m2ZWC"

from kkachi import ApiLlm, refine

print("=" * 70)
print("Testing FUELIX API Connection")
print("=" * 70)

# Try different base URLs
test_urls = [
    ("Standard Anthropic", "https://api.anthropic.com"),
    ("FUELIX API", "https://api.fuelix.ai"),
    ("FUELIX Alternative", "https://fuelix.ai/api"),
]

for name, base_url in test_urls:
    print(f"\n🔍 Trying {name}: {base_url}")
    os.environ["KKACHI_BASE_URL"] = base_url

    try:
        # Try to create LLM and make a simple call
        llm = ApiLlm.from_env()
        result = refine(llm, "Say 'test'").max_iter(1).go()
        print(f"   ✅ SUCCESS! Output: {result.output[:50]}...")
        print(f"   Tokens: {result.tokens}")
        break
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower():
            print(f"   ❌ Authentication failed (wrong endpoint or key)")
        elif "404" in error_msg:
            print(f"   ❌ Endpoint not found")
        elif "Connection" in error_msg:
            print(f"   ❌ Connection failed")
        else:
            print(f"   ❌ Error: {error_msg[:100]}...")

print("\n" + "=" * 70)
