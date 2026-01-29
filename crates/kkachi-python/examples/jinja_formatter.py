"""Example demonstrating JinjaFormatter with refinement loops."""

from kkachi import (
    ApiLlm,
    JinjaTemplate,
    JinjaFormatter,
    reason,
    Checks,
)

def main():
    print("="*60)
    print("JinjaFormatter with Refinement")
    print("="*60)

    # Create LLM (mock for example)
    try:
        llm = ApiLlm.from_env()
        print("Using real LLM from environment")
    except Exception:
        print("Warning: Using mock LLM (no API key found)")
        def llm(prompt, feedback=None):
            return "fn parse() -> Result<String> { Ok(String::new()) }"

    # Example 1: Basic formatting
    print("\n1. Basic Formatter:")
    template = JinjaTemplate.from_str("basic", """
## Your Task
{{ task }}

{% if feedback %}
## Previous Attempt Feedback
{{ feedback }}

Please address the issues above.
{% endif %}

## Requirements
- Must compile
- Include error handling
""")

    formatter = JinjaFormatter(template)

    result = reason(llm, "Write a URL parser in Rust") \
        .with_formatter(formatter) \
        .require("fn ") \
        .require("Result") \
        .max_iter(3) \
        .go()

    print(f"Score: {result.score:.2f}")
    print(f"Iterations: {result.iterations}")
    print(f"Output:\n{result.output[:200]}..." if len(result.output) > 200 else f"Output:\n{result.output}")

    # Example 2: Iteration-aware formatting
    print("\n2. Iteration-Aware Formatter:")
    template = JinjaTemplate.from_str("iter_aware", """
# Iteration {{ iteration + 1 }}

Task: {{ task }}

{% if iteration == 0 %}
This is your first attempt. Focus on correctness.
{% elif iteration == 1 %}
This is your second attempt. Address the feedback below.
{% else %}
Final attempt - make sure all requirements are met!
{% endif %}

{% if feedback %}
**Feedback**: {{ feedback }}
{% endif %}
""")

    formatter = JinjaFormatter(template)

    result = reason(llm, "Write a JSON validator") \
        .with_formatter(formatter) \
        .require("fn ") \
        .forbid("panic!") \
        .max_iter(3) \
        .go()

    print(f"Final score: {result.score:.2f}")

    # Example 3: Complex formatting with metadata
    print("\n3. Complex Formatting:")
    template = JinjaTemplate.from_str("complex", """
╔═══════════════════════════════════════╗
║  CODE GENERATION - ITERATION {{ iteration + 1 }}  ║
╔═══════════════════════════════════════╝

📋 Task: {{ task }}

{% if feedback %}
⚠️  Issues from Previous Attempt:
{{ feedback }}

💡 Suggestions:
- Review the feedback carefully
- Ensure all patterns are included
- Test edge cases
{% endif %}

✅ Requirements Checklist:
[ ] Function signature with proper types
[ ] Result type for error handling
[ ] No panic! or unwrap()
[ ] Comprehensive logic

Begin your solution below:
""")

    formatter = JinjaFormatter(template)

    result = reason(llm, "Write a configuration file parser") \
        .with_formatter(formatter) \
        .require("fn ") \
        .require("Result") \
        .forbid("panic!") \
        .forbid(".unwrap()") \
        .max_iter(5) \
        .target(1.0) \
        .go()

    print(f"Achieved score: {result.score:.2f} after {result.iterations} iterations")

if __name__ == "__main__":
    main()
