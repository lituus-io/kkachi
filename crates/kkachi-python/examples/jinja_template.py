"""Example demonstrating JinjaTemplate standalone rendering."""

from kkachi import JinjaTemplate

def main():
    print("="*60)
    print("JinjaTemplate Standalone Rendering")
    print("="*60)

    # Example 1: Simple variable substitution
    print("\n1. Simple Variables:")
    template = JinjaTemplate.from_str("simple", """
## Task
Generate a {{ language }} {{ item_type }}.
""")
    output = template.render({"language": "Rust", "item_type": "parser"})
    print(output)

    # Example 2: Conditionals
    print("\n2. Conditionals:")
    template = JinjaTemplate.from_str("conditional", """
Task: {{ task }}
{% if hint %}
Hint: {{ hint }}
{% endif %}
""")

    with_hint = template.render({"task": "Write code", "hint": "Use error handling"})
    print("With hint:", with_hint)

    without_hint = template.render({"task": "Write code"})
    print("Without hint:", without_hint)

    # Example 3: Loops
    print("\n3. Loops:")
    template = JinjaTemplate.from_str("loop", """
Requirements:
{% for req in requirements %}
- {{ req }}
{% endfor %}
""")
    output = template.render({
        "requirements": ["Must compile", "Handle errors", "Include tests"]
    })
    print(output)

    # Example 4: Nested structures
    print("\n4. Nested Structures:")
    template = JinjaTemplate.from_str("nested", """
Configuration:
  Debug: {{ config.debug }}
  Port: {{ config.port }}

Features:
{% for feature in config.features %}
  - {{ feature }}
{% endfor %}
""")
    output = template.render({
        "config": {
            "debug": True,
            "port": 8080,
            "features": ["auth", "logging", "metrics"]
        }
    })
    print(output)

    # Example 5: Filters
    print("\n5. Filters:")
    template = JinjaTemplate.from_str("filters", """
Name: {{ name | upper }}
Language: {{ language | lower }}
Items: {{ items | length }}
""")
    output = template.render({
        "name": "alice",
        "language": "RUST",
        "items": [1, 2, 3, 4, 5]
    })
    print(output)

    # Example 6: render_strings convenience method
    print("\n6. String Convenience Method:")
    template = JinjaTemplate.from_str("strings", "Hello {{ name }}, your role is {{ role }}.")
    output = template.render_strings(name="Bob", role="developer")
    print(output)

    # Example 7: Load from file
    print("\n7. File Loading:")
    # Create a sample template file
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.join(tmpdir, "test.j2")
        with open(template_path, 'w') as f:
            f.write("""
## Code Generation Task
Language: {{ language }}
Task: {{ task }}

{% if constraints %}
Constraints:
{% for c in constraints %}
- {{ c }}
{% endfor %}
{% endif %}
""")

        template = JinjaTemplate.from_file(template_path)
        output = template.render({
            "language": "Python",
            "task": "Write a web scraper",
            "constraints": ["Use requests", "Handle rate limiting"]
        })
        print(output)

if __name__ == "__main__":
    main()
