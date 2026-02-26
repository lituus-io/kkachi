"""Tests for JinjaTemplate and JinjaFormatter."""

import pytest
import tempfile
import os
from kkachi import JinjaTemplate, JinjaFormatter


def test_jinja_template_from_str():
    """Test creating template from string."""
    template = JinjaTemplate.from_str("test", "Hello {{ name }}")
    assert template.name() == "test"

    output = template.render({"name": "World"})
    assert output == "Hello World"


def test_jinja_template_render_strings():
    """Test convenience render_strings method."""
    template = JinjaTemplate.from_str("test", "{{ a }} + {{ b }}")
    output = template.render_strings(a="foo", b="bar")
    assert output == "foo + bar"


def test_jinja_template_conditionals():
    """Test Jinja conditionals."""
    template = JinjaTemplate.from_str("cond", """
{% if show %}
Visible
{% endif %}
""")

    with_show = template.render({"show": True})
    assert "Visible" in with_show

    without_show = template.render({"show": False})
    assert "Visible" not in without_show


def test_jinja_template_loops():
    """Test Jinja loops."""
    template = JinjaTemplate.from_str("loop", """
{% for item in items %}
- {{ item }}
{% endfor %}
""")

    output = template.render({"items": ["a", "b", "c"]})
    assert "- a" in output
    assert "- b" in output
    assert "- c" in output


def test_jinja_template_nested_structures():
    """Test nested dicts and lists."""
    template = JinjaTemplate.from_str("nested", """
Name: {{ config.name }}
Count: {{ config.count }}
Items:
{% for item in config.items %}
  - {{ item }}
{% endfor %}
""")

    output = template.render({
        "config": {
            "name": "test",
            "count": 42,
            "items": ["x", "y", "z"]
        }
    })

    assert "Name: test" in output
    assert "Count: 42" in output
    assert "- x" in output


def test_jinja_template_filters():
    """Test Jinja filters."""
    template = JinjaTemplate.from_str("filters", "{{ name | upper }}")
    output = template.render({"name": "alice"})
    assert output == "ALICE"


def test_jinja_template_from_file():
    """Test loading template from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.join(tmpdir, "test.j2")
        with open(template_path, 'w') as f:
            f.write("File content: {{ value }}")

        template = JinjaTemplate.from_file(template_path)
        output = template.render({"value": "success"})
        assert output == "File content: success"


def test_jinja_template_error_invalid_syntax():
    """Test error on invalid template syntax."""
    with pytest.raises(RuntimeError, match="Failed to parse template"):
        JinjaTemplate.from_str("bad", "{{ unclosed")


def test_jinja_template_error_undefined_variable():
    """Test error on undefined variable."""
    template = JinjaTemplate.from_str("test", "{{ undefined }}")
    with pytest.raises(RuntimeError, match="render error"):
        template.render({})


def test_jinja_formatter_creation():
    """Test creating JinjaFormatter."""
    template = JinjaTemplate.from_str("fmt", "Task: {{ task }}")
    formatter = JinjaFormatter(template)
    assert formatter is not None


def test_jinja_template_default_filter():
    """Test default filter."""
    template = JinjaTemplate.from_str("default", "{{ value | default('fallback') }}")

    with_value = template.render({"value": "actual"})
    assert with_value == "actual"

    without_value = template.render({})
    assert without_value == "fallback"


def test_jinja_template_length_filter():
    """Test length filter."""
    template = JinjaTemplate.from_str("len", "Count: {{ items | length }}")
    output = template.render({"items": [1, 2, 3, 4, 5]})
    assert "Count: 5" in output


def test_jinja_template_complex_expression():
    """Test complex Jinja expressions."""
    template = JinjaTemplate.from_str("complex", """
{% for i in range(3) %}
Item {{ i + 1 }}
{% endfor %}
""")

    output = template.render({})
    assert "Item 1" in output
    assert "Item 2" in output
    assert "Item 3" in output


def test_jinja_template_repr():
    """Test __repr__ method."""
    template = JinjaTemplate.from_str("mytemplate", "{{ test }}")
    assert "JinjaTemplate" in repr(template)
    assert "mytemplate" in repr(template)


def test_jinja_template_multiple_vars():
    """Test template with multiple variables."""
    template = JinjaTemplate.from_str("multi", "{{ first }} and {{ second }}")
    output = template.render({"first": "Alice", "second": "Bob"})
    assert output == "Alice and Bob"


def test_jinja_template_integer_values():
    """Test template with integer values."""
    template = JinjaTemplate.from_str("int", "Value: {{ num }}")
    output = template.render({"num": 42})
    assert "Value: 42" in output


def test_jinja_template_boolean_values():
    """Test template with boolean values."""
    template = JinjaTemplate.from_str("bool", "{{ flag }}")
    output = template.render({"flag": True})
    assert "true" in output.lower()


def test_jinja_template_empty_list():
    """Test template with empty list."""
    template = JinjaTemplate.from_str("empty", """
{% for item in items %}
- {{ item }}
{% else %}
Empty list
{% endfor %}
""")
    output = template.render({"items": []})
    assert "Empty list" in output


def test_jinja_template_list_of_dicts():
    """Test template with list of dictionaries."""
    template = JinjaTemplate.from_str("list_dicts", """
{% for person in people %}
- {{ person.name }}: {{ person.age }}
{% endfor %}
""")
    output = template.render({
        "people": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
    })
    assert "Alice: 30" in output
    assert "Bob: 25" in output
