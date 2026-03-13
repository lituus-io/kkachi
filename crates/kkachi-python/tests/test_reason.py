# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Tests for reason() multi-line output fix."""

import pytest
from kkachi import reason, Checks


def test_reason_multiline_no_marker_preserves_full():
    """Bug fix test: multi-line without marker preserves full content."""
    def mock_llm(prompt, feedback=None):
        return "Line 1\nLine 2\nLine 3\nLine 4"

    result = reason(mock_llm, "Generate content").go()

    # Should preserve ALL lines (not just last line)
    assert result.output == "Line 1\nLine 2\nLine 3\nLine 4"
    assert len(result.output.split('\n')) == 4
    # No reasoning when no marker
    assert result.reasoning is None


def test_reason_with_marker_still_extracts():
    """Marker-based extraction still works."""
    def mock_llm(prompt, feedback=None):
        return "Step 1\nStep 2\nStep 3\nTherefore: 42"

    result = reason(mock_llm, "Solve problem").go()

    # Should extract only answer after marker
    assert result.output == "42"
    # Reasoning should contain steps
    assert "Step 1" in result.reasoning
    assert "Step 3" in result.reasoning


def test_reason_yaml_template_automatic():
    """Real-world: YAML generation works automatically."""
    def mock_llm(prompt, feedback=None):
        return """name: template
runtime: yaml
description: Multi-line YAML

config:
  project: test
  region: us-central1

resources:
  bucket:
    type: storage.v1.bucket
    properties:
      name: test-bucket"""

    validator = Checks().min_len(100)
    result = reason(mock_llm, "Generate YAML") \
        .validate(validator) \
        .go()

    # Full YAML automatically preserved (no .full_output() needed!)
    assert "resources:" in result.output
    assert "bucket:" in result.output
    assert len(result.output) > 100
    assert result.score == 1.0


def test_reason_single_line_unchanged():
    """Single line without marker."""
    def mock_llm(prompt, feedback=None):
        return "Simple answer"

    result = reason(mock_llm, "Question").go()

    assert result.output == "Simple answer"
    assert result.reasoning is None
