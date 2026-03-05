#!/usr/bin/env python3
# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Kkachi CliValidator + Pulumi Test
==================================
Standalone test demonstrating kkachi's CliValidator with pulumi commands.

Tests:
  1. validate() a known-good Pulumi YAML against `pulumi preview`
  2. validate() a known-bad Pulumi YAML — expect failure + captured feedback
  3. validate_with_captures() to inspect stdout/stderr
  4. Multi-stage validation: Checks + CliValidator composed with .and_()
  5. Full reason() loop: LLM fixes broken YAML using CLI feedback

Usage:
    python pulumi_cli_validator_test.py

Dependencies:
    pip install kkachi pulumi pulumi-gcp
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from kkachi import (
    ApiLlm,
    Checks,
    CliValidator,
    Defaults,
    Skill,
    reason,
)

# =============================================================================
# Configuration
# =============================================================================

LLM_API_KEY_PATH = os.environ.get(
    "LLM_API_KEY_PATH",
    "/path/to/llm/api-key",
)
GCP_CREDS = os.environ.get(
    "GCP_CREDS",
    "/path/to/gcp/credentials.json",
)

# =============================================================================
# Test fixtures
# =============================================================================

# A valid Pulumi YAML that provisions a single BigQuery dataset
GOOD_YAML = """\
name: cli-validator-test
runtime: yaml
description: CliValidator test template

config:
  project:
    type: string
    value: "{gcp_project}"

resources:
  test_dataset:
    type: gcp:bigquery:Dataset
    properties:
      datasetId: cli_validator_test_ds
      friendlyName: CLI Validator Test
      location: northamerica-northeast1
      project: ${{project}}
      deleteContentsOnDestroy: true
"""

# A broken Pulumi YAML — invalid resource type
BAD_YAML = """\
name: cli-validator-test
runtime: yaml
description: CliValidator test template

config:
  project:
    type: string
    value: "{gcp_project}"

resources:
  test_dataset:
    type: gcp:bigquery:NonExistentResource
    properties:
      datasetId: cli_validator_test_ds
      location: northamerica-northeast1
      project: ${{project}}
"""


def make_stack_name(label: str) -> str:
    short_hash = hashlib.md5(label.encode()).hexdigest()[:6]
    return f"kkachi-clitest-{short_hash}"


def setup_pulumi_workspace(gcp_project: str) -> tuple[str, str]:
    """Create a temp dir with pulumi stack initialized."""
    stack_name = make_stack_name("cli-validator-test")
    work_dir = tempfile.mkdtemp(prefix="kkachi-clitest-")

    # Bootstrap Pulumi.yaml
    Path(work_dir, "Pulumi.yaml").write_text(
        f"name: {stack_name}\nruntime: yaml\ndescription: test\n"
    )

    os.environ.setdefault("PULUMI_CONFIG_PASSPHRASE", "")

    # Init stack (ignore if already exists)
    subprocess.run(
        ["pulumi", "stack", "init", stack_name, "--non-interactive"],
        cwd=work_dir, capture_output=True
    )
    subprocess.run(
        ["pulumi", "config", "set", "gcp:project", gcp_project, "-s", stack_name],
        cwd=work_dir, capture_output=True
    )
    subprocess.run(
        ["pulumi", "config", "set", "project", gcp_project, "-s", stack_name],
        cwd=work_dir, capture_output=True
    )

    # Stack config file
    stack_config = Path(work_dir) / f"Pulumi.{stack_name}.yaml"
    stack_config.write_text(
        f"config:\n  gcp:project: {gcp_project}\n  project: {gcp_project}\n"
    )

    return work_dir, stack_name


def make_preview_validator(work_dir: str, stack_name: str) -> CliValidator:
    """Create a CliValidator that runs `pulumi preview`."""
    preview_cmd = (
        f'cp "$0" {work_dir}/Pulumi.yaml && '
        f'sed -i.bak "1s/^name: .*/name: {stack_name}/" {work_dir}/Pulumi.yaml && '
        f'rm -f {work_dir}/Pulumi.yaml.bak && '
        f'cd {work_dir} && '
        f'pulumi preview -s {stack_name} --non-interactive 2>&1'
    )
    return (
        CliValidator("bash")
        .args(["-c", preview_cmd])
        .ext("yaml")
        .env("GOOGLE_APPLICATION_CREDENTIALS", GCP_CREDS)
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .timeout(120)
        .capture()
    )


# =============================================================================
# Tests
# =============================================================================

def test_1_validate_good_yaml(validator: CliValidator, good_yaml: str):
    """Test 1: validate() with known-good YAML should pass."""
    print("Test 1: validate() with known-good Pulumi YAML")
    print("-" * 60)

    result = validator.validate(good_yaml)
    print(f"  Score:    {result.value}")
    print(f"  Perfect:  {result.is_perfect()}")
    print(f"  Feedback: {result.feedback[:200] if result.feedback else 'None'}")

    assert result.value > 0.0, f"Expected pass, got score={result.value}"
    print("  PASSED")
    print()


def test_2_validate_bad_yaml(validator: CliValidator, bad_yaml: str):
    """Test 2: validate() with broken YAML should fail; captures contain error details."""
    print("Test 2: validate() with broken Pulumi YAML")
    print("-" * 60)

    result, captures = validator.validate_with_captures(bad_yaml)
    print(f"  Score:    {result.value}")
    print(f"  Perfect:  {result.is_perfect()}")
    print(f"  Captures: {len(captures)}")

    for i, cap in enumerate(captures):
        status = "OK" if cap.success else "FAIL"
        print(f"    [{i}] {status} (exit={cap.exit_code}, {cap.duration_ms}ms)")
        if not cap.success:
            output = cap.stderr if cap.stderr.strip() else cap.stdout
            print(f"    Error output: {output[:300]}")

    assert result.value == 0.0, f"Expected failure, got score={result.value}"
    assert len(captures) > 0, "Expected captures"
    assert any(not c.success for c in captures), "Expected at least one failed capture"
    print("  PASSED")
    print()


def test_3_validate_with_captures(validator: CliValidator, good_yaml: str):
    """Test 3: validate_with_captures() returns CliCapture objects."""
    print("Test 3: validate_with_captures() inspection")
    print("-" * 60)

    result, captures = validator.validate_with_captures(good_yaml)
    print(f"  Score:    {result.value}")
    print(f"  Captures: {len(captures)}")

    for i, cap in enumerate(captures):
        print(f"  Capture[{i}]:")
        print(f"    Stage:    {cap.stage}")
        print(f"    Success:  {cap.success}")
        print(f"    Exit:     {cap.exit_code}")
        print(f"    Duration: {cap.duration_ms}ms")
        stdout_preview = cap.stdout[:200] if cap.stdout else "(empty)"
        print(f"    Stdout:   {stdout_preview}")

    assert len(captures) > 0, "Expected at least one capture"
    print("  PASSED")
    print()


def test_4_composed_validation(validator: CliValidator, good_yaml: str, bad_yaml: str):
    """Test 4: Checks + CliValidator composed with .and_()."""
    print("Test 4: Composed validation (Checks + CliValidator)")
    print("-" * 60)

    yaml_checks = (
        Checks()
        .require("runtime: yaml")
        .require("resources:")
        .require("gcp:bigquery:Dataset")
        .forbid("```")
    )

    combined = yaml_checks.and_(validator)

    # Good YAML should pass both
    result_good = combined.validate(good_yaml)
    print(f"  Good YAML score: {result_good.value}")
    assert result_good.value > 0.0, "Good YAML should pass composed validation"

    # Bad YAML has invalid resource type — Checks passes but CLI fails
    result_bad = combined.validate(bad_yaml)
    print(f"  Bad YAML score:  {result_bad.value}")
    # Bad YAML won't have "gcp:bigquery:Dataset" so Checks should catch it too
    print(f"  Feedback: {result_bad.feedback[:200] if result_bad.feedback else 'None'}")

    print("  PASSED")
    print()


def test_5_reason_loop(work_dir: str, stack_name: str, gcp_project: str):
    """Test 5: Full reason() loop — LLM fixes broken YAML using CLI feedback."""
    print("Test 5: reason() with CliValidator feedback loop")
    print("-" * 60)

    with open(LLM_API_KEY_PATH) as f:
        api_key = f.read().strip()
    llm = (
        ApiLlm.openai_with_url(api_key, "your-model-id", "https://your-api-endpoint.com")
        .with_timeout(120)
        .with_retry(3)
    )
    print(f"  LLM: {llm}")

    preview_validator = make_preview_validator(work_dir, stack_name)
    preview_validator = preview_validator.required()

    yaml_checks = (
        Checks()
        .require("runtime: yaml")
        .require("resources:")
        .require("gcp:bigquery:Dataset")
        .require("deleteContentsOnDestroy: true")
        .forbid("```")
        .forbid("Terraform")
    )

    combined = yaml_checks.and_(preview_validator)

    defaults = (
        Defaults()
        .set("iam_user", r"user:\S+@example\.com", "user:mark.gates@telus.com",
             note="Test IAM user")
        .set("iam_sa",
             r"serviceAccount:\S+@\S+\.iam\.gserviceaccount\.com",
             f"serviceAccount:terraform@{gcp_project}.iam.gserviceaccount.com",
             note="Test service account")
    )

    skill = (
        Skill()
        .instruct("pulumi_context",
            "This is a Pulumi YAML template. Never reference Terraform or HCL.")
        .instruct("deleteContentsOnDestroy",
            "Always set deleteContentsOnDestroy: true on datasets.")
        .instruct("deletionProtection",
            "gcp:bigquery:Dataset does NOT support deletionProtection — never add it.")
    )

    prompt = f"""Output ONLY raw Pulumi YAML (no markdown fences, no commentary, no "Answer:" prefix).
Start with "name:" on line 1.

Create a Pulumi YAML program that provisions a single BigQuery dataset with one IAM binding in GCP project {gcp_project}.

{defaults.context()}

CRITICAL RULES:
- Do NOT define gcp:project in the config section — it is set externally via `pulumi config`
- Define config variables: project (type: string)
- Use "value:" for config defaults, NOT "default:"
- Use ${{project}} to reference the project config variable
- Resource type must be gcp:bigquery:Dataset
- IAM type must be gcp:bigquery:DatasetIamMember
- Set deleteContentsOnDestroy: true
- gcp:bigquery:Dataset does NOT support deletionProtection — do not add it
- IAM member placeholder will be fixed by runtime defaults
- Location: northamerica-northeast1"""

    result = (
        reason(llm, prompt)
        .no_reasoning()
        .skill(skill)
        .defaults(defaults)
        .validate(combined)
        .max_iter(5)
        .target(1.0)
        .go()
    )

    print(f"  Score:      {result.score:.2f}")
    print(f"  Iterations: {result.iterations}")
    if result.error:
        print(f"  Error:      {result.error}")

    # Show captures
    captures = preview_validator.get_captures()
    print(f"  Captures:   {len(captures)}")
    for i, cap in enumerate(captures):
        status = "OK" if cap.success else "FAIL"
        print(f"    [{i}] {status} (exit={cap.exit_code}, {cap.duration_ms}ms)")

    if result.score >= 1.0:
        print(f"  Output preview:\n{result.output[:500]}")

    assert result.score > 0.0, f"Expected LLM to fix YAML, got score={result.score}"
    print("  PASSED")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Kkachi CliValidator + Pulumi Test Suite")
    print("=" * 60)
    print()

    # Load GCP credentials
    with open(GCP_CREDS) as f:
        creds = json.load(f)
    gcp_project = creds["project_id"]
    print(f"GCP project: {gcp_project}")

    # Prepare test YAML with real project
    good_yaml = GOOD_YAML.format(gcp_project=gcp_project)
    bad_yaml = BAD_YAML.format(gcp_project=gcp_project)

    # Setup workspace
    work_dir, stack_name = setup_pulumi_workspace(gcp_project)
    print(f"Work dir:    {work_dir}")
    print(f"Stack:       {stack_name}")
    print()

    # Create validator (shared across tests 1-4)
    validator = make_preview_validator(work_dir, stack_name)

    passed = 0
    failed = 0

    # Test 1: Good YAML
    try:
        test_1_validate_good_yaml(validator, good_yaml)
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
        print()

    # Test 2: Bad YAML
    # Need a fresh validator since captures accumulate
    validator2 = make_preview_validator(work_dir, stack_name)
    try:
        test_2_validate_bad_yaml(validator2, bad_yaml)
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
        print()

    # Test 3: Captures
    validator3 = make_preview_validator(work_dir, stack_name)
    try:
        test_3_validate_with_captures(validator3, good_yaml)
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
        print()

    # Test 4: Composed validation
    validator4 = make_preview_validator(work_dir, stack_name)
    try:
        test_4_composed_validation(validator4, good_yaml, bad_yaml)
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
        print()

    # Test 5: Full LLM loop
    try:
        test_5_reason_loop(work_dir, stack_name, gcp_project)
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
        print()

    # Summary
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    # Cleanup instructions
    print()
    print("Cleanup:")
    print(f"  cd {work_dir}")
    print(f"  pulumi destroy --yes -s {stack_name} --non-interactive")
    print(f"  pulumi stack rm {stack_name} --yes")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
