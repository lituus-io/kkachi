#!/usr/bin/env python3
# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Kkachi Pulumi Template Pipeline
================================
7-step pipeline demonstrating kkachi's end-to-end capabilities:

1. Read source folder — locate Pulumi.yaml / Pulumi.*.yaml + child code
2. RAG similarity search — DuckDB-backed store for existing templates
3. Recursive optimization with `pulumi preview` CLI validation
4. Recursive optimization with `pulumi up` CLI validation
5. Generate structured GitHub Markdown README.md (sections A–E)
6. RAG writeback — update if >95% similar, else add new
7. Write output files — Pulumi.yaml, child code, and README.md to source folder

Inputs:  Source root folder containing Pulumi YAML templates
Outputs: Corrected Pulumi.yaml + child code + README.md written to source folder

Usage:
    python pulumi_template_pipeline.py

Dependencies:
    pip install kkachi pulumi pulumi-gcp
"""

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

from _pipeline_common import (
    is_pulumi_yaml,
    make_stack_name,
    strip_config_section,
    read_source_folder,
    format_files,
    rag_lookup,
    rag_writeback,
    extract_errors_from_captures,
    verify_output_files,
)

# =============================================================================
# Configuration
# =============================================================================

SOURCE_FOLDER = os.environ.get(
    "SOURCE_FOLDER",
    "/path/to/pulumi/templates",
)
LLM_API_KEY_PATH = os.environ.get(
    "LLM_API_KEY_PATH",
    "/path/to/llm/api-key",
)
GCP_CREDS = os.environ.get(
    "GCP_CREDS",
    "/path/to/gcp/credentials.json",
)
RAG_DB = os.environ.get("KKACHI_RAG_DB", "./template_knowledge.db")
DIST_DIR = os.environ.get("KKACHI_DIST_DIR", str(Path(SOURCE_FOLDER) / "dist"))


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("Kkachi Pulumi Template Pipeline")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Read source folder
    # -------------------------------------------------------------------------
    print("Step 1: Read source folder")
    print("-" * 70)

    if not Path(SOURCE_FOLDER).exists():
        print(f"  Source folder not found: {SOURCE_FOLDER}")
        print("  Set SOURCE_FOLDER env var and retry.")
        sys.exit(1)

    pulumi_files, context_files = read_source_folder(SOURCE_FOLDER)

    if not pulumi_files:
        print(f"  No Pulumi.yaml or Pulumi.*.yaml found in {SOURCE_FOLDER}")
        sys.exit(1)

    print(f"  Pulumi templates ({len(pulumi_files)}):")
    for path in pulumi_files:
        print(f"    - {path}")
    if context_files:
        print(f"  Supporting files ({len(context_files)}):")
        for path in context_files:
            print(f"    - {path}")

    # Build context — strip config sections to prevent LLM from copying
    # provider-level config (gcp:project) into generated output
    stripped_files = {k: strip_config_section(v) for k, v in pulumi_files.items()}
    pulumi_context = format_files(stripped_files, lang="yaml")

    # Identify the root Pulumi.yaml (primary template)
    root_template = pulumi_files.get("Pulumi.yaml", "")
    if not root_template:
        first_key = next(iter(pulumi_files))
        root_template = pulumi_files[first_key]
        print(f"  Primary template: {first_key}")
    else:
        print("  Primary template: Pulumi.yaml")
    print()

    # -------------------------------------------------------------------------
    # Step 2: RAG similarity search
    # -------------------------------------------------------------------------
    print("Step 2: RAG similarity search")
    print("-" * 70)

    rag_results = rag_lookup(RAG_DB, root_template[:500])
    rag_section = ""

    if rag_results:
        print(f"  Found {len(rag_results)} similar templates:")
        rag_parts = []
        for r in rag_results:
            print(f"    - {r.id} (score: {r.score:.3f})")
            rag_parts.append(
                f"### RAG Example (similarity: {r.score:.2f})\n"
                f"```yaml\n{r.content[:800]}\n```"
            )
        rag_section = "\nEXISTING SIMILAR TEMPLATES:\n" + "\n\n".join(rag_parts)
    else:
        print("  No similar templates found (empty store or first run)")
    print()

    # -------------------------------------------------------------------------
    # Setup: LLM + Pulumi workspace
    # -------------------------------------------------------------------------
    print("Setup: LLM + Pulumi workspace")
    print("-" * 70)

    with open(LLM_API_KEY_PATH) as f:
        api_key = f.read().strip()
    llm = (
        ApiLlm.openai_with_url(api_key, "your-model-id", "https://your-api-endpoint.com")
        .with_timeout(120)
        .with_retry(3)
    )
    print(f"  LLM: {llm}")

    # Read GCP project ID and service account email from credentials
    with open(GCP_CREDS) as f:
        creds = json.load(f)
    gcp_project = creds["project_id"]
    gcp_service_account = creds["client_email"]
    print(f"  GCP project: {gcp_project}")
    print(f"  Service account: {gcp_service_account}")

    # Unique stack name for this template
    stack_name = make_stack_name(SOURCE_FOLDER)
    print(f"  Stack name: {stack_name}")

    # Create working directory
    work_dir = tempfile.mkdtemp(prefix="kkachi-pulumi-")
    print(f"  Work dir: {work_dir}")

    # Bootstrap Pulumi.yaml so stack init succeeds
    Path(work_dir, "Pulumi.yaml").write_text(
        f"name: {stack_name}\nruntime: yaml\ndescription: kkachi pipeline\n"
    )

    # Set passphrase for local state (non-interactive)
    os.environ.setdefault("PULUMI_CONFIG_PASSPHRASE", "")

    # Initialize stack
    subprocess.run(
        ["pulumi", "stack", "init", stack_name, "--non-interactive"],
        cwd=work_dir, capture_output=True
    )
    # Set provider-level GCP project (used by all gcp: resources)
    subprocess.run(
        ["pulumi", "config", "set", "gcp:project", gcp_project, "-s", stack_name],
        cwd=work_dir, capture_output=True
    )
    # Set the template's own 'project' config variable (used in ${project} refs)
    subprocess.run(
        ["pulumi", "config", "set", "project", gcp_project, "-s", stack_name],
        cwd=work_dir, capture_output=True
    )
    # Set builder config variable
    subprocess.run(
        ["pulumi", "config", "set", "builder", "kkachi-pipeline", "-s", stack_name],
        cwd=work_dir, capture_output=True
    )

    # Also create the stack config file directly to be safe
    stack_config_file = Path(work_dir) / f"Pulumi.{stack_name}.yaml"
    stack_config_file.write_text(
        f"config:\n  gcp:project: {gcp_project}\n"
        f"  project: {gcp_project}\n"
        f"  builder: kkachi-pipeline\n"
    )
    print(f"  Stack config: {stack_config_file.name}")

    # Runtime defaults — replace LLM-generated placeholders before validation
    defaults = (
        Defaults()
        .set("iam_user", r"user:\S+@example\.com", "user:spicyzhug@test.com",
             note="Replace with actual IAM user email for production")
        .set("iam_group", r"group:\S+@example\.com", "user:spicyzhug@test.com",
             note="Replace with actual group or user for production")
        .set("iam_sa",
             r"serviceAccount:\S+@\S+\.iam\.gserviceaccount\.com",
             f"serviceAccount:{gcp_service_account}",
             note="Service account from GCP credentials JSON")
        .from_env("gcp_project", r"my-gcp-project",
                   "GOOGLE_CLOUD_PROJECT", fallback=gcp_project,
                   note="Target GCP project ID")
    )
    print(f"  Defaults: {len(defaults)} entries")
    for ann in defaults.annotations():
        note = f" — {ann.note}" if ann.note else ""
        print(f"    {ann.key}: {ann.replacement} ({ann.source}){note}")
    print()

    # Domain skill — persistent instructions for all LLM calls
    skill = (
        Skill()
        .instruct("pulumi_context",
            "This is a Pulumi YAML template — always use Pulumi resource types and "
            "Pulumi terminology. Never reference Terraform, HCL, or Terraform providers. "
            "All resource types use the Pulumi GCP provider (e.g. gcp:bigquery:Dataset).")
        .instruct("deleteContentsOnDestroy",
            "For datasets, always set 'deleteContentsOnDestroy: true'. "
            "Omitting it limits the capability to delete the resources through CICD.")
        .instruct("deletionProtection",
            "gcp:bigquery:Dataset does NOT support deletionProtection — "
            "do not add it to Dataset resources.")
        .instruct("staging_expiry",
            "Staging expiry should be set to YYYY-MM-DD format. "
            "Example config entry:\n"
            "  staging_expiry:\n"
            "    type: string\n"
            "    value: 2026-12-31\n"
            "Do NOT use milliseconds or epoch timestamps for staging_expiry.")
        .instruct("iam_unresolvable",
            "If an IAM binding references an identity (user, group, or serviceAccount) "
            "that may not exist or is unresolvable, comment out the entire IAM resource "
            "block in the YAML with '#' and add a note: "
            "'# TODO: Uncomment and change identity to a valid principal'. "
            "Only comment out IAM resources with unresolvable identities, not datasets.")
        .instruct("naming",
            "Use snake_case for all Pulumi resource logical names.")
    )
    print(f"  Skill: {len(skill)} instructions")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Recursive optimization with pulumi preview
    # -------------------------------------------------------------------------
    print("Step 3: Recursive optimization — pulumi preview")
    print("-" * 70)

    # CliValidator: copy candidate → work_dir/Pulumi.yaml, fix the name field
    # to match the stack name, then run pulumi preview.
    # On macOS (BSD sed), -i requires a backup extension argument.
    preview_cmd = (
        f'cp "$0" {work_dir}/Pulumi.yaml && '
        f'sed -i.bak "1s/^name: .*/name: {stack_name}/" {work_dir}/Pulumi.yaml && '
        f'rm -f {work_dir}/Pulumi.yaml.bak && '
        f'cd {work_dir} && '
        f'pulumi preview -s {stack_name} --non-interactive 2>&1'
    )
    preview_validator = (
        CliValidator("bash")
        .args(["-c", preview_cmd])
        .ext("yaml")
        .env("GOOGLE_APPLICATION_CREDENTIALS", GCP_CREDS)
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .timeout(120)
        .capture()
        .required()
    )

    yaml_checks = (
        Checks()
        .require("runtime: yaml")
        .require("resources:")
        .require("gcp:bigquery:Dataset")
        .forbid("bilayer-sa@")
        .forbid("```")         # No markdown fences in raw YAML output
        .forbid("Terraform")   # Must use Pulumi terminology, not Terraform
    )

    preview_combined = yaml_checks.and_(preview_validator)

    # Build the optimization prompt
    preview_prompt = f"""Output ONLY raw Pulumi YAML (no markdown fences, no commentary, no "Answer:" prefix).
Start with "name:" on line 1.

Create a Pulumi YAML program that provisions BigQuery datasets and IAM bindings in GCP project {gcp_project}.

{defaults.context()}

CRITICAL RULES:
- Do NOT define gcp:project in the config section — it is set externally via `pulumi config`
- You MUST define these config variables: project (type: string), builder (type: string), staging_expiry (type: string)
- The "project" config variable is REQUIRED — without it, ${{project}} references in resources will fail
- Use "value:" for config defaults, NOT "default:"
- staging_expiry must use YYYY-MM-DD date format. Example: staging_expiry: {{type: string, value: "2026-12-31"}}
- Use ${{project}} to reference the project config variable in resource properties
- All resource types must be valid Pulumi GCP types (gcp:bigquery:Dataset, gcp:bigquery:DatasetIamMember)
- IAM members will be fixed automatically by runtime defaults — use any placeholder
- gcp:bigquery:Dataset does NOT support deletionProtection — never add it to Dataset resources
- Always set deleteContentsOnDestroy: true on all datasets
- If an IAM resource references an identity that may not exist, comment it out with a TODO note
- This is a Pulumi YAML template — do NOT reference Terraform or HCL

RESOURCE STRUCTURE (adapt from source templates below):
- A default dataset with deleteContentsOnDestroy: true
- Dataset-level IAM bindings (gcp:bigquery:DatasetIamMember) for user, group, service account
- A logical storage billing dataset with IAM
- A physical storage billing dataset with IAM
- Use deleteContentsOnDestroy: true for all datasets
- Location variable: northamerica-northeast1

SOURCE TEMPLATES (for reference — adapt the structure, NOT the config):
{pulumi_context}
{rag_section}"""

    preview_result = (
        reason(llm, preview_prompt)
        .no_reasoning()
        .skill(skill)
        .defaults(defaults)
        .validate(preview_combined)
        .max_iter(8)
        .target(1.0)
        .go()
    )

    print(f"  Result: score={preview_result.score:.2f}, "
          f"iterations={preview_result.iterations}")
    if preview_result.error:
        print(f"  Error: {preview_result.error}")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Recursive optimization with pulumi up
    # -------------------------------------------------------------------------
    print("Step 4: Recursive optimization — pulumi up")
    print("-" * 70)

    up_cmd = (
        f'cp "$0" {work_dir}/Pulumi.yaml && '
        f'sed -i.bak "1s/^name: .*/name: {stack_name}/" {work_dir}/Pulumi.yaml && '
        f'rm -f {work_dir}/Pulumi.yaml.bak && '
        f'cd {work_dir} && '
        f'pulumi up --yes -s {stack_name} --non-interactive 2>&1'
    )
    up_validator = (
        CliValidator("bash")
        .args(["-c", up_cmd])
        .ext("yaml")
        .env("GOOGLE_APPLICATION_CREDENTIALS", GCP_CREDS)
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .timeout(300)
        .capture()
        .required()
    )

    # Use the best output from preview as starting point
    up_prompt = f"""Output ONLY raw Pulumi YAML (no markdown fences, no commentary, no "Answer:" prefix).
Start with "name:" on line 1.

Fix the following Pulumi YAML so it deploys successfully via `pulumi up` to GCP project {gcp_project}.

{defaults.context()}

CRITICAL RULES:
- Do NOT define gcp:project in the config section — it is set externally
- Fix any deployment errors reported in the feedback
- All resource types must be valid Pulumi GCP provider types
- IAM members will be fixed automatically by runtime defaults
- staging_expiry must use YYYY-MM-DD date format (e.g. "2026-12-31"), not milliseconds
- gcp:bigquery:Dataset does NOT support deletionProtection — never add it to Dataset resources
- For resources that DO support deletionProtection, set deletionProtection: false
- Always set deleteContentsOnDestroy: true on all datasets
- If an IAM resource fails because the identity does not exist, comment it out with: # TODO: Uncomment and change identity to a valid principal

CURRENT TEMPLATE:
{preview_result.output}"""

    up_combined = yaml_checks.and_(up_validator)

    up_result = (
        reason(llm, up_prompt)
        .no_reasoning()
        .skill(skill)
        .defaults(defaults)
        .validate(up_combined)
        .max_iter(5)
        .target(1.0)
        .go()
    )

    print(f"  Result: score={up_result.score:.2f}, "
          f"iterations={up_result.iterations}")
    if up_result.error:
        print(f"  Error: {up_result.error}")
    print()

    # Select best final template
    final_yaml = up_result.output if up_result.score >= 1.0 else preview_result.output

    # Write corrected Pulumi.yaml
    final_yaml_path = Path(work_dir) / "Pulumi.yaml"
    final_yaml_path.write_text(final_yaml)
    print(f"  Final template: {final_yaml_path}")

    # -------------------------------------------------------------------------
    # Step 5: Generate README.md documentation
    # -------------------------------------------------------------------------
    print()
    print("Step 5: Generate README.md documentation")
    print("-" * 70)

    # Collect captured stdout/stderr from CLI validations (accumulated across
    # all iterations, not just the last one).
    preview_captures = preview_validator.get_captures()
    up_captures = up_validator.get_captures()

    print(f"  Preview captures: {len(preview_captures)}")
    print(f"  Deploy captures:  {len(up_captures)}")

    # Build errors & solutions section from captured CLI output.
    errors_entries = extract_errors_from_captures(preview_captures, "preview")
    errors_entries += extract_errors_from_captures(up_captures, "up")

    if errors_entries:
        errors_table_rows = "\n".join(
            f"| {err} | Fixed during recursive optimization | {stage} |"
            for stage, err in errors_entries[:10]  # Limit to 10 most relevant
        )
    else:
        errors_table_rows = "| No errors | Clean validation | preview/up |"

    folder_name = Path(SOURCE_FOLDER).name

    # Build defaults annotations for README
    defaults_annotations = []
    for ann in defaults.annotations():
        note = f" — {ann.note}" if ann.note else ""
        defaults_annotations.append(
            f"| {ann.key} | `{ann.original_pattern}` | `{ann.replacement}` | {ann.source}{note} |"
        )
    defaults_table = "\n".join(defaults_annotations) if defaults_annotations else \
        "| — | — | — | No runtime defaults configured |"

    doc_prompt = f"""Output the complete markdown document directly — do not prefix with "Answer:" or similar labels.

Generate a GitHub Markdown README.md for a Pulumi YAML template that provisions BigQuery datasets with IAM bindings.

IMPORTANT INSTRUCTIONS:
- This is a Pulumi YAML template — always say "Pulumi", never "Terraform" or "HCL"
- Do NOT mention any specific GCP project name or project ID in the README
- The template is designed to be reusable across any GCP project
- Use ${{{{\"project\"}}}} as the placeholder when referring to the project variable

TEMPLATE CODE:
```yaml
{final_yaml}
```

ERRORS ENCOUNTERED DURING VALIDATION:
| Error | Solution | Stage |
|-------|----------|-------|
{errors_table_rows}

RUNTIME DEFAULTS APPLIED:
| Key | Pattern | Replacement | Source |
|-----|---------|-------------|--------|
{defaults_table}

Write the document with EXACTLY these six sections using ## headers:

## A. Use Case
Describe what this template provisions, when to use it, and typical business context.
Be specific to BigQuery datasets and IAM in GCP.

## B. Code
Provide the final working Pulumi YAML in a ```yaml code block.
Add brief inline YAML comments explaining key resource properties.

## C. Errors & Solutions
Include a markdown table with columns: | Error | Solution | Stage |
Use the errors from the validation above. If no errors, note that validation passed cleanly.

## D. Runtime Defaults
Include a markdown table with columns: | Key | Pattern | Replacement | Source |
Use the runtime defaults table above. Explain that these values are applied automatically
via regex substitution before deployment validation, ensuring deployment-specific values
(IAM users, service accounts, project IDs) are correct regardless of LLM output.

## E. Links
Two subsections:
**Google Cloud Documentation**
- Link to BigQuery documentation at cloud.google.com/bigquery/docs
- Link to IAM documentation at cloud.google.com/iam/docs

**Pulumi Documentation**
- Link to gcp:bigquery:Dataset at pulumi.com/registry/packages/gcp/api-docs/bigquery/dataset
- Link to gcp:bigquery:DatasetIamMember at pulumi.com/registry/packages/gcp/api-docs/bigquery/datasetiammember

## F. Metadata
A ```yaml code block containing:
- category: data-analytics
- subcategory: bigquery
- template_name: {folder_name}
- keywords: [bigquery, dataset, iam, gcp, pulumi]
- gcp_services: [BigQuery, IAM]
- pulumi_resources: [gcp:bigquery:Dataset, gcp:bigquery:DatasetIamMember]"""

    print(f"  Doc prompt length: {len(doc_prompt)} chars")

    doc_checks = (
        Checks()
        .require("## A. Use Case")
        .require("## B. Code")
        .require("## C. Errors & Solutions")
        .require("## D. Runtime Defaults")
        .require("## E. Links")
        .require("## F. Metadata")
        .require("```yaml")
        .require("| Error")
        .require("pulumi.com")
        .require("cloud.google.com")
        .forbid("Terraform")   # Must use Pulumi terminology, not Terraform
        .forbid(gcp_project)   # Must not mention the specific project name
        .min_len(500)
    )

    doc_result = (
        reason(llm, doc_prompt)
        .no_reasoning()
        .skill(skill)
        .validate(doc_checks)
        .max_iter(3)
        .target(1.0)
        .go()
    )

    readme_path = Path(work_dir) / "README.md"
    readme_path.write_text(doc_result.output)

    print(f"  Result: score={doc_result.score:.2f}, "
          f"iterations={doc_result.iterations}")
    if doc_result.error:
        print(f"  Error: {doc_result.error}")
    print(f"  Written to: {readme_path}")
    print()

    # -------------------------------------------------------------------------
    # Step 6: RAG writeback
    # -------------------------------------------------------------------------
    print("Step 6: RAG writeback")
    print("-" * 70)

    # Write back the corrected Pulumi YAML (the template itself, not the docs)
    tag = f"pulumi-{Path(SOURCE_FOLDER).name}"
    mem = rag_writeback(RAG_DB, final_yaml, tag=tag)
    print()

    # -------------------------------------------------------------------------
    # Step 7: Write output files to source folder
    # -------------------------------------------------------------------------
    print("Step 7: Write output files")
    print("-" * 70)

    output_dir = Path(SOURCE_FOLDER)
    written_files = []

    # 7a. Write the corrected root Pulumi.yaml
    out_pulumi = output_dir / "Pulumi.yaml"
    out_pulumi.write_text(final_yaml)
    written_files.append(str(out_pulumi))
    print(f"  Pulumi.yaml -> {out_pulumi}")

    # 7b. Copy child folders and code files (.sql, .py, etc.) from source
    #     These are the supporting files read in Step 1 — preserved as-is.
    for rel_path, content in context_files.items():
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        written_files.append(str(out_path))
        print(f"  {rel_path} -> {out_path}")

    # 7c. Write child Pulumi.yaml files (sub-stacks) — preserved from source
    for rel_path, content in pulumi_files.items():
        if rel_path == "Pulumi.yaml":
            continue  # Already written as the corrected root template
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        written_files.append(str(out_path))
        print(f"  {rel_path} -> {out_path}")

    # 7d. Write the generated README.md
    out_readme = output_dir / "README.md"
    out_readme.write_text(doc_result.output)
    written_files.append(str(out_readme))
    print(f"  README.md -> {out_readme}")

    print(f"  Total: {len(written_files)} files written")

    verify_output_files(written_files)
    print()

    # -------------------------------------------------------------------------
    # Step 8: Package RAG knowledge base as pip wheel
    # -------------------------------------------------------------------------
    print("Step 8: Package RAG knowledge base")
    print("-" * 70)

    try:
        pkg_result = mem.package(
            f"kkachi_kb_{Path(SOURCE_FOLDER).name}",
            version="0.1.0",
            output_dir=DIST_DIR,
            description=f"Kkachi knowledge base for {Path(SOURCE_FOLDER).name}",
            author="kkachi-pipeline",
        )
        print(f"  Wheel: {pkg_result.wheel_name}")
        print(f"  Path:  {pkg_result.wheel_path}")
        print(f"  Size:  {pkg_result.size_bytes} bytes (DB: {pkg_result.db_size_bytes} bytes)")
        print(f"  Files: {pkg_result.file_count}")
        print(f"  Install: pip install {pkg_result.wheel_path}")
    except RuntimeError as e:
        print(f"  Skipped: {e}")
    print()

    # -------------------------------------------------------------------------
    # Cleanup instructions
    # -------------------------------------------------------------------------
    print("Cleanup")
    print("-" * 70)
    print(f"  To destroy deployed resources:")
    print(f"    cd {work_dir}")
    print(f"    pulumi destroy --yes -s {stack_name} --non-interactive")
    print(f"    pulumi stack rm {stack_name} --yes")
    print()

    print("=" * 70)
    print("Pipeline complete!")
    print(f"  Source folder: {output_dir}")
    print(f"  Files written: {len(written_files)}")
    for f in written_files:
        print(f"    - {f}")
    print(f"  Work dir:   {work_dir}")
    print(f"  RAG store:  {RAG_DB}")
    print(f"  Dist dir:   {DIST_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
