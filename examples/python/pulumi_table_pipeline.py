#!/usr/bin/env python3
# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Kkachi Pulumi Table Pipeline (v0.6.0 — Composable Orchestration)
================================================================
Rewritten using kkachi v0.6.0 composable orchestration APIs:

  - pipeline() for chaining refine + map steps
  - concurrent() / ConcurrentRunner for parallel deploy + README generation
  - Shared LLM callable (no copies, passed by reference through closures)

7-step pipeline:
1. Read source folder — locate Pulumi.yaml + schema.json + child code
2. RAG similarity search — DuckDB-backed store for existing templates
3. Pipeline composition: refine with `pulumi preview` CLI validation
4-5. ConcurrentRunner: deploy (pulumi up) + README generation in parallel
6. RAG writeback — update if >95% similar, else add new
7. Write output files — Pulumi.yaml, schema.json, and README.md to source folder

Inputs:  Source root folder containing Pulumi YAML templates + schema.json
Outputs: Corrected Pulumi.yaml + schema.json + README.md written to source folder

Usage:
    python pulumi_table_pipeline.py

Dependencies:
    pip install kkachi pulumi pulumi-gcp
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kkachi import (
    ApiLlm,
    Checks,
    CliValidator,
    Defaults,
    Memory,
    Skill,
    # v0.6.0: Pipeline composition + concurrent execution
    pipeline,
    concurrent,
)

# =============================================================================
# Configuration
# =============================================================================

SOURCE_FOLDER = os.environ.get(
    "SOURCE_FOLDER",
    "/Users/gatema/Desktop/telus/bi-layer-docs/stacks/big_query/tables",
)
FUELIX_KEY_PATH = os.environ.get(
    "FUELIX_KEY_PATH",
    "/Users/gatema/Desktop/drive/git/code/creds/fuelix",
)
GCP_CREDS = os.environ.get(
    "GCP_CREDS",
    "/Users/gatema/Desktop/drive/git/code/creds/terraform.json",
)
RAG_DB = os.environ.get("KKACHI_RAG_DB", "./template_knowledge.db")

# Pulumi YAML file patterns
PULUMI_PATTERNS = ("Pulumi.yaml", "Pulumi.*.yaml")


# =============================================================================
# Helpers
# =============================================================================

def is_pulumi_yaml(name: str) -> bool:
    """Check if a filename matches Pulumi.yaml or Pulumi.*.yaml."""
    return name == "Pulumi.yaml" or (name.startswith("Pulumi.") and name.endswith(".yaml"))


def make_stack_name(folder: str) -> str:
    """Create a unique pulumi stack name from the source folder path."""
    slug = Path(folder).name.replace(" ", "-").lower()
    short_hash = hashlib.md5(folder.encode()).hexdigest()[:6]
    return f"kkachi-{slug}-{short_hash}"


def strip_config_section(yaml_text: str) -> str:
    """Remove the config: section from Pulumi YAML to avoid confusing the LLM."""
    lines = yaml_text.splitlines()
    result = []
    skip = False
    for line in lines:
        if re.match(r'^config:', line):
            skip = True
            continue
        if skip and re.match(r'^[a-zA-Z]', line):
            skip = False
        if not skip:
            result.append(line)
    return "\n".join(result)


CANONICAL_CONFIG = """\
config:
  # Automatically managed by BI Layer Builder
  project:
    type: string
  # Automatically managed by BI Layer Builder
  builder:
    type: string
  # Staging expiry in YYYY-MM-DD format — for cleanup / testing in staging
  staging_expiry:
    type: string
    value: "2026-12-31"
"""


def rewrite_config_section(yaml_text: str) -> str:
    """Replace the config: section with the canonical, project-agnostic version."""
    lines = yaml_text.splitlines()
    before: list[str] = []
    after: list[str] = []
    skip = False
    found = False
    for line in lines:
        if re.match(r'^config:', line):
            skip = True
            found = True
            continue
        if skip and re.match(r'^[a-zA-Z]', line):
            skip = False
        if skip:
            continue
        if found:
            after.append(line)
        else:
            before.append(line)

    result = "\n".join(before)
    if found:
        result += "\n" + CANONICAL_CONFIG
    if after:
        result += "\n".join(after) + "\n"
    return result


# =============================================================================
# Step 1: Read Source Folder
# =============================================================================

def read_source_folder(root: str) -> tuple[dict[str, str], dict[str, str]]:
    """Read source folder recursively.

    Returns:
        pulumi_files: {relative_path: content} for Pulumi.yaml / Pulumi.*.yaml
        context_files: {relative_path: content} for all other code files
    """
    root_path = Path(root)
    pulumi_files = {}
    context_files = {}

    for f in sorted(root_path.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() in (".md", ".mdx"):
            continue
        try:
            rel = str(f.relative_to(root_path))
            content = f.read_text()
            if not content.strip():
                print(f"  ERROR: Input file is empty: {rel}")
                sys.exit(1)
            if is_pulumi_yaml(f.name):
                pulumi_files[rel] = content
            else:
                context_files[rel] = content
        except (UnicodeDecodeError, PermissionError):
            pass

    return pulumi_files, context_files


def format_files(files: dict[str, str], lang: str = "yaml") -> str:
    """Format file contents into a single context string."""
    parts = []
    for path, content in files.items():
        ext_lang = "json" if path.endswith(".json") else lang
        parts.append(f"### {path}\n```{ext_lang}\n{content}\n```")
    return "\n\n".join(parts)


# =============================================================================
# Step 2: RAG Similarity Search
# =============================================================================

def rag_lookup(query: str, k: int = 2) -> list:
    """Search RAG for similar Pulumi templates. Returns top-k results."""
    mem = Memory().persist(RAG_DB)
    if mem.is_empty():
        return []
    results = mem.search(query, k=k)
    return [r for r in results if r.score > 0.3]


# =============================================================================
# Step 6: RAG Writeback
# =============================================================================

def rag_writeback(content: str, tag: str) -> "Memory":
    """Write result back to RAG store. Update if >95% similar, else add new."""
    mem = Memory().persist(RAG_DB)
    results = mem.search(content, k=1)

    if results and results[0].score > 0.95:
        mem.update(results[0].id, content)
        print(f"  Updated existing RAG entry {results[0].id} "
              f"(similarity: {results[0].score:.3f})")
    else:
        doc_id = mem.add_tagged(tag, content)
        similarity = results[0].score if results else 0.0
        print(f"  Added new RAG entry {doc_id} "
              f"(closest similarity: {similarity:.3f})")
    return mem


# =============================================================================
# Build skill and defaults
# =============================================================================

def build_skill() -> Skill:
    """Build domain-specific skill with Pulumi/BigQuery instructions."""
    return (
        Skill()
        .instruct("pulumi_context",
            "This is a Pulumi YAML template — always use Pulumi resource types and "
            "Pulumi terminology. Never reference Terraform, HCL, or Terraform providers. "
            "All resource types use the Pulumi GCP provider (e.g. gcp:bigquery:Dataset).")
        .instruct("deleteContentsOnDestroy",
            "For datasets, always set 'deleteContentsOnDestroy: true'. "
            "Omitting it limits the capability to delete the resources through CICD.")
        .instruct("deletionProtection",
            "For gcp:bigquery:Table resources, always set 'deletionProtection: false'. "
            "This is required so tables can be deleted through CICD. "
            "NOTE: gcp:bigquery:Dataset does NOT support deletionProtection — "
            "do not add it to Dataset resources. Only add it to Table resources.")
        .instruct("staging_expiry",
            "Staging expiry should be set to YYYY-MM-DD format. "
            "Example config entry:\n"
            "  staging_expiry:\n"
            "    type: string\n"
            "    value: 2026-12-31\n"
            "Do NOT use milliseconds or epoch timestamps for staging_expiry.")
        .instruct("iam_roles",
            "IAM roles MUST be limited to the following predefined roles:\n"
            "- For BigQuery datasets (DatasetIamMember): roles/bigquery.dataViewer\n"
            "- For BigQuery tables (IamMember): roles/bigquery.dataViewer or roles/bigquery.dataEditor\n"
            "Do NOT use any other roles. Do NOT use custom roles for active IAM bindings.")
        .instruct("iam_unresolvable",
            "If an IAM binding references an identity (user, group, or serviceAccount) "
            "that may not exist or is unresolvable, comment out the entire IAM resource "
            "block in the YAML with '#' and add a note: "
            "'# TODO: Uncomment and change identity to a valid principal'. "
            "Only comment out IAM resources with unresolvable identities, not datasets or tables.")
        .instruct("naming",
            "Use snake_case for all Pulumi resource logical names.")
        .instruct("schema_json",
            "Tables that use fn::readFile for schema must reference './schema.json'. "
            "The schema variable should use: fn::readFile: ./schema.json "
            "The table's schema property should reference the variable with a block scalar: "
            "schema: |\\n  ${schema}")
        .instruct("table_options",
            "For gcp:bigquery:Table resources, include options with "
            "deleteBeforeReplace: true and replaceOnChanges for schema and tableId. "
            "This ensures schema changes trigger resource replacement.")
        .instruct("custom_roles",
            "Custom IAM roles like 'biLayerBigQueryWriter.customRole' may not exist "
            "in the target GCP project. IAM bindings that reference custom roles "
            "should be commented out with "
            "'# TODO: Uncomment when custom role is created in target project'. "
            "Replace with standard predefined roles for active bindings: "
            "roles/bigquery.dataViewer for read access, roles/bigquery.dataEditor for write.")
    )


def build_defaults(gcp_project: str, gcp_service_account: str) -> Defaults:
    """Build runtime defaults for IAM, project, and custom role substitution."""
    return (
        Defaults()
        .set("iam_user", r"user:\S+@example\.com", "user:mark.gates@telus.com",
             note="Replace with actual IAM user email for production")
        .set("iam_group", r"group:\S+@example\.com", "user:mark.gates@telus.com",
             note="Replace with actual group or user for production")
        .set("iam_sa",
             r"serviceAccount:\S+@\S+\.iam\.gserviceaccount\.com",
             f"serviceAccount:{gcp_service_account}",
             note="Service account from GCP credentials JSON")
        .from_env("gcp_project", r"my-gcp-project",
                   "GOOGLE_CLOUD_PROJECT", fallback=gcp_project,
                   note="Target GCP project ID")
        .set("custom_role",
             r"projects/\$\{project\}/roles/biLayerBigQueryWriter\.customRole",
             "roles/bigquery.dataEditor",
             note="Replace custom role with standard role for testing")
    )


def build_yaml_checks() -> Checks:
    """Build structural YAML checks for Pulumi templates."""
    return (
        Checks()
        .require("runtime: yaml")
        .require("resources:")
        .require("gcp:bigquery:Dataset")
        .require("gcp:bigquery:Table")
        .require("deletionProtection: false")
        .require("fn::readFile")
        .forbid("bilayer-sa@")
        .forbid("```")
        .forbid("Terraform")
    )


# =============================================================================
# Main Pipeline — Composable Orchestration (v0.6.0)
# =============================================================================

def main():
    print("=" * 70)
    print("Kkachi Pulumi Table Pipeline (Python v0.6.0 — Composable Orchestration)")
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

    # Build context — strip config sections
    stripped_files = {k: strip_config_section(v) for k, v in pulumi_files.items()}
    pulumi_context = format_files(stripped_files, lang="yaml")
    context_section = format_files(context_files) if context_files else ""

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

    rag_results = rag_lookup(root_template[:500])
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

    with open(FUELIX_KEY_PATH) as f:
        api_key = f.read().strip()
    llm = (
        ApiLlm.openai_with_url(api_key, "claude-sonnet-4-6", "https://api.fuelix.ai")
        .with_timeout(120)
        .with_retry(3)
    )
    print(f"  LLM: claude-sonnet-4-6 via fuelix.ai")

    with open(GCP_CREDS) as f:
        creds = json.load(f)
    gcp_project = creds["project_id"]
    gcp_service_account = creds["client_email"]
    print(f"  GCP project: {gcp_project}")
    print(f"  Service account: {gcp_service_account}")

    stack_name = make_stack_name(SOURCE_FOLDER)
    print(f"  Stack name: {stack_name}")

    work_dir = tempfile.mkdtemp(prefix="kkachi-pulumi-table-")
    print(f"  Work dir: {work_dir}")

    # Copy schema.json to work_dir
    source_schema = Path(SOURCE_FOLDER) / "schema.json"
    if source_schema.exists():
        shutil.copy2(source_schema, Path(work_dir) / "schema.json")
        print("  Copied schema.json to work dir")
    else:
        print("  WARNING: schema.json not found in source folder")

    # Bootstrap Pulumi.yaml
    Path(work_dir, "Pulumi.yaml").write_text(
        f"name: {stack_name}\nruntime: yaml\ndescription: kkachi pipeline\n"
    )

    os.environ.setdefault("PULUMI_CONFIG_PASSPHRASE", "")

    # Initialize or select stack
    subprocess.run(
        ["pulumi", "stack", "init", stack_name, "--non-interactive"],
        cwd=work_dir, capture_output=True
    )
    for key, value in [
        ("gcp:project", gcp_project),
        ("project", gcp_project),
        ("builder", gcp_service_account),
    ]:
        subprocess.run(
            ["pulumi", "config", "set", key, value, "-s", stack_name],
            cwd=work_dir, capture_output=True
        )

    # Create stack config file
    stack_config_file = Path(work_dir) / f"Pulumi.{stack_name}.yaml"
    stack_config_file.write_text(
        f"config:\n  gcp:project: {gcp_project}\n"
        f"  project: {gcp_project}\n"
        f"  builder: {gcp_service_account}\n"
    )
    config_flags = (
        f'--config project={gcp_project} '
        f'--config builder={gcp_service_account}'
    )
    print(f"  Stack config: {stack_config_file.name}")

    # Runtime defaults
    defaults = build_defaults(gcp_project, gcp_service_account)
    print(f"  Defaults: {len(defaults)} entries")
    for ann in defaults.annotations():
        note = f" — {ann.note}" if ann.note else ""
        print(f"    {ann.key}: {ann.replacement} ({ann.source}){note}")
    print()

    # Domain skill
    skill = build_skill()
    print(f"  Skill: {len(skill)} instructions")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Pipeline composition — refine with pulumi preview
    #
    # Uses pipeline() to chain: .refine(yaml_checks AND cli_validator)
    # -------------------------------------------------------------------------
    print("Step 3: Pipeline composition — pulumi preview")
    print("-" * 70)

    preview_cmd = (
        f'cp "$0" {work_dir}/Pulumi.yaml && '
        f'sed -i.bak "1s/^name: .*/name: {stack_name}/" {work_dir}/Pulumi.yaml && '
        f'rm -f {work_dir}/Pulumi.yaml.bak && '
        f'cd {work_dir} && '
        f'pulumi preview -s {stack_name} {config_flags} --non-interactive 2>&1'
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

    yaml_checks = build_yaml_checks()
    preview_combined = yaml_checks.and_(preview_validator)

    schema_json_content = ""
    if source_schema.exists():
        schema_json_content = source_schema.read_text()

    preview_prompt = f"""Output ONLY raw Pulumi YAML (no markdown fences, no commentary, no "Answer:" prefix).
Start with "name:" on line 1.

Create a Pulumi YAML program that provisions BigQuery datasets, tables, and IAM bindings in GCP project {gcp_project}.

{defaults.context()}

CRITICAL RULES:
- Do NOT define gcp:project in the config section — it is set externally via `pulumi config`
- You MUST define these config variables: project (type: string), builder (type: string), staging_expiry (type: string)
- The "project" config variable is REQUIRED — without it, ${{project}} references in resources will fail
- Use "value:" for config defaults, NOT "default:"
- staging_expiry must use YYYY-MM-DD date format. Example: staging_expiry: {{type: string, value: "2026-12-31"}}
- Use ${{project}} to reference the project config variable in resource properties
- All resource types must be valid Pulumi GCP types (gcp:bigquery:Dataset, gcp:bigquery:DatasetIamMember, gcp:bigquery:Table, gcp:bigquery:IamMember)
- IAM members will be fixed automatically by runtime defaults — use any placeholder
- gcp:bigquery:Dataset does NOT support deletionProtection — never add it to Dataset resources
- gcp:bigquery:Table DOES support deletionProtection — always set deletionProtection: false on Table resources
- Always set deleteContentsOnDestroy: true on all datasets
- If an IAM resource references an identity that may not exist, comment it out with a TODO note
- The bq_writer custom role (biLayerBigQueryWriter.customRole) does NOT exist in this project — comment out any IAM bindings that use ${{bq_writer}} or custom roles, with a TODO note
- IAM roles MUST be limited to: roles/bigquery.dataViewer (datasets & table read), roles/bigquery.dataEditor (table write)
- This is a Pulumi YAML template — do NOT reference Terraform or HCL

RESOURCE STRUCTURE (adapt from source templates below):
- A dataset with deleteContentsOnDestroy: true
- Dataset-level IAM bindings (gcp:bigquery:DatasetIamMember) for user, group, service account
- A table with:
  - deletionProtection: false
  - timePartitioning (type: DAY, field: part_dt)
  - schema loaded via fn::readFile: ./schema.json (use a variable with fn::readFile)
  - options: deleteBeforeReplace: true, replaceOnChanges: [schema, tableId]
- Table-level IAM bindings (gcp:bigquery:IamMember) for user, group, service account
  - Note: table-level IAM uses gcp:bigquery:IamMember (NOT DatasetIamMember)
  - Table IAM requires: project, datasetId, tableId, role, member
- A bq_writer variable: projects/${{project}}/roles/biLayerBigQueryWriter.customRole
- Location variable: northamerica-northeast1

SCHEMA FILE (schema.json — already exists in work directory):
```json
{schema_json_content}
```

SOURCE TEMPLATES (for reference — adapt the structure, NOT the config):
{pulumi_context}

{context_section}
{rag_section}"""

    # v0.6.0: Use pipeline() composition instead of reason()
    preview_result = (
        pipeline(llm, preview_prompt)
        .refine(preview_combined, max_iter=8, target=1.0)
        .go()
    )

    print(f"  Pipeline result:")
    print(f"    Steps: {preview_result.steps_count}")
    print(f"    Tokens: {preview_result.total_tokens}")
    print(f"    Elapsed: {preview_result.elapsed_ms}ms")
    print(f"    Output: {len(preview_result.output)}B")
    print()

    # -------------------------------------------------------------------------
    # Steps 4-5: ConcurrentRunner — deploy + README generation in parallel
    #
    # Uses concurrent() to run two pipelines concurrently on a shared LLM:
    #   1. "deploy" — refine with `pulumi up`
    #   2. "readme" — refine with doc_checks
    # -------------------------------------------------------------------------
    print("Steps 4-5: ConcurrentRunner — deploy + README in parallel")
    print("-" * 70)

    up_cmd = (
        f'cp "$0" {work_dir}/Pulumi.yaml && '
        f'sed -i.bak "1s/^name: .*/name: {stack_name}/" {work_dir}/Pulumi.yaml && '
        f'rm -f {work_dir}/Pulumi.yaml.bak && '
        f'cd {work_dir} && '
        f'pulumi up --yes -s {stack_name} {config_flags} --non-interactive 2>&1'
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

    up_combined = build_yaml_checks().and_(up_validator)

    # Capture preview output for concurrent tasks
    preview_output = preview_result.output

    # Collect preview captures for README
    preview_captures = preview_validator.get_captures()
    print(f"  Preview captures: {len(preview_captures)}")

    # Build error entries from preview captures
    errors_entries = []
    for cap in preview_captures:
        if not cap.success:
            output = cap.stderr if cap.stderr.strip() else cap.stdout
            for line in output.strip().splitlines():
                line = line.strip()
                if line and ("error" in line.lower() or "failed" in line.lower()
                             or "invalid" in line.lower()):
                    errors_entries.append(("preview", line[:200]))

    # Build README prompt components
    final_yaml_canonical = rewrite_config_section(preview_output)

    if errors_entries:
        errors_table_rows = "\n".join(
            f"| {err} | Fixed during recursive optimization | {stage} |"
            for stage, err in errors_entries[:10]
        )
    else:
        errors_table_rows = "| No errors | Clean validation | preview/up |"

    folder_name = Path(SOURCE_FOLDER).name

    defaults_annotations = []
    for ann in defaults.annotations():
        note = f" — {ann.note}" if ann.note else ""
        defaults_annotations.append(
            f"| {ann.key} | `{ann.original_pattern}` | `{ann.replacement}` | {ann.source}{note} |"
        )
    defaults_table = "\n".join(defaults_annotations) if defaults_annotations else \
        "| — | — | — | No runtime defaults configured |"

    # Build deploy prompt
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
- gcp:bigquery:Table DOES support deletionProtection — always set deletionProtection: false
- Always set deleteContentsOnDestroy: true on all datasets
- Table schema must use fn::readFile: ./schema.json via a variable
- Table must have options: deleteBeforeReplace: true, replaceOnChanges: [schema, tableId]
- If an IAM resource fails because the identity does not exist, comment it out with: # TODO: Uncomment and change identity to a valid principal
- Custom IAM roles do NOT exist — comment out IAM bindings using custom roles
- IAM roles MUST be limited to: roles/bigquery.dataViewer, roles/bigquery.dataEditor
- If an IAM resource fails because the identity does not exist, comment it out with a TODO note

CURRENT TEMPLATE:
{preview_output}"""

    # Build README prompt
    doc_prompt = f"""Output the complete markdown document directly — do not prefix with "Answer:" or similar labels.

Generate a GitHub Markdown README.md for a Pulumi YAML template that provisions BigQuery datasets, tables, and IAM bindings.

IMPORTANT INSTRUCTIONS:
- This is a Pulumi YAML template — always say "Pulumi", never "Terraform" or "HCL"
- Do NOT mention any specific GCP project name or project ID in the README
- The template is designed to be reusable across any GCP project
- Use ${{{{"project"}}}} as the placeholder when referring to the project variable

TEMPLATE CODE:
```yaml
{final_yaml_canonical}
```

SCHEMA FILE:
```json
{schema_json_content}
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

## B. Code
Provide the final working Pulumi YAML in a ```yaml code block.
Also show the schema.json in a separate ```json code block.

## C. Errors & Solutions
Include a markdown table with columns: | Error | Solution | Stage |

## D. Runtime Defaults
Include a markdown table with columns: | Key | Pattern | Replacement | Source |

## E. Links
Two subsections:
**Google Cloud Documentation**
- Link to BigQuery documentation at cloud.google.com/bigquery/docs
- Link to IAM documentation at cloud.google.com/iam/docs

**Pulumi Documentation**
- Link to gcp:bigquery:Dataset at pulumi.com/registry/packages/gcp/api-docs/bigquery/dataset
- Link to gcp:bigquery:Table at pulumi.com/registry/packages/gcp/api-docs/bigquery/table

## F. Metadata
A ```yaml code block with category, subcategory, template_name: {folder_name}, keywords, gcp_services, pulumi_resources"""

    doc_checks = (
        Checks()
        .require("## A. Use Case")
        .require("## B. Code")
        .require("## C. Errors & Solutions")
        .require("## D. Runtime Defaults")
        .require("## E. Links")
        .require("## F. Metadata")
        .require("```yaml")
        .require("```json")
        .require("pulumi.com")
        .require("cloud.google.com")
        .require("schema.json")
        .forbid("Terraform")
    )

    print("  Launching concurrent() runner: deploy + README generation...")

    # v0.6.0: Use concurrent() to run deploy and README pipelines in parallel
    concurrent_results = (
        concurrent(llm)
        .task("deploy", up_prompt, lambda p: p.refine(up_combined, max_iter=5, target=1.0))
        .task("readme", doc_prompt, lambda p: p.refine(doc_checks, max_iter=3, target=1.0))
        .max_concurrency(2)
        .go()
    )

    # -------------------------------------------------------------------------
    # Process concurrent results
    # -------------------------------------------------------------------------
    print()
    print("Concurrent results:")
    print("-" * 70)

    for result in concurrent_results:
        status = f"{'OK' if result.success else 'FAIL'} (output={len(result.output)}B)"
        print(f"  {result.label}: {status} (elapsed={result.elapsed_ms}ms)")
    print()

    # Extract deploy and README results
    deploy_result = next((r for r in concurrent_results if r.label == "deploy"), None)
    readme_result = next((r for r in concurrent_results if r.label == "readme"), None)

    # Select best final template
    deploy_output = deploy_result.output if deploy_result and deploy_result.success else ""
    final_yaml = deploy_output if deploy_output else preview_output

    # Collect deploy captures
    up_captures = up_validator.get_captures()
    print(f"  Deploy captures: {len(up_captures)}")

    # Write corrected Pulumi.yaml to work dir
    final_yaml_path = Path(work_dir) / "Pulumi.yaml"
    final_yaml_path.write_text(final_yaml)
    print(f"  Final template: {final_yaml_path}")
    print()

    # -------------------------------------------------------------------------
    # Step 6: RAG writeback
    # -------------------------------------------------------------------------
    print("Step 6: RAG writeback")
    print("-" * 70)

    tag = f"pulumi-{Path(SOURCE_FOLDER).name}"
    mem = rag_writeback(rewrite_config_section(final_yaml), tag=tag)
    print()

    # -------------------------------------------------------------------------
    # Step 7: Write output files to source folder
    # -------------------------------------------------------------------------
    print("Step 7: Write output files")
    print("-" * 70)

    output_dir = Path(SOURCE_FOLDER)
    written_files = []

    # 7a. Write corrected root Pulumi.yaml (with canonical portable config)
    final_yaml_portable = rewrite_config_section(final_yaml)
    out_pulumi = output_dir / "Pulumi.yaml"
    out_pulumi.write_text(final_yaml_portable)
    written_files.append(str(out_pulumi))
    print(f"  Pulumi.yaml -> {out_pulumi}")

    # 7b. Copy context files (preserved as-is)
    for rel_path, content in context_files.items():
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        written_files.append(str(out_path))
        print(f"  {rel_path} -> {out_path}")

    # 7c. Write child Pulumi.yaml files
    for rel_path, content in pulumi_files.items():
        if rel_path == "Pulumi.yaml":
            continue
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        written_files.append(str(out_path))
        print(f"  {rel_path} -> {out_path}")

    # 7d. Write README.md
    readme_text = readme_result.output if readme_result and readme_result.success else "# README\n\nGeneration pending."
    out_readme = output_dir / "README.md"
    out_readme.write_text(readme_text)
    written_files.append(str(out_readme))
    print(f"  README.md -> {out_readme}")

    print(f"  Total: {len(written_files)} files written")

    # Verify no output file is empty
    errors = []
    for f in written_files:
        p = Path(f)
        if not p.exists():
            errors.append(f"  ERROR: Output file missing: {f}")
        elif p.stat().st_size == 0 or not p.read_text().strip():
            errors.append(f"  ERROR: Output file is empty: {f}")
    if errors:
        print()
        for err in errors:
            print(err)
        print(f"\n  Pipeline failed: {len(errors)} empty/missing output file(s)")
        sys.exit(1)
    else:
        print("  Verified: all output files are non-empty")
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
    print("Pipeline complete! (v0.6.0 Composable Orchestration)")
    print(f"  Source folder: {output_dir}")
    print(f"  Files written: {len(written_files)}")
    for f in written_files:
        print(f"    - {f}")
    print(f"  Work dir:   {work_dir}")
    print(f"  RAG store:  {RAG_DB}")
    print()
    print("  New v0.6.0 features used:")
    print("    - pipeline() composition (refine with chained validators)")
    print("    - concurrent() runner (deploy + README in parallel)")
    print("    - Shared LLM callable (passed through closure to concurrent tasks)")
    print("=" * 70)


if __name__ == "__main__":
    main()
