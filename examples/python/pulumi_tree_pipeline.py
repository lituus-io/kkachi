#!/usr/bin/env python3
# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Kkachi Pulumi Tree Pipeline
================================
Walks a directory tree, discovers every folder containing Pulumi.yaml (or
Pulumi.*.yaml), and runs the full 7-step pipeline on each one:

1. Read root folder — Pulumi.yaml + supporting files (schema.json, .sql, .js, .py)
2. RAG similarity search — DuckDB-backed store
3. Recursive optimization with `pulumi preview` CLI validation
4. Recursive optimization with `pulumi up` CLI validation
5. Generate structured GitHub Markdown README.md (sections A–F)
6. RAG writeback — update if >95% similar, else add new
7. Write output files — corrected Pulumi.yaml + README.md back to source folder

The tree walker treats each directory containing a Pulumi.yaml as an independent
root.  Child subdirectories that also have their own Pulumi.yaml are treated as
separate roots (not merged into the parent).

Usage:
    python pulumi_tree_pipeline.py

Dependencies:
    pip install kkachi pulumi pulumi-gcp
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from kkachi import (
    ApiLlm,
    Checks,
    CliValidator,
    Defaults,
    Memory,
    Skill,
    reason,
)

from _pipeline_common import (
    is_pulumi_yaml,
    make_stack_name,
    strip_config_section,
    rewrite_config_section,
    format_files,
    rag_lookup,
    rag_writeback,
    extract_errors_from_captures,
    verify_output_files,
    CANONICAL_CONFIG,
)

# =============================================================================
# Configuration
# =============================================================================

SCAN_ROOT = os.environ.get(
    "SCAN_ROOT",
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
DIST_DIR = os.environ.get("KKACHI_DIST_DIR", str(Path(SCAN_ROOT) / "dist"))


# =============================================================================
# Template introspection
# =============================================================================

# Resource types we can detect in YAML content
RESOURCE_PATTERNS = {
    "gcp:bigquery:Dataset":          "Dataset",
    "gcp:bigquery:Table":            "Table",
    "gcp:bigquery:Routine":          "Routine",
    "gcp:bigquery:DatasetIamMember": "DatasetIamMember",
    "gcp:bigquery:IamMember":        "IamMember",
    "gcp:storage:Bucket":            "Bucket",
    "gcp:storage/bucketIAMMember:BucketIAMMember": "BucketIAMMember",
    "gcp:cloudrunv2:Service":        "CloudRun",
    "gcp:secretmanager:Secret":      "Secret",
    "gcp:workflows:Workflow":        "Workflow",
    "gcp:cloudscheduler/job:Job":    "Scheduler",
    "google-native:bigquery/v2:Routine": "NativeRoutine",
}

# Pulumi registry base URL fragments for documentation links
PULUMI_DOC_URLS = {
    "gcp:bigquery:Dataset":          "pulumi.com/registry/packages/gcp/api-docs/bigquery/dataset",
    "gcp:bigquery:Table":            "pulumi.com/registry/packages/gcp/api-docs/bigquery/table",
    "gcp:bigquery:Routine":          "pulumi.com/registry/packages/gcp/api-docs/bigquery/routine",
    "gcp:bigquery:DatasetIamMember": "pulumi.com/registry/packages/gcp/api-docs/bigquery/datasetiammember",
    "gcp:bigquery:IamMember":        "pulumi.com/registry/packages/gcp/api-docs/bigquery/iammember",
    "gcp:storage:Bucket":            "pulumi.com/registry/packages/gcp/api-docs/storage/bucket",
    "gcp:cloudrunv2:Service":        "pulumi.com/registry/packages/gcp/api-docs/cloudrunv2/service",
    "gcp:workflows:Workflow":        "pulumi.com/registry/packages/gcp/api-docs/workflows/workflow",
    "gcp:cloudscheduler/job:Job":    "pulumi.com/registry/packages/gcp/api-docs/cloudscheduler/job",
}


@dataclass
class TemplateInfo:
    """Introspected metadata about a single Pulumi root folder."""
    root: str                                    # absolute path
    pulumi_files: dict[str, str] = field(default_factory=dict)   # rel → content
    context_files: dict[str, str] = field(default_factory=dict)  # rel → content
    resource_types: set[str] = field(default_factory=set)
    has_table: bool = False
    has_routine: bool = False
    has_view: bool = False
    has_materialized_view: bool = False
    has_external_table: bool = False
    has_bucket: bool = False
    has_workflow: bool = False
    has_cloud_run: bool = False
    has_fn_read_file: bool = False
    has_schema_json: bool = False
    has_query_sql: bool = False
    has_script_js: bool = False
    has_python_file: bool = False
    has_custom_role: bool = False
    has_deletion_protection: bool = False



# =============================================================================
# Tree discovery
# =============================================================================

def discover_roots(scan_root: str) -> list[str]:
    """Walk the tree and return every directory that contains a Pulumi.yaml."""
    roots = []
    for dirpath, _dirnames, filenames in os.walk(scan_root):
        for fn in filenames:
            if is_pulumi_yaml(fn):
                roots.append(dirpath)
                break  # one match per directory is enough
    return sorted(roots)


def read_root_folder(root: str) -> tuple[dict[str, str], dict[str, str]]:
    """Read ONLY the files in *this* directory — do NOT recurse into child
    directories that have their own Pulumi.yaml (those are separate roots)."""
    root_path = Path(root)
    pulumi_files = {}
    context_files = {}

    for f in sorted(root_path.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() in (".md", ".mdx"):
            continue
        # Skip dist / output directories
        if f.name == "dist":
            continue
        try:
            content = f.read_text()
            if not content.strip():
                continue
            rel = f.name
            if is_pulumi_yaml(f.name):
                pulumi_files[rel] = content
            else:
                context_files[rel] = content
        except (UnicodeDecodeError, PermissionError):
            pass

    # Also pick up files in immediate subdirectories that do NOT have their own
    # Pulumi.yaml (e.g., stored_procedures/pyspark_stored_proc/src/spark/python/main.py)
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith(".") or subdir.name == "dist":
            continue
        # Skip child roots — they have their own pipeline run
        child_has_pulumi = any(is_pulumi_yaml(c.name) for c in subdir.iterdir() if c.is_file())
        if child_has_pulumi:
            continue
        for f in sorted(subdir.rglob("*")):
            if not f.is_file() or f.name.startswith("."):
                continue
            if f.suffix.lower() in (".md", ".mdx"):
                continue
            try:
                content = f.read_text()
                if not content.strip():
                    continue
                rel = str(f.relative_to(root_path))
                if is_pulumi_yaml(f.name):
                    pulumi_files[rel] = content
                else:
                    context_files[rel] = content
            except (UnicodeDecodeError, PermissionError):
                pass

    return pulumi_files, context_files


def introspect(root: str, pulumi_files: dict[str, str],
               context_files: dict[str, str]) -> TemplateInfo:
    """Analyze the template content and detect what resource types, features,
    and supporting files are present."""
    info = TemplateInfo(root=root, pulumi_files=pulumi_files, context_files=context_files)
    all_yaml = "\n".join(pulumi_files.values())

    for res_type in RESOURCE_PATTERNS:
        if res_type in all_yaml:
            info.resource_types.add(res_type)

    info.has_table = "gcp:bigquery:Table" in info.resource_types
    info.has_routine = ("gcp:bigquery:Routine" in info.resource_types or
                        "google-native:bigquery/v2:Routine" in info.resource_types)
    info.has_view = "view:" in all_yaml and "materializedView:" not in all_yaml and info.has_table
    info.has_materialized_view = "materializedView:" in all_yaml
    info.has_external_table = "externalDataConfiguration:" in all_yaml
    info.has_bucket = "gcp:storage:Bucket" in info.resource_types
    info.has_workflow = "gcp:workflows:Workflow" in info.resource_types
    info.has_cloud_run = "gcp:cloudrunv2:Service" in info.resource_types
    info.has_fn_read_file = "fn::readFile" in all_yaml
    info.has_schema_json = "schema.json" in context_files
    info.has_query_sql = "query.sql" in context_files
    info.has_script_js = "script.js" in context_files
    info.has_python_file = any(f.endswith(".py") for f in context_files)
    info.has_custom_role = "customRole" in all_yaml or "bq_writer" in all_yaml
    info.has_deletion_protection = "deletionProtection" in all_yaml

    return info


# =============================================================================
# Dynamic prompt / skill / checks builders
# =============================================================================

def build_skill(info: TemplateInfo) -> Skill:
    """Build a Skill with instructions relevant to the detected template features."""
    skill = (
        Skill()
        .instruct("pulumi_context",
            "This is a Pulumi YAML template — always use Pulumi resource types and "
            "Pulumi terminology. Never reference Terraform, HCL, or Terraform providers. "
            "All resource types use the Pulumi GCP provider (e.g. gcp:bigquery:Dataset).")
        .instruct("deleteContentsOnDestroy",
            "For datasets, always set 'deleteContentsOnDestroy: true'. "
            "Omitting it limits the capability to delete the resources through CICD.")
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
            "- For Cloud Storage buckets (BucketIAMMember): roles/storage.objectViewer or roles/storage.objectAdmin\n"
            "Do NOT use any other roles. Do NOT use custom roles for active IAM bindings.")
        .instruct("iam_unresolvable",
            "If an IAM binding references an identity (user, group, or serviceAccount) "
            "that may not exist or is unresolvable, comment out the entire IAM resource "
            "block in the YAML with '#' and add a note: "
            "'# TODO: Uncomment and change identity to a valid principal'. "
            "Only comment out IAM resources with unresolvable identities.")
        .instruct("naming",
            "Use snake_case for all Pulumi resource logical names.")
    )

    if info.has_table and not info.has_view and not info.has_materialized_view:
        skill = skill.instruct("deletionProtection",
            "For gcp:bigquery:Table resources, always set 'deletionProtection: false'. "
            "This is required so tables can be deleted through CICD. "
            "NOTE: gcp:bigquery:Dataset does NOT support deletionProtection — "
            "do not add it to Dataset resources. Only add it to Table resources.")

    if info.has_table and not info.has_view and not info.has_materialized_view and not info.has_external_table:
        skill = skill.instruct("table_options",
            "For gcp:bigquery:Table resources with schema, include options with "
            "deleteBeforeReplace: true and replaceOnChanges for schema and tableId.")

    if info.has_fn_read_file and info.has_schema_json:
        skill = skill.instruct("schema_json",
            "Tables that use fn::readFile for schema must reference './schema.json'. "
            "The schema variable should use: fn::readFile: ./schema.json")

    if info.has_fn_read_file and info.has_query_sql:
        skill = skill.instruct("query_sql",
            "Views, materialized views, routines, or workflows that use fn::readFile "
            "for SQL must reference './query.sql'. "
            "The query variable should use: fn::readFile: ./query.sql")

    if info.has_fn_read_file and info.has_script_js:
        skill = skill.instruct("script_js",
            "Scalar functions that use fn::readFile for JavaScript must reference "
            "'./script.js'. The code variable should use: fn::readFile: ./script.js")

    if info.has_custom_role:
        skill = skill.instruct("custom_roles",
            "Custom IAM roles like 'biLayerBigQueryWriter.customRole' may not exist "
            "in the target GCP project. IAM bindings that reference custom roles "
            "should be commented out with "
            "'# TODO: Uncomment when custom role is created in target project'. "
            "Replace with standard predefined roles for active bindings: "
            "roles/bigquery.dataViewer for read access, roles/bigquery.dataEditor for write.")

    if info.has_view:
        skill = skill.instruct("views",
            "BigQuery views use gcp:bigquery:Table with a 'view' property. "
            "The view block should include 'useLegacySql: false' and a query. "
            "Use replaceOnChanges: [query, tableId] in options.")

    if info.has_materialized_view:
        skill = skill.instruct("materialized_views",
            "BigQuery materialized views use gcp:bigquery:Table with a "
            "'materializedView' property containing 'enableRefresh: true' and "
            "'refreshIntervalMs'. Use a query loaded via fn::readFile if available.")

    if info.has_external_table:
        skill = skill.instruct("external_tables",
            "External BigQuery tables use gcp:bigquery:Table with "
            "'externalDataConfiguration'. Include sourceFormat, sourceUris, "
            "and autodetect or schema as appropriate.")

    if info.has_routine:
        skill = skill.instruct("routines",
            "BigQuery routines (stored procedures, scalar functions, table functions) "
            "use gcp:bigquery:Routine. Include routineType, language, "
            "definitionBody, and arguments as appropriate. "
            "Use replaceOnChanges: [definitionBody] in options.")

    if info.has_workflow:
        skill = skill.instruct("workflows",
            "Cloud Workflows use gcp:workflows:Workflow with a sourceContents "
            "property containing the workflow YAML definition.")

    if info.has_bucket:
        skill = skill.instruct("buckets",
            "Cloud Storage buckets should include forceDestroy: true for CICD, "
            "a location, and uniformBucketLevelAccess: true.\n"
            "Bucket names MUST be prefixed with ${project}-${sanitized_process_nm}.\n"
            "Use a sanitized_process_nm variable with fn::str:replace to convert "
            "underscores to hyphens:\n"
            "  sanitized_process_nm:\n"
            "    fn::str:replace:\n"
            "      string: ${process_nm}\n"
            "      old: \"_\"\n"
            "      new: \"-\"\n"
            "This fn::str:replace function is a Pulumi YAML built-in and MUST be preserved.")

    return skill


def build_defaults(gcp_project: str, gcp_service_account: str,
                   info: TemplateInfo) -> Defaults:
    """Build runtime Defaults relevant to the detected template features."""
    defaults = (
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
    )
    if info.has_custom_role:
        defaults = defaults.set("custom_role",
            r"projects/\$\{project\}/roles/biLayerBigQueryWriter\.customRole",
            "roles/bigquery.dataEditor",
            note="Replace custom role with standard role for testing")
    return defaults


def build_yaml_checks(info: TemplateInfo) -> Checks:
    """Build Checks validators based on detected resource types."""
    checks = (
        Checks()
        .require("runtime: yaml")
        .require("resources:")
        .forbid("bilayer-sa@")
        .forbid("```")
    )
    if "gcp:bigquery:Dataset" in info.resource_types:
        checks = checks.require("gcp:bigquery:Dataset")
    if info.has_table and not info.has_view and not info.has_materialized_view:
        checks = checks.require("gcp:bigquery:Table")
        checks = checks.require("deletionProtection: false")
    if info.has_fn_read_file:
        checks = checks.require("fn::readFile")
    if info.has_routine:
        checks = checks.require("gcp:bigquery:Routine")
    if info.has_view:
        checks = checks.require("view:")
    if info.has_materialized_view:
        checks = checks.require("materializedView:")
    if info.has_bucket:
        checks = checks.require("fn::str:replace")
    return checks


def describe_template(info: TemplateInfo) -> str:
    """Generate a human-readable description of what the template provisions."""
    parts = []
    if "gcp:bigquery:Dataset" in info.resource_types:
        parts.append("BigQuery datasets")
    if info.has_table and not info.has_view and not info.has_materialized_view and not info.has_external_table:
        parts.append("tables with schema and time partitioning")
    if info.has_view:
        parts.append("BigQuery views")
    if info.has_materialized_view:
        parts.append("materialized views")
    if info.has_external_table:
        parts.append("external tables")
    if info.has_routine:
        parts.append("BigQuery routines (stored procedures / functions)")
    if info.has_bucket:
        parts.append("Cloud Storage buckets")
    if info.has_workflow:
        parts.append("Cloud Workflows")
    if info.has_cloud_run:
        parts.append("Cloud Run services")
    parts.append("IAM bindings")
    return ", ".join(parts)


def build_resource_structure(info: TemplateInfo) -> str:
    """Build a RESOURCE STRUCTURE section tailored to this template."""
    lines = []

    if "gcp:bigquery:Dataset" in info.resource_types:
        lines.append("- A dataset with deleteContentsOnDestroy: true")
    if "gcp:bigquery:DatasetIamMember" in info.resource_types:
        lines.append("- Dataset-level IAM bindings (gcp:bigquery:DatasetIamMember)")

    if info.has_table and not info.has_view and not info.has_materialized_view and not info.has_external_table:
        lines.append("- A table with:")
        lines.append("  - deletionProtection: false")
        if info.has_schema_json:
            lines.append("  - schema loaded via fn::readFile: ./schema.json")
        lines.append("  - options: deleteBeforeReplace: true, replaceOnChanges: [schema, tableId]")
    if info.has_view:
        lines.append("- A BigQuery view (gcp:bigquery:Table with view property):")
        lines.append("  - view.useLegacySql: false")
        if info.has_query_sql:
            lines.append("  - query loaded via fn::readFile: ./query.sql")
        lines.append("  - options: replaceOnChanges: [query, tableId]")
    if info.has_materialized_view:
        lines.append("- A materialized view (gcp:bigquery:Table with materializedView property):")
        lines.append("  - enableRefresh: true, refreshIntervalMs")
        if info.has_query_sql:
            lines.append("  - query loaded via fn::readFile: ./query.sql")
    if info.has_external_table:
        lines.append("- An external table (gcp:bigquery:Table with externalDataConfiguration):")
        lines.append("  - sourceFormat, sourceUris, autodetect or schema")
    if info.has_routine:
        lines.append("- A BigQuery routine (gcp:bigquery:Routine):")
        lines.append("  - routineType, language, definitionBody, arguments")
        if info.has_query_sql:
            lines.append("  - definitionBody from fn::readFile: ./query.sql")
        if info.has_script_js:
            lines.append("  - definitionBody from fn::readFile: ./script.js")
        if info.has_python_file:
            lines.append("  - definitionBody from fn::readFile for Python code")
        lines.append("  - options: replaceOnChanges: [definitionBody]")
    if info.has_bucket:
        lines.append("- A Cloud Storage bucket with forceDestroy: true")
        lines.append("  - Bucket name prefixed with ${project}-${sanitized_process_nm}")
        lines.append("  - sanitized_process_nm variable using fn::str:replace (underscores → hyphens)")
    if info.has_workflow:
        lines.append("- A Cloud Workflow with sourceContents")
    if info.has_cloud_run:
        lines.append("- A Cloud Run service")
    if "gcp:bigquery:IamMember" in info.resource_types:
        lines.append("- Table-level IAM bindings (gcp:bigquery:IamMember)")
    if info.has_custom_role:
        lines.append("- A bq_writer variable for custom role (comment out bindings using it)")

    lines.append("- Location variable: northamerica-northeast1")
    return "\n".join(lines)


def build_pulumi_doc_links(info: TemplateInfo) -> str:
    """Build the Pulumi documentation links section for the README prompt."""
    lines = []
    for res_type in sorted(info.resource_types):
        url = PULUMI_DOC_URLS.get(res_type)
        if url:
            lines.append(f"- Link to {res_type} at {url}")
    return "\n".join(lines) if lines else "- Link to Pulumi GCP provider at pulumi.com/registry/packages/gcp"



# =============================================================================
# Per-root pipeline
# =============================================================================

@dataclass
class PipelineResult:
    root: str
    preview_score: float = 0.0
    preview_iters: int = 0
    up_score: float = 0.0
    up_iters: int = 0
    doc_score: float = 0.0
    doc_iters: int = 0
    files_written: int = 0
    work_dir: str = ""
    stack_name: str = ""
    error: str = ""


def run_pipeline(root: str, llm, gcp_project: str, gcp_service_account: str) -> PipelineResult:
    """Run the full 7-step pipeline on a single root folder."""
    result = PipelineResult(root=root)
    folder_name = Path(root).name
    rel_root = str(Path(root).relative_to(SCAN_ROOT))

    print(f"\n  {'=' * 66}")
    print(f"  Pipeline: {rel_root}")
    print(f"  {'=' * 66}")

    # -- Step 1: Read --------------------------------------------------------
    print(f"  Step 1: Read source folder")
    pulumi_files, context_files = read_root_folder(root)
    if not pulumi_files:
        result.error = "No Pulumi.yaml found"
        print(f"    SKIP — no Pulumi.yaml found in {root}")
        return result

    print(f"    Pulumi: {list(pulumi_files.keys())}")
    if context_files:
        print(f"    Supporting: {list(context_files.keys())}")

    info = introspect(root, pulumi_files, context_files)
    description = describe_template(info)
    print(f"    Resources: {description}")

    stripped_files = {k: strip_config_section(v) for k, v in pulumi_files.items()}
    pulumi_context = format_files(stripped_files, lang="yaml")
    context_section = format_files(context_files) if context_files else ""

    root_template = pulumi_files.get("Pulumi.yaml", "")
    if not root_template:
        root_template = next(iter(pulumi_files.values()))

    # -- Step 2: RAG ---------------------------------------------------------
    print(f"  Step 2: RAG lookup")
    rag_results = rag_lookup(RAG_DB, root_template[:500])
    rag_section = ""
    if rag_results:
        print(f"    Found {len(rag_results)} similar (best: {rag_results[0].score:.3f})")
        rag_parts = [
            f"### RAG Example (similarity: {r.score:.2f})\n```yaml\n{r.content[:800]}\n```"
            for r in rag_results
        ]
        rag_section = "\nEXISTING SIMILAR TEMPLATES:\n" + "\n\n".join(rag_parts)
    else:
        print(f"    No similar templates")

    # -- Setup ---------------------------------------------------------------
    stack_name = make_stack_name(root)
    result.stack_name = stack_name
    work_dir = tempfile.mkdtemp(prefix=f"kkachi-tree-{folder_name}-")
    result.work_dir = work_dir

    # Copy supporting files so fn::readFile works
    for rel, content in context_files.items():
        dest = Path(work_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    # Bootstrap
    Path(work_dir, "Pulumi.yaml").write_text(
        f"name: {stack_name}\nruntime: yaml\ndescription: kkachi pipeline\n"
    )
    os.environ.setdefault("PULUMI_CONFIG_PASSPHRASE", "")
    subprocess.run(["pulumi", "stack", "init", stack_name, "--non-interactive"],
                   cwd=work_dir, capture_output=True)
    subprocess.run(["pulumi", "config", "set", "gcp:project", gcp_project, "-s", stack_name],
                   cwd=work_dir, capture_output=True)
    subprocess.run(["pulumi", "config", "set", "project", gcp_project, "-s", stack_name],
                   cwd=work_dir, capture_output=True)
    subprocess.run(["pulumi", "config", "set", "builder", gcp_service_account, "-s", stack_name],
                   cwd=work_dir, capture_output=True)
    stack_cfg = Path(work_dir) / f"Pulumi.{stack_name}.yaml"
    stack_cfg.write_text(
        f"config:\n  gcp:project: {gcp_project}\n"
        f"  project: {gcp_project}\n  builder: {gcp_service_account}\n"
    )

    # Config flags passed to pulumi preview/up CLI commands
    config_flags = (
        f'--config project={gcp_project} '
        f'--config builder={gcp_service_account}'
    )
    print(f"    Stack: {stack_name}  Work: {work_dir}")

    # Build dynamic components
    defaults = build_defaults(gcp_project, gcp_service_account, info)
    skill = build_skill(info)
    yaml_checks = build_yaml_checks(info)
    resource_structure = build_resource_structure(info)

    # Collect content of specific supporting files for the prompt
    supporting_files_prompt = ""
    for fname in ("schema.json", "query.sql", "script.js"):
        if fname in context_files:
            ext = Path(fname).suffix.lstrip(".")
            lang = {"json": "json", "sql": "sql", "js": "javascript"}.get(ext, "text")
            supporting_files_prompt += (
                f"\nSUPPORTING FILE ({fname} — already exists in work directory):\n"
                f"```{lang}\n{context_files[fname]}\n```\n"
            )
    # For Python files in subdirs
    for fpath, content in context_files.items():
        if fpath.endswith(".py"):
            supporting_files_prompt += (
                f"\nSUPPORTING FILE ({fpath} — already exists in work directory):\n"
                f"```python\n{content}\n```\n"
            )

    # -- Step 3: Preview -----------------------------------------------------
    print(f"  Step 3: pulumi preview")
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
    preview_combined = yaml_checks.and_(preview_validator)

    preview_prompt = f"""Output ONLY raw Pulumi YAML (no markdown fences, no commentary, no "Answer:" prefix).
Start with "name:" on line 1.

Create a Pulumi YAML program that provisions {description} in GCP project {gcp_project}.

{defaults.context()}

CRITICAL RULES:
- Do NOT define gcp:project in the config section — it is set externally via `pulumi config`
- You MUST define these config variables: project (type: string), builder (type: string), staging_expiry (type: string)
- The "project" config variable is REQUIRED — without it, ${{project}} references in resources will fail
- Use "value:" for config defaults, NOT "default:"
- staging_expiry must use YYYY-MM-DD date format. Example: staging_expiry: {{type: string, value: "2026-12-31"}}
- Use ${{project}} to reference the project config variable in resource properties
- All resource types must be valid Pulumi GCP types
- IAM members will be fixed automatically by runtime defaults — use any placeholder
- gcp:bigquery:Dataset does NOT support deletionProtection — never add it to Dataset resources
- For gcp:bigquery:Table (non-view, non-materialized), set deletionProtection: false
- Always set deleteContentsOnDestroy: true on all datasets
- If an IAM resource references an identity that may not exist, comment it out with a TODO note
- Custom IAM roles (biLayerBigQueryWriter.customRole) do NOT exist in this project — comment out any IAM bindings using custom roles
- IAM roles MUST be limited to: roles/bigquery.dataViewer (datasets & table read), roles/bigquery.dataEditor (table write), roles/storage.objectViewer or roles/storage.objectAdmin (buckets)
- If the template uses fn::readFile, the referenced file already exists in the work directory — preserve the fn::readFile reference
- This is a Pulumi YAML template — do NOT reference Terraform or HCL

RESOURCE STRUCTURE:
{resource_structure}
{supporting_files_prompt}

SOURCE TEMPLATES (for reference — adapt the structure, NOT the config):
{pulumi_context}

{context_section}
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
    result.preview_score = preview_result.score
    result.preview_iters = preview_result.iterations
    print(f"    score={preview_result.score:.2f}  iters={preview_result.iterations}")

    # -- Step 4: Up ----------------------------------------------------------
    print(f"  Step 4: pulumi up")
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
    up_combined = yaml_checks.and_(up_validator)

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
- For gcp:bigquery:Table (non-view, non-materialized), set deletionProtection: false
- Always set deleteContentsOnDestroy: true on all datasets
- If fn::readFile is used, the referenced files exist — do not remove fn::readFile references
- Custom IAM roles do NOT exist — comment out IAM bindings using custom roles
- IAM roles MUST be limited to: roles/bigquery.dataViewer, roles/bigquery.dataEditor, roles/storage.objectViewer, roles/storage.objectAdmin
- If an IAM resource fails because the identity does not exist, comment it out with a TODO note

CURRENT TEMPLATE:
{preview_result.output}"""

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
    result.up_score = up_result.score
    result.up_iters = up_result.iterations
    print(f"    score={up_result.score:.2f}  iters={up_result.iterations}")

    final_yaml = up_result.output if up_result.score >= 1.0 else preview_result.output
    Path(work_dir, "Pulumi.yaml").write_text(final_yaml)

    # -- Step 5: README ------------------------------------------------------
    print(f"  Step 5: README.md")
    preview_captures = preview_validator.get_captures()
    up_captures = up_validator.get_captures()

    errors_entries = []
    errors_entries = extract_errors_from_captures(preview_captures, "preview")
    errors_entries += extract_errors_from_captures(up_captures, "up")

    errors_table_rows = "\n".join(
        f"| {err} | Fixed during recursive optimization | {stage} |"
        for stage, err in errors_entries[:10]
    ) if errors_entries else "| No errors | Clean validation | preview/up |"

    defaults_annotations = []
    for ann in defaults.annotations():
        note = f" — {ann.note}" if ann.note else ""
        defaults_annotations.append(
            f"| {ann.key} | `{ann.original_pattern}` | `{ann.replacement}` | {ann.source}{note} |"
        )
    defaults_table = "\n".join(defaults_annotations) or "| — | — | — | No defaults |"

    pulumi_doc_links = build_pulumi_doc_links(info)
    resource_type_list = sorted(info.resource_types)

    # Rewrite config to canonical form before generating README
    final_yaml_canonical = rewrite_config_section(final_yaml)

    doc_prompt = f"""Output the complete markdown document directly — do not prefix with "Answer:" or similar labels.

Generate a GitHub Markdown README.md for a Pulumi YAML template that provisions {description}.

IMPORTANT INSTRUCTIONS:
- This is a Pulumi YAML template — always say "Pulumi", never "Terraform" or "HCL"
- Do NOT mention any specific GCP project name or project ID in the README
- The template is designed to be reusable across any GCP project
- Use ${{{{"project"}}}} as the placeholder when referring to the project variable

TEMPLATE CODE:
```yaml
{final_yaml_canonical}
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
Be specific to the resource types: {description}.
Do NOT mention any specific project name — the template is project-agnostic.

## B. Code
Provide the final working Pulumi YAML in a ```yaml code block.
Add brief inline YAML comments explaining key resource properties.
The config section should show the portable version (no hardcoded project values).

## C. Errors & Solutions
Include a markdown table with columns: | Error | Solution | Stage |
Use the errors from the validation above. If no errors, note that validation passed cleanly.

## D. Runtime Defaults
Include a markdown table with columns: | Key | Pattern | Replacement | Source |
Explain that these values are applied automatically via regex substitution before
Pulumi deployment validation, ensuring deployment-specific values are correct.

## E. Links
Two subsections:
**Google Cloud Documentation**
- Link to BigQuery documentation at cloud.google.com/bigquery/docs
- Link to IAM documentation at cloud.google.com/iam/docs

**Pulumi Documentation**
{pulumi_doc_links}

## F. Metadata
A ```yaml code block containing:
- category: data-analytics
- subcategory: bigquery
- template_name: {folder_name}
- keywords: [bigquery, {folder_name}, iam, gcp, pulumi]
- gcp_services: [BigQuery, IAM]
- pulumi_resources: [{", ".join(resource_type_list)}]"""

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
    result.doc_score = doc_result.score
    result.doc_iters = doc_result.iterations
    print(f"    score={doc_result.score:.2f}  iters={doc_result.iterations}")

    # -- Step 6: RAG writeback -----------------------------------------------
    print(f"  Step 6: RAG writeback")
    tag = f"pulumi-{rel_root.replace('/', '-')}"
    mem = rag_writeback(RAG_DB, rewrite_config_section(final_yaml), tag=tag)

    # -- Step 7: Write outputs -----------------------------------------------
    print(f"  Step 7: Write outputs")

    # Rewrite the config section to the canonical, project-agnostic form.
    # During preview/up the LLM-generated config had hardcoded project values
    # needed for validation; the final persisted template must be portable.
    final_yaml_portable = rewrite_config_section(final_yaml)

    output_dir = Path(root)
    written = []

    out_pulumi = output_dir / "Pulumi.yaml"
    out_pulumi.write_text(final_yaml_portable)
    written.append(str(out_pulumi))

    out_readme = output_dir / "README.md"
    out_readme.write_text(doc_result.output)
    written.append(str(out_readme))

    result.files_written = len(written)
    print(f"    Written {len(written)} files to {output_dir}")

    # Verify
    for f in written:
        p = Path(f)
        if not p.exists() or p.stat().st_size == 0:
            print(f"    WARNING: empty or missing — {f}")

    # -- Destroy after successful deploy to free GCP resources ---------------
    if up_result.score >= 1.0:
        print(f"  Cleanup: destroying deployed resources")
        subprocess.run(
            ["pulumi", "stack", "select", stack_name],
            cwd=work_dir, capture_output=True
        )
        subprocess.run(
            ["pulumi", "destroy", "--yes", "--non-interactive"],
            cwd=work_dir, capture_output=True,
            env={**os.environ, "GOOGLE_APPLICATION_CREDENTIALS": GCP_CREDS,
                 "PULUMI_CONFIG_PASSPHRASE": ""},
            timeout=120
        )
        subprocess.run(
            ["pulumi", "stack", "rm", stack_name, "--yes"],
            cwd=work_dir, capture_output=True
        )
        print(f"    Destroyed and removed stack {stack_name}")
    else:
        print(f"  NOTE: Stack {stack_name} left intact (up score < 1.0)")
        print(f"    cd {work_dir} && pulumi destroy --yes --non-interactive")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Kkachi Pulumi Tree Pipeline")
    print(f"Scanning: {SCAN_ROOT}")
    print("=" * 70)

    if not Path(SCAN_ROOT).exists():
        print(f"Scan root not found: {SCAN_ROOT}")
        sys.exit(1)

    # Discover all roots
    roots = discover_roots(SCAN_ROOT)
    print(f"\nDiscovered {len(roots)} Pulumi root(s):")
    for i, r in enumerate(roots, 1):
        rel = str(Path(r).relative_to(SCAN_ROOT))
        print(f"  {i:2d}. {rel}")
    print()

    # Setup LLM (shared across all pipeline runs)
    with open(LLM_API_KEY_PATH) as f:
        api_key = f.read().strip()
    llm = (
        ApiLlm.openai_with_url(api_key, "your-model-id", "https://your-api-endpoint.com")
        .with_timeout(120)
        .with_retry(3)
    )

    with open(GCP_CREDS) as f:
        creds = json.load(f)
    gcp_project = creds["project_id"]
    gcp_service_account = creds["client_email"]
    print(f"LLM: {llm}")
    print(f"GCP project: {gcp_project}")
    print(f"Service account: {gcp_service_account}")

    # Run pipeline on each root
    results: list[PipelineResult] = []
    for i, root in enumerate(roots, 1):
        rel = str(Path(root).relative_to(SCAN_ROOT))
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(roots)}] {rel}")
        print(f"{'#' * 70}")

        try:
            pr = run_pipeline(root, llm, gcp_project, gcp_service_account)
            results.append(pr)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append(PipelineResult(root=root, error=str(e)))

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n\n{'=' * 70}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 70}")
    print()
    print(f"{'Root':<45} {'Preview':>10} {'Up':>10} {'Doc':>10} {'Files':>6}")
    print(f"{'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for pr in results:
        rel = str(Path(pr.root).relative_to(SCAN_ROOT))
        if pr.error and not pr.preview_iters:
            status = "SKIP"
            total_skip += 1
            print(f"{rel:<45} {'SKIP':>10} {'SKIP':>10} {'SKIP':>10} {0:>6}  ({pr.error})")
        else:
            prev = f"{pr.preview_score:.2f}/{pr.preview_iters}"
            up = f"{pr.up_score:.2f}/{pr.up_iters}"
            doc = f"{pr.doc_score:.2f}/{pr.doc_iters}"
            print(f"{rel:<45} {prev:>10} {up:>10} {doc:>10} {pr.files_written:>6}")
            if pr.up_score >= 1.0:
                total_pass += 1
            else:
                total_fail += 1

    print()
    print(f"Total: {len(results)} roots — "
          f"{total_pass} passed, {total_fail} failed, {total_skip} skipped")
    print(f"RAG store: {RAG_DB}")
    print(f"Dist dir:  {DIST_DIR}")

    # Package at the end
    print(f"\nPackaging RAG knowledge base...")
    try:
        mem = Memory().persist(RAG_DB)
        pkg = mem.package(
            "kkachi_kb_big_query",
            version="0.1.0",
            output_dir=DIST_DIR,
            description="Kkachi knowledge base for BigQuery Pulumi templates",
            author="kkachi-pipeline",
        )
        print(f"  Wheel: {pkg.wheel_name}")
        print(f"  Path:  {pkg.wheel_path}")
        print(f"  Size:  {pkg.size_bytes} bytes")
    except RuntimeError as e:
        print(f"  Skipped: {e}")

    print(f"\n{'=' * 70}")
    print("Tree pipeline complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
