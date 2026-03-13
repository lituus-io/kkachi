#!/usr/bin/env python3
# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""
Shared utilities for Pulumi pipeline examples.

Contains common helper functions used across pulumi_table_pipeline.py,
pulumi_template_pipeline.py, and pulumi_tree_pipeline.py.
"""

import hashlib
import re
import sys
from pathlib import Path

from kkachi import Memory


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
    """Remove the config: section from Pulumi YAML to avoid confusing the LLM.

    The config section often contains gcp:project which the LLM then copies
    into generated output, causing pulumi validation errors (gcp:project must
    be set externally via `pulumi config set gcp:project`).
    """
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
# File reading
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
        ext = Path(path).suffix.lstrip(".")
        ext_lang = {"json": "json", "sql": "sql", "js": "javascript", "py": "python"}.get(ext, lang)
        parts.append(f"### {path}\n```{ext_lang}\n{content}\n```")
    return "\n\n".join(parts)


# =============================================================================
# RAG operations
# =============================================================================

def rag_lookup(rag_db: str, query: str, k: int = 2) -> list:
    """Search RAG for similar Pulumi templates. Returns top-k results."""
    mem = Memory().persist(rag_db)
    if mem.is_empty():
        return []
    results = mem.search(query, k=k)
    return [r for r in results if r.score > 0.3]


def rag_writeback(rag_db: str, content: str, tag: str) -> "Memory":
    """Write result back to RAG store. Update if >95% similar, else add new."""
    mem = Memory().persist(rag_db)
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
# Error extraction
# =============================================================================

def extract_errors_from_captures(captures, stage: str) -> list[tuple[str, str]]:
    """Extract error lines from CLI validation captures.

    Returns list of (stage, error_line) tuples.
    """
    entries = []
    for cap in captures:
        if not cap.success:
            output = cap.stderr if cap.stderr.strip() else cap.stdout
            for line in output.strip().splitlines():
                line = line.strip()
                if line and ("error" in line.lower() or "failed" in line.lower()
                             or "invalid" in line.lower()):
                    entries.append((stage, line[:200]))
    return entries


# =============================================================================
# Output verification
# =============================================================================

def verify_output_files(written_files: list[str]) -> None:
    """Verify no output file is empty. Exits with error if any are."""
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
