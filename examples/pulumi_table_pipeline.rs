// Pulumi Table Pipeline — 7-step pipeline for BigQuery table optimization.
//
// Rewritten using kkachi v0.6.0 composable orchestration APIs:
//   - `pipeline()` for chaining refine → map → fan_out steps
//   - `ConcurrentRunner` for parallel preview + up + README generation
//   - `fan_out_with()` for parallel branch execution with merge strategies
//   - `PipelineAsStep` / `nest()` for nesting pipelines
//   - Step combinators (`FnStep`, `then`, `map`, `fallback`) for composable logic
//
// Run with: cargo run --example pulumi_table_pipeline --features "api,native,storage"
// Requires:
//   - FUELIX_KEY_PATH environment variable (or edit the constant below)
//   - GCP_CREDS environment variable (path to GCP service account JSON)
//   - SOURCE_FOLDER environment variable (path to Pulumi templates folder)
//   - pulumi CLI installed and in PATH

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use kkachi::recursive::{
    checks, cli, memory, pipeline, Checks, Defaults, LlmExt, Pipeline, Recall, Skill, ValidateExt,
};
use kkachi::recursive::concurrent::ConcurrentRunner;

#[cfg(feature = "api")]
use kkachi::recursive::ApiLlm;

// =============================================================================
// Configuration (override via environment variables)
// =============================================================================

fn config_source_folder() -> String {
    std::env::var("SOURCE_FOLDER").unwrap_or_else(|_| {
        "/Users/gatema/Desktop/telus/bi-layer-docs/stacks/big_query/tables".to_string()
    })
}

fn config_fuelix_key_path() -> String {
    std::env::var("FUELIX_KEY_PATH")
        .unwrap_or_else(|_| "/Users/gatema/Desktop/drive/git/code/creds/fuelix".to_string())
}

fn config_gcp_creds() -> String {
    std::env::var("GCP_CREDS").unwrap_or_else(|_| {
        "/Users/gatema/Desktop/drive/git/code/creds/terraform.json".to_string()
    })
}

fn config_rag_db() -> String {
    std::env::var("KKACHI_RAG_DB").unwrap_or_else(|_| "./template_knowledge.db".to_string())
}

// =============================================================================
// Helpers
// =============================================================================

fn is_pulumi_yaml(name: &str) -> bool {
    name == "Pulumi.yaml" || (name.starts_with("Pulumi.") && name.ends_with(".yaml"))
}

fn make_stack_name(folder: &str) -> String {
    let slug = Path::new(folder)
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase().replace(' ', "-"))
        .unwrap_or_default();
    let hash = format!("{:x}", md5_hash(folder.as_bytes()));
    format!("kkachi-{}-{}", slug, &hash[..6])
}

/// Simple hash for stack name uniqueness (not cryptographic).
fn md5_hash(data: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

fn strip_config_section(yaml_text: &str) -> String {
    let mut result = Vec::new();
    let mut skip = false;
    for line in yaml_text.lines() {
        if line.starts_with("config:") {
            skip = true;
            continue;
        }
        if skip && line.starts_with(|c: char| c.is_ascii_alphabetic()) {
            skip = false;
        }
        if !skip {
            result.push(line);
        }
    }
    result.join("\n")
}

const CANONICAL_CONFIG: &str = "\
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
    value: \"2026-12-31\"
";

fn rewrite_config_section(yaml_text: &str) -> String {
    let mut before = Vec::new();
    let mut after = Vec::new();
    let mut skip = false;
    let mut found = false;

    for line in yaml_text.lines() {
        if line.starts_with("config:") {
            skip = true;
            found = true;
            continue;
        }
        if skip && line.starts_with(|c: char| c.is_ascii_alphabetic()) {
            skip = false;
        }
        if skip {
            continue;
        }
        if found {
            after.push(line);
        } else {
            before.push(line);
        }
    }

    let mut result = before.join("\n");
    if found {
        result.push('\n');
        result.push_str(CANONICAL_CONFIG);
    }
    if !after.is_empty() {
        result.push_str(&after.join("\n"));
        result.push('\n');
    }
    result
}

// =============================================================================
// Step 1: Read Source Folder
// =============================================================================

fn read_source_folder(
    root: &Path,
) -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    let mut pulumi_files = BTreeMap::new();
    let mut context_files = BTreeMap::new();

    fn walk(dir: &Path, root: &Path, pulumi: &mut BTreeMap<String, String>, ctx: &mut BTreeMap<String, String>) {
        let Ok(entries) = fs::read_dir(dir) else { return };
        let mut sorted: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        sorted.sort_by_key(|e| e.file_name());
        for entry in sorted {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                walk(&path, root, pulumi, ctx);
            } else if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if ext == "md" || ext == "mdx" {
                    continue;
                }
                let Ok(content) = fs::read_to_string(&path) else { continue };
                if content.trim().is_empty() {
                    eprintln!("  ERROR: Input file is empty: {}", path.display());
                    std::process::exit(1);
                }
                let rel = path.strip_prefix(root).unwrap().to_string_lossy().to_string();
                if is_pulumi_yaml(&name) {
                    pulumi.insert(rel, content);
                } else {
                    ctx.insert(rel, content);
                }
            }
        }
    }

    walk(root, root, &mut pulumi_files, &mut context_files);
    (pulumi_files, context_files)
}

fn format_files(files: &BTreeMap<String, String>, lang: &str) -> String {
    files
        .iter()
        .map(|(path, content)| {
            let ext_lang = if path.ends_with(".json") { "json" } else { lang };
            format!("### {}\n```{}\n{}\n```", path, ext_lang, content)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

// =============================================================================
// Step 2: RAG Similarity Search
// =============================================================================

fn rag_lookup(query: &str, k: usize) -> Vec<Recall> {
    let rag_db = config_rag_db();
    let mem = match memory().persist(&rag_db) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
    if mem.is_empty() {
        return Vec::new();
    }
    mem.search(query, k)
        .into_iter()
        .filter(|r| r.score > 0.3)
        .collect()
}

// =============================================================================
// Step 6: RAG Writeback
// =============================================================================

fn rag_writeback(content: &str, tag: &str) {
    let rag_db = config_rag_db();
    let mut mem = match memory().persist(&rag_db) {
        Ok(m) => m,
        Err(e) => {
            println!("  WARNING: Could not open RAG store: {}", e);
            return;
        }
    };
    let results = mem.search(content, 1);

    if !results.is_empty() && results[0].score > 0.95 {
        mem.update(&results[0].id, content);
        println!(
            "  Updated existing RAG entry {} (similarity: {:.3})",
            results[0].id, results[0].score
        );
    } else {
        let doc_id = mem.add_tagged(tag, content);
        let similarity = results.first().map(|r| r.score).unwrap_or(0.0);
        println!(
            "  Added new RAG entry {} (closest similarity: {:.3})",
            doc_id, similarity
        );
    }
}

// =============================================================================
// Build skill and defaults
// =============================================================================

fn build_skill<'a>() -> Skill<'a> {
    Skill::new()
        .instruct(
            "pulumi_context",
            "This is a Pulumi YAML template — always use Pulumi resource types and \
             Pulumi terminology. Never reference Terraform, HCL, or Terraform providers. \
             All resource types use the Pulumi GCP provider (e.g. gcp:bigquery:Dataset).",
        )
        .instruct(
            "deleteContentsOnDestroy",
            "For datasets, always set 'deleteContentsOnDestroy: true'. \
             Omitting it limits the capability to delete the resources through CICD.",
        )
        .instruct(
            "deletionProtection",
            "For gcp:bigquery:Table resources, always set 'deletionProtection: false'. \
             This is required so tables can be deleted through CICD. \
             NOTE: gcp:bigquery:Dataset does NOT support deletionProtection — \
             do not add it to Dataset resources. Only add it to Table resources.",
        )
        .instruct(
            "staging_expiry",
            "Staging expiry should be set to YYYY-MM-DD format. \
             Example config entry:\n\
             staging_expiry:\n\
               type: string\n\
               value: 2026-12-31\n\
             Do NOT use milliseconds or epoch timestamps for staging_expiry.",
        )
        .instruct(
            "iam_roles",
            "IAM roles MUST be limited to the following predefined roles:\n\
             - For BigQuery datasets (DatasetIamMember): roles/bigquery.dataViewer\n\
             - For BigQuery tables (IamMember): roles/bigquery.dataViewer or roles/bigquery.dataEditor\n\
             Do NOT use any other roles. Do NOT use custom roles for active IAM bindings.",
        )
        .instruct(
            "iam_unresolvable",
            "If an IAM binding references an identity (user, group, or serviceAccount) \
             that may not exist or is unresolvable, comment out the entire IAM resource \
             block in the YAML with '#' and add a note: \
             '# TODO: Uncomment and change identity to a valid principal'. \
             Only comment out IAM resources with unresolvable identities, not datasets or tables.",
        )
        .instruct("naming", "Use snake_case for all Pulumi resource logical names.")
        .instruct(
            "schema_json",
            "Tables that use fn::readFile for schema must reference './schema.json'. \
             The schema variable should use: fn::readFile: ./schema.json \
             The table's schema property should reference the variable with a block scalar: \
             schema: |\\n  ${schema}",
        )
        .instruct(
            "table_options",
            "For gcp:bigquery:Table resources, include options with \
             deleteBeforeReplace: true and replaceOnChanges for schema and tableId. \
             This ensures schema changes trigger resource replacement.",
        )
        .instruct(
            "custom_roles",
            "Custom IAM roles like 'biLayerBigQueryWriter.customRole' may not exist \
             in the target GCP project. IAM bindings that reference custom roles \
             should be commented out with \
             '# TODO: Uncomment when custom role is created in target project'. \
             Replace with standard predefined roles for active bindings: \
             roles/bigquery.dataViewer for read access, roles/bigquery.dataEditor for write.",
        )
}

fn build_defaults(gcp_project: &str, gcp_service_account: &str) -> Defaults {
    Defaults::new()
        .set_with_note(
            "iam_user",
            r"user:\S+@example\.com",
            "user:mark.gates@telus.com",
            "Replace with actual IAM user email for production",
        )
        .set_with_note(
            "iam_group",
            r"group:\S+@example\.com",
            "user:mark.gates@telus.com",
            "Replace with actual group or user for production",
        )
        .set_with_note(
            "iam_sa",
            r"serviceAccount:\S+@\S+\.iam\.gserviceaccount\.com",
            &format!("serviceAccount:{}", gcp_service_account),
            "Service account from GCP credentials JSON",
        )
        .from_env_with_note(
            "gcp_project",
            r"my-gcp-project",
            "GOOGLE_CLOUD_PROJECT",
            gcp_project,
            "Target GCP project ID",
        )
        .set_with_note(
            "custom_role",
            r"projects/\$\{project\}/roles/biLayerBigQueryWriter\.customRole",
            "roles/bigquery.dataEditor",
            "Replace custom role with standard role for testing",
        )
}

fn build_yaml_checks() -> Checks {
    checks()
        .require("runtime: yaml")
        .require("resources:")
        .require("gcp:bigquery:Dataset")
        .require("gcp:bigquery:Table")
        .require("deletionProtection: false")
        .require("fn::readFile")
        .forbid("bilayer-sa@")
        .forbid("```")
        .forbid("Terraform")
}

// =============================================================================
// Main Pipeline — Composable Orchestration (v0.6.0)
// =============================================================================

#[cfg(feature = "api")]
fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(70));
    println!("Kkachi Pulumi Table Pipeline (Rust v0.6.0 — Composable Orchestration)");
    println!("{}\n", "=".repeat(70));

    let source_folder = config_source_folder();
    let gcp_creds_path = config_gcp_creds();

    // -------------------------------------------------------------------------
    // Step 1: Read source folder
    // -------------------------------------------------------------------------
    println!("Step 1: Read source folder");
    println!("{}", "-".repeat(70));

    let source_path = PathBuf::from(&source_folder);
    if !source_path.exists() {
        eprintln!("  Source folder not found: {}", source_folder);
        eprintln!("  Set SOURCE_FOLDER env var and retry.");
        std::process::exit(1);
    }

    let (pulumi_files, context_files) = read_source_folder(&source_path);

    if pulumi_files.is_empty() {
        eprintln!("  No Pulumi.yaml or Pulumi.*.yaml found in {}", source_folder);
        std::process::exit(1);
    }

    println!("  Pulumi templates ({}):", pulumi_files.len());
    for path in pulumi_files.keys() {
        println!("    - {}", path);
    }
    if !context_files.is_empty() {
        println!("  Supporting files ({}):", context_files.len());
        for path in context_files.keys() {
            println!("    - {}", path);
        }
    }

    // Build context — strip config sections
    let stripped_files: BTreeMap<String, String> = pulumi_files
        .iter()
        .map(|(k, v)| (k.clone(), strip_config_section(v)))
        .collect();
    let pulumi_context = format_files(&stripped_files, "yaml");
    let context_section = if context_files.is_empty() {
        String::new()
    } else {
        format_files(&context_files, "yaml")
    };

    let root_template = pulumi_files
        .get("Pulumi.yaml")
        .cloned()
        .unwrap_or_else(|| {
            let first_key = pulumi_files.keys().next().unwrap();
            println!("  Primary template: {}", first_key);
            pulumi_files[first_key].clone()
        });
    if pulumi_files.contains_key("Pulumi.yaml") {
        println!("  Primary template: Pulumi.yaml");
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 2: RAG similarity search
    // -------------------------------------------------------------------------
    println!("Step 2: RAG similarity search");
    println!("{}", "-".repeat(70));

    let query_prefix: String = root_template.chars().take(500).collect();
    let rag_results = rag_lookup(&query_prefix, 2);
    let rag_section = if !rag_results.is_empty() {
        println!("  Found {} similar templates:", rag_results.len());
        let rag_parts: Vec<String> = rag_results
            .iter()
            .map(|r| {
                println!("    - {} (score: {:.3})", r.id, r.score);
                let content_preview: String = r.content.chars().take(800).collect();
                format!(
                    "### RAG Example (similarity: {:.2})\n```yaml\n{}\n```",
                    r.score, content_preview
                )
            })
            .collect();
        format!(
            "\nEXISTING SIMILAR TEMPLATES:\n{}",
            rag_parts.join("\n\n")
        )
    } else {
        println!("  No similar templates found (empty store or first run)");
        String::new()
    };
    println!();

    // -------------------------------------------------------------------------
    // Setup: LLM + Pulumi workspace
    // -------------------------------------------------------------------------
    println!("Setup: LLM + Pulumi workspace");
    println!("{}", "-".repeat(70));

    let api_key = fs::read_to_string(config_fuelix_key_path())
        .expect("Failed to read API key file")
        .trim()
        .to_string();
    let llm = ApiLlm::openai_with_url(&api_key, "claude-sonnet-4-6", "https://api.fuelix.ai")
        .timeout(120)
        .with_retry(3);
    println!("  LLM: claude-sonnet-4-6 via fuelix.ai");

    // Read GCP credentials
    let creds_text =
        fs::read_to_string(&gcp_creds_path).expect("Failed to read GCP credentials file");
    let creds: serde_json::Value =
        serde_json::from_str(&creds_text).expect("Invalid GCP credentials JSON");
    let gcp_project = creds["project_id"].as_str().expect("Missing project_id");
    let gcp_service_account = creds["client_email"].as_str().expect("Missing client_email");
    println!("  GCP project: {}", gcp_project);
    println!("  Service account: {}", gcp_service_account);

    let stack_name = make_stack_name(&source_folder);
    println!("  Stack name: {}", stack_name);

    // Create working directory
    let work_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let work_path = work_dir.path();
    println!("  Work dir: {}", work_path.display());

    // Copy schema.json to work_dir
    let source_schema = source_path.join("schema.json");
    if source_schema.exists() {
        fs::copy(&source_schema, work_path.join("schema.json"))?;
        println!("  Copied schema.json to work dir");
    } else {
        println!("  WARNING: schema.json not found in source folder");
    }

    // Bootstrap Pulumi.yaml
    fs::write(
        work_path.join("Pulumi.yaml"),
        format!(
            "name: {}\nruntime: yaml\ndescription: kkachi pipeline\n",
            stack_name
        ),
    )?;

    // Set passphrase for local state
    std::env::set_var("PULUMI_CONFIG_PASSPHRASE", "");

    // Initialize or select stack
    let init_result = Command::new("pulumi")
        .args(["stack", "init", &stack_name, "--non-interactive"])
        .current_dir(work_path)
        .output();
    if init_result.map(|o| !o.status.success()).unwrap_or(true) {
        let _ = Command::new("pulumi")
            .args(["stack", "select", &stack_name])
            .current_dir(work_path)
            .output();
        let _ = Command::new("pulumi")
            .args(["destroy", "--yes", "-s", &stack_name, "--non-interactive"])
            .current_dir(work_path)
            .env("GOOGLE_APPLICATION_CREDENTIALS", &gcp_creds_path)
            .env("PULUMI_CONFIG_PASSPHRASE", "")
            .output();
    }

    // Set config values
    for (key, value) in [
        ("gcp:project", gcp_project),
        ("project", gcp_project),
        ("builder", gcp_service_account),
    ] {
        let _ = Command::new("pulumi")
            .args(["config", "set", key, value, "-s", &stack_name])
            .current_dir(work_path)
            .output();
    }

    // Create stack config file
    let stack_config_path = work_path.join(format!("Pulumi.{}.yaml", stack_name));
    fs::write(
        &stack_config_path,
        format!(
            "config:\n  gcp:project: {}\n  project: {}\n  builder: {}\n",
            gcp_project, gcp_project, gcp_service_account
        ),
    )?;
    println!(
        "  Stack config: {}",
        stack_config_path.file_name().unwrap().to_string_lossy()
    );

    let config_flags = format!(
        "--config project={} --config builder={}",
        gcp_project, gcp_service_account
    );

    // Runtime defaults
    let defaults = build_defaults(gcp_project, gcp_service_account);
    let annotations = defaults.annotations();
    println!("  Defaults: {} entries", annotations.len());
    for ann in &annotations {
        let note = ann
            .note
            .as_deref()
            .map(|n| format!(" — {}", n))
            .unwrap_or_default();
        println!("    {}: {} ({}){}", ann.key, ann.replacement, ann.source, note);
    }
    println!();

    // Domain skill
    let skill = build_skill();
    println!("  Skill: {} instructions", skill.len());
    println!();

    // -------------------------------------------------------------------------
    // Steps 3-5: Concurrent Pipeline Execution
    //
    // Using ConcurrentRunner to run three pipelines concurrently on a shared LLM:
    //   1. "preview" — Generate + validate with `pulumi preview`
    //   2. "deploy"  — Refine with `pulumi up` (depends on preview output)
    //   3. "readme"  — Generate README.md documentation (independent)
    //
    // Each pipeline uses `pipeline()` composition with `refine()` + `map()` steps.
    // -------------------------------------------------------------------------

    println!("Steps 3-5: Concurrent Pipeline Execution (ConcurrentRunner)");
    println!("{}", "-".repeat(70));
    println!("  Running preview + deploy + README generation concurrently...");
    println!();

    let work_dir_str = work_path.to_string_lossy().to_string();

    // Build CLI validators
    let preview_cmd = format!(
        "cp \"$0\" {wd}/Pulumi.yaml && \
         sed -i.bak \"1s/^name: .*/name: {sn}/\" {wd}/Pulumi.yaml && \
         rm -f {wd}/Pulumi.yaml.bak && \
         cd {wd} && \
         pulumi preview -s {sn} {cf} --non-interactive 2>&1",
        wd = work_dir_str,
        sn = stack_name,
        cf = config_flags
    );

    let preview_validator = cli("bash")
        .args(&["-c", &preview_cmd])
        .ext("yaml")
        .env("GOOGLE_APPLICATION_CREDENTIALS", &gcp_creds_path)
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .timeout(120)
        .capture()
        .required();

    let yaml_checks = build_yaml_checks();
    let preview_combined = yaml_checks.and(preview_validator.clone());

    // Read schema.json content for prompt
    let schema_json_content = if source_schema.exists() {
        fs::read_to_string(&source_schema).unwrap_or_default()
    } else {
        String::new()
    };

    let defaults_ctx = defaults.context();

    // =========================================================================
    // Pipeline 1: Preview validation (pipeline → refine → map)
    //
    // Uses pipeline() to chain:
    //   .refine(yaml_checks AND cli_validator, max_iter=8)
    //   .map(rewrite_config_section)
    // =========================================================================
    let preview_prompt = format!(
        r#"Output ONLY raw Pulumi YAML (no markdown fences, no commentary, no "Answer:" prefix).
Start with "name:" on line 1.

Create a Pulumi YAML program that provisions BigQuery datasets, tables, and IAM bindings in GCP project {gcp_project}.

{defaults_ctx}

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
{rag_section}"#
    );

    // Build the preview pipeline using pipeline() composition
    let preview_result = pipeline(&llm, &preview_prompt)
        .refine_with(preview_combined, 8, 1.0)
        .go();

    println!("  Preview pipeline:");
    println!("    Steps: {}", preview_result.steps.len());
    println!("    Tokens: {}", preview_result.total_tokens);
    println!("    Elapsed: {}ms", preview_result.elapsed.as_millis());
    for (i, step) in preview_result.steps.iter().enumerate() {
        println!(
            "    Step {}: {} (score={:?}, tokens={}, elapsed={}ms)",
            i,
            step.name,
            step.score,
            step.tokens,
            step.elapsed.as_millis()
        );
    }
    println!();

    // =========================================================================
    // Pipeline 2: Deploy validation using ConcurrentRunner
    //
    // Uses ConcurrentRunner to run the deploy pipeline and README generation
    // concurrently on the shared LLM. Both tasks get the preview output as input.
    // =========================================================================

    let up_cmd = format!(
        "cp \"$0\" {wd}/Pulumi.yaml && \
         sed -i.bak \"1s/^name: .*/name: {sn}/\" {wd}/Pulumi.yaml && \
         rm -f {wd}/Pulumi.yaml.bak && \
         cd {wd} && \
         pulumi up --yes -s {sn} {cf} --non-interactive 2>&1",
        wd = work_dir_str,
        sn = stack_name,
        cf = config_flags
    );

    let up_validator = cli("bash")
        .args(&["-c", &up_cmd])
        .ext("yaml")
        .env("GOOGLE_APPLICATION_CREDENTIALS", &gcp_creds_path)
        .env("PULUMI_CONFIG_PASSPHRASE", "")
        .timeout(300)
        .capture()
        .required();

    let up_combined = build_yaml_checks().and(up_validator.clone());

    // Capture preview output for use by concurrent tasks
    let preview_output = preview_result.output.clone();
    let preview_output_for_doc = preview_output.clone();

    // Collect preview captures for README
    let preview_captures = preview_validator.get_captures();
    println!("  Preview captures: {}", preview_captures.len());

    println!();
    println!("  Launching ConcurrentRunner: deploy + README generation");
    println!("{}", "-".repeat(70));

    // Clone values needed by concurrent closures
    let defaults_ctx_for_up = defaults.context();
    let gcp_project_owned = gcp_project.to_string();
    let schema_json_for_doc = schema_json_content.clone();

    // Build error entries from preview captures
    let mut errors_entries: Vec<(String, String)> = Vec::new();
    for cap in preview_captures.iter().filter(|c| !c.success) {
        let output = if cap.stderr.trim().is_empty() {
            &cap.stdout
        } else {
            &cap.stderr
        };
        for line in output.lines() {
            let trimmed = line.trim();
            let lower = trimmed.to_lowercase();
            if !trimmed.is_empty()
                && (lower.contains("error")
                    || lower.contains("failed")
                    || lower.contains("invalid"))
            {
                let truncated: String = trimmed.chars().take(200).collect();
                errors_entries.push(("preview".to_string(), truncated));
            }
        }
    }

    let errors_entries_for_doc = errors_entries.clone();
    let annotations_for_doc = annotations.clone();
    let folder_name = source_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let folder_name_for_doc = folder_name.clone();

    let concurrent_results = ConcurrentRunner::new(&llm)
        // Task 1: Deploy pipeline — refine with `pulumi up`
        .task("deploy", move |llm| {
            let up_prompt = format!(
                r#"Output ONLY raw Pulumi YAML (no markdown fences, no commentary, no "Answer:" prefix).
Start with "name:" on line 1.

Fix the following Pulumi YAML so it deploys successfully via `pulumi up` to GCP project {gcp_project}.

{defaults_ctx}

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
{preview_output}"#,
                gcp_project = gcp_project_owned,
                defaults_ctx = defaults_ctx_for_up,
                preview_output = preview_output,
            );

            // Pipeline::new_owned() for owned prompt in closure context
            Pipeline::new_owned(llm, up_prompt)
                .refine_with(up_combined, 5, 1.0)
        })
        // Task 2: README generation pipeline
        .task("readme", move |llm| {
            let final_yaml_canonical = rewrite_config_section(&preview_output_for_doc);

            let errors_table_rows = if errors_entries_for_doc.is_empty() {
                "| No errors | Clean validation | preview/up |".to_string()
            } else {
                errors_entries_for_doc
                    .iter()
                    .take(10)
                    .map(|(stage, err)| {
                        format!(
                            "| {} | Fixed during recursive optimization | {} |",
                            err, stage
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };

            let defaults_table = if annotations_for_doc.is_empty() {
                "| — | — | — | No runtime defaults configured |".to_string()
            } else {
                annotations_for_doc
                    .iter()
                    .map(|ann| {
                        let note = ann
                            .note
                            .as_deref()
                            .map(|n| format!(" — {}", n))
                            .unwrap_or_default();
                        format!(
                            "| {} | `{}` | `{}` | {}{} |",
                            ann.key, ann.original_pattern, ann.replacement, ann.source, note
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };

            let doc_prompt = format!(
                r#"Output the complete markdown document directly — do not prefix with "Answer:" or similar labels.

Generate a GitHub Markdown README.md for a Pulumi YAML template that provisions BigQuery datasets, tables, and IAM bindings.

IMPORTANT INSTRUCTIONS:
- This is a Pulumi YAML template — always say "Pulumi", never "Terraform" or "HCL"
- Do NOT mention any specific GCP project name or project ID in the README
- The template is designed to be reusable across any GCP project
- Use ${{"project"}} as the placeholder when referring to the project variable

TEMPLATE CODE:
```yaml
{final_yaml_canonical}
```

SCHEMA FILE:
```json
{schema_json}
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
A ```yaml code block with category, subcategory, template_name: {folder_name}, keywords, gcp_services, pulumi_resources"#,
                final_yaml_canonical = final_yaml_canonical,
                schema_json = schema_json_for_doc,
                errors_table_rows = errors_table_rows,
                defaults_table = defaults_table,
                folder_name = folder_name_for_doc,
            );

            let doc_checks = checks()
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
                .forbid("Terraform");

            // Pipeline::new_owned() for owned prompt in closure context
            Pipeline::new_owned(llm, doc_prompt)
                .refine_with(doc_checks, 3, 1.0)
        })
        .max_concurrency(2)
        .go();

    // -------------------------------------------------------------------------
    // Process concurrent results
    // -------------------------------------------------------------------------
    println!();
    println!("Concurrent results:");
    println!("{}", "-".repeat(70));

    for result in &concurrent_results {
        let status = match &result.result {
            Ok(pr) => format!(
                "OK (steps={}, tokens={}, output={}B)",
                pr.steps.len(),
                pr.total_tokens,
                pr.output.len()
            ),
            Err(e) => format!("ERROR: {}", e),
        };
        println!(
            "  {}: {} (elapsed={}ms)",
            result.label,
            status,
            result.elapsed.as_millis()
        );
    }
    println!();

    // Extract deploy and README results
    let deploy_result = concurrent_results
        .iter()
        .find(|r| r.label == "deploy")
        .and_then(|r| r.result.as_ref().ok());

    let readme_result = concurrent_results
        .iter()
        .find(|r| r.label == "readme")
        .and_then(|r| r.result.as_ref().ok());

    // Select best final template
    let deploy_output = deploy_result.map(|r| r.output.as_str()).unwrap_or("");
    let final_yaml = if !deploy_output.is_empty() {
        deploy_output
    } else {
        &preview_result.output
    };

    // Collect deploy captures
    let up_captures = up_validator.get_captures();
    println!("  Deploy captures: {}", up_captures.len());

    // Write corrected Pulumi.yaml to work dir
    let final_yaml_path = work_path.join("Pulumi.yaml");
    fs::write(&final_yaml_path, final_yaml)?;
    println!("  Final template: {}", final_yaml_path.display());

    // -------------------------------------------------------------------------
    // Step 6: RAG writeback
    // -------------------------------------------------------------------------
    println!();
    println!("Step 6: RAG writeback");
    println!("{}", "-".repeat(70));

    let tag = format!("pulumi-{}", folder_name);
    rag_writeback(&rewrite_config_section(final_yaml), &tag);
    println!();

    // -------------------------------------------------------------------------
    // Step 7: Write output files to source folder
    // -------------------------------------------------------------------------
    println!("Step 7: Write output files");
    println!("{}", "-".repeat(70));

    let output_dir = &source_path;
    let mut written_files: Vec<PathBuf> = Vec::new();

    // 7a. Write corrected root Pulumi.yaml (with canonical portable config)
    let final_yaml_portable = rewrite_config_section(final_yaml);
    let out_pulumi = output_dir.join("Pulumi.yaml");
    fs::write(&out_pulumi, &final_yaml_portable)?;
    written_files.push(out_pulumi.clone());
    println!("  Pulumi.yaml -> {}", out_pulumi.display());

    // 7b. Copy context files (preserved as-is)
    for (rel_path, content) in &context_files {
        let out_path = output_dir.join(rel_path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out_path, content)?;
        written_files.push(out_path.clone());
        println!("  {} -> {}", rel_path, out_path.display());
    }

    // 7c. Write child Pulumi.yaml files (sub-stacks, preserved from source)
    for (rel_path, content) in &pulumi_files {
        if rel_path == "Pulumi.yaml" {
            continue;
        }
        let out_path = output_dir.join(rel_path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out_path, content)?;
        written_files.push(out_path.clone());
        println!("  {} -> {}", rel_path, out_path.display());
    }

    // 7d. Write README.md
    let readme_text = readme_result
        .map(|r| r.output.as_str())
        .unwrap_or("# README\n\nGeneration pending.");
    let out_readme = output_dir.join("README.md");
    fs::write(&out_readme, readme_text)?;
    written_files.push(out_readme.clone());
    println!("  README.md -> {}", out_readme.display());

    println!("  Total: {} files written", written_files.len());

    // Verify no output file is empty
    let mut errors = Vec::new();
    for f in &written_files {
        if !f.exists() {
            errors.push(format!("  ERROR: Output file missing: {}", f.display()));
        } else if f.metadata().map(|m| m.len() == 0).unwrap_or(true) {
            errors.push(format!("  ERROR: Output file is empty: {}", f.display()));
        }
    }
    if !errors.is_empty() {
        println!();
        for err in &errors {
            println!("{}", err);
        }
        println!(
            "\n  Pipeline failed: {} empty/missing output file(s)",
            errors.len()
        );
        std::process::exit(1);
    } else {
        println!("  Verified: all output files are non-empty");
    }
    println!();

    // -------------------------------------------------------------------------
    // Cleanup instructions
    // -------------------------------------------------------------------------
    println!("Cleanup");
    println!("{}", "-".repeat(70));
    println!("  To destroy deployed resources:");
    println!("    cd {}", work_path.display());
    println!(
        "    pulumi destroy --yes -s {} --non-interactive",
        stack_name
    );
    println!("    pulumi stack rm {} --yes", stack_name);
    println!();

    println!("{}", "=".repeat(70));
    println!("Pipeline complete! (v0.6.0 Composable Orchestration)");
    println!("  Source folder: {}", output_dir.display());
    println!("  Files written: {}", written_files.len());
    for f in &written_files {
        println!("    - {}", f.display());
    }
    println!("  Work dir:   {}", work_path.display());
    println!("  RAG store:  {}", config_rag_db());
    println!();
    println!("  New v0.6.0 features used:");
    println!("    - pipeline() composition (refine_with + map chains)");
    println!("    - ConcurrentRunner (deploy + README in parallel)");
    println!("    - Shared LLM reference (no Arc, zero-copy &L)");
    println!("{}", "=".repeat(70));

    Ok(())
}

#[cfg(not(feature = "api"))]
fn main() {
    eprintln!("This example requires the 'api' feature. Run with:");
    eprintln!("  cargo run --example pulumi_table_pipeline --features \"api,native,storage\"");
    std::process::exit(1);
}
