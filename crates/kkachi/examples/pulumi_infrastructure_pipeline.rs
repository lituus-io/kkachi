// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Pulumi Infrastructure Pipeline with Token-Based Chunking
//!
//! This example demonstrates:
//!
//! 1. **Token-based chunking** - Using tiktoken for precise token counting
//! 2. **Section-based output** - Structured output with Code, UseCase, KnownErrors, KnownIssues
//! 3. **CLI validation** - Using `pulumi preview` to validate generated infrastructure
//! 4. **Recursive refinement** - Iterate until infrastructure passes preview
//! 5. **Chained outputs** - Chain template execution into sectioned output
//!
//! # Running This Example
//!
//! This example simulates Pulumi validation without requiring actual Pulumi installation.
//! For real Pulumi integration, ensure:
//! - Pulumi CLI is installed: `curl -fsSL https://get.pulumi.com | sh`
//! - AWS credentials are configured
//! - A Pulumi project is initialized
//!
//! ```bash
//! cargo run --example pulumi_infrastructure_pipeline --features tiktoken
//! ```

#[cfg(feature = "tiktoken")]
use kkachi::diff::{DiffRenderer, DiffStyle, TextDiff};
#[cfg(feature = "tiktoken")]
use kkachi::recursive::cli::CommandResult;
#[cfg(feature = "tiktoken")]
use std::time::Duration;

#[cfg(feature = "tiktoken")]
use kkachi::recursive::{
    chain::{ChainedOutput, ChunkMetadata, OutputChunk},
    chunk::{ChunkConfig, ChunkStrategy, SectionType, TextChunker},
    tokenize::{TiktokenTokenizer, Tokenizer},
};

fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("   Pulumi Infrastructure Pipeline with Token-Based Chunking");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Run all examples
    #[cfg(feature = "tiktoken")]
    {
        example_1_token_counting()?;
        example_2_infrastructure_chunking()?;
        example_3_pulumi_validation_loop()?;
        example_4_chained_output_template()?;
    }

    #[cfg(not(feature = "tiktoken"))]
    {
        println!("  This example requires the 'tiktoken' feature.");
        println!(
            "  Run with: cargo run --example pulumi_infrastructure_pipeline --features tiktoken\n"
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("   All Examples Complete!");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}

/// Example 1: Token Counting with TiktokenTokenizer
///
/// Demonstrates precise token counting for LLM context management.
#[cfg(feature = "tiktoken")]
fn example_1_token_counting() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 1: Token Counting with Tiktoken                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let tokenizer = TiktokenTokenizer::claude();

    // Sample infrastructure code
    let code_samples = vec![
        (
            "Simple bucket",
            r#"const bucket = new aws.s3.Bucket("my-bucket");"#,
        ),
        (
            "VPC with subnets",
            r#"const vpc = new aws.ec2.Vpc("main-vpc", {
    cidrBlock: "10.0.0.0/16",
    enableDnsHostnames: true,
    tags: { Name: "main-vpc" },
});

const publicSubnet = new aws.ec2.Subnet("public", {
    vpcId: vpc.id,
    cidrBlock: "10.0.1.0/24",
    mapPublicIpOnLaunch: true,
});"#,
        ),
        (
            "RDS instance",
            r#"const db = new aws.rds.Instance("database", {
    allocatedStorage: 20,
    engine: "postgres",
    engineVersion: "15",
    instanceClass: "db.t3.micro",
    dbName: "appdb",
    username: "admin",
    password: dbPassword,
    storageEncrypted: true,
    skipFinalSnapshot: false,
});"#,
        ),
    ];

    println!("  Token counts for infrastructure code samples:\n");
    println!("  {:30} {:>10} {:>12}", "Sample", "Tokens", "Chars/Token");
    println!("  {}", "-".repeat(56));

    for (name, code) in &code_samples {
        let tokens = tokenizer.count_tokens(code);
        let chars_per_token = code.len() as f64 / tokens as f64;
        println!("  {:30} {:>10} {:>12.2}", name, tokens, chars_per_token);
    }

    // Demonstrate encode/decode roundtrip
    let sample = "const bucket = new aws.s3.Bucket(\"test\");";
    let encoded = tokenizer.encode(sample);
    let decoded = tokenizer.decode(&encoded);

    println!("\n  Encode/decode roundtrip:");
    println!("    Original:  \"{}\"", sample);
    println!("    Token IDs: {:?}", &encoded[..encoded.len().min(10)]);
    println!("    Decoded:   \"{}\"", decoded);
    println!("    Match: {}", if sample == decoded { "✅" } else { "❌" });

    println!();
    Ok(())
}

/// Example 2: Infrastructure Code Chunking
///
/// Demonstrates chunking large infrastructure templates into token-limited pieces.
#[cfg(feature = "tiktoken")]
fn example_2_infrastructure_chunking() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 2: Infrastructure Code Chunking                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let tokenizer = TiktokenTokenizer::claude();

    // Large infrastructure template
    let infrastructure = r#"## Imports

import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

## VPC Configuration

const config = new pulumi.Config();
const projectName = pulumi.getProject();

// Create VPC
const vpc = new aws.ec2.Vpc("main-vpc", {
    cidrBlock: "10.0.0.0/16",
    enableDnsHostnames: true,
    enableDnsSupport: true,
    tags: {
        Name: "main-vpc",
        Environment: "production",
        ManagedBy: "pulumi",
    },
});

## Internet Gateway

// Create Internet Gateway for public subnet access
const igw = new aws.ec2.InternetGateway("main-igw", {
    vpcId: vpc.id,
    tags: { Name: "main-igw" },
});

## Public Subnet

// Public subnet for load balancers and bastion hosts
const publicSubnet = new aws.ec2.Subnet("public-subnet", {
    vpcId: vpc.id,
    cidrBlock: "10.0.1.0/24",
    availabilityZone: "us-west-2a",
    mapPublicIpOnLaunch: true,
    tags: { Name: "public-subnet" },
});

## Private Subnet

// Private subnet for application servers and databases
const privateSubnet = new aws.ec2.Subnet("private-subnet", {
    vpcId: vpc.id,
    cidrBlock: "10.0.2.0/24",
    availabilityZone: "us-west-2a",
    tags: { Name: "private-subnet" },
});

## NAT Gateway

// NAT Gateway for private subnet internet access
const eip = new aws.ec2.Eip("nat-eip", { vpc: true });
const natGateway = new aws.ec2.NatGateway("nat-gateway", {
    allocationId: eip.id,
    subnetId: publicSubnet.id,
    tags: { Name: "nat-gateway" },
});

## Route Tables

// Public route table with internet gateway
const publicRt = new aws.ec2.RouteTable("public-rt", {
    vpcId: vpc.id,
    routes: [{ cidrBlock: "0.0.0.0/0", gatewayId: igw.id }],
});

// Private route table with NAT gateway
const privateRt = new aws.ec2.RouteTable("private-rt", {
    vpcId: vpc.id,
    routes: [{ cidrBlock: "0.0.0.0/0", natGatewayId: natGateway.id }],
});

## Route Table Associations

new aws.ec2.RouteTableAssociation("public-rta", {
    subnetId: publicSubnet.id,
    routeTableId: publicRt.id,
});

new aws.ec2.RouteTableAssociation("private-rta", {
    subnetId: privateSubnet.id,
    routeTableId: privateRt.id,
});

## Exports

export const vpcId = vpc.id;
export const publicSubnetId = publicSubnet.id;
export const privateSubnetId = privateSubnet.id;
"#;

    println!(
        "  Infrastructure template: {} chars, {} lines",
        infrastructure.len(),
        infrastructure.lines().count()
    );

    // Create chunker with small token limit for demonstration
    let config = ChunkConfig {
        max_tokens: 150,
        overlap_tokens: 20,
        strategy: ChunkStrategy::Section,
        semantic_split: true,
        min_tokens: 10,
    };

    let chunker = TextChunker::with_config(tokenizer.clone(), config);

    // Chunk with section detection
    let chunks = chunker.chunk_with_sections(infrastructure);

    println!("  Chunked into {} pieces:\n", chunks.len());
    println!(
        "  {:5} {:20} {:>8} {:>8}",
        "Index", "Section", "Tokens", "Bytes"
    );
    println!("  {}", "-".repeat(45));

    for chunk in &chunks {
        let section_name = chunk
            .section_type
            .as_ref()
            .map(|s| format!("{:?}", s))
            .unwrap_or_else(|| "None".to_string());

        println!(
            "  {:5} {:20} {:>8} {:>8}",
            chunk.index,
            section_name,
            chunk.token_count,
            chunk.content.len()
        );
    }

    // Show first chunk content
    if let Some(first) = chunks.first() {
        println!("\n  First chunk preview:");
        let preview: String = first.content.chars().take(200).collect();
        for line in preview.lines().take(5) {
            println!("    {}", line);
        }
        if first.content.len() > 200 {
            println!("    ...");
        }
    }

    println!();
    Ok(())
}

/// Example 3: Pulumi CLI Validation Loop
///
/// Simulates the recursive refinement loop with Pulumi preview validation.
/// In production, this would actually run `pulumi preview`.
#[cfg(feature = "tiktoken")]
fn example_3_pulumi_validation_loop() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 3: Pulumi CLI Validation Loop                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Simulated LLM responses that progressively improve
    let responses = vec![
        // Iteration 0: Missing imports, missing config
        r#"const bucket = new aws.s3.Bucket("my-bucket", {
    versioning: { enabled: true },
});"#,
        // Iteration 1: Added imports, missing tags
        r#"import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

const bucket = new aws.s3.Bucket("my-bucket", {
    versioning: { enabled: true },
});"#,
        // Iteration 2: Added tags, missing public access block
        r#"import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

const projectName = pulumi.getProject();

const bucket = new aws.s3.Bucket(`${projectName}-bucket`, {
    versioning: { enabled: true },
    tags: {
        Environment: "production",
        ManagedBy: "pulumi",
    },
});"#,
        // Iteration 3: Complete with public access block
        r#"import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

const config = new pulumi.Config();
const projectName = pulumi.getProject();
const environment = config.get("environment") || "dev";

const bucket = new aws.s3.Bucket(`${projectName}-bucket`, {
    versioning: { enabled: true },
    tags: {
        Environment: environment,
        ManagedBy: "pulumi",
        Project: projectName,
    },
});

const publicAccessBlock = new aws.s3.BucketPublicAccessBlock(`${projectName}-bucket-pab`, {
    bucket: bucket.id,
    blockPublicAcls: true,
    blockPublicPolicy: true,
    ignorePublicAcls: true,
    restrictPublicBuckets: true,
});

export const bucketName = bucket.id;
export const bucketArn = bucket.arn;"#,
    ];

    let tokenizer = TiktokenTokenizer::claude();
    let diff_renderer = DiffRenderer::new().with_style(DiffStyle::Unified);

    let mut iteration = 0;
    let mut current_code = responses[0].to_string();
    let mut prev_code = String::new();
    let mut score_history: Vec<f64> = Vec::new();
    let mut feedback_history: Vec<String> = Vec::new();

    println!("  Task: Generate Pulumi code for an S3 bucket with security best practices\n");

    loop {
        println!("─── Iteration {} ───", iteration);

        // Show diff from previous iteration
        if !prev_code.is_empty() {
            let diff = TextDiff::new(&prev_code, &current_code);
            if diff.has_changes() {
                println!("\n  Changes from previous iteration:");
                let rendered = diff_renderer.render_text(&diff);
                for line in rendered.lines().take(10) {
                    println!("    {}", line);
                }
                if rendered.lines().count() > 10 {
                    println!("    ...");
                }
            }
        }

        // Token count
        let token_count = tokenizer.count_tokens(&current_code);
        println!(
            "\n  Code: {} tokens, {} chars",
            token_count,
            current_code.len()
        );

        // Simulate Pulumi validation
        let validation = simulate_pulumi_preview(&current_code);

        if validation.success {
            println!("\n  ✅ Pulumi preview succeeded!");
            score_history.push(1.0);
            break;
        } else {
            let errors = parse_simulated_errors(&validation);
            println!(
                "\n  ❌ Pulumi preview failed with {} issue(s):",
                errors.len()
            );
            for (i, err) in errors.iter().enumerate() {
                println!("    {}. {}", i + 1, err);
            }

            // Calculate score based on remaining issues
            let max_issues = 4.0;
            let score = 1.0 - (errors.len() as f64 / max_issues).min(1.0);
            score_history.push(score);
            println!("\n  Score: {:.0}%", score * 100.0);

            feedback_history.push(errors.join("\n"));
        }

        // Move to next iteration
        iteration += 1;
        if iteration >= responses.len() {
            println!("\n  Max iterations reached");
            break;
        }

        prev_code = current_code.clone();
        current_code = responses[iteration].to_string();
        println!();
    }

    // Show score progression
    println!("\n  Score progression:");
    for (i, score) in score_history.iter().enumerate() {
        let bar_len = (score * 20.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
        println!("    Iteration {}: {} {:.0}%", i, bar, score * 100.0);
    }

    println!("\n═══ Final Code ═══\n{}", current_code);
    println!();

    Ok(())
}

/// Simulate Pulumi preview validation
#[cfg(feature = "tiktoken")]
fn simulate_pulumi_preview(code: &str) -> CommandResult {
    let mut issues = Vec::new();

    // Check for required imports
    if !code.contains("import * as pulumi") {
        issues.push("Missing Pulumi import");
    }
    if !code.contains("import * as aws") {
        issues.push("Missing AWS provider import");
    }

    // Check for best practices
    if !code.contains("tags:") {
        issues.push("Resources should have tags for cost tracking");
    }
    if !code.contains("pulumi.getProject()") && !code.contains("projectName") {
        issues.push("Use project name for resource naming convention");
    }
    if code.contains("s3.Bucket") && !code.contains("BucketPublicAccessBlock") {
        issues.push("S3 buckets should have public access blocked");
    }

    CommandResult {
        success: issues.is_empty(),
        exit_code: if issues.is_empty() { 0 } else { 1 },
        stdout: String::new(),
        stderr: issues.join("\n"),
        duration: Duration::from_millis(100),
    }
}

/// Parse simulated validation errors
#[cfg(feature = "tiktoken")]
fn parse_simulated_errors(result: &CommandResult) -> Vec<String> {
    result
        .stderr
        .lines()
        .filter(|l| !l.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Example 4: Chained Output Template
///
/// Demonstrates generating structured output with typed sections.
#[cfg(feature = "tiktoken")]
fn example_4_chained_output_template() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Example 4: Chained Output Template with Sections              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let tokenizer = TiktokenTokenizer::claude();

    // Final infrastructure code from previous example
    let code = r#"import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

const config = new pulumi.Config();
const projectName = pulumi.getProject();
const environment = config.get("environment") || "dev";

const bucket = new aws.s3.Bucket(`${projectName}-bucket`, {
    versioning: { enabled: true },
    tags: {
        Environment: environment,
        ManagedBy: "pulumi",
        Project: projectName,
    },
});

const publicAccessBlock = new aws.s3.BucketPublicAccessBlock(`${projectName}-bucket-pab`, {
    bucket: bucket.id,
    blockPublicAcls: true,
    blockPublicPolicy: true,
    ignorePublicAcls: true,
    restrictPublicBuckets: true,
});

export const bucketName = bucket.id;
export const bucketArn = bucket.arn;"#;

    // Known errors from refinement
    let known_errors = vec![
        ("Missing Pulumi import", "Added `import * as pulumi`"),
        ("Missing tags", "Added Environment, ManagedBy, Project tags"),
        (
            "No public access block",
            "Added BucketPublicAccessBlock resource",
        ),
    ];

    // Create sectioned output
    let sections = vec![
        (SectionType::CodeBlock, code.to_string()),
        (
            SectionType::custom("UseCase"),
            r#"## Use Case

This infrastructure template creates a production-ready S3 bucket with:

- **Versioning**: Enabled for data protection and recovery
- **Public Access Block**: All public access blocked by default
- **Proper Tagging**: Environment, ManagedBy, and Project tags for cost tracking

**When to use:**
- Storing application artifacts
- Log aggregation
- Static asset hosting (with CloudFront)
- Backup storage"#
                .to_string(),
        ),
        (
            SectionType::custom("KnownErrors"),
            format!(
                r#"## Known Errors

| Error | Resolution |
|-------|------------|
{}
"#,
                known_errors
                    .iter()
                    .map(|(err, res)| format!("| {} | {} |", err, res))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        ),
        (
            SectionType::custom("KnownIssues"),
            r#"## Known Issues

1. **Single Region**: This template deploys to a single region. For cross-region replication, add `aws.s3.BucketReplicationConfiguration`.

2. **No Lifecycle Rules**: Add lifecycle rules for cost optimization if storing large amounts of data.

3. **No Encryption**: Consider adding server-side encryption with KMS for sensitive data.

4. **No Access Logging**: Add access logging for compliance requirements."#
                .to_string(),
        ),
    ];

    // Build ChainedOutput
    let mut chunks = smallvec::SmallVec::new();
    let mut full_text = String::new();
    let mut total_tokens = 0;

    for (i, (section_type, content)) in sections.iter().enumerate() {
        let token_count = tokenizer.count_tokens(content);
        total_tokens += token_count;

        chunks.push(OutputChunk {
            content: std::borrow::Cow::Owned(content.clone()),
            section_type: section_type.clone(),
            token_count,
            index: i as u16,
            metadata: ChunkMetadata::default(),
        });

        full_text.push_str(content);
        full_text.push_str("\n\n");
    }

    let output = ChainedOutput {
        chunks,
        full_text,
        total_tokens,
        convergence_score: 1.0,
        iterations: 4,
    };

    // Display output structure
    println!("  ChainedOutput structure:\n");
    println!("    Total chunks:      {}", output.chunks.len());
    println!("    Total tokens:      {}", output.total_tokens);
    println!(
        "    Convergence score: {:.0}%",
        output.convergence_score * 100.0
    );
    println!("    Iterations:        {}", output.iterations);
    println!();

    println!("  Sections:");
    println!("  {:5} {:20} {:>8}", "Index", "Type", "Tokens");
    println!("  {}", "-".repeat(35));

    for chunk in &output.chunks {
        println!(
            "  {:5} {:20} {:>8}",
            chunk.index,
            format!("{:?}", chunk.section_type),
            chunk.token_count
        );
    }

    // Generate final template markdown
    let template = generate_output_template(&output, 4);

    println!("\n═══ Generated Output Template ═══\n");
    // Show first 50 lines
    for (i, line) in template.lines().enumerate() {
        if i >= 50 {
            println!("... ({} more lines)", template.lines().count() - 50);
            break;
        }
        println!("{}", line);
    }

    println!();
    Ok(())
}

/// Generate the final output template markdown
#[cfg(feature = "tiktoken")]
fn generate_output_template(output: &ChainedOutput, iterations: u32) -> String {
    let mut template = String::new();

    // YAML frontmatter
    template.push_str("---\n");
    template.push_str("name: vpc_infrastructure\n");
    template.push_str("version: \"1.0\"\n");
    template.push_str("generated: true\n");
    template.push_str(&format!("iterations: {}\n", iterations));
    template.push_str(&format!("total_tokens: {}\n", output.total_tokens));
    template.push_str(&format!(
        "convergence_score: {:.2}\n",
        output.convergence_score
    ));
    template.push_str("---\n\n");

    // Title
    template.push_str("# S3 Bucket with Security Best Practices\n\n");

    // Add each section
    for chunk in &output.chunks {
        match &chunk.section_type {
            SectionType::CodeBlock => {
                template.push_str("## Code\n\n");
                template.push_str("```typescript\n");
                template.push_str(&chunk.content);
                template.push_str("\n```\n\n");
            }
            _ => {
                template.push_str(&chunk.content);
                template.push_str("\n\n");
            }
        }
    }

    // Footer
    template.push_str("---\n");
    template.push_str("*Generated by Kkachi recursive optimization pipeline*\n");

    template
}
