// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Code generation for optimized prompts

/// Generate Rust code for optimized prompts
pub fn generate_optimized_code(signature: &str, demos: &[&str]) -> String {
    let demos_code = demos
        .iter()
        .map(|d| format!("    \"{}\"", d.replace('"', "\\\"")))
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        "// Auto-generated optimized prompts\npub const OPTIMIZED_SIGNATURE: &str = \"{}\";\n\npub const OPTIMIZED_DEMOS: &[&str] = &[\n{}\n];\n",
        signature.replace('"', "\\\""),
        demos_code
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen() {
        let code = generate_optimized_code("question -> answer", &["demo1", "demo2"]);
        assert!(code.contains("OPTIMIZED_SIGNATURE"));
        assert!(code.contains("OPTIMIZED_DEMOS"));
    }
}
