// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Security tests for the DuckDB RAG auto-packager.

#![cfg(feature = "storage")]

use kkachi::recursive::PackagerBuilder;

/// Names containing `../` must not cause files to be written outside the output directory.
/// The packager should either sanitize the name or produce output strictly within output_dir.
#[test]
fn test_no_path_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    std::fs::write(&db_path, b"fake-db-content").unwrap();

    let out_dir = dir.path().join("dist");

    let result = PackagerBuilder::new(&db_path)
        .name("../evil")
        .version("0.1.0")
        .output_dir(out_dir.to_str().unwrap())
        .compress(false)
        .build();

    match result {
        Ok(pkg) => {
            // If the build succeeds, the wheel must reside inside output_dir.
            let canonical_out = out_dir.canonicalize().unwrap();
            let canonical_whl = pkg.wheel_path.canonicalize().unwrap();
            assert!(
                canonical_whl.starts_with(&canonical_out),
                "Wheel escaped output_dir! wheel_path={}, output_dir={}",
                canonical_whl.display(),
                canonical_out.display(),
            );

            // Also verify that no file was created in the parent of output_dir
            // that looks like it came from path traversal.
            let parent = dir.path();
            let stray_files: Vec<_> = std::fs::read_dir(parent)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name();
                    let s = name.to_string_lossy();
                    s.contains("evil") || s.ends_with(".whl")
                })
                .collect();
            assert!(
                stray_files.is_empty(),
                "Path traversal created stray files in parent dir: {:?}",
                stray_files.iter().map(|e| e.path()).collect::<Vec<_>>(),
            );

            // Inspect the zip entries — none should contain `..`
            let file = std::fs::File::open(&pkg.wheel_path).unwrap();
            let mut archive = zip::ZipArchive::new(file).unwrap();
            for i in 0..archive.len() {
                let entry = archive.by_index(i).unwrap();
                let entry_name = entry.name().to_string();
                assert!(
                    !entry_name.contains("../"),
                    "ZIP entry contains path traversal: {}",
                    entry_name,
                );
            }
        }
        Err(_) => {
            // Rejecting the name outright is also acceptable behaviour.
        }
    }
}

/// Special characters in the package name should be sanitized in the wheel filename
/// so the resulting file is safe for filesystem use and PEP 427 compliant.
#[test]
fn test_wheel_name_sanitization() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    std::fs::write(&db_path, b"fake-db-content").unwrap();

    let out_dir = dir.path().join("dist");

    let tricky_names = [
        "my-package", // hyphens → underscores per PEP 503
        "name with space",
        "name/slash",
        "name\\backslash",
        "name@version",
        "name!bang",
    ];

    for name in &tricky_names {
        let result = PackagerBuilder::new(&db_path)
            .name(*name)
            .version("0.1.0")
            .output_dir(out_dir.to_str().unwrap())
            .compress(false)
            .build();

        match result {
            Ok(pkg) => {
                // The wheel filename must not contain problematic characters.
                assert!(
                    !pkg.wheel_name.contains('/'),
                    "Wheel name for {:?} contains '/': {}",
                    name,
                    pkg.wheel_name,
                );
                assert!(
                    !pkg.wheel_name.contains('\\'),
                    "Wheel name for {:?} contains '\\': {}",
                    name,
                    pkg.wheel_name,
                );
                assert!(
                    !pkg.wheel_name.contains(' '),
                    "Wheel name for {:?} contains space: {}",
                    name,
                    pkg.wheel_name,
                );

                // The wheel file must actually exist at the reported path.
                assert!(
                    pkg.wheel_path.exists(),
                    "Wheel file does not exist for name {:?}: {}",
                    name,
                    pkg.wheel_path.display(),
                );

                // PEP 427: filename is {name}-{version}-{tags}.whl
                assert!(
                    pkg.wheel_name.ends_with("-py3-none-any.whl"),
                    "Wheel name for {:?} is not PEP 427 compliant: {}",
                    name,
                    pkg.wheel_name,
                );
            }
            Err(_) => {
                // Rejecting names with unsupported characters is also acceptable.
            }
        }
    }
}
