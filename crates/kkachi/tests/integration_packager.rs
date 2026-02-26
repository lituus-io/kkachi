// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Integration tests for the DuckDB RAG auto-packager.

#[cfg(feature = "storage")]
mod tests {
    use kkachi::recursive::{memory, PackagerBuilder};
    use std::io::Read;

    #[test]
    fn test_memory_persist_then_package() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_kb.db");
        let out_dir = dir.path().join("dist");

        let mut mem = memory().persist(db_path.to_str().unwrap()).unwrap();
        mem.add("Q: What is Rust? A: A systems programming language.");
        mem.add("Q: What is kkachi? A: An LLM optimization library.");

        let result = mem
            .package("test_kb")
            .unwrap()
            .version("0.2.0")
            .output_dir(out_dir.to_str().unwrap())
            .description("Test KB")
            .author("Test Author")
            .build()
            .unwrap();

        assert_eq!(result.wheel_name, "test_kb-0.2.0-py3-none-any.whl");
        assert!(result.wheel_path.exists());
        assert!(result.size_bytes > 0);
        assert!(result.db_size_bytes > 0);
        assert_eq!(result.file_count, 5);
    }

    #[test]
    fn test_package_empty_memory() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("empty.db");
        let out_dir = dir.path().join("dist");

        // Create an empty persistent memory
        let _mem = memory().persist(db_path.to_str().unwrap()).unwrap();

        let result = PackagerBuilder::new(&db_path)
            .name("empty_kb")
            .version("0.1.0")
            .output_dir(out_dir.to_str().unwrap())
            .build()
            .unwrap();

        assert!(result.wheel_path.exists());
        assert!(result.db_size_bytes > 0); // DuckDB file has header even when empty
    }

    #[test]
    fn test_package_in_memory_fails() {
        let mem = memory();
        let result = mem.package("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_package_large_db() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("large.db");
        let out_dir = dir.path().join("dist");

        let mut mem = memory().persist(db_path.to_str().unwrap()).unwrap();
        for i in 0..1000 {
            mem.add(&format!("Document number {} with some content to make it bigger", i));
        }

        let result = mem
            .package("large_kb")
            .unwrap()
            .output_dir(out_dir.to_str().unwrap())
            .build()
            .unwrap();

        assert!(result.wheel_path.exists());
        assert!(result.db_size_bytes > 0);

        // Verify the DB in the wheel has content
        let file = std::fs::File::open(&result.wheel_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut db_entry = archive.by_name("large_kb/data/knowledge.db").unwrap();
        let mut buf = Vec::new();
        db_entry.read_to_end(&mut buf).unwrap();
        assert_eq!(buf.len() as u64, result.db_size_bytes);
    }

    #[test]
    fn test_package_with_tagged_docs() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("tagged.db");
        let out_dir = dir.path().join("dist");

        let mut mem = memory().persist(db_path.to_str().unwrap()).unwrap();
        mem.add_tagged("rust", "Rust memory safety concepts");
        mem.add_tagged("python", "Python packaging with pip");
        mem.add("Untagged document");

        assert_eq!(mem.len(), 3);

        let result = mem
            .package("tagged_kb")
            .unwrap()
            .output_dir(out_dir.to_str().unwrap())
            .build()
            .unwrap();

        assert!(result.wheel_path.exists());
    }

    #[test]
    fn test_package_wheel_is_valid_zip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("valid.db");
        let out_dir = dir.path().join("dist");

        std::fs::write(&db_path, b"fake-db").unwrap();

        let result = PackagerBuilder::new(&db_path)
            .name("valid_pkg")
            .output_dir(out_dir.to_str().unwrap())
            .build()
            .unwrap();

        // Verify it's a valid ZIP by opening it
        let file = std::fs::File::open(&result.wheel_path).unwrap();
        let archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 5);
    }
}
