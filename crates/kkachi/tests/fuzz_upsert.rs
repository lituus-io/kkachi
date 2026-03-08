// Property-based tests for Memory::upsert and upsert_tagged.

use kkachi::recursive::memory;
use proptest::prelude::*;

proptest! {
    #[test]
    fn upsert_idempotent_for_any_content(content in "\\PC{1,200}") {
        let mut mem = memory();
        let id1 = mem.upsert(&content).unwrap();
        let id2 = mem.upsert(&content).unwrap();
        prop_assert_eq!(id1, id2, "same content must always produce the same ID");
    }

    #[test]
    fn upsert_tagged_preserves_tag(
        tag in "[a-z]{1,20}",
        content in "\\PC{1,200}",
    ) {
        let mut mem = memory();
        mem.upsert_tagged(&tag, &content).unwrap();
        let tags = mem.tags().unwrap();
        prop_assert!(
            tags.contains(&tag),
            "tag {:?} must appear in tags() after upsert_tagged, got {:?}",
            tag,
            tags,
        );
    }
}
