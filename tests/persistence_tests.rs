//! Integration tests for the persistence feature.
//!
//! These tests exercise the full lifecycle:
//! prover → operations → persist → reload → verify

#[cfg(feature = "persistence")]
mod persistence_tests {
    use ergo_avltree_rust::authenticated_tree_ops::*;
    use ergo_avltree_rust::batch_avl_prover::*;
    use ergo_avltree_rust::batch_avl_verifier::*;
    use ergo_avltree_rust::batch_node::*;
    use ergo_avltree_rust::operation::*;
    use ergo_avltree_rust::persistence::RedbAVLStorage;
    use ergo_avltree_rust::versioned_avl_storage::*;
    use bytes::Bytes;
    use std::sync::Arc;
    use tempfile::tempdir;

    const KEY_LENGTH: usize = 32;

    fn blake2b_key(i: usize) -> Bytes {
        use blake2::digest::Digest;
        let mut hasher = Blake2b256::new();
        hasher.update(&i.to_string());
        Bytes::copy_from_slice(&hasher.finalize())
    }

    fn dummy_resolver(digest: &Digest32) -> Node {
        Node::LabelOnly(NodeHeader::new(Some(*digest), None))
    }

    fn new_prover() -> BatchAVLProver {
        let tree = AVLTree::new(Arc::new(dummy_resolver), KEY_LENGTH, None);
        BatchAVLProver::new(tree, true)
    }

    fn new_verifier(
        digest: &ADDigest,
        proof: &SerializedAdProof,
    ) -> BatchAVLVerifier {
        let tree = AVLTree::new(Arc::new(dummy_resolver), KEY_LENGTH, None);
        BatchAVLVerifier::new(digest, proof, tree, None, None).unwrap()
    }

    #[test]
    fn test_persist_and_reload_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("avl.redb");

        // Phase 1: Build tree with 10 elements and persist
        let digest_v1;
        {
            let mut storage = RedbAVLStorage::open(&db_path, KEY_LENGTH, None).unwrap();
            let mut prover = new_prover();

            for i in 0..10usize {
                let key = blake2b_key(i);
                let value = Bytes::from(format!("value_{}", i));
                prover
                    .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                    .unwrap();
            }
            prover.generate_proof();
            storage.update(&mut prover, vec![]).unwrap();
            digest_v1 = storage.version().unwrap();
        }

        // Phase 2: Reopen storage and verify the version persisted
        {
            let storage = RedbAVLStorage::open(&db_path, KEY_LENGTH, None).unwrap();
            assert_eq!(storage.version().unwrap(), digest_v1);
        }
    }

    #[test]
    fn test_persist_multi_version_and_rollback() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("avl.redb");

        let mut storage = RedbAVLStorage::open(&db_path, KEY_LENGTH, None).unwrap();
        let mut prover = new_prover();

        // Version 1: insert 5 elements
        for i in 0..5usize {
            let key = blake2b_key(i);
            let value = Bytes::from(format!("v1_{}", i));
            prover
                .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                .unwrap();
        }
        prover.generate_proof();
        storage.update(&mut prover, vec![]).unwrap();
        let digest_v1 = storage.version().unwrap();

        // Version 2: insert 5 more elements
        for i in 5..10usize {
            let key = blake2b_key(i);
            let value = Bytes::from(format!("v2_{}", i));
            prover
                .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                .unwrap();
        }
        prover.generate_proof();
        storage.update(&mut prover, vec![]).unwrap();
        let _digest_v2 = storage.version().unwrap();

        // Rollback to v1
        let (root, height) = storage.rollback(&digest_v1).unwrap();
        assert_eq!(storage.version().unwrap(), digest_v1);

        // Verify the root and height were recovered
        assert!(height > 0);
        assert!(!root.borrow().is_leaf() || height == 0);
    }

    #[test]
    fn test_persist_proof_still_verifies() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("avl.redb");

        let mut storage = RedbAVLStorage::open(&db_path, KEY_LENGTH, None).unwrap();
        let mut prover = new_prover();

        // Insert initial data
        let keys: Vec<Bytes> = (0..20usize).map(|i| blake2b_key(i)).collect();
        for key in &keys {
            let value = Bytes::copy_from_slice(&key[..8]);
            prover
                .perform_one_operation(&Operation::Insert(KeyValue {
                    key: key.clone(),
                    value,
                }))
                .unwrap();
        }
        let _initial_proof = prover.generate_proof();
        let initial_digest = prover.digest().unwrap();
        storage.update(&mut prover, vec![]).unwrap();

        // Perform lookups and generate proof
        for key in &keys[..5] {
            prover
                .perform_one_operation(&Operation::Lookup(key.clone()))
                .unwrap();
        }
        let lookup_proof = prover.generate_proof();
        let post_lookup_digest = prover.digest().unwrap();

        // Verify the proof works
        let mut verifier = new_verifier(&initial_digest, &lookup_proof);
        for key in &keys[..5] {
            let result = verifier.perform_one_operation(&Operation::Lookup(key.clone()));
            assert!(result.is_ok());
            assert!(result.unwrap().is_some());
        }
        let verified_digest = verifier.digest();
        assert!(verified_digest.is_some());
        assert_eq!(verified_digest.unwrap(), post_lookup_digest);
    }

    #[test]
    fn test_rollback_versions_chain() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("avl.redb");

        let mut storage = RedbAVLStorage::open(&db_path, KEY_LENGTH, None).unwrap();
        let mut prover = new_prover();

        // Create 3 versions
        let mut digests = Vec::new();
        for batch in 0..3usize {
            for i in 0..3usize {
                let key = blake2b_key(batch * 3 + i);
                let value = Bytes::from(format!("batch_{}_item_{}", batch, i));
                prover
                    .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                    .unwrap();
            }
            prover.generate_proof();
            storage.update(&mut prover, vec![]).unwrap();
            digests.push(storage.version().unwrap());
        }

        // rollback_versions should return all 3 versions in reverse order
        let versions = storage.rollback_versions();
        let versions: Vec<_> = versions.collect();
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0], digests[2]);
        assert_eq!(versions[1], digests[1]);
        assert_eq!(versions[2], digests[0]);
    }

    #[test]
    fn test_large_tree_persistence() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("avl.redb");

        let mut storage = RedbAVLStorage::open(&db_path, KEY_LENGTH, None).unwrap();
        let mut prover = new_prover();

        // Insert 500 elements (should stress test the undo-log and node store)
        for i in 0..500usize {
            let key = blake2b_key(i);
            let value = Bytes::from(i.to_string());
            prover
                .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                .unwrap();
        }
        prover.generate_proof();
        storage.update(&mut prover, vec![]).unwrap();

        let digest = storage.version().unwrap();
        assert!(digest.len() > 0);

        // Lookup a key to make sure tree is functional
        let key = blake2b_key(250);
        let result = prover.unauthenticated_lookup(&key);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), Bytes::from("250"));
    }
}
