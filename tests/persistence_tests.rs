//! Integration tests for the persistence feature.
//!
//! These tests exercise the full lifecycle:
//! prover → operations → persist → reload → verify

#[cfg(feature = "persistence")]
mod persistence_tests {
    use bytes::Bytes;
    use ergo_avltree_rust::authenticated_tree_ops::*;
    use ergo_avltree_rust::batch_avl_prover::*;
    use ergo_avltree_rust::batch_avl_verifier::*;
    use ergo_avltree_rust::batch_node::*;
    use ergo_avltree_rust::operation::*;
    use ergo_avltree_rust::persistence::RedbAVLStorage;
    use ergo_avltree_rust::persistent_batch_avl_prover::PersistentBatchAVLProver;
    use ergo_avltree_rust::versioned_avl_storage::*;
    use std::cell::RefCell;
    use std::path::Path;
    use std::rc::Rc;
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
        let tree = AVLTree::new(dummy_resolver, KEY_LENGTH, None);
        BatchAVLProver::new(tree, true)
    }

    fn new_verifier(digest: &ADDigest, proof: &SerializedAdProof) -> BatchAVLVerifier {
        let tree = AVLTree::new(dummy_resolver, KEY_LENGTH, None);
        BatchAVLVerifier::new(digest, proof, tree, None, None).unwrap()
    }

    fn new_persistent_prover(path: &Path) -> PersistentBatchAVLProver {
        let storage = RedbAVLStorage::open(path, KEY_LENGTH, None).unwrap();
        let tree = AVLTree::with_resolver(storage.create_resolver(), KEY_LENGTH, None);
        let prover = BatchAVLProver::new(tree, true);
        PersistentBatchAVLProver::new(prover, Box::new(storage), vec![]).unwrap()
    }

    fn initialize_storage(storage: &mut RedbAVLStorage, prover: &mut BatchAVLProver) -> ADDigest {
        storage.update(prover, vec![]).unwrap();
        let baseline = storage.version().unwrap();
        let _proof = prover.generate_proof();
        baseline
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
            initialize_storage(&mut storage, &mut prover);

            for i in 0..10usize {
                let key = blake2b_key(i);
                let value = Bytes::from(format!("value_{}", i));
                prover
                    .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                    .unwrap();
            }
            storage.update(&mut prover, vec![]).unwrap();
            storage.flush().unwrap();
            let _proof = prover.generate_proof();
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
        initialize_storage(&mut storage, &mut prover);

        // Version 1: insert 5 elements
        for i in 0..5usize {
            let key = blake2b_key(i);
            let value = Bytes::from(format!("v1_{}", i));
            prover
                .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                .unwrap();
        }
        storage.update(&mut prover, vec![]).unwrap();
        let _proof = prover.generate_proof();
        let digest_v1 = storage.version().unwrap();

        // Version 2: insert 5 more elements
        for i in 5..10usize {
            let key = blake2b_key(i);
            let value = Bytes::from(format!("v2_{}", i));
            prover
                .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                .unwrap();
        }
        storage.update(&mut prover, vec![]).unwrap();
        let _proof = prover.generate_proof();
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
        initialize_storage(&mut storage, &mut prover);

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
        let initial_digest = prover.digest().unwrap();
        storage.update(&mut prover, vec![]).unwrap();
        let _initial_proof = prover.generate_proof();

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
        let baseline_digest = initialize_storage(&mut storage, &mut prover);

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
            storage.update(&mut prover, vec![]).unwrap();
            let _proof = prover.generate_proof();
            digests.push(storage.version().unwrap());
        }

        // rollback_versions includes the initialized empty baseline and all 3
        // mutation versions in reverse order.
        let versions = storage.rollback_versions();
        let versions: Vec<_> = versions.collect();
        assert_eq!(versions.len(), 4);
        assert_eq!(versions[0], digests[2]);
        assert_eq!(versions[1], digests[1]);
        assert_eq!(versions[2], digests[0]);
        assert_eq!(versions[3], baseline_digest);
    }

    #[test]
    fn test_large_tree_persistence() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("avl.redb");

        let mut storage = RedbAVLStorage::open(&db_path, KEY_LENGTH, None).unwrap();
        let mut prover = new_prover();
        initialize_storage(&mut storage, &mut prover);

        // Insert 500 elements (should stress test the undo-log and node store)
        for i in 0..500usize {
            let key = blake2b_key(i);
            let value = Bytes::from(i.to_string());
            prover
                .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
                .unwrap();
        }
        storage.update(&mut prover, vec![]).unwrap();
        let _proof = prover.generate_proof();

        let digest = storage.version().unwrap();
        assert!(digest.len() > 0);

        // Lookup a key to make sure tree is functional
        let key = blake2b_key(250);
        let result = prover.unauthenticated_lookup(&key);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), Bytes::from("250"));
    }

    #[test]
    fn test_restart_remove_rollback_and_next_proof() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("avl.redb");
        let keys: Vec<Bytes> = (0..32usize).map(blake2b_key).collect();
        let values: Vec<Bytes> = (0..32usize)
            .map(|i| Bytes::from(format!("value_{i}")))
            .collect();

        let mut persistent = new_persistent_prover(&db_path);
        let empty_digest = persistent.digest();
        for (key, value) in keys.iter().zip(&values) {
            persistent
                .perform_one_operation(&Operation::Insert(KeyValue {
                    key: key.clone(),
                    value: value.clone(),
                }))
                .unwrap();
        }
        let insert_proof = persistent
            .generate_proof_and_update_storage(vec![])
            .unwrap();
        let digest_v1 = persistent.digest();

        let mut insert_verifier = new_verifier(&empty_digest, &insert_proof);
        for (key, value) in keys.iter().zip(&values) {
            insert_verifier
                .perform_one_operation(&Operation::Insert(KeyValue {
                    key: key.clone(),
                    value: value.clone(),
                }))
                .unwrap();
        }
        assert_eq!(insert_verifier.digest().unwrap(), digest_v1);

        let removed_index = 7usize;
        assert_eq!(
            persistent
                .perform_one_operation(&Operation::Remove(keys[removed_index].clone()))
                .unwrap(),
            Some(values[removed_index].clone())
        );
        let remove_proof = persistent
            .generate_proof_and_update_storage(vec![])
            .unwrap();
        let digest_v2 = persistent.digest();

        let mut remove_verifier = new_verifier(&digest_v1, &remove_proof);
        assert_eq!(
            remove_verifier
                .perform_one_operation(&Operation::Remove(keys[removed_index].clone()))
                .unwrap(),
            Some(values[removed_index].clone())
        );
        assert_eq!(remove_verifier.digest().unwrap(), digest_v2);
        persistent.storage.flush().unwrap();
        drop(persistent);

        // A resolver must preserve the content-address key on the materialized
        // node. Copy-on-write mutation then leaves that old deletion identity
        // available to persistence instead of turning it into a new digest.
        let storage = RedbAVLStorage::open(&db_path, KEY_LENGTH, None).unwrap();
        let mut persisted_root_digest = [0u8; 32];
        persisted_root_digest.copy_from_slice(&digest_v2[..32]);
        let resolved_root = Rc::new(RefCell::new(storage.create_resolver()(
            &persisted_root_digest,
        )));
        assert_eq!(resolved_root.borrow().get_label(), persisted_root_digest);
        let resolved_snapshot = resolved_root.borrow().clone();
        let copied_root = match resolved_snapshot {
            Node::Internal(node) => {
                InternalNode::update(&resolved_root, &node.left, &node.right, node.balance)
            }
            Node::Leaf(node) => LeafNode::update(
                &resolved_root,
                node.hdr.key.as_ref().unwrap(),
                &node.value,
                &node.next_node_key,
            ),
            Node::LabelOnly(_) => panic!("resolver returned an unresolved root"),
        };
        assert!(!Rc::ptr_eq(&resolved_root, &copied_root));
        assert_eq!(resolved_root.borrow().get_label(), persisted_root_digest);
        drop(copied_root);
        drop(resolved_root);
        drop(storage);

        let mut reloaded = new_persistent_prover(&db_path);
        assert_eq!(reloaded.digest(), digest_v2);
        assert_eq!(reloaded.unauthenticated_lookup(&keys[removed_index]), None);
        assert_eq!(
            reloaded.unauthenticated_lookup(&keys[8]),
            Some(values[8].clone())
        );

        reloaded.rollback(&digest_v1).unwrap();
        assert_eq!(reloaded.digest(), digest_v1);
        assert_eq!(
            reloaded.unauthenticated_lookup(&keys[removed_index]),
            Some(values[removed_index].clone())
        );

        let branch_key = blake2b_key(1_000);
        let branch_value = Bytes::from("after_rollback");
        reloaded
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: branch_key.clone(),
                value: branch_value.clone(),
            }))
            .unwrap();
        let branch_proof = reloaded.generate_proof_and_update_storage(vec![]).unwrap();
        let branch_digest = reloaded.digest();

        let mut branch_verifier = new_verifier(&digest_v1, &branch_proof);
        branch_verifier
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: branch_key,
                value: branch_value,
            }))
            .unwrap();
        assert_eq!(branch_verifier.digest().unwrap(), branch_digest);
        reloaded.storage.flush().unwrap();
    }
}
