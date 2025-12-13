#[cfg(all(test, feature = "disk-persistence"))]
mod disk_persistence_tests {
    use bytes::Bytes;
    use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
    use ergo_avltree_rust::batch_node::*;
    use ergo_avltree_rust::operation::*;
    use ergo_avltree_rust::persistent_batch_avl_prover::*;
    use ergo_avltree_rust::storage::DiskBackedAVLStorage;
    use ergo_avltree_rust::versioned_avl_storage::VersionedAVLStorage;
    use std::boxed::Box;
    use tempfile::TempDir;

    fn create_test_storage(
        key_length: usize,
        value_length: Option<usize>,
    ) -> (TempDir, Box<DiskBackedAVLStorage>) {
        let temp_dir = TempDir::new().unwrap();
        let storage = Box::new(
            DiskBackedAVLStorage::new(temp_dir.path(), key_length, value_length, 10).unwrap(),
        );
        (temp_dir, storage)
    }

    #[test]
    fn test_basic_disk_persistence() {
        let (_temp_dir, storage) = create_test_storage(32, Some(8));
        let resolver = DiskBackedAVLStorage::get_resolver();
        let tree = AVLTree::new(resolver, 32, Some(8));
        let prover = BatchAVLProver::new(tree, true);

        let mut persistent_prover =
            PersistentBatchAVLProver::new(prover, storage, vec![]).unwrap();

        // Insert some data
        let key1 = Bytes::from(vec![1u8; 32]);
        let value1 = Bytes::from(10u64.to_be_bytes().to_vec());

        let op1 = Operation::Insert(KeyValue {
            key: key1.clone(),
            value: value1.clone(),
        });

        persistent_prover.perform_one_operation(&op1).unwrap();
        let digest1 = persistent_prover.digest();

        // Generate proof and update storage
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();

        // Verify storage has the version
        assert_eq!(persistent_prover.storage.version(), Some(digest1.clone()));

        // Verify we can lookup the value
        let result = persistent_prover.unauthenticated_lookup(&key1);
        assert_eq!(result, Some(value1));
    }

    #[test]
    fn test_multiple_operations_and_versions() {
        let (_temp_dir, storage) = create_test_storage(32, Some(8));
        let resolver = DiskBackedAVLStorage::get_resolver();
        let tree = AVLTree::new(resolver, 32, Some(8));
        let prover = BatchAVLProver::new(tree, true);

        let mut persistent_prover =
            PersistentBatchAVLProver::new(prover, storage, vec![]).unwrap();

        // Version 1: Insert key1
        let key1 = Bytes::from(vec![1u8; 32]);
        let value1 = Bytes::from(10u64.to_be_bytes().to_vec());
        let op1 = Operation::Insert(KeyValue {
            key: key1.clone(),
            value: value1.clone(),
        });
        persistent_prover.perform_one_operation(&op1).unwrap();
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();
        let digest1 = persistent_prover.digest();

        // Version 2: Insert key2
        let key2 = Bytes::from(vec![2u8; 32]);
        let value2 = Bytes::from(20u64.to_be_bytes().to_vec());
        let op2 = Operation::Insert(KeyValue {
            key: key2.clone(),
            value: value2.clone(),
        });
        persistent_prover.perform_one_operation(&op2).unwrap();
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();
        let digest2 = persistent_prover.digest();

        // Version 3: Update key1
        let value1_updated = Bytes::from(15u64.to_be_bytes().to_vec());
        let op3 = Operation::Update(KeyValue {
            key: key1.clone(),
            value: value1_updated.clone(),
        });
        persistent_prover.perform_one_operation(&op3).unwrap();
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();
        let digest3 = persistent_prover.digest();

        // Verify current state
        assert_eq!(persistent_prover.storage.version(), Some(digest3.clone()));
        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key1),
            Some(value1_updated.clone())
        );
        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key2),
            Some(value2.clone())
        );

        // Rollback to version 2
        persistent_prover.rollback(&digest2).unwrap();
        assert_eq!(persistent_prover.storage.version(), Some(digest2.clone()));
        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key1),
            Some(value1.clone())
        );
        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key2),
            Some(value2.clone())
        );

        // Rollback to version 1
        persistent_prover.rollback(&digest1).unwrap();
        assert_eq!(persistent_prover.storage.version(), Some(digest1));
        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key1),
            Some(value1)
        );
        assert_eq!(persistent_prover.unauthenticated_lookup(&key2), None);
    }

    #[test]
    fn test_persistence_across_reopens() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        let key1 = Bytes::from(vec![1u8; 32]);
        let value1 = Bytes::from(10u64.to_be_bytes().to_vec());
        let digest1;

        // First session: create and populate tree
        {
            let storage = Box::new(DiskBackedAVLStorage::new(&path, 32, Some(8), 10).unwrap());
            let resolver = DiskBackedAVLStorage::get_resolver();
            let tree = AVLTree::new(resolver, 32, Some(8));
            let prover = BatchAVLProver::new(tree, true);
            let mut persistent_prover =
                PersistentBatchAVLProver::new(prover, storage, vec![]).unwrap();

            let op1 = Operation::Insert(KeyValue {
                key: key1.clone(),
                value: value1.clone(),
            });
            persistent_prover.perform_one_operation(&op1).unwrap();
            persistent_prover
                .generate_proof_and_update_storage(vec![])
                .unwrap();
            digest1 = persistent_prover.digest();
        }

        // Second session: reopen and verify
        {
            let storage = Box::new(DiskBackedAVLStorage::new(&path, 32, Some(8), 10).unwrap());
            
            // Verify storage has the version
            assert_eq!(storage.version(), Some(digest1.clone()));

            // Create new prover with existing storage
            let resolver = DiskBackedAVLStorage::get_resolver();
            let tree = AVLTree::new(resolver, 32, Some(8));
            let prover = BatchAVLProver::new(tree, true);
            let mut persistent_prover =
                PersistentBatchAVLProver::new(prover, storage, vec![]).unwrap();

            // Verify data is still there
            assert_eq!(persistent_prover.digest(), digest1);
            assert_eq!(
                persistent_prover.unauthenticated_lookup(&key1),
                Some(value1)
            );
        }
    }

    #[test]
    fn test_remove_operations() {
        let (_temp_dir, storage) = create_test_storage(32, Some(8));
        let resolver = DiskBackedAVLStorage::get_resolver();
        let tree = AVLTree::new(resolver, 32, Some(8));
        let prover = BatchAVLProver::new(tree, true);

        let mut persistent_prover =
            PersistentBatchAVLProver::new(prover, storage, vec![]).unwrap();

        // Insert three keys
        let key1 = Bytes::from(vec![1u8; 32]);
        let key2 = Bytes::from(vec![2u8; 32]);
        let key3 = Bytes::from(vec![3u8; 32]);
        let value = Bytes::from(10u64.to_be_bytes().to_vec());

        for key in &[key1.clone(), key2.clone(), key3.clone()] {
            let op = Operation::Insert(KeyValue {
                key: key.clone(),
                value: value.clone(),
            });
            persistent_prover.perform_one_operation(&op).unwrap();
        }
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();
        let digest_all = persistent_prover.digest();

        // Remove key2
        let op_remove = Operation::Remove(key2.clone());
        persistent_prover
            .perform_one_operation(&op_remove)
            .unwrap();
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();

        // Verify key2 is gone
        assert_eq!(persistent_prover.unauthenticated_lookup(&key2), None);
        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key1),
            Some(value.clone())
        );
        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key3),
            Some(value.clone())
        );

        // Rollback to before removal
        persistent_prover.rollback(&digest_all).unwrap();

        // Verify key2 is back
        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key2),
            Some(value)
        );
    }

    #[test]
    fn test_large_tree_persistence() {
        let (_temp_dir, storage) = create_test_storage(32, Some(8));
        let resolver = DiskBackedAVLStorage::get_resolver();
        let tree = AVLTree::new(resolver, 32, Some(8));
        let prover = BatchAVLProver::new(tree, true);

        let mut persistent_prover =
            PersistentBatchAVLProver::new(prover, storage, vec![]).unwrap();

        // Insert 1000 keys
        for i in 1..1001u32 {
            let mut key = vec![0u8; 32];
            key[28..32].copy_from_slice(&i.to_be_bytes());
            let key = Bytes::from(key);
            let value = Bytes::from((i as u64).to_be_bytes().to_vec());

            let op = Operation::Insert(KeyValue {
                key: key.clone(),
                value: value.clone(),
            });
            persistent_prover.perform_one_operation(&op).unwrap();
        }

        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();
        let digest = persistent_prover.digest();

        // Verify some random keys
        for i in [1u32, 100, 500, 1000] {
            let mut key = vec![0u8; 32];
            key[28..32].copy_from_slice(&i.to_be_bytes());
            let key = Bytes::from(key);
            let expected_value = Bytes::from((i as u64).to_be_bytes().to_vec());

            assert_eq!(
                persistent_prover.unauthenticated_lookup(&key),
                Some(expected_value)
            );
        }

        // Verify tree height is reasonable (should be around log2(1000) = ~10)
        let height = persistent_prover.height();
        assert!(height >= 10 && height <= 20, "Height: {}", height);

        // Verify digest is stable
        assert_eq!(persistent_prover.digest(), digest);
    }

    #[test]
    fn test_update_long_by_operation() {
        let (_temp_dir, storage) = create_test_storage(32, Some(8));
        let resolver = DiskBackedAVLStorage::get_resolver();
        let tree = AVLTree::new(resolver, 32, Some(8));
        let prover = BatchAVLProver::new(tree, true);

        let mut persistent_prover =
            PersistentBatchAVLProver::new(prover, storage, vec![]).unwrap();

        let key1 = Bytes::from(vec![1u8; 32]);

        // Increment non-existing key (should create it)
        let op1 = Operation::UpdateLongBy(KeyDelta {
            key: key1.clone(),
            delta: 10,
        });
        persistent_prover.perform_one_operation(&op1).unwrap();
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();

        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key1),
            Some(Bytes::from(10i64.to_be_bytes().to_vec()))
        );

        // Increment existing key
        let op2 = Operation::UpdateLongBy(KeyDelta {
            key: key1.clone(),
            delta: 5,
        });
        persistent_prover.perform_one_operation(&op2).unwrap();
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();

        assert_eq!(
            persistent_prover.unauthenticated_lookup(&key1),
            Some(Bytes::from(15i64.to_be_bytes().to_vec()))
        );

        // Decrement to zero (should remove key)
        let op3 = Operation::UpdateLongBy(KeyDelta {
            key: key1.clone(),
            delta: -15,
        });
        persistent_prover.perform_one_operation(&op3).unwrap();
        persistent_prover
            .generate_proof_and_update_storage(vec![])
            .unwrap();

        assert_eq!(persistent_prover.unauthenticated_lookup(&key1), None);
    }
}
